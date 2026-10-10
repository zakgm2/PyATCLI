"""
analyses.py
-----------
Every analysis PyAT offers, as a function of a recording and a plain parameter dict — so the same
analysis can be run from the app, from a script (`run_analysis`), from the command line
(`python -m PyATCLI run ...`), and replayed from a provenance record (reproduce.py), and all
four give the same numbers.

    result = pl.run_analysis("event_peth", cache, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
    result.tables["zscore"]        # a DataFrame
    result.parameters              # the parameters as run, defaults filled in — what a provenance record stores

`cache` is a recording as recording.py builds it (load_tdt_recording(...)["cache"], make_oxysoft_cache, ...).
Each table has the same columns, in the same order, as the CSV the app exports for that analysis, so
a script's output can be compared with an export column for column (reproduce.py does exactly that).

The analyses (describe_kinds() lists them with every parameter):

    zscore_window    z-score a window around one time (the Analysis menu's Z-Score)
    auc_window       area under the curve in a window (AUC)
    fft_window       spectrum and top peaks of a window (FFT)
    curve_fit        fit a model to the stretch between two times (Curve Fit)
    event_peth       every occurrence of one event aligned to it: trials, mean/SEM, AUC, peak, bins (Event PETH)
    peak_finder      statistically significant peaks (Find Significant Peaks)
    event_intervals  time between markers (Measure Intervals)
    process          the processed traces themselves (dF/F and the raw channels)

Group analysis is the one analysis that is not a function of one recording: it runs from a GroupSpec over
several (run_group_from_spec below, analysis/group/run.py's run_group_analysis, `python -m PyATCLI group`).

Not recording data: `parameters` and `summary` hold what was asked and a few numbers; the arrays are in the tables.
"""

import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .analysis import (
    compute_auc_from_trace, compute_auc_matrix, compute_auc_window, compute_event_zscore_peth, compute_fft_slice,
    compute_group_stats, compute_marker_intervals, compute_peri_event_from_trace, compute_peri_event_matrix,
    find_fft_peaks, find_peak_near_events, find_significant_peaks, fit_model_to_segment, get_zscore_slice,
    mean_channels, smooth_signal,
)
from .analysis.group import GroupSpec
from .analysis.group.report import trace_frame
from .analysis.group.run import GROUP_SETTINGS_FILENAME, run_group_analysis
from .markers import group_markers_by_name
from .models import CURVE_FIT_MODELS
from .numeric import as_float64
from .recording import normalized_signal

DEFAULT_WINDOW_S = 15.0           # the app's own default analysis window, each side of the point


class ParameterError(ValueError):
    """A parameter is missing, unknown, or has a value the analysis cannot use."""


class AnalysisError(RuntimeError):
    """The parameters were fine but the recording does not hold enough to run the analysis (a window past
    the end of the recording, an event with no usable trial, ...)."""


@dataclass
class AnalysisResult:
    kind: str
    parameters: dict
    tables: dict
    summary: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


@dataclass(frozen=True)
class Param:
    name: str
    type: str                       # float | int | str | bool | floats | ints | markers | groups | choice
    default: object = None
    required: bool = False
    doc: str = ""
    choices: tuple = ()


@dataclass(frozen=True)
class Analysis:
    kind: str
    summary: str
    params: tuple
    fn: object
    tables: dict = field(default_factory=dict)


_REGISTRY = {}

# What each analysis returns: its tables by name, and the columns of each that are ALWAYS there — a reader may ask for them by name and
# count on them. A table can have more columns than are listed (a column per time bin, a column per signal, columns that only some settings
# make), and more can be added at any time. Removing or renaming a listed column, or a table, breaks callers: that is a new PROTOCOL_VERSION
# (ops/__init__.py), and engine/contract.py and tests/test_engine_contract.py are what notice it.
TABLES = {
    "zscore_window": {
        "window": ["Channel", "time_from_center_s", "z_score"],
        "peaks": ["Channel", "Peak Z-score", "Peak time from center (s)"],
    },
    "auc_window": {
        "auc": ["Channel", "Total AUC", "Positive AUC", "Negative AUC", "Window Pre (s)", "Window Post (s)"],
        "window": ["Channel", "time_from_center_s", "value"],
    },
    "fft_window": {
        "peaks": ["Channel", "Peak Rank", "Frequency (Hz)", "BPM", "Power"],
        "spectrum": ["Channel", "frequency_hz", "power"],
    },
    "curve_fit": {
        "fit": ["Channel", "Model", "Parameter", "Value", "R2", "t1 (s)", "t2 (s)", "dt (s)"],
        "curve": ["Channel", "Model", "time_s", "value", "fit"],
        "context": ["Channel", "time_s", "value"],
    },
    "event_peth": {
        "zscore": ["time_from_event_s", "z_score"],
        "mean_sem": ["time_from_event_s", "mean", "sem"],
        "trials": ["Trial", "Trial Event Time (s)", "time_from_event_s", "z_score"],
        "auc": ["Trial", "Trial Event Time (s)", "Total AUC", "Positive AUC", "Negative AUC", "Checked"],
        "peak": ["Trial", "Trial Event Time (s)", "Peak Amplitude", "Time-to-Peak (s)", "Checked"],
        "bins": ["Trial", "Trial Event Time (s)", "Checked"],                       # (and a "Bin a-bs Mean" column per time bin)
    },
    "peak_finder": {                                                                 # which tables there are depends on `scope`
        "peaks": ["time_s", "z_score", "kind"],                                      # whole
        "alignment": ["Event Time (s)", "Found", "Latency (s)", "Z-score"],          # one_event, all_same_name, scan_all
        "scan": ["Event", "Occurrences", "Found", "Hit Rate", "Avg Z (found only)"],  # scan_all
    },
    "event_intervals": {
        "intervals": ["time_s", "store", "label", "phase", "dt_since_prev_same_store_s", "dt_since_prev_any_s"],
    },
    "process": {
        "trace": ["time_s"],                                                         # (and a column per signal)
    },
}


# What else the library writes as tables, besides an analysis's own: a group analysis's folder and an Export View's data. Declared like TABLES
# (the columns that are always there) and, for a folder, the file each table is written to; in the engine's contract (engine/contract.py) too.
VIEW_TABLES = {
    "data": [],                          # the x column (time_s for a TDT or Oxysoft recording), then a column per trace
    "markers": ["label", "color"],       # (and the x column first); only when there are markers in view
}


def describe_outputs():
    """{name: {"summary", "tables": {table: [the columns that are always there]}, "files": {table or role: file name in the folder}}} for what
    is written besides an analysis's tables: "group_analysis" (a group's results folder) and "view" (an Export View's tables)."""
    from .analysis.group.report import GROUP_TABLES, NOTES_FILENAME, REPORT_FILENAME, TABLE_FILES

    return {
        GROUP_KIND: {"summary": "a group analysis's results folder: every table as a CSV beside the settings it was run from, a report and its notes",
                     "tables": {name: list(cols) for name, cols in GROUP_TABLES.items()},
                     "files": {**TABLE_FILES, "settings": GROUP_SETTINGS_FILENAME, "report": REPORT_FILENAME, "notes": NOTES_FILENAME}},
        VIEW_KIND: {"summary": "an Export View's numbers: every sample in view, and the markers in view",
                    "tables": {name: list(cols) for name, cols in VIEW_TABLES.items()}, "files": {}},
    }


def analysis(kind, summary, params):
    def register(fn):
        _REGISTRY[kind] = Analysis(kind, summary, tuple(params), fn, {name: tuple(cols) for name, cols in TABLES.get(kind, {}).items()})
        return fn
    return register


def kinds():
    """The analyses that can be run, in the order they are listed above."""
    return tuple(_REGISTRY)


def describe_kinds():
    """{kind: {"summary", "parameters": [{"name", "type", "default", "required", "doc", "choices"}], "tables": {name: [the columns that are
    always there]}} — for help text, documentation and the engine's contract (engine/contract.py)."""
    return {
        k: {"summary": a.summary, "parameters": [
            {"name": p.name, "type": p.type, "default": p.default, "required": p.required, "doc": p.doc,
             "choices": list(p.choices)} for p in a.params],
            "tables": {name: list(cols) for name, cols in a.tables.items()}}
        for k, a in _REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def _coerce(p, value):
    try:
        if p.type == "float":
            v = float(value)
            if not math.isfinite(v):
                raise ValueError("not finite")
            return v
        if p.type == "int":
            if isinstance(value, float) and not value.is_integer():
                raise ValueError("not a whole number")
            return int(value)
        if p.type == "bool":
            if isinstance(value, str):
                if value.lower() in ("true", "yes", "1", "on"):
                    return True
                if value.lower() in ("false", "no", "0", "off"):
                    return False
                raise ValueError("not true/false")
            return bool(value)
        if p.type == "str":
            return str(value)
        if p.type == "floats":
            if isinstance(value, str):
                value = [v for v in re.split(r"[,\s]+", value.strip()) if v]
            vals = [float(v) for v in value]
            if not all(math.isfinite(v) for v in vals):
                raise ValueError("not finite")
            return vals
        if p.type == "groups":
            return {str(k): [float(t) for t in v] for k, v in dict(value).items()}
        if p.type == "markers":
            marks = [dict(m) for m in value]
            for m in marks:
                m["time"] = float(m["time"])
            return marks
        if p.type == "ints":
            if isinstance(value, str):
                value = [v for v in re.split(r"[,\s]+", value.strip()) if v]
            return [int(v) for v in value]
        if p.type == "choice":
            if value not in p.choices:
                raise ValueError(f"must be one of {', '.join(map(str, p.choices))}")
            return value
    except (TypeError, ValueError) as e:
        raise ParameterError(f"{p.name}: {e} (got {value!r})") from None
    raise ParameterError(f"{p.name}: unknown parameter type {p.type}")


def normalize_parameters(kind, parameters=None):
    """`parameters` checked against the analysis's parameter list, with defaults filled in. Raises
    ParameterError for a missing required parameter, an unknown name or a bad value."""
    if kind not in _REGISTRY:
        raise ParameterError(f"Unknown analysis {kind!r}; expected one of {', '.join(_REGISTRY)}")
    spec = {p.name: p for p in _REGISTRY[kind].params}
    given = dict(parameters or {})
    unknown = sorted(set(given) - set(spec))
    if unknown:
        raise ParameterError(f"{kind}: unknown parameter(s) {', '.join(unknown)}; valid: {', '.join(spec)}")
    out = {}
    for name, p in spec.items():
        if name in given and given[name] is not None:
            out[name] = _coerce(p, given[name])
        elif p.required:
            raise ParameterError(f"{kind}: {name} is required ({p.doc})")
        else:
            out[name] = p.default
    return out


def run_analysis(kind, cache, parameters=None, *, store_labels=None, progress=None):
    """Runs one analysis. `cache` is a recording (see the module docstring);
    `store_labels` the store renames in effect, for analyses that name events. Returns an AnalysisResult."""
    p = normalize_parameters(kind, parameters)
    ctx = _Context(cache, store_labels or {}, progress)
    tables, summary, notes = _REGISTRY[kind].fn(ctx, p)
    p.update(ctx.resolved)                  # what the analysis looked up (event times, markers): the record is self-sufficient
    result = AnalysisResult(kind, p, tables, summary, notes)
    return result


@dataclass
class _Context:
    cache: object
    store_labels: dict
    progress: object
    resolved: dict = field(default_factory=dict)      # parameters an analysis fills in from the recording


# ---------------------------------------------------------------------------
# What an analysis looks at
# ---------------------------------------------------------------------------

def signal_channels(cache, signal=None):
    """The (label, y) traces the click-triggered analyses work on: Oxysoft's mean O2Hb / HHb (/ tHb); for a TDT
    recording the one signal named by `signal` (a key of cache["signals"], default the first); for a recording
    with only a corrected trace, that; for a generic table its first column."""
    if cache.get("source") == "Oxysoft":
        out = [("Mean O2Hb", mean_channels(cache["o2hb"])), ("Mean HHb", mean_channels(cache["hhb"]))]
        if "thb" in cache:
            out.append(("Mean tHb", mean_channels(cache["thb"])))
        return out
    if "corr" in cache or "raw" in cache or cache.get("signals"):
        signals = cache.get("signals")
        if signals:
            key = signal if signal in signals else next(iter(signals))
            return [(signals[key]["label"], signals[key]["y"])]
        return [("dF/F (corrected)", cache["corr"] if "corr" in cache else cache["raw"])]
    return [("Signal", next(iter(cache["y_columns"].values())))]


def _frame(columns, rows):
    return pd.DataFrame(rows, columns=columns)


_SIGNAL = Param("signal", "str", "normalized", doc="which of the recording's signals (TDT: normalized, main_driver, isosbestic)")
_PRE = Param("pre_s", "float", DEFAULT_WINDOW_S, doc="seconds before the point")
_POST = Param("post_s", "float", DEFAULT_WINDOW_S, doc="seconds after the point")


# ---------------------------------------------------------------------------
# Click-triggered window analyses
# ---------------------------------------------------------------------------

@analysis("zscore_window", "z-score a window around one time against its own pre-event baseline",
          [Param("center_s", "float", required=True, doc="the time the window is centred on"), _PRE, _POST, _SIGNAL])
def _zscore_window(ctx, p):
    cache, center = ctx.cache, p["center_s"]
    x, fs = cache["x"], cache["fs"]
    rows, peaks, notes = [], [], []
    for label, y in signal_channels(cache, p["signal"]):
        slice_x, z = get_zscore_slice(x, smooth_signal(y, fs), center, pre=p["pre_s"], post=p["post_s"])
        if z is None or len(z) == 0:
            notes.append(f"{label}: no samples in this window")
            continue
        i = int(np.argmax(np.abs(z)))
        peaks.append([label, float(z[i]), float(slice_x[i] - center)])
        rows += [[label, float(t - center), float(v)] for t, v in zip(slice_x, z)]
    if not peaks:
        raise AnalysisError("Not enough data in this window to compute a Z-score.")
    tables = {"window": _frame(["Channel", "time_from_center_s", "z_score"], rows),
              "peaks": _frame(["Channel", "Peak Z-score", "Peak time from center (s)"], peaks)}
    return tables, {"peak_z": {r[0]: r[1] for r in peaks}}, notes


@analysis("auc_window", "area under the curve in a window, split by sign",
          [Param("center_s", "float", required=True, doc="the time the window is centred on"), _PRE, _POST, _SIGNAL])
def _auc_window(ctx, p):
    cache, center = ctx.cache, p["center_s"]
    rows, window = [], []
    for label, y in signal_channels(cache, p["signal"]):
        r = compute_auc_window(cache["x"], y, center, p["pre_s"], p["post_s"])
        rows.append([label, r["total_auc"], r["positive_auc"], r["negative_auc"], p["pre_s"], p["post_s"]])
        window += [[label, float(t - center), float(v)] for t, v in zip(r["seg_x"], r["seg_y"])]
    tables = {"auc": _frame(["Channel", "Total AUC", "Positive AUC", "Negative AUC", "Window Pre (s)", "Window Post (s)"], rows),
              "window": _frame(["Channel", "time_from_center_s", "value"], window)}
    return tables, {"total_auc": {r[0]: r[1] for r in rows}}, []


@analysis("fft_window", "power spectrum of a window (mean and linear trend removed) and its top peaks",
          [Param("center_s", "float", required=True, doc="the time the window is centred on"), _PRE, _POST, _SIGNAL])
def _fft_window(ctx, p):
    cache = ctx.cache
    peaks, spectrum = [], []
    for label, y in signal_channels(cache, p["signal"]):
        freqs, power, _, _ = compute_fft_slice(cache["x"], y, p["center_s"], cache["fs"], pre=p["pre_s"], post=p["post_s"])
        found = find_fft_peaks(freqs, power) if len(freqs) > 0 else []
        peaks += [[label, i + 1, pk["freq_hz"], pk["bpm"], pk["power"]] for i, pk in enumerate(found)]
        spectrum += [[label, float(f), float(w)] for f, w in zip(freqs, power)]
    tables = {"peaks": _frame(["Channel", "Peak Rank", "Frequency (Hz)", "BPM", "Power"], peaks),
              "spectrum": _frame(["Channel", "frequency_hz", "power"], spectrum)}
    return tables, {"n_peaks": len(peaks)}, []


def curve_model_ids():
    """{short id: registry name} for the curve-fit models, e.g. {"linear": "Linear  (y = mx + b)", ...}."""
    return {re.sub(r"[^a-z0-9]+", "_", name.split("(")[0].lower()).strip("_"): name for name in CURVE_FIT_MODELS}


def resolve_curve_model(name):
    """A curve-fit model's registry name from its registry name or short id ("exponential decay",
    "exponential_decay", "Exponential Decay" all work). Raises ParameterError for an unknown one."""
    if name in CURVE_FIT_MODELS:
        return name
    ids = curve_model_ids()
    key = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if key in ids:
        return ids[key]
    raise ParameterError(f"model: unknown curve-fit model {name!r}; expected one of {', '.join(ids)}")


@analysis("curve_fit", "fit a model to the stretch of the signal between two times",
          [Param("t1_s", "float", required=True, doc="start of the stretch (snapped to the nearest sample)"),
           Param("t2_s", "float", required=True, doc="end of the stretch"),
           Param("model", "str", "linear", doc="one of the curve_model_ids()"), _SIGNAL,
           Param("window_s", "float", 30.0, doc="seconds of the signal around the stretch that the figure shows for context "
                                                "(the `context` table); at least 1")])
def _curve_fit(ctx, p):
    cache = ctx.cache
    model_name = resolve_curve_model(p["model"])
    model_fn, p0_fn, pnames = CURVE_FIT_MODELS[model_name]
    x = cache["x"]
    i1, i2 = sorted((int(np.argmin(np.abs(x - p["t1_s"]))), int(np.argmin(np.abs(x - p["t2_s"])))))
    t1, t2 = float(x[i1]), float(x[i2])
    # The wider view the figure shows around the stretch, as the app's Curve Fit window does: `window_s` seconds
    # centred on the stretch, never narrower than the stretch itself.
    half = max(1.0, p["window_s"]) / 2
    t_mid = (t1 + t2) / 2
    view_start = min(max(float(x[0]), t_mid - half), t1)
    view_end = max(min(float(x[-1]), t_mid + half), t2)
    w_start = int(np.searchsorted(x, view_start))
    w_end = min(int(np.searchsorted(x, view_end)), len(x) - 1)
    rows, curve, context, notes, r2s = [], [], [], [], {}
    for label, y in signal_channels(cache, p["signal"]):
        y = as_float64(y)
        x_seg, y_seg = x[i1:i2 + 1], y[i1:i2 + 1]
        if len(x_seg) < 4:
            notes.append(f"{label}: fewer than 4 samples between the two times")
            continue
        context += [[label, float(tt), float(yy)] for tt, yy in zip(x[w_start:w_end + 1], y[w_start:w_end + 1])]
        res = fit_model_to_segment(x_seg, y_seg, model_fn, p0_fn)
        if not res["success"]:
            notes.append(f"{label}: fit failed: {res['error']}")
            curve += [[label, model_name, float(tt), float(yy), float("nan")] for tt, yy in zip(x_seg, y_seg)]   # the stretch, no fit
            continue
        r2s[label] = float(res["r2"])
        rows += [[label, model_name, n, float(v), float(res["r2"]), t1, t2, t2 - t1] for n, v in zip(pnames, res["popt"])]
        curve += [[label, model_name, float(tt), float(yy), float(ff)] for tt, yy, ff in zip(x_seg, y_seg, res["y_fit"])]
    tables = {"fit": _frame(["Channel", "Model", "Parameter", "Value", "R2", "t1 (s)", "t2 (s)", "dt (s)"], rows),
              "curve": _frame(["Channel", "Model", "time_s", "value", "fit"], curve),
              "context": _frame(["Channel", "time_s", "value"], context)}
    return tables, {"r2": r2s, "model": model_name, "view": [view_start, view_end]}, notes


# ---------------------------------------------------------------------------
# Event-locked analyses
# ---------------------------------------------------------------------------

def _event_times(ctx, event, given):
    if given:
        return sorted(float(t) for t in given)
    groups = group_markers_by_name(ctx.cache, ctx.store_labels)
    if event not in groups:
        shown = ", ".join(sorted(groups)[:20]) + (" ..." if len(groups) > 20 else "")
        raise ParameterError(f"event: no event named {event!r} in this recording; it has: {shown or 'none'}")
    return groups[event]


@analysis("event_peth", "align every occurrence of one event: trials, mean/SEM, AUC, peak and time bins per trial and for the group",
          [Param("event", "str", required=True, doc="the event's name as listed (e.g. L1P¹, or a note's text)"),
           Param("pre_s", "float", DEFAULT_WINDOW_S, doc="seconds before each event"),
           Param("post_s", "float", DEFAULT_WINDOW_S, doc="seconds after each event"),
           Param("bin_s", "float", 1.0, doc="width of the post-event mean-time bins"),
           Param("selected_trials", "ints", None, doc="1-based trial numbers to include in the group statistics (default: all)"),
           Param("event_times", "floats", None, doc="use these event times instead of looking the event up (what a record stores)")])
def _event_peth(ctx, p):
    cache = ctx.cache
    pre, post, bin_s = p["pre_s"], p["post_s"], max(0.01, p["bin_s"])
    if pre <= 0 or post <= 0:
        raise ParameterError("pre_s and post_s must be positive")
    times = _event_times(ctx, p["event"], p["event_times"])
    ctx.resolved["event_times"] = [float(t) for t in times]
    _, _, y = normalized_signal(cache)
    clean = smooth_signal(y, cache["fs"])
    res = compute_event_zscore_peth(cache["x"], clean, times, pre, post, progress=ctx.progress)
    matrix, axis, trial_times = res["trial_matrix"], res["time_axis"], res["trial_event_times"]
    n_total = matrix.shape[0]
    if n_total == 0:
        raise AnalysisError(f"No usable trials for '{p['event']}' — all occurrences were too close to the "
                            "start/end of the recording for the current window.")
    wanted = p["selected_trials"]
    if wanted is None:
        selected = list(range(n_total))
    else:
        bad = [n for n in wanted if not 1 <= n <= n_total]
        if bad:
            raise ParameterError(f"selected_trials: trial(s) {bad} do not exist; there are {n_total} usable trials")
        selected = sorted({n - 1 for n in wanted})
    sub = matrix[selected] if selected else matrix[:0]
    mean, sem = compute_group_stats(sub)
    per_auc = compute_auc_matrix(axis, matrix)["per_trial"]
    per_peri = compute_peri_event_matrix(axis, matrix, post, bin_s)["per_trial"]
    g_auc = compute_auc_from_trace(axis, mean) if selected else {"total_auc": 0.0, "positive_auc": 0.0, "negative_auc": 0.0}
    g_peri = compute_peri_event_from_trace(axis, mean, post, bin_s) if selected else {"peak_amplitude": 0.0, "latency": 0.0, "mean_bins": []}
    checked = set(selected)
    GROUP = "GROUP (checked only)"

    trials = [[i + 1, float(trial_times[i]), float(t), float(v)] for i in selected for t, v in zip(axis, matrix[i])]
    auc_rows = [[i + 1, float(t), a["total_auc"], a["positive_auc"], a["negative_auc"], i in checked]
                for i, (t, a) in enumerate(zip(trial_times, per_auc))]
    auc_rows.append([GROUP, None, g_auc["total_auc"], g_auc["positive_auc"], g_auc["negative_auc"], None])
    peak_rows = [[i + 1, float(t), q["peak_amplitude"], q["latency"], i in checked] for i, (t, q) in enumerate(zip(trial_times, per_peri))]
    peak_rows.append([GROUP, None, g_peri["peak_amplitude"], g_peri["latency"], None])
    edges = per_peri[0]["mean_bins"]
    bin_cols = [f"Bin {b['bin_start']:.1f}-{b['bin_end']:.1f}s Mean" for b in edges]
    bin_rows = [[i + 1, float(t)] + [b["mean"] for b in q["mean_bins"]] + [i in checked] for i, (t, q) in enumerate(zip(trial_times, per_peri))]
    g_bins = [b["mean"] for b in g_peri["mean_bins"]]
    bin_rows.append([GROUP, None] + g_bins + [None] * (len(edges) - len(g_bins)) + [None])

    tables = {
        "zscore": _frame(["time_from_event_s", "z_score"], list(zip(map(float, axis), map(float, mean)))),
        "mean_sem": _frame(["time_from_event_s", "mean", "sem"], list(zip(map(float, axis), map(float, mean), map(float, sem)))),
        "trials": _frame(["Trial", "Trial Event Time (s)", "time_from_event_s", "z_score"], trials),
        "auc": _frame(["Trial", "Trial Event Time (s)", "Total AUC", "Positive AUC", "Negative AUC", "Checked"], auc_rows),
        "peak": _frame(["Trial", "Trial Event Time (s)", "Peak Amplitude", "Time-to-Peak (s)", "Checked"], peak_rows),
        "bins": _frame(["Trial", "Trial Event Time (s)"] + bin_cols + ["Checked"], bin_rows),
    }
    summary = {"n_trials": n_total, "n_selected": len(selected), "n_events": len(times),
               "group_peak": g_peri["peak_amplitude"], "group_latency_s": g_peri["latency"], "group_auc": g_auc["total_auc"]}
    notes = [f"{len(times) - n_total} of {len(times)} occurrences were too close to the recording's edges and were skipped"] \
        if n_total < len(times) else []
    return tables, summary, notes


@analysis("peak_finder", "statistically significant peaks: across the whole recording, or near events",
          [Param("scope", "choice", "whole", choices=("whole", "one_event", "all_same_name", "scan_all"),
                 doc="whole: peaks anywhere; one_event / all_same_name: is there a peak near the event(s); scan_all: every event type"),
           Param("z_threshold", "float", 5.0, doc="minimum z-score for a peak to count"),
           Param("min_distance_s", "float", 1.0, doc="minimum spacing between peaks (scope whole)"),
           Param("include_troughs", "bool", False, doc="also consider negative dips"),
           Param("event", "str", None, doc="the event's name (scope all_same_name)"),
           Param("event_time", "float", None, doc="the event's time (scope one_event)"),
           Param("event_times", "floats", None, doc="use these event times instead of looking the event up"),
           Param("event_groups", "groups", None, doc="scope scan_all: {event name: times} to scan instead of the recording's own "
                                                      "(what a record stores)"),
           _PRE, _POST])
def _peak_finder(ctx, p):
    cache = ctx.cache
    _, _, y = normalized_signal(cache)
    clean = smooth_signal(y, cache["fs"])
    x, scope, thr, tr = cache["x"], p["scope"], p["z_threshold"], p["include_troughs"]
    ALIGN = ["Event Time (s)", "Found", "Latency (s)", "Z-score"]

    def align_rows(results):
        return [[r["event_time"], r["found"], r["latency"] if r["found"] else None, r["z_score"] if r["found"] else None] for r in results]

    if scope == "whole":
        peaks = find_significant_peaks(x, clean, z_threshold=thr, min_distance_sec=p["min_distance_s"], include_troughs=tr)
        return ({"peaks": _frame(["time_s", "z_score", "kind"], [[q["time"], q["z_score"], q["kind"]] for q in peaks])},
                {"n_peaks": len(peaks)}, [])
    if scope == "one_event":
        if p["event_time"] is None and not p["event_times"]:
            raise ParameterError("peak_finder: scope one_event needs event_time")
        times = p["event_times"] or [p["event_time"]]
    elif scope == "all_same_name":
        if not p["event"] and not p["event_times"]:
            raise ParameterError("peak_finder: scope all_same_name needs event")
        times = _event_times(ctx, p["event"], p["event_times"])
    else:
        times = None
    if times is not None:
        ctx.resolved["event_times"] = [float(t) for t in times]
        results = find_peak_near_events(x, clean, times, p["pre_s"], p["post_s"], z_threshold=thr, include_troughs=tr, progress=ctx.progress)
        return ({"alignment": _frame(ALIGN, align_rows(results))},
                {"n_events": len(results), "n_found": sum(1 for r in results if r["found"])}, [])
    groups = p["event_groups"] if p["event_groups"] is not None else group_markers_by_name(cache, ctx.store_labels)
    ctx.resolved["event_groups"] = {name: [float(t) for t in times] for name, times in groups.items()}
    by_event = {name: find_peak_near_events(x, clean, t, p["pre_s"], p["post_s"], z_threshold=thr, include_troughs=tr)
                for name, t in groups.items()}
    scan = []
    for name, results in by_event.items():
        found = [r for r in results if r["found"]]
        scan.append([name, len(results), len(found), len(found) / len(results) if results else 0.0,
                     sum(r["z_score"] for r in found) / len(found) if found else None])
    scan.sort(key=lambda r: r[3], reverse=True)
    detail = [[name] + row for name, results in by_event.items() for row in align_rows(results)]
    return ({"scan": _frame(["Event", "Occurrences", "Found", "Hit Rate", "Avg Z (found only)"], scan),
             "alignment": _frame(["Event"] + ALIGN, detail)},
            {"n_event_types": len(by_event)}, [])


@analysis("event_intervals", "the time between markers: since the previous of the same store, and of any",
          [Param("markers", "markers", None, doc="the markers to measure, [{time, store, label, phase}, ...] "
                                                 "(default: the recording's own markers; a record stores what was measured)")])
def _event_intervals(ctx, p):
    markers = p["markers"] if p["markers"] is not None else ctx.cache.get("markers", [])
    if not markers:
        raise AnalysisError("No markers currently on the plot to measure.")
    rows = compute_marker_intervals(markers)
    ctx.resolved["markers"] = [{k: m.get(k) for k in ("time", "store", "label", "phase")} for m in markers]
    table = _frame(["time_s", "store", "label", "phase", "dt_since_prev_same_store_s", "dt_since_prev_any_s"],
                   [[r["time"], r["store"], r["label"], r["phase"], r["dt_store"], r["dt_global"]] for r in rows])
    return {"intervals": table}, {"n_markers": len(rows)}, []


# ---------------------------------------------------------------------------
# Whole-recording outputs
# ---------------------------------------------------------------------------

@analysis("process", "the processed traces themselves: dF/F and the raw per-wavelength channels",
          [Param("signals", "str", None, doc="comma-separated signal keys (default: every one the recording offers)")])
def _process(ctx, p):
    cache = ctx.cache
    signals = cache.get("signals")
    if signals:
        keys = [k.strip() for k in p["signals"].split(",")] if p["signals"] else list(signals)
        missing = [k for k in keys if k not in signals]
        if missing:
            raise ParameterError(f"signals: no signal(s) {missing}; the recording has {list(signals)}")
        cols = {k: as_float64(signals[k]["y"]) for k in keys}
    elif cache.get("source") == "Oxysoft":
        cols = {f"O2Hb {i + 1}": row for i, row in enumerate(cache["o2hb"])}
        cols.update({f"HHb {i + 1}": row for i, row in enumerate(cache["hhb"])})
    elif "y_columns" in cache:
        cols = dict(cache["y_columns"])
    else:
        cols = {"signal": normalized_signal(cache)[2]}
    table = pd.DataFrame({"time_s": as_float64(cache["x"]), **cols})
    return {"trace": table}, {"n_samples": len(table), "columns": list(cols)}, []


def run_group_from_spec(spec, out_dir=None, progress=None):
    """The group analysis for a GroupSpec or its dict (what analysis_settings.json holds), as an AnalysisResult
    whose tables are the group's result tables. Writes the tables, report to `out_dir` if given."""
    spec = spec if isinstance(spec, GroupSpec) else GroupSpec.from_dict(spec)
    problems = spec.problems()
    if problems:
        raise ParameterError("The group settings are not usable: " + " ".join(problems))
    results = run_group_analysis(spec, out_dir=out_dir, progress=progress)
    tables = {k: v for k, v in results.frames().items() if len(v)}
    traces = trace_frame(results) if isinstance(getattr(results, "traces", None), dict) else None
    if traces is not None and len(traces):
        tables["mean_traces"] = traces                           # what mean_traces.csv holds; PyAT draws it
    notes = [n["text"] for n in results.notes]
    result = AnalysisResult("group_analysis", spec.to_dict(), tables, {"files": results.files}, notes)
    return result, results


GROUP_KIND = "group_analysis"     # the kind a provenance record of a group analysis carries
VIEW_KIND = "view"                # the kind a provenance record of an Export View carries
VIEW_PARAMETERS = ("x_range", "y_range", "signals", "show_grid", "attributes", "traces")


def run_view(cache, parameters=None, *, store_labels=None):
    """The numbers of an Export View, as an AnalysisResult: `tables["data"]` is the CSV beside the figure — every sample in view, the
    time and a column per trace — and `tables["markers"]` the markers in view (when there are any). `parameters` are a view's
    (viewplan.ViewPlan.parameters(): "x_range", "y_range", "signals", "show_grid", "attributes"); none shows all of the recording.
    "attributes" is whatever the viewer drew the view with (a title, labels, colours): kept as it is, it changes no number.
    This is what `reproduce` runs for a "view" record: nothing is drawn."""
    from . import viewplan

    p = dict(parameters or {})
    unknown = sorted(set(p) - set(VIEW_PARAMETERS))
    if unknown:
        raise ParameterError(f"a view has no parameter {', '.join(map(repr, unknown))}; its parameters are: {', '.join(VIEW_PARAMETERS)}")
    view = viewplan.plan_view(cache, signals=p.get("signals"), xlim=p.get("x_range"), ylim=p.get("y_range"), show_grid=p.get("show_grid", True),
                              store_labels=store_labels, attributes=p.get("attributes"))
    tables = {"data": view.data_table()}
    if view.markers:
        tables["markers"] = view.markers_table()
    return AnalysisResult(VIEW_KIND, view.parameters(), tables, {"samples": len(tables["data"]), "traces": [t.csv_name for t in view.traces]}, [])
