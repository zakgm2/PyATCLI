"""
provenance.py
-------------
How a result was produced, as data: the software that ran (versions of this library, Python, the
numerical packages), what the pipeline did that no caller chose (the filter, the bleaching model, the
z-score baseline rule), which recording went in and in what state (splices, markers, the motion-correction
method), and the parameters of the analysis itself. Written next to every export as
<export name>.provenance.json, and enough — with the recording — to run the analysis again
(PyATCLI.reproduce, and the `python -m PyATCLI` command line).

A record is a plain JSON-able dict:

    schema     SCHEMA_VERSION of this layout
    kind       what was produced ("event_peth", "zscore_window", "group_analysis", "figure", ...)
    created    ISO date-time
    software   software_versions(): this library, Python, platform, the numerical packages, the app
    source     file_fingerprint() of the recording's folder / file: names, sizes, hashes
    recording  recording_state(): its type, length, splices, markers, motion-correction method
    pipeline   pipeline_parameters(): the processing no caller chose, from the constants the code uses
    analysis   {"type": kind, "parameters": {...}} — what was asked of that analysis
    outputs    the files written, each with its size and hash
    notes      anything worth saying in words

Nothing in a record is recording *data*: no samples, no trace values. A record holds names, counts,
times of events, parameters and hashes. File paths are included (they are what lets the analysis be run
again) and can be left out with `portable=True` in file_fingerprint.
"""

import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime

from ._version import resolve_version

SCHEMA_VERSION = 1
SIDECAR_SUFFIX = ".provenance.json"

# A file larger than this is identified by its name, size and date rather than hashed: a TDT block's
# .tev can be gigabytes, and its small index/header files already say whether it is the same block.
HASH_LIMIT_BYTES = 16 * 2 ** 20

# Distributions whose versions can change a number (or how a figure looks); reported when installed.
PACKAGES = ("numpy", "scipy", "pandas", "statsmodels", "scikit-learn", "tdt", "openpyxl", "tqdm",
            "PyQt6", "pyqtgraph")


# ---------------------------------------------------------------------------
# Software
# ---------------------------------------------------------------------------

def library_version():
    """This library's running version (see _version.py for why that is not always the installed one)."""
    return resolve_version()


def software_versions(app=None, packages=PACKAGES):
    """What is running: Python, the OS, this library, the numerical packages that are installed, and the
    program using it (`app`, e.g. {"name": "PyAT", "version": "3.2.0"}). Never raises."""
    from importlib import metadata

    pkgs = {}
    for name in packages:
        try:
            pkgs[name] = metadata.version(name)
        except Exception:                                    # noqa: BLE001 - not installed / unreadable: just absent
            pass
    try:
        plat = platform.platform()
    except Exception:                                        # noqa: BLE001
        plat = sys.platform
    out = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "bits": 64 if sys.maxsize > 2 ** 32 else 32,
        "platform": plat,
        "frozen": bool(getattr(sys, "frozen", False)),
        "library": library_version(),
        "packages": pkgs,
    }
    if app:
        out["app"] = dict(app)
    return out


def check_environment(record, packages=("numpy", "scipy", "pandas", "statsmodels", "scikit-learn")):
    """Differences between the software a record was produced with and the software running now that
    could matter, as plain sentences ([] = none worth mentioning). Library version differences and a
    different Python minor version are named, and so is a different major.minor of a numerical package:
    results can differ in the last digits across those, rarely more."""
    then = record.get("software", {})
    now = software_versions()
    out = []
    if then.get("library") and then["library"] != now["library"]:
        out.append(f"PyATCLI was {then['library']} when this was produced; this is {now['library']}.")

    def minor(v):
        return ".".join(str(v).split(".")[:2])

    if then.get("python") and minor(then["python"]) != minor(now["python"]):
        out.append(f"Python was {then['python']} when this was produced; this is {now['python']}.")
    for name in packages:
        a, b = then.get("packages", {}).get(name), now["packages"].get(name)
        if a and b and minor(a) != minor(b):
            out.append(f"{name} was {a} when this was produced; this is {b} — numbers can differ in the last digits.")
    return out


# ---------------------------------------------------------------------------
# What the pipeline does that no caller chose
# ---------------------------------------------------------------------------

def pipeline_parameters(regression_method="ols"):
    """The processing a TDT recording goes through and the analysis conventions, read from the constants
    the code itself uses (so this cannot drift from what ran)."""
    from .sources import processing_TDT as P
    from .analysis import shared, zscore_peth

    if regression_method not in P.REGRESSION_METHODS:
        raise ValueError(f"Unknown regression method {regression_method!r}; expected one of {P.REGRESSION_METHODS}")
    return {
        "dtype": "float64",
        "motion_correction": {
            "method": regression_method,
            "signal_stream": "the stream whose name contains '465'",
            "reference_stream": "the stream whose name contains '415' (none: no motion correction)",
            "fit": "linear regression of the 415 reference onto the 465 signal; the fitted 415 is subtracted",
            "ransac": {"residual_threshold": "3 x robust noise (MAD of an OLS fit's residuals / 0.6745)", "random_state": 0},
            "huber": {"epsilon": 1.35, "fitted on median/MAD-standardised data": True},
        },
        "photobleaching": {
            "model": "double exponential fitted to the 465 stream",
            "fit": "scipy.optimize.curve_fit on the samples above the stream's median, in units of its own maximum",
            "fallback": "log-linear trend if the fit fails",
        },
        "dff": {
            "with_reference": "motion-corrected residual / bleaching baseline",
            "without_reference": "(465 - bleaching baseline) / bleaching baseline",
        },
        "denoise": {
            "filter": "Butterworth low-pass, zero-phase (scipy filtfilt)",
            "cutoff_hz": P.DENOISE_CUTOFF_HZ,
            "order": P.DENOISE_ORDER,
        },
        "smoothing": {
            "method": "moving average (applied before event-locked analyses)",
            "window_s": shared.SMOOTH_WINDOW_SEC,
        },
        "zscore": {
            "baseline": "the samples before the event (t < event time)",
            "baseline_artifact_guard": f"samples beyond {zscore_peth.BASELINE_ROBUST_SD:g} robust SDs (1.4826 x MAD) "
                                       "of the baseline median are pulled in before its mean and SD are taken",
            "robust_sd_limit": zscore_peth.BASELINE_ROBUST_SD,
        },
    }


# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------

def _sha256_file(path, limit=None):
    limit = HASH_LIMIT_BYTES if limit is None else limit            # looked up now, so changing the constant takes effect
    try:
        if os.path.getsize(path) > limit:
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _iso(ts):
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def file_fingerprint(path, max_files=200, portable=False):
    """Identifies a recording's folder or file without reading gigabytes: for a folder, every file
    directly inside it (not its JSON saves/ or other subfolders) with its size and, if small enough, its
    SHA-256; for a file, the same for it. `digest` condenses the list, so two records can be compared
    at a glance.

    `portable=True` leaves the absolute path out (the name stays), for a record that will leave the
    machine; running the analysis again then needs the recording's location given explicitly."""
    full = os.path.abspath(path)
    out = {"name": os.path.basename(full.rstrip("/\\"))}
    if not portable:
        out["path"] = full
    if os.path.isdir(full):
        files, total, newest = [], 0, 0.0
        for name in sorted(os.listdir(full)):
            p = os.path.join(full, name)
            if not os.path.isfile(p) or len(files) >= max_files:
                continue
            size = os.path.getsize(p)
            total += size
            newest = max(newest, os.path.getmtime(p))
            files.append({"name": name, "size": size, "sha256": _sha256_file(p)})
        out.update(kind="folder", n_files=len(files), size_bytes=total, files=files)
        if newest:
            out["modified"] = _iso(newest)
    elif os.path.isfile(full):
        size = os.path.getsize(full)
        files = [{"name": out["name"], "size": size, "sha256": _sha256_file(full)}]
        out.update(kind="file", n_files=1, size_bytes=size, files=files, modified=_iso(os.path.getmtime(full)))
    else:
        out.update(kind="missing", n_files=0, size_bytes=0, files=[])
        return out
    lines = "\n".join(f"{f['name']}|{f['size']}|{f['sha256'] or '-'}" for f in files)
    out["digest"] = hashlib.sha256(lines.encode("utf-8")).hexdigest()
    return out


def markers_summary(markers, detected_markers=None):
    """How many markers there are and a digest of them (time to the microsecond, label, store, phase;
    order does not matter), so a record can say which markers an analysis saw without holding them."""
    def digest(ms):
        keys = sorted((round(float(m["time"]), 6), str(m.get("label", "")), str(m.get("store", "")),
                       str(m.get("phase", ""))) for m in ms)
        return hashlib.sha256(json.dumps(keys).encode("utf-8")).hexdigest()[:16]

    out = {"n_user": len(markers), "user_digest": digest(markers)}
    if detected_markers is not None:
        out.update(n_detected=len(detected_markers), detected_digest=digest(detected_markers))
    return out


def recording_state(cache, regression_method=None, splices=(), store_labels=None, signal=None):
    """The state of a loaded recording that decides what an analysis sees: its type, length and sampling
    rate, the motion-correction method, every splice applied (in order), the markers (counts and
    digests, see markers_summary), the store renames in effect, and which signal was analysed. `cache` is
    the dict of recording.py."""
    x = cache["x"]
    store = cache.get("store")
    base_store = re.sub(r" \[(?:cut|kept) [^\]]*\]$", "", store) if isinstance(store, str) else store
    out = {
        "source_type": cache.get("source"),
        "store": store,
        "base_store": base_store,
        "n_samples": int(len(x)),
        "fs": float(cache["fs"]) if cache.get("fs") is not None else None,
        "duration_s": float(x[-1] - x[0]) if len(x) else 0.0,
        "splices": [dict(s) for s in splices],
        "markers": markers_summary(cache.get("markers", []), cache.get("detected_markers")),
    }
    if cache.get("source") == "Generic":
        # which table and columns of the file were opened — the choice a person makes in the loader's dialog
        out["generic"] = {"table": base_store, "x_column": cache.get("x_label"), "y_columns": list(cache.get("y_columns", {}))}
    if cache.get("signals"):
        out["signal_labels"] = {k: sig["label"] for k, sig in cache["signals"].items()}
    if regression_method is not None:
        out["regression_method"] = regression_method
    if store_labels:
        out["store_labels"] = dict(store_labels)
    if signal is not None:
        out["signal"] = signal
    return out


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def output_entry(path):
    """One entry of a record's `outputs`: the file's name, size and (if small enough) SHA-256."""
    return {"name": os.path.basename(path), "size": os.path.getsize(path), "sha256": _sha256_file(path)}


def build_record(kind, parameters=None, *, source=None, recording=None, pipeline=None, app=None,
                 outputs=None, notes=None, now=None):
    """Assembles a provenance record (see the module docstring). `source` is a path (fingerprinted here)
    or an already-built fingerprint dict; `outputs` a list of file paths or entries; empty sections are left out."""
    rec = {
        "schema": SCHEMA_VERSION,
        "kind": kind,
        "created": (now or datetime.now()).isoformat(timespec="seconds"),
        "software": software_versions(app),
    }
    if source is not None:
        rec["source"] = source if isinstance(source, dict) else file_fingerprint(source)
    if recording:
        rec["recording"] = recording
    if pipeline:
        rec["pipeline"] = pipeline
    rec["analysis"] = {"type": kind, "parameters": dict(parameters or {})}
    if outputs:
        rec["outputs"] = [o if isinstance(o, dict) else output_entry(o) for o in outputs]
    if notes:
        rec["notes"] = list(notes)
    return rec


def sidecar_path_for(export_path):
    """Where the record for an export goes: beside it, same name, `.provenance.json` in place of its
    extension — `Event PETH.png` -> `Event PETH.provenance.json`."""
    return os.path.splitext(export_path)[0] + SIDECAR_SUFFIX


def json_default(o):
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def write_record(path, record):
    """Writes a record as indented UTF-8 JSON (NumPy numbers and arrays are converted); returns `path`."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=json_default)
    return path


def read_record(path):
    """Reads a record back. Raises ValueError if it is not one (no `schema` / `kind`) or is from a newer
    layout than this library understands."""
    with open(path, "r", encoding="utf-8") as f:
        rec = json.load(f)
    if not isinstance(rec, dict) or "schema" not in rec or "kind" not in rec:
        raise ValueError(f"{os.path.basename(path)} is not a provenance record")
    if rec["schema"] > SCHEMA_VERSION:
        raise ValueError(f"{os.path.basename(path)} uses provenance layout {rec['schema']}; this library reads "
                         f"up to {SCHEMA_VERSION} — update PyATCLI")
    return rec
