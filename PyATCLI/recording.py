"""
recording.py
------------
A recording as PyAT holds it, without the GUI: everything between "open a recording" and "run an
analysis" that isn't drawing. The GUI keeps this state in `ctx.cache`; this module builds, edits and
queries exactly that dict, so a script gets the same numbers the app shows.

The cache ("recording") dict
    source            'TDT' | 'Oxysoft' | 'Generic'
    source_path       the folder / file it was read from
    store             a display name (the folder / file / table name, plus a tag per splice)
    x, fs             time axis (s) and sampling rate (Hz)
    markers           user-placed markers (list of dict, see markers.py)
    highlights, text_annotations   shaded stretches and text boxes, when the recording has them saved (saved_state.py)
    -- TDT:     raw, corr (dF/F), detected_markers, signals {key: {"label", "y"}}, motion_correction_inlier_fraction
    -- Oxysoft: o2hb, hhb, optional thb (each (n_channels, n_samples)), detected_markers, fit_factor_mean
    -- Generic: y_columns {name: y}, x_label

What is here
    - the saved state next to a recording ("JSON saves/": markers, splices, the names given to its signals, highlights,
      text boxes) — where it lives, and reading it back (saved_state.py writes it);
    - building a cache from a load (make_tdt_cache / make_oxysoft_cache / make_generic_cache) and
      the one-call headless load_tdt_recording;
    - splicing (apply_splice for one, apply_splices for a saved list — Cut Out and Keep Inside both
      re-fit the dF/F on what is left, exactly as the app does);
    - picking the signal analyses run on (normalized_signal), and finding the recordings under a folder
      (find_tdt_recordings).

The splice arithmetic itself (cutting/stitching arrays and markers) is in splice.py; the dF/F
pipeline is in sources/processing_TDT.py.
"""

import json
import os

import numpy as np

from .markers import group_markers_by_name  # noqa: F401 — re-exported: the usual next step after loading
from .sources.processing_TDT import compute_dff, process_tdt_folder, validate_tdt_folder
from .progress import Plan
from .splice import MODE_CUT_OUT, MODE_KEEP_INSIDE, splice_cut_out, splice_keep_inside

# What a loaded recording (its `cache`) always has, by where it came from: a caller may count on these keys (ENGINE_CONTRACT.json lists them; removing
# or renaming one is a breaking change, see engine/contract.py). There are more (a TDT recording also holds the pieces it was built from, an Oxysoft
# one its fit factor, any of them the highlights and text boxes once there are some), and more can be added.
CACHE_KEYS = {
    "always": ["x", "fs", "source", "source_path", "store", "markers"],
    "TDT": ["signals", "corr", "raw", "detected_markers", "motion_correction_inlier_fraction"],
    "Oxysoft": ["o2hb", "hhb", "detected_markers"],
    "Generic": ["x_label", "y_columns"],
}

SAVES_FOLDER = "JSON saves"
HIGHLIGHTS_FILE = "highlights.json"
TEXT_ANNOTATIONS_FILE = "text_annotations.json"


# ---------------------------------------------------------------------------
# Where the saved state lives
# ---------------------------------------------------------------------------

def saved_state_dir(source_path):
    """The "JSON saves" folder for a recording: inside a folder source (TDT), next to a file source
    (Oxysoft, Generic)."""
    base = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
    return os.path.join(base, SAVES_FOLDER)


def marker_sidecar_path(source_path):
    """Where the recording's markers are saved: JSON saves/markers.json inside a folder source,
    JSON saves/<file name>.markers.json next to a file source."""
    if os.path.isdir(source_path):
        return os.path.join(saved_state_dir(source_path), "markers.json")
    return os.path.join(saved_state_dir(source_path), os.path.basename(source_path) + ".markers.json")


def legacy_marker_sidecar_path(source_path):
    """Pre-JSON-saves-folder location: a sibling file next to the raw data folder/file, e.g.
    <source>.markers.json. Only read, for markers saved before they moved into the JSON saves/
    folder, so old studies don't lose theirs."""
    return source_path + ".markers.json"


def annotation_sidecar_path(source_path, filename):
    """Same JSON saves/ folder as markers.json, for any other list saved the same way
    (highlights.json, text_annotations.json) — no legacy fallback, they're newer than the move."""
    if os.path.isdir(source_path):
        return os.path.join(saved_state_dir(source_path), filename)
    return os.path.join(saved_state_dir(source_path), os.path.basename(source_path) + f".{filename}")


def splice_sidecar_path(source_path):
    """splice.json, next to markers.json in the same JSON saves/ folder — every splice stacked on
    the recording, in order."""
    return os.path.join(os.path.dirname(marker_sidecar_path(source_path)), "splice.json")


def read_saved_splices(source_path):
    """The recording's saved splices, in the order they were applied ([] if none). Accepts the
    current list format and the old single-splice dict, like the app's loader does."""
    path = splice_sidecar_path(source_path)
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        saved = json.load(f)
    return [saved] if isinstance(saved, dict) else list(saved)


def signal_labels_path(source_path):
    """Where the names given to the recording's signals are saved (signal_labels.json, in JSON saves/)."""
    return annotation_sidecar_path(source_path, "signal_labels.json")


def read_saved_signal_labels(source_path):
    """{signal key: the name it was given} saved for the recording ({} if none). `ValueError` if the file
    does not hold an object."""
    path = signal_labels_path(source_path)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    if not isinstance(saved, dict):
        raise ValueError("signal_labels.json does not hold an object of names")
    return {str(k): str(v) for k, v in saved.items()}


def read_saved_annotations(source_path, filename):
    """A list of dicts saved in its own file in the saved state (HIGHLIGHTS_FILE, TEXT_ANNOTATIONS_FILE); [] if there is none.
    `ValueError` if the file does not hold a list."""
    path = annotation_sidecar_path(source_path, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    if not isinstance(saved, list):
        raise ValueError(f"{filename} does not hold a list")
    return saved


def apply_signal_labels(cache, labels):
    """Gives the recording's signals the names in `labels` ({signal key: name}; a key the recording does
    not have, or an empty name, is ignored). The cache dict is updated in place (its arrays are not copied);
    returns it. The name is what figures, legends and the "Channel" column of exports show."""
    signals = cache.get("signals")
    if signals and labels:
        cache["signals"] = {k: ({**sig, "label": labels[k]} if labels.get(k) else sig) for k, sig in signals.items()}
    return cache


def current_signal_labels(cache):
    """{signal key: name in effect} for the recording ({} if it has no signals map) — what a provenance record
    stores so a replay names the signals the same."""
    return {k: sig["label"] for k, sig in (cache.get("signals") or {}).items()}


def name_problem(final_labels):
    """Why {signal key: name} can't be the recording's names, or None if it can: a name must say something, and two signals can't
    share one (the legend, the colours and the exports tell them apart by name)."""
    if any(not str(name).strip() for name in final_labels.values()):
        return "A signal needs a name."
    names = [str(name).strip() for name in final_labels.values()]
    for name in names:
        if names.count(name) > 1:
            return f"Two signals can't both be called \"{name}\"."
    return None


def plan_signal_renames(cache, renames):
    """{signal key: new name} for what would change: a key the recording doesn't have, or a name that is empty or unchanged, is
    skipped. ValueError, with nothing changed, if the result would give two signals the same name."""
    signals = cache.get('signals') if cache else None
    if not signals:
        return {}
    current = current_signal_labels(cache)
    changes = {key: str(name).strip() for key, name in renames.items()
               if key in signals and name and str(name).strip() and str(name).strip() != current[key]}
    if changes:
        problem = name_problem({**current, **changes})
        if problem:
            raise ValueError(problem)
    return changes


def read_saved_markers(source_path):
    """The markers saved for the recording (JSON saves/markers.json, else the older sibling
    <source>.markers.json); [] if there are none. Same lookup as the app's restore."""
    path = marker_sidecar_path(source_path)
    if not path or not os.path.exists(path):
        path = legacy_marker_sidecar_path(source_path)
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        saved = json.load(f)
    if not isinstance(saved, list):
        raise ValueError("markers.json does not hold a list of markers")
    return saved


# ---------------------------------------------------------------------------
# Building a cache from a load
# ---------------------------------------------------------------------------

def build_signal_options(result):
    """The signals a TDT recording offers for plotting/analysis: the computed normalized trace plus
    whatever raw per-wavelength channels process_tdt_folder() found, as
    {key: {"label", "y"}} (the app adds a display colour to each). Not hardcoded to exactly
    "Normalized/Isosbestic/Main Driver" — a block with no isosbestic reference stream just won't
    have that key."""
    options = {
        "normalized": {"label": "Normalized (dF/F)", "y": result["corr"]},
    }
    for ch in result.get("channels", []):
        options[ch["key"]] = {"label": ch["label"], "y": ch["y"]}
    return options


def make_tdt_cache(folder_path, result, signals=None):
    """The cache for a freshly processed TDT folder (`result` is process_tdt_folder's return value),
    before any splice or saved marker. `signals` replaces the default build_signal_options(result)
    (the app passes its own, with colours added)."""
    return {
        'source':           'TDT',
        'source_path':      folder_path,
        'store':            os.path.basename(folder_path.rstrip('/\\')),
        'x':                result['x'],
        'raw':              result['raw'],
        'corr':             result['corr'],
        'fs':               result['fs'],
        'detected_markers': result.get('markers', []),
        'markers':          [],
        'signals':          signals if signals is not None else build_signal_options(result),
        # the share of samples the motion-correction fit kept (None for a recording with no reference stream): not
        # drawn anywhere, but the number to compare regression methods by, so a window can say it after a load
        'motion_correction_inlier_fraction': result.get('motion_correction_inlier_fraction'),
    }


def make_oxysoft_cache(file_path, dataset):
    """The cache for a loaded Oxysoft file (`dataset` is load_dataset_file's Dataset)."""
    n_ch = dataset.metadata.get('n_channels', dataset.num_channels // 2)
    cache = {
        'source':          'Oxysoft',
        'source_path':     file_path,
        'store':           dataset.folder_name,
        'x':               np.arange(dataset.num_samples) / dataset.sample_rate,
        'o2hb':            dataset.signals[:n_ch],
        'hhb':             dataset.signals[n_ch:],
        'fs':              dataset.sample_rate,
        'fit_factor_mean': dataset.metadata.get('fit_factor_mean'),
        'detected_markers': [
            {"time": ev["sample"] / dataset.sample_rate, "label": ev["label"],
             "color": "black", "store": "Events"}
            for ev in dataset.events
        ],
        'markers': [],
    }
    if 'thb' in dataset.metadata:
        cache['thb'] = dataset.metadata['thb']
    return cache


def make_generic_cache(path, table, x_index=0, y_indices=None):
    """The cache for one table of a generic tabular file (`table` is one of load_any_file's
    GenericTables): column `x_index` is the x axis (rows with no x value are dropped) and each of
    `y_indices` (default: every other column) becomes a trace, keyed by its header."""
    if y_indices is None:
        y_indices = [i for i in range(len(table.headers)) if i != x_index]
    x_data = table.data[:, x_index]
    valid = ~np.isnan(x_data)
    x_data = x_data[valid]
    y_columns = {table.headers[yi]: table.data[valid, yi] for yi in y_indices}
    from .analysis.shared import estimate_sample_rate
    fs = estimate_sample_rate(x_data) if len(x_data) > 1 else 1.0
    return {
        "source":      "Generic",
        "source_path": path,
        "x":           x_data,
        "y_columns":   y_columns,
        "x_label":     table.headers[x_index],
        "store":       table.name,
        "fs":          fs,
        "markers":     [],
    }


def normalized_signal(cache):
    """The (key, label, y) of the trace every analysis that works on the corrected signal uses
    (Event PETH, the peak finder, group analysis, ...): "normalized" (dF/F) if the recording offers
    it, whatever comes first if not; for a source with no signals map, its corrected trace, else its
    raw one."""
    signals = cache.get('signals')
    if not signals:
        y = cache['corr'] if 'corr' in cache else cache['raw']
        return "normalized", "dF/F (corrected)", y
    key = "normalized" if "normalized" in signals else next(iter(signals))
    return key, signals[key]['label'], signals[key]['y']


# ---------------------------------------------------------------------------
# Splicing
# ---------------------------------------------------------------------------

def apply_splice(source_cache, mode, start, end, regression_method="ols", progress=None):
    """Pure computation: one splice applied to source_cache, returning the new spliced cache dict, or
    None if the range isn't usable. Nothing is mutated.

    Branches on the cache's source since each one carries the actual signal in different keys. All
    three paths funnel every source-specific array through splice.py's extra_channels (sliced along
    each array's own last axis, so Oxysoft's 2D per-detector arrays and Generic's 1D columns both
    just work) so the trim/cut-and-stitch math itself lives in exactly one place.

    For TDT, both modes re-run motion/bleaching correction (compute_dff) on whatever raw signal
    survived: an artifact dragging the original fit can still be included or excluded by either
    operation, so a fresh fit — not just a shorter view of the old one — is what a splice means.
    regression_method only matters there (Oxysoft/Generic have no correction pipeline to re-run), and
    so does `progress` (a PyATCLI progress callback): only that recompute is slow enough to
    report on.
    """
    splice_fn = splice_cut_out if mode == MODE_CUT_OUT else splice_keep_inside
    source = source_cache.get('source')
    x = source_cache['x']
    markers = source_cache['markers']
    detected_markers = source_cache.get('detected_markers', [])

    if source == 'TDT':
        # The raw per-wavelength channels (main driver, isosbestic, ...) are sample-aligned with
        # x/raw/corr and must be trimmed/cut identically or they'd end up the wrong length (and
        # showing the wrong span of samples) after this splice — 'normalized' is excluded since it's
        # already the freshly-spliced 'corr' below.
        source_signals = source_cache.get('signals', {})
        extra_channels = {k: sig['y'] for k, sig in source_signals.items() if k != 'normalized'}

        result = splice_fn(x, source_cache['raw'], source_cache['corr'],
                           markers, detected_markers, start, end, extra_channels=extra_channels)
        if result is None:
            return None

        raw_out, corr_out = result['raw'], result['corr']

        # Only possible when the raw main-driver channel survived the splice above (every TDT
        # recording has one — see process_tdt_folder — so this is really just a defensive check).
        if 'main_driver' in result['extra_channels']:
            y_465 = result['extra_channels']['main_driver']
            y_415 = result['extra_channels'].get('isosbestic')
            computed = compute_dff(y_465, y_415, source_cache['fs'],
                                   regression_method=regression_method, progress=progress)
            raw_out, corr_out = computed['raw'], computed['corr']
            inlier_fraction = computed.get('motion_correction_inlier_fraction')
        else:
            inlier_fraction = source_cache.get('motion_correction_inlier_fraction')

        spliced = dict(source_cache)
        spliced['motion_correction_inlier_fraction'] = inlier_fraction            # of the fit on what is left
        spliced['x'] = result['x']
        spliced['raw'] = raw_out
        spliced['corr'] = corr_out
        spliced['markers'] = result['markers']
        spliced['detected_markers'] = result['detected_markers']
        if source_signals:
            spliced['signals'] = {
                key: {**sig, 'y': corr_out if key == 'normalized' else result['extra_channels'][key]}
                for key, sig in source_signals.items()
            }
        return spliced

    if source == 'Oxysoft':
        # No single raw/corr pair here — the detector arrays (each shape (n_channels, n_samples)) ARE
        # the signal, so they go through extra_channels instead; x is passed as a throwaway raw/corr
        # placeholder since the function requires something 1D there, but result['raw']/['corr'] are
        # never read back out below.
        extra_channels = {'o2hb': source_cache['o2hb'], 'hhb': source_cache['hhb']}
        if 'thb' in source_cache:
            extra_channels['thb'] = source_cache['thb']

        result = splice_fn(x, x, x, markers, detected_markers, start, end,
                           extra_channels=extra_channels)
        if result is None:
            return None

        spliced = dict(source_cache)
        spliced['x'] = result['x']
        spliced['o2hb'] = result['extra_channels']['o2hb']
        spliced['hhb'] = result['extra_channels']['hhb']
        if 'thb' in source_cache:
            spliced['thb'] = result['extra_channels']['thb']
        spliced['markers'] = result['markers']
        spliced['detected_markers'] = result['detected_markers']
        return spliced

    if source == 'Generic':
        # y_columns are the signal(s) — same throwaway-placeholder reasoning as Oxysoft above.
        extra_channels = dict(source_cache['y_columns'])

        result = splice_fn(x, x, x, markers, detected_markers, start, end,
                           extra_channels=extra_channels)
        if result is None:
            return None

        spliced = dict(source_cache)
        spliced['x'] = result['x']
        spliced['y_columns'] = result['extra_channels']
        spliced['markers'] = result['markers']
        spliced['detected_markers'] = result['detected_markers']
        return spliced

    return None


def spliced_store_name(base_name, splices):
    """Store label reflecting every splice currently applied, in order — derived fresh from the
    splice list each time (rather than appending a tag onto whatever the previous label happened to
    be) so it stays correct after one is removed."""
    if not splices:
        return base_name
    tags = ", ".join(
        f"{'cut' if s['mode'] == MODE_CUT_OUT else 'kept'} {s['start']:.1f}s-{s['end']:.1f}s"
        for s in splices
    )
    return f"{base_name} [{tags}]"


def apply_splices(cache, splices, regression_method="ols", progress=None):
    """Replays a saved list of splices on `cache`, in order, each on top of the one before (splices
    stack). A splice whose range leaves too little (fewer than 2 samples) is skipped, as the app does.

    Returns (spliced_cache, applied): the final cache (its 'store' tagged with every applied splice;
    `cache` itself is untouched) and the list of {"mode", "start", "end"} that took effect. With
    nothing applied, the cache comes back as given."""
    plan = Plan(progress, [(f"Splice {i + 1} of {len(splices)}", 1) for i in range(len(splices))])
    base_name = cache['store']
    current, applied = cache, []
    for i, s in enumerate(splices):
        spliced = apply_splice(current, s["mode"], s["start"], s["end"], regression_method=regression_method,
                               progress=plan.sub(f"Splice {i + 1} of {len(splices)}"))
        if spliced is None:
            continue
        applied.append({"mode": s["mode"], "start": s["start"], "end": s["end"]})
        spliced['store'] = spliced_store_name(base_name, applied)
        current = spliced
    return current, applied


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tdt_recording(folder_path, regression_method="ols", use_saved=True, progress=None):
    """Loads a TDT block the way the app does: processed to dF/F, the recording's saved splices
    replayed (each re-fits the dF/F), then its saved markers put on top. Nothing is written.

    Parameters
    ----------
    use_saved : bool
        False skips the JSON saves/ folder entirely (the recording exactly as recorded).
    progress : None, True or callable(fraction, message)
        See PyATCLI.progress.

    Returns
    -------
    dict with
        cache           the recording after splices and saved markers (see the module docstring)
        original_cache  the unspliced cache, or None if no splice was applied (what the app keeps to
                        be able to restore the full recording)
        splices         the splices that took effect, in order
        warnings        saved files that could not be read (the recording is then used without them)
    Raises ValueError if the folder is not a readable TDT block.
    """
    plan = Plan(progress, [("Reading and processing", 8), ("Applying saved splices", 2)])
    valid, message = validate_tdt_folder(folder_path)
    if not valid:
        raise ValueError(message)

    result = process_tdt_folder(folder_path, regression_method=regression_method,
                                progress=plan.sub("Reading and processing"))
    cache = make_tdt_cache(folder_path, result)
    original_cache, applied, warnings = None, [], []
    if use_saved:
        cache, original_cache, applied, warnings = _restore_saved_state(
            cache, folder_path, regression_method, progress=plan.sub("Applying saved splices"))
    plan.done()
    return {"cache": cache, "original_cache": original_cache, "splices": applied, "warnings": warnings}


def _restore_saved_state(cache, source_path, regression_method, progress=None):
    """What every loader does last: replay the recording's saved splices (each re-fits the dF/F for TDT) and
    put its saved markers on top. Returns (cache, original_cache, applied, warnings)."""
    warnings = []
    try:
        splices = read_saved_splices(source_path)
    except Exception as e:                                    # noqa: BLE001 - an unreadable file must not stop the load
        warnings.append(f"Saved splices could not be read ({e}); the recording is used as recorded.")
        splices = []
    original, applied = None, []
    if splices:
        spliced, applied = apply_splices(cache, splices, regression_method=regression_method, progress=progress)
        if applied:
            original, cache = cache, spliced
    try:
        cache["markers"] = read_saved_markers(source_path)    # replaces the (empty) list, after the splices
    except Exception as e:                                    # noqa: BLE001
        warnings.append(f"Saved markers could not be read ({e}); only the recorded events are used.")
    for key, filename, noun in (("highlights", HIGHLIGHTS_FILE, "highlights"), ("text_annotations", TEXT_ANNOTATIONS_FILE, "text boxes")):
        try:                                                  # not set at all when nothing is saved, as the app leaves them
            if os.path.exists(annotation_sidecar_path(source_path, filename)):
                cache[key] = read_saved_annotations(source_path, filename)
        except Exception as e:                                # noqa: BLE001
            warnings.append(f"Saved {noun} could not be read ({e}); the recording is used without them.")
    try:
        labels = read_saved_signal_labels(source_path)
        apply_signal_labels(cache, labels)
        if original is not None:
            apply_signal_labels(original, labels)             # so restoring the full recording keeps the names
    except Exception as e:                                    # noqa: BLE001
        warnings.append(f"Saved signal names could not be read ({e}); the default names are used.")
    return cache, original, applied, warnings


def load_recording(path, regression_method="ols", use_saved=True, table=None, x_column=None, y_columns=None, progress=None):
    """Opens any recording the app can: a TDT folder, an Oxysoft .txt export, or a generic table file, the way
    the app does — with its saved splices replayed and its saved markers on top (`use_saved=False` skips both).
    Returns what load_tdt_recording returns: {"cache", "original_cache", "splices", "warnings"}.

    For a generic file (Excel, CSV, TSV, text) `table` names which of its tables to open (needed only if it has
    more than one), `x_column` its x column (default the first) and `y_columns` the traces (default every other
    column). `regression_method` matters for TDT only. Raises ValueError with what is wrong if the path is
    not a recording it can open."""
    from .sources.dataset import DataFormat, detect_format_file
    from .sources.load import load_dataset_file
    from .sources.tables import load_any_file

    if os.path.isdir(path):
        return load_tdt_recording(path, regression_method=regression_method, use_saved=use_saved, progress=progress)
    if not os.path.isfile(path):
        raise ValueError(f"{path} does not exist")
    if detect_format_file(path) == DataFormat.OXYSOFT:
        cache = make_oxysoft_cache(path, load_dataset_file(path, progress=progress))
    else:
        tables = load_any_file(path, progress=progress)
        if table is None and len(tables) > 1:
            raise ValueError(f"{os.path.basename(path)} holds {len(tables)} tables ({', '.join(t.name for t in tables)}); "
                             "say which one to open")
        chosen = next((t for t in tables if t.name == table), None) if table is not None else tables[0]
        if chosen is None:
            raise ValueError(f"{os.path.basename(path)} has no table named {table!r}; it has: {', '.join(t.name for t in tables)}")
        headers = list(chosen.headers)
        x_name = x_column if x_column is not None else headers[0]
        bad = [c for c in [x_name] + list(y_columns or []) if c not in headers]
        if bad:
            raise ValueError(f"column(s) {bad} are not in table {chosen.name!r}; its columns: {', '.join(headers)}")
        y_idx = [headers.index(c) for c in y_columns] if y_columns else None
        cache = make_generic_cache(path, chosen, headers.index(x_name), y_idx)
    if not use_saved:
        return {"cache": cache, "original_cache": None, "splices": [], "warnings": []}
    cache, original, applied, warnings = _restore_saved_state(cache, path, regression_method, progress)
    return {"cache": cache, "original_cache": original, "splices": applied, "warnings": warnings}


# ---------------------------------------------------------------------------
# Finding recordings
# ---------------------------------------------------------------------------

def find_tdt_recordings(path):
    """Every valid TDT block anywhere under `path`, at any depth, sorted — a cohort folder
    containing per-subcohort folders that each contain the actual per-animal/per-session TDT blocks
    is a real lab layout, not just a flat one level down, so this walks the whole subtree rather than
    just path's immediate children. Anything that's never a valid TDT block anywhere in the tree
    (other folders, loose files of any type) is silently skipped, not reported, same as
    validate_tdt_folder already does for a single folder. `path` itself is not tested (a caller that
    wants it checks validate_tdt_folder first).

    Stops descending once a directory is confirmed to be a TDT block itself (topdown os.walk,
    pruning `dirs` in place) — its own internal files/subfolders are TDT's, not further recordings
    to find, and there's no reason to keep walking into them.

    Guards against a directory cycle (a folder that links back to one of its own ancestors, so the
    "further down" it looks like never bottoms out) by tracking each directory's *real*,
    symlink/junction-resolved identity rather than trusting os.walk's own followlinks=False default —
    confirmed by direct testing that that default does NOT catch a Windows NTFS junction cycle
    (os.path.islink() simply returns False for a junction, unlike a real symlink), which would
    otherwise have os.walk descend into the same real folder over and over, reporting it as more and
    more distinct-looking "finds" each time, stopped only by accidentally hitting the OS's
    path-length limit rather than by any real protection."""
    found = []
    visited_real_dirs = {os.path.realpath(path)}
    for root, dirs, _files in os.walk(path):
        kept = []
        for name in dirs:
            real = os.path.realpath(os.path.join(root, name))
            if real in visited_real_dirs:
                continue  # cycle — this real directory was already walked
            visited_real_dirs.add(real)
            kept.append(name)
        dirs[:] = kept

        if root == path:
            continue  # the picked folder itself is the caller's to check
        valid, _ = validate_tdt_folder(root)
        if valid:
            found.append(root)
            dirs[:] = []  # don't walk into a confirmed TDT block's own contents
    found.sort()
    return found
