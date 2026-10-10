"""
analysis/group/scan.py
------------------------
Finds out what each recording of a group contains — its event markers and how long it is — without
processing the signal, so a setup step can list the markers the recordings share before anything slow
happens.

A recording's markers are built the way the single-recording analyses build them: the events TDT
detected (with the recording's saved splices applied, as PyAT does when it restores splice.json) plus
any markers saved next to it (JSON saves/markers.json), grouped by name with markers.group_markers_by_name.
No GUI is involved, so this is safe on a worker thread.
"""

import os

from ...markers import group_markers_by_name
from ...sources.processing_TDT import scan_tdt_markers, validate_tdt_folder
from ...progress import Plan
from ...recording import read_saved_markers, read_saved_splices


def default_subject_names(folders, base_path):
    """One subject name per recording: the recording's own folder name (Synapse's
    Subject-YYMMDD-HHMMSS). Two recordings that share a folder name (in different
    subfolders) get their path from the searched folder on, so names start out unique."""
    names = [os.path.basename(f.rstrip("/\\")) for f in folders]
    out = []
    for f, n in zip(folders, names):
        out.append(n if names.count(n) == 1 else os.path.relpath(f, base_path).replace("\\", "/"))
    return out


def scan_recording(folder, base_path, store_labels=None, progress=None):
    """
    What one recording contains.

    Returns a dict with
        folder, name (its folder name), label (path from the searched folder),
        groups      {marker name: sorted event times} — every event the single-recording
                    analyses would offer, under the same names
        t_range     (first, last sample time) after any saved splices, or None
        n_events, n_splices (applied), n_splices_saved, n_saved_markers,
        block_name, start_time, warnings (things skipped, in words), error (None, or why the
        recording could not be read at all — then `groups` is empty).
    """
    result = {
        "folder": folder,
        "name": os.path.basename(folder.rstrip("/\\")),
        "label": os.path.relpath(folder, base_path).replace("\\", "/"),
        "groups": {}, "t_range": None, "n_events": 0,
        "n_splices": 0, "n_splices_saved": 0, "n_saved_markers": 0,
        "block_name": None, "start_time": None, "warnings": [], "error": None,
    }
    try:
        valid, message = validate_tdt_folder(folder)
        if not valid:
            raise ValueError(message)
        splices = []
        try:
            splices = read_saved_splices(folder)
        except Exception as e:
            result["warnings"].append(f"Saved splices could not be read ({e}); the recording is used as recorded.")
        result["n_splices_saved"] = len(splices)

        scan = scan_tdt_markers(folder, splices=splices or None, progress=progress)

        saved = []
        try:
            saved = read_saved_markers(folder)
        except Exception as e:
            result["warnings"].append(f"Saved markers could not be read ({e}); only the recorded events are used.")

        groups = group_markers_by_name({"detected_markers": scan["markers"], "markers": saved}, store_labels)
        result.update(groups=groups, t_range=scan["t_range"], n_events=sum(len(t) for t in groups.values()),
                      n_splices=scan["n_splices"], n_saved_markers=len(saved),
                      block_name=scan["block_name"], start_time=scan["start_time"])
    except Exception as e:
        result["error"] = str(e) or type(e).__name__
    return result


def scan_recordings(folders, base_path, store_labels=None, progress=None):
    """scan_recording for each folder, in order, one step of `progress` per recording. A recording
    that cannot be read comes back with its `error` set instead of stopping the rest."""
    plan = Plan(progress, [(f"rec{i}", 1) for i in range(len(folders))])
    out = []
    for i, folder in enumerate(folders):
        name = os.path.basename(folder.rstrip("/\\"))
        plan.begin(f"rec{i}", f"Scanning {name} ({i + 1} of {len(folders)})")
        out.append(scan_recording(folder, base_path, store_labels))
    plan.done()
    return out
