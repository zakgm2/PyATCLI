"""
analysis/group/run.py
-----------------------
Runs a group analysis end to end, headless: each recording of the spec is loaded (processed to dF/F,
its saved splices replayed, its saved markers put on top), its trials cut and measured, the models
fitted on all of them together (trials.extract_group_trials, stats.fit_group_models), and
the tables and report written (report.write_group_results). Also reads and writes the group's
settings file — analysis_settings.json, the spec plus what was in the recordings — which is what makes
a group analysis repeatable: run_group_analysis(GroupSpec.from_dict(spec_from_that_file)).

Recordings are handled one at a time and dropped again, so a large group never has more than one
recording's signal in memory. Figures are not drawn here (this library has no plotting); a caller that
wants them passes `extra_writers`.
"""

import json
import os
from datetime import datetime

import pandas as pd

from ... import provenance
from ...markers import group_markers_by_name
from ...progress import Plan
from ...recording import load_tdt_recording, normalized_signal
from . import GroupSpec
from .report import write_group_results
from .stats import fit_group_models
from .trials import TRIAL_COLUMNS, extract_group_trials

GROUP_SETTINGS_FILENAME = "analysis_settings.json"


def load_group_recording(folder, regression_method="ols", store_labels=None, progress=None):
    """
    Loads one recording for group analysis exactly the way the single-recording analyses see it,
    without a window: TDT block processed to dF/F, the recording's saved splices replayed (Cut Out
    re-fits the dF/F, as in the app), its saved markers put on top, and the events grouped by name
    with Event PETH's own grouping. The signal handed back is the normalized dF/F, the trace every
    analysis uses.

    Returns a dict with
        x, y      the time axis and the normalized dF/F after any saved splices
        fs        sampling rate in Hz
        groups    {marker name: sorted event times}, as Event PETH lists them
        n_splices how many saved splices were applied
        warnings  saved files that could not be read (the recording is then used without them)
    Raises ValueError if the folder is not a readable TDT block.
    """
    loaded = load_tdt_recording(folder, regression_method=regression_method, progress=progress)
    cache = loaded["cache"]
    _key, _label, y = normalized_signal(cache)
    return {"x": cache["x"], "y": y, "fs": cache["fs"],
            "groups": group_markers_by_name(cache, store_labels),
            "n_splices": len(loaded["splices"]), "warnings": loaded["warnings"]}


def run_group_analysis(spec, out_dir=None, progress=None, load=load_group_recording, extra_writers=()):
    """
    The whole analysis for `spec` (a GroupSpec). If `out_dir` is given the tables and the report are
    written there (write_group_results), then each of `extra_writers` — callables
    f(results, out_dir) -> list of paths written — runs (the app draws and saves its figures this way).

    `load(folder, regression_method, store_labels, progress)` returns what load_group_recording
    returns; it is a parameter so the analysis can be run on recordings that are not TDT blocks on
    disk (and so the tests can). Raises RuntimeError naming the subject if a recording cannot be
    loaded: leaving one out silently would change the group, so the person is told and can leave it
    out on purpose instead.

    Returns the GroupResults.
    """
    subjects = list(spec.subjects)
    stages = [(f"rec{i}", 1.0) for i in range(len(subjects))] + [("models", 1.5), ("writing", 0.3)]
    plan = Plan(progress, stages)

    frames, traces, excluded, notes, grid = [], {}, {m: 0 for m in spec.markers}, [], None
    for i, s in enumerate(subjects):
        name, folder = s["subject"], s["folder"]
        report = plan.sub(f"rec{i}")

        def relay(fraction, message="", _name=name):
            if report is not None:
                report(fraction, f"{_name}: {message}" if message else _name)

        plan.begin(f"rec{i}", f"Loading {name} ({i + 1} of {len(subjects)})")
        try:
            data = load(folder, spec.regression_method, spec.store_labels, relay if report is not None else None)
        except Exception as e:
            raise RuntimeError(f"Could not load '{name}' ({folder}): {e}") from e
        out = extract_group_trials(data["x"], data["y"], data["groups"], spec, subject=name, recording=folder)
        frames.append(out["trials"])
        traces[name] = out["traces"]
        grid = out["grid"]
        for marker, n in out["excluded"].items():
            excluded[marker] = excluded.get(marker, 0) + n
        for w in data.get("warnings", []):
            notes.append({"level": "warning", "measure": "", "text": f"{name}: {w}"})
        del data, out                                       # the signal is not needed again: free it before the next one

    plan.begin("models", "Fitting models")
    trials = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRIAL_COLUMNS)
    results = fit_group_models(trials, spec, progress=plan.sub("models"), traces=traces, trace_grid=grid,
                               excluded=excluded)
    results.notes = notes + results.notes

    if out_dir:
        plan.begin("writing", "Writing the tables and the report")
        results.files = write_group_results(results, out_dir)
        for writer in extra_writers:
            results.files += list(writer(results, out_dir))
    plan.done()
    return results


def write_group_settings(group_dir, spec, scans, app=None):
    """Writes the group's analysis settings (the spec, and a summary of each recording used) to
    <group_dir>/analysis_settings.json, creating the folder; returns the file's path.

    `app` names the program doing the writing, e.g. {"name": "PyAT", "version": "3.2.0"} — recorded
    with this library's own version, since both decide the numbers. The file is also a provenance record, so
    `reproduce` can run it again; each recording carries the digest of its files at that time."""
    from ... import __version__ as library_version

    included = {s["folder"] for s in spec.subjects}
    recordings = []
    for sc in scans:
        if sc["folder"] not in included:
            continue
        subject = next(s["subject"] for s in spec.subjects if s["folder"] == sc["folder"])
        recordings.append({
            "subject": subject, "folder": sc["folder"], "block_name": sc["block_name"],
            "start_time": sc["start_time"], "events_per_marker": {m: len(t) for m, t in sc["groups"].items()},
            "saved_splices_applied": sc["n_splices"], "saved_markers": sc["n_saved_markers"],
        })
    for r in recordings:
        r["digest"] = provenance.file_fingerprint(r["folder"]).get("digest")
    doc = {
        "app": {**(app or {"name": "PyATCLI"}), "library_version": library_version},
        "created": datetime.now().isoformat(timespec="seconds"),
        "spec": spec.to_dict(),
        "recordings": recordings,
        # The same file is a provenance record (provenance.py): reproduce.reproduce() and `python -m PyATCLI
        # reproduce` replay it, and the versions it was written with are checked against the ones running.
        "schema": provenance.SCHEMA_VERSION,
        "kind": "group_analysis",
        "software": provenance.software_versions(app),
        "pipeline": provenance.pipeline_parameters(spec.regression_method),
        "analysis": {"type": "group_analysis", "parameters": spec.to_dict()},
    }
    os.makedirs(group_dir, exist_ok=True)
    path = os.path.join(group_dir, GROUP_SETTINGS_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return path


def read_group_settings(path):
    """Reads an analysis_settings.json back: returns (GroupSpec, the whole document). `path` may be
    the file or the group's folder."""
    if os.path.isdir(path):
        path = os.path.join(path, GROUP_SETTINGS_FILENAME)
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return GroupSpec.from_dict(doc["spec"]), doc
