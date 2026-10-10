"""ops/group.py — the Hypothesis Testing wizard's steps, and running a group analysis."""

import os
import shutil

from .. import outputs
from ..analyses import run_group_from_spec
from ..analysis import group as design
from ..analysis.group import GroupSpec
from ..analysis.group.run import GROUP_SETTINGS_FILENAME, read_group_settings, write_group_settings
from ..analysis.group.scan import default_subject_names, scan_recordings
from ..sources.processing_TDT import REGRESSION_METHODS
from . import OpError, operation
from .analysis import _where
from .common import recordings_under, require_exists, tables_wire


def _keep_settings_with_results(settings, out_dir):
    """Puts a copy of the settings file the tables were made from in their folder, so the folder says how it was made: it can be plotted
    with the group's own name and markers, and reproduced, without the folder the settings came from."""
    source = settings if os.path.isfile(settings) else os.path.join(settings, GROUP_SETTINGS_FILENAME)
    target = os.path.join(out_dir, GROUP_SETTINGS_FILENAME)
    os.makedirs(out_dir, exist_ok=True)
    if not (os.path.exists(target) and os.path.samefile(source, target)):
        shutil.copyfile(source, target)


def _group_dir(out, output_folder, name):
    if output_folder:                                                    # <folder>/Group Analysis/<group name>, new if the name was used
        folder = outputs.group_dir(output_folder, name)
        if folder is None:
            raise OpError(f"the group name {name!r} cannot be used as a folder name", "invalid_input")
        return folder
    return out


def _run(ctx, spec, out_dir, tables="shapes"):
    """Runs the analysis (it writes tables, a report and a record; it never draws). `tables` says how
    the result tables travel back as "table_data": "shapes" (their columns and rows), "inline", or "files" (CSVs in the engine's data folder,
    for a window that wants the numbers); "note_records" and "software" are what a window needs to rebuild the results from them."""
    result, results = run_group_from_spec(spec, out_dir=out_dir, progress=ctx.progress())
    for n in result.notes:
        ctx.note(n)
    files = list(result.summary.get("files", []))
    for f in files:
        ctx.written(f)
    return {"name": spec.group_name, "tables": len(result.tables), "table_names": list(result.tables), "notes": list(result.notes),
            "files": files, "directory": out_dir, "table_data": tables_wire(ctx, result.tables, tables, prefix="group"),
            "note_records": list(results.notes), "software": results.software, "spec": spec.to_dict()}


@operation("group.run", "run a group analysis from its analysis_settings.json (or a settings dict; write_settings saves that dict, "
                         "with the scans it was set up from, as analysis_settings.json in the results folder first)", result=["name", "tables", "table_names", "notes", "note_records", "software", "spec", "table_data", "files", "directory"])
def group_run(ctx, *, settings, out=None, output_folder=None, tables="shapes", scans=None, write_settings=False):
    out, output_folder = _where(out, output_folder)
    if isinstance(settings, dict):
        spec, path = GroupSpec.from_dict(settings), None
    else:
        path = require_exists(settings)
        spec, _doc = read_group_settings(path)
    out_dir = _group_dir(out, output_folder, spec.group_name)
    written = None
    if write_settings and out_dir:                                        # the wizard's choices, saved first: a failed run still leaves them
        written = write_group_settings(out_dir, spec, scans or [], app=ctx.app)
        ctx.written(written)
    result = _run(ctx, spec, out_dir, tables=tables)
    if out_dir and path:
        _keep_settings_with_results(path, out_dir)
    if written:
        result["settings"] = written
    return result


@operation("group.scan", "what each recording of a group holds: its markers, length and saved splices (no signal processing)", result=["base", "scans"])
def group_scan(ctx, *, folders, base_path=None, store_labels=None):
    folders = [os.path.abspath(require_exists(f)) for f in folders]
    base = base_path or (os.path.commonpath(folders) if len(folders) > 1 else os.path.dirname(folders[0]))
    scans = scan_recordings(folders, base, store_labels or {}, progress=ctx.progress())
    for sc, name in zip(scans, default_subject_names(folders, base)):
        sc["subject"] = name
    return {"base": base, "scans": scans}


def _marker_rows(scans):
    index = design.marker_index(scans)
    shared = set(design.common_markers(scans))
    return [{"name": name, "recordings": e["recordings"], "events": e["events"], "shared": name in shared}
            for name, e in sorted(index.items(), key=lambda kv: (-kv[1]["recordings"], kv[0]))]


@operation("group.setup", "set up a group analysis as the wizard does: find the recordings, list their markers, check the design, write the settings")
def group_setup(ctx, *, folders, name, markers=None, exclude=None, store_labels=None, pre=design.DEFAULT_PRE, post=design.DEFAULT_POST,
                baseline=design.DEFAULT_BASELINE, response=design.DEFAULT_RESPONSE, signal="dff", smooth=0.5, measures=None,
                peak_direction="absolute", decay_fraction=0.5, correction="holm", alpha=0.05, regression="ols", out=None,
                output_folder=None, run=False):
    out, output_folder = _where(out, output_folder)
    if regression not in REGRESSION_METHODS:
        raise OpError(f"regression must be one of {', '.join(REGRESSION_METHODS)}, not {regression!r}", "bad_parameter")
    found = []
    for path in folders:
        found += [os.path.abspath(f) for f in recordings_under(path)]
    found = sorted(dict.fromkeys(found))
    store_labels = dict(store_labels or {})
    scanned = group_scan(ctx, folders=found, store_labels=store_labels)
    scans, base = scanned["scans"], scanned["base"]
    skipped = [{"subject": sc["subject"], "error": sc["error"]} for sc in scans if sc["error"]]
    for s in skipped:
        ctx.warn(f"{s['subject']} skipped: {s['error']}")
    names = {sc["subject"] for sc in scans}
    unknown = [n for n in (exclude or []) if n not in names]
    if unknown:
        raise OpError(f"no recording named {', '.join(map(repr, unknown))}; the recordings are: " + ", ".join(sorted(names)), "invalid_input")
    chosen = [sc for sc in scans if not sc["error"] and sc["subject"] not in (exclude or [])]
    if not chosen:
        raise OpError("none of the recordings could be read", "invalid_input")

    rows = _marker_rows(chosen)
    base_result = {"base": base, "recordings": [sc["subject"] for sc in chosen], "skipped": skipped, "markers": rows}
    if not markers:
        return {**base_result, "stage": "markers", "settings": None, "directory": None, "design_warnings": [], "ran": False}
    if not (out or output_folder):                                       # only listing the markers needs no place to write
        raise OpError("say where the settings go: --out DIR, or --output-folder DIR (the group gets its own folder under it)", "invalid_input")
    have = {r["name"] for r in rows}
    missing = [m for m in markers if m not in have]
    if missing:
        raise OpError(f"no recording has the marker {', '.join(map(repr, missing))}; the markers are: " + ", ".join(sorted(have)), "invalid_input")

    spec = GroupSpec(group_name=name, subjects=[{"subject": sc["subject"], "folder": sc["folder"]} for sc in chosen], markers=list(markers),
                     store_labels=store_labels, pre=pre, post=post, baseline=tuple(baseline), response=tuple(response), signal=signal,
                     smooth_seconds=smooth, metrics=list(measures or design.METRICS), peak_direction=peak_direction,
                     decay_fraction=decay_fraction, correction=correction, alpha=alpha, regression_method=regression)
    problems = spec.problems()
    if problems:
        raise OpError(" ".join(problems), "invalid_input")
    warnings = design.design_summary(chosen, spec.markers, spec.pre, spec.post)["warnings"]
    out_dir = _group_dir(out, output_folder, name)
    path = write_group_settings(out_dir, spec, scans, app=ctx.app)
    ctx.written(path)
    result = {**base_result, "stage": "settings", "settings": path, "directory": out_dir, "design_warnings": warnings, "ran": False,
              "subjects": len(spec.subjects), "marker_names": list(spec.markers)}
    if run:
        result["run"] = _run(ctx, spec, out_dir)
        result["ran"] = True
    return result
