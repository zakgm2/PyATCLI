"""ops/analysis.py — run an analysis, run it again from a record, run many."""

import json
import os

from .. import outputs, provenance
from ..analyses import ParameterError, kinds, normalize_parameters, run_analysis
from ..progress import Plan
from ..reproduce import reproduce as reproduce_record
from ..reproduce import safe_name, write_result
from . import OpError, operation
from .common import open_recording, recording_name, recordings_under, tables_wire


def _where(out, output_folder):
    if out and output_folder:
        raise OpError("give --out or --output-folder, not both", "invalid_input")
    return out, output_folder


def _write_result(ctx, result, cache, path, stem, out, output_folder, load, splices):
    """Writes the result's tables and a provenance record: into `out`, or under the Output folder's <data type>/<recording> (Analysis)
    folder, under a name no earlier result has. Returns (files written, the folder)."""
    stem = stem or safe_name(f"{recording_name(path)}_{result.kind}")
    if out is None:
        out = outputs.recording_dir(output_folder, cache.get("source") or "Generic", path)
        stem = outputs.unique_stem(out, stem, [f"_{safe_name(t)}.csv" for t in result.tables] + [".provenance.json"])
    written = write_result(result, out, stem)
    method = (load or {}).get("regression_method") or "ols"
    record = provenance.build_record(
        result.kind, result.parameters, source=provenance.file_fingerprint(path),
        recording=provenance.recording_state(cache, method, splices, signal=result.parameters.get("signal")),
        pipeline=provenance.pipeline_parameters(method), app=ctx.app, outputs=written, notes=result.notes)
    written.append(provenance.write_record(os.path.join(out, f"{stem}.provenance.json"), record))
    for f in written:
        ctx.written(f)
    return written, out


@operation("run", "run one analysis on a recording; writes CSV tables and a provenance record when told where", result=["kind", "recording", "parameters", "summary", "notes", "tables", "files", "directory"])
def run(ctx, *, kind, recording, parameters=None, load=None, store_labels=None, out=None, output_folder=None, tables="shapes", stem=None):
    out, output_folder = _where(out, output_folder)
    if kind not in kinds():
        raise ParameterError(f"unknown analysis {kind!r}; the analyses are: {', '.join(kinds())}")
    params = dict(parameters or {})
    normalize_parameters(kind, params)                                  # fail on a bad parameter before loading anything slow
    # Opening the recording is most of the work when it is read from a path and none of it when the engine already holds it; the analysis
    # that follows may report nothing of its own (a fit is one call), so it is announced at its start and finished at its end.
    opening, running = "Opening the recording", f"Running {kind.replace('_', ' ')}"
    plan = Plan(ctx.progress(), [(opening, 1 if isinstance(recording, dict) else 4), (running, 3 if isinstance(recording, dict) else 1)])
    plan.begin(opening)
    loaded, path = open_recording(ctx, recording, load, progress=plan.sub(opening))
    cache = loaded["cache"]
    plan.begin(running)
    result = run_analysis(kind, cache, params, store_labels=store_labels, progress=plan.sub(running))
    plan.done("Done")
    written, directory = [], None
    if out or output_folder:
        written, directory = _write_result(ctx, result, cache, path, stem, out, output_folder, load, loaded["splices"])
    return {"kind": result.kind, "recording": path, "parameters": result.parameters, "summary": result.summary, "notes": result.notes,
            "tables": tables_wire(ctx, result.tables, tables, prefix=f"{kind}"), "files": written, "directory": directory}


@operation("reproduce", "run an analysis again from its provenance record and check the numbers against exports")
def reproduce(ctx, *, record, recording=None, out=None, compare=None, tolerance=1e-4, strict=False, rounded=None, tables="shapes"):
    rep = reproduce_record(record, recording=recording, out_dir=out, compare=compare or (), tolerance=tolerance, strict=strict,
                           progress=ctx.progress(), rounded=rounded)
    for w in rep.warnings:
        ctx.warn(w)
    for f in rep.written:
        ctx.written(f)
    return {"ok": rep.ok, "kind": rep.result.kind, "warnings": list(rep.warnings), "comparisons": rep.comparisons, "files": list(rep.written),
            "directory": out, "tables": tables_wire(ctx, rep.result.tables, tables, prefix=f"{rep.result.kind}")}


def _read_jobs(jobs):
    """(jobs list, defaults dict) from a path to a jobs file, or the document itself."""
    if isinstance(jobs, (str, os.PathLike)):
        with open(jobs, "r", encoding="utf-8") as f:
            jobs = json.load(f)
    jobs_list = jobs["jobs"] if isinstance(jobs, dict) else jobs
    defaults = jobs.get("defaults", {}) if isinstance(jobs, dict) else {}
    if not isinstance(jobs_list, list) or not all(isinstance(j, dict) and "kind" in j and "recording" in j for j in jobs_list):
        raise OpError('a jobs file is {"defaults": {...}, "jobs": [{"kind", "recording", "parameters"}, ...]} (or just the list of jobs)',
                      "invalid_input")
    return jobs_list, defaults


@operation("batch", "run many analyses over many recordings (a folder of TDT blocks counts as many)")
def batch(ctx, *, jobs, out=None, output_folder=None, load=None):
    import pandas as pd

    out, output_folder = _where(out, output_folder)
    if not (out or output_folder):
        raise OpError("say where the results go: --out DIR, or --output-folder DIR", "invalid_input")
    jobs_list, defaults = _read_jobs(jobs)
    load = dict(load or {})
    if "regression_method" in defaults:
        load["regression_method"] = defaults["regression_method"]
    if "use_saved" in defaults:
        load["use_saved"] = bool(defaults["use_saved"])
    for n, job in enumerate(jobs_list, 1):                              # a bad parameter anywhere fails before the first slow load
        normalize_parameters(job["kind"], {**defaults.get("parameters", {}), **job.get("parameters", {})})
    summary, failed, combined_files = [], [], []
    for n, job in enumerate(jobs_list, 1):
        kind = job["kind"]
        parameters = {**defaults.get("parameters", {}), **job.get("parameters", {})}
        job_dir = os.path.join(output_folder, "Batch", f"{n:02d}_{kind}") if output_folder else os.path.join(out, f"{n:02d}_{kind}")
        combined, files, names = {}, [], []
        for path in recordings_under(job["recording"]):
            ctx.check_cancel()
            name = safe_name(recording_name(path))
            try:
                loaded, _ = open_recording(ctx, path, load)
                result = run_analysis(kind, loaded["cache"], parameters, progress=ctx.progress())
                written, _dir = _write_result(ctx, result, loaded["cache"], path, f"{name}_{kind}", None if output_folder else os.path.join(job_dir, name),
                                              output_folder, load, loaded["splices"])
            except Exception as e:                                       # noqa: BLE001 — one recording failing must not stop the rest
                from . import is_expected, error_info
                if not is_expected(e):
                    raise
                failed.append({"job": n, "kind": kind, "recording": name, "error": error_info(e)["message"]})
                continue
            files += written
            names.append(name)
            ctx.note(f"job {n} {kind}: {name}: {len(written)} files")
            for tname, table in result.tables.items():
                combined.setdefault(tname, []).append(table.assign(recording=name))
        for tname, frames in combined.items():
            path = os.path.join(job_dir, f"{kind}_{safe_name(tname)}_all.csv")
            os.makedirs(job_dir, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8")
            combined_files.append(path)
            ctx.written(path)
        summary.append({"job": n, "kind": kind, "recordings": names, "files": files})
    return {"jobs": summary, "failed": failed, "combined": combined_files, "ok": not failed}
