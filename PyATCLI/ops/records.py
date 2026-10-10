"""
ops/records.py — the provenance record that goes beside an export.

    provenance.write   build the record of an export (the software, the recording's fingerprint and state, the pipeline's settings, the
                       analysis' parameters, the files made) and write it as <export>.provenance.json

A window that exports a table or a figure it drew itself asks for this afterwards: the work is reading the recording's files to fingerprint
them (hashing every small file in a TDT block's folder — remembered while the folder is unchanged) and the export's own bytes.
"""

import os

from .. import provenance
from . import operation
from .common import open_recording, require_exists

FIGURE_EXTENSIONS = (".png", ".pdf", ".svg", ".tif", ".tiff", ".jpg", ".jpeg")

_fingerprints = {}              # recording path -> (its modification stamp, fingerprint): hashing is quick, but not free per export


def source_fingerprint(path):
    """provenance.file_fingerprint(path), remembered while the folder or file is unchanged."""
    try:
        stamp = os.stat(path).st_mtime_ns
    except OSError:
        return provenance.file_fingerprint(path)
    hit = _fingerprints.get(path)
    if hit is None or hit[0] != stamp:
        hit = _fingerprints[path] = (stamp, provenance.file_fingerprint(path))
    return hit[1]


def default_kind(export_path):
    """What an export is when its caller did not say: a figure or a table, by extension."""
    return "figure" if os.path.splitext(export_path)[1].lower() in FIGURE_EXTENSIONS else "table"


@operation("provenance.write", "write the provenance record beside an export (<export>.provenance.json); returns where", result=["record"])
def write(ctx, *, export, kind=None, parameters=None, notes=None, recording=None, source_path=None, extra_outputs=None, store_labels=None,
          regression_method=None, load=None):
    """`export` is the file that was written; `kind` and `parameters` say what made it (an analyses.py kind, so `reproduce` can replay it;
    none: a figure or a table, by extension). `recording` names the recording the export came from (a path or an engine reference, as
    `run` takes it) for its state in the record; `source_path` fingerprints a source that is not a recording the engine opened (the PT2
    viewer, a text field study). `regression_method` is the motion-correction method in force, for a TDT recording."""
    export = require_exists(export)
    parameters = dict(parameters or {})
    cache, splices, recording_path = None, (), None
    if recording is not None:
        loaded, recording_path = open_recording(ctx, recording, load)
        cache, splices = loaded["cache"], loaded["splices"]
    source = source_path or recording_path
    method = regression_method if cache is not None and cache.get("source") == "TDT" else None
    doc = provenance.build_record(
        kind or default_kind(export), parameters,
        source=source_fingerprint(source) if source else None,
        recording=provenance.recording_state(cache, method, splices, store_labels=store_labels or None, signal=parameters.get("signal"))
        if cache is not None else None,
        pipeline=provenance.pipeline_parameters(method or "ols") if cache is not None else None,
        app=ctx.app, outputs=[export, *(extra_outputs or [])], notes=notes)
    path = provenance.write_record(provenance.sidecar_path_for(export), doc)
    ctx.written(path)
    return {"record": path}
