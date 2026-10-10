"""
ops/recording.py — the recordings the engine keeps open (engine only: a one-shot command opens, works and exits).

    recording.open     read a recording (saved splices and markers applied) and hold it; returns its id and its arrays as .npy references
    recording.view     the same recording with a given list of splices (Ctrl+Z, removing a splice and Restore Full Recording are this);
                       markers given on the recording as recorded come back moved through them
    recording.splice   stack one splice on a list of splices; the markers given are carried through it
    recording.labels   the names the recording's signals are given
    recording.close    let go of it

A request that works on a recording (`run`, `view`, ...) names it as {"id": ..., "splices": [...], "markers": [...]} (see engine/store.py).
"""

import os

from ..analyses import ParameterError
from ..sources.tables import load_any_file
from ..splice import MODE_CUT_OUT, MODE_KEEP_INSIDE
from . import OpError, operation
from .common import load_options, require_exists


def _store(ctx):
    if ctx.store is None:
        raise OpError("recordings are only kept open by the engine (python -m PyATCLI serve)", "no_recording")
    return ctx.store


@operation("recording.open", "open a recording and keep it: its id, its structure as JSON and its arrays as .npy files to map (fresh: read the disk again)", result=["id", "path", "cache", "directory", "original_cache", "splices", "signal_labels", "restored", "warnings", "regression_method"])
def open_(ctx, *, path, load=None, fresh=False):
    store = _store(ctx)
    entry = store.open(require_exists(path), load_options(load), ctx.progress(), fresh=bool(fresh))
    return entry.opened(ctx.progress())


def _check_markers(markers):
    if markers is not None and not all(isinstance(m, dict) and "time" in m for m in markers):
        raise ParameterError("markers are objects with a time, like the ones in a recording: {\"time\": 12.5, \"label\": ...}")


@operation("recording.view", "the open recording with a list of splices applied (computed once per list, then remembered); markers on the "
                             "recording as recorded come back moved through them", result=["id", "path", "cache", "directory", "splices", "warnings"])
def view(ctx, *, id, splices=None, markers=None):
    _check_markers(markers)
    entry = _store(ctx).get(id)
    result = entry.wire(splices, ctx.progress())
    if markers is not None:
        result["markers"] = entry.move_markers(markers, result["splices"])
    return result


@operation("recording.splice", "stack a splice on a list of splices; markers (on the timeline before it) come back moved through it", result=["id", "path", "cache", "directory", "splices", "warnings"])
def splice(ctx, *, id, mode, start, end, splices=None, markers=None):
    if mode not in (MODE_CUT_OUT, MODE_KEEP_INSIDE):
        raise ParameterError(f"mode must be {MODE_CUT_OUT!r} or {MODE_KEEP_INSIDE!r}, not {mode!r}")
    _check_markers(markers)
    entry = _store(ctx).get(id)
    before = list(entry.initial_splices if splices is None else splices)
    after = before + [{"mode": mode, "start": float(start), "end": float(end)}]
    result = entry.wire(after, ctx.progress())                      # raises if the range leaves too little
    if markers is not None:
        result["markers"] = entry.move_markers(markers, after, start=len(before))
    return result


@operation("recording.labels", "the names the open recording's signals are given ({signal key: name})")
def labels(ctx, *, id, signal_labels):
    entry = _store(ctx).get(id)
    entry.set_signal_labels(signal_labels)
    return {"id": id, "signal_labels": dict(signal_labels)}


@operation("recording.close", "let go of an open recording", result=["id", "open"])
def close(ctx, *, id):
    _store(ctx).close(id)
    return {"id": id, "open": _store(ctx).ids()}


@operation("recording.list", "the recordings the engine has open", result=["open"])
def list_(ctx):
    store = _store(ctx)
    return {"open": [{"id": i, "path": store.get(i).path} for i in store.ids()]}


@operation("file.tables", "the tables of a generic file (Excel, CSV, TSV, text): their names, columns, sizes and first rows", result=["path", "tables"])
def tables(ctx, *, path, preview=8):
    found = load_any_file(require_exists(path), progress=ctx.progress())
    return {"path": os.fspath(path), "tables": [
        {"name": t.name, "headers": [str(h) for h in t.headers], "rows": int(t.data.shape[0]),
         "preview": [[None if v != v else float(v) for v in row] for row in t.data[:max(0, int(preview))]]} for t in found]}
