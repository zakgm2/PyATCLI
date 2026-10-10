"""
ops/common.py
-------------
What the operations share: how a request names a recording and its load options, how tables travel in a result, and the small
checks every operation makes before it does anything slow.
"""

import os

from ..analyses import ParameterError
from . import OpError

LOAD_KEYS = ("regression_method", "use_saved", "table", "x_column", "y_columns")


def load_options(load):
    """The keyword arguments of load_recording from a request's `load` dict ({} -> its defaults). An unknown key is a ParameterError."""
    load = dict(load or {})
    unknown = sorted(set(load) - set(LOAD_KEYS))
    if unknown:
        raise ParameterError(f"unknown load option(s) {', '.join(unknown)}; valid: {', '.join(LOAD_KEYS)}")
    return {k: v for k, v in load.items() if v is not None}


def recording_name(path):
    """A recording's own name: its folder's or file's basename."""
    return os.path.basename(os.fspath(path).rstrip("/" + chr(92)))


def require_exists(path):
    path = os.fspath(path)
    if not os.path.exists(path):
        raise OpError(f"{path} does not exist", "not_found")
    return path


def open_recording(ctx, recording, load=None, progress=None):
    """({"cache", "original_cache", "splices", "warnings"}, path) for the recording a request names: a path (the file as it is on disk now,
    opened fresh) or a reference {"id", "splices"?, "markers"?, "signal_labels"?, ...} to a recording the engine has open (engine/store.py).
    `progress` is where opening reports, if not the request's own."""
    if isinstance(recording, dict):
        if "id" not in recording:
            raise OpError("a recording is a path, or {\"id\": ...} for one the engine has open", "invalid_input")
        if ctx.store is None:
            raise OpError("recordings are only kept open by the engine (python -m PyATCLI serve); give a path", "no_recording")
        entry = ctx.store.get(recording["id"])
        return entry.resolve(recording, progress if progress is not None else ctx.progress()), entry.path
    path = require_exists(recording)
    return ctx.load(path, progress=progress, **load_options(load)), path


def recordings_under(path):
    """[path] if it is a recording itself, else every TDT block found under it."""
    from ..sources.processing_TDT import validate_tdt_folder
    from ..recording import find_tdt_recordings

    path = require_exists(path)
    if os.path.isdir(path) and not validate_tdt_folder(path)[0]:
        found = find_tdt_recordings(path)
        if not found:
            raise OpError(f"no TDT recordings found under {path}", "invalid_input")
        return found
    return [path]


def table_summary(table, name=None):
    return {"columns": [str(c) for c in table.columns], "rows": int(len(table))}


def tables_wire(ctx, tables, mode="shapes", prefix="table"):
    """How a result's tables (DataFrames) travel: "shapes" says each one's columns and number of rows; "inline" adds the values as
    {column: [values]} (small tables); "files" writes each as a CSV in ctx.data_dir and gives its path (a table too big for a message)."""
    if mode not in ("shapes", "inline", "files"):
        raise ParameterError(f"tables must be shapes, inline or files, not {mode!r}")
    out = {}
    for name, table in tables.items():
        entry = table_summary(table, name)
        if mode == "inline":
            entry["data"] = {str(c): table[c].tolist() for c in table.columns}
        elif mode == "files":
            if ctx.data_dir is None:
                raise OpError("tables='files' needs somewhere to put them (the engine's data folder)", "invalid_input")
            os.makedirs(ctx.data_dir, exist_ok=True)
            path = os.path.join(ctx.data_dir, f"{prefix}_{_safe(name)}.csv")
            table.to_csv(path, index=False, encoding="utf-8")
            entry["path"] = path
        out[str(name)] = entry
    return out


def _safe(text):
    import re
    return re.sub(r'[<>:"/\\|?*\s]+', "_", str(text)).strip("_") or "table"
