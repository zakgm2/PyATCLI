"""
ops/results.py — reading a finished analysis's results back.

    results.open   what a results folder, a provenance record or a table file holds: what made it, how it was asked, its tables, what the
                   analysis said and the files they came from

The data side of looking at a result (results.py): a window that shows one asks for this and draws the tables itself.
"""

from ..results import NoResults, open_results
from . import OpError, operation
from .common import tables_wire


@operation("results.open", "what a results folder (a group analysis's, or one holding records or tables), a provenance record or a table file holds: "
                           "its kind, parameters, tables, notes and the files they were read from",
           result=["path", "directory", "kind", "parameters", "tables", "notes", "software", "files"])
def open_(ctx, *, path, tables="shapes"):
    """`tables` says how the tables travel back, as `group.run` does: "shapes" (columns and number of rows), "inline" (and the values) or "files"
    (CSVs in the engine's data folder, for a window that wants the numbers)."""
    try:
        opened = open_results(path)
    except NoResults as e:
        raise OpError(str(e), "invalid_input") from e
    return {"path": opened.path, "directory": opened.directory, "kind": opened.kind, "parameters": opened.parameters,
            "tables": tables_wire(ctx, opened.tables, tables, prefix="results"), "notes": opened.notes, "software": opened.software,
            "files": opened.files}
