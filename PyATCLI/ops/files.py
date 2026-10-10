"""
ops/files.py — looking at files and folders on disk before (or instead of) opening a recording.

    files.inspect   what a path is: a TDT block, a folder with TDT blocks somewhere under it, an Oxysoft export, a table file
    pt2.load        a Terranova Prospa .pt2 image (the EFNMR viewer's data), its array handed over as a .npy file

A directory walk over a cohort folder or a read of a file on a network drive can take seconds, which is why a window asks for these
instead of doing them on its own thread.
"""

import os

from ..sources.dataset import DataFormat, detect_format, detect_format_file
from ..sources.pt2 import load_pt2
from ..sources.processing_TDT import validate_tdt_folder
from ..recording import find_tdt_recordings
from . import operation
from .common import require_exists
from .compute import arrays_wire


@operation("files.inspect", "what a path is: a TDT block, a folder with TDT blocks under it (searched at any depth), an Oxysoft export or a table file", result=["path", "kind", "valid", "message", "recordings"])
def inspect(ctx, *, path, search=True):
    """`recordings` are the TDT blocks to open: the path itself if it is one, else every block found under it (`search=False` skips the
    walk). `kind`: tdt (a block), oxysoft (an export file, or a folder holding one), table (any other file), folder (anything else)."""
    path = require_exists(path)
    if os.path.isdir(path):
        valid, message = validate_tdt_folder(path)
        if valid:
            return {"path": path, "kind": "tdt", "valid": True, "message": None, "recordings": [path]}
        found = find_tdt_recordings(path) if search else []
        kind = "oxysoft" if detect_format(path) == DataFormat.OXYSOFT else "folder"
        return {"path": path, "kind": kind, "valid": False, "message": message, "recordings": found}
    kind = "oxysoft" if detect_format_file(path) == DataFormat.OXYSOFT else "table"
    return {"path": path, "kind": kind, "valid": True, "message": None, "recordings": []}


@operation("pt2.load", "read a Terranova Prospa .pt2 image: its pixels as an .npy file to map", result=["path", "shape", "image"])
def pt2_load(ctx, *, path):
    path = require_exists(path)
    image = load_pt2(path)
    return {"path": path, "shape": list(image.shape), "image": arrays_wire(ctx, {"image": image}, "pt2")}
