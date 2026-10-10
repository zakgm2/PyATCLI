"""
engine/session.py
-----------------
One engine session: the recordings it holds open (store.py), the folder its bulk results are written to, and `handle`, which runs one
operation. The JSON-lines server (server.py) is a transport around this, and so is PyAT's in-process engine (used when no second process
is wanted: tests, debugging) — the operations see the same context either way.
"""

import os
import shutil
import tempfile

from .. import ops
from ..ops import OpContext
from .store import RecordingStore


class Session:
    def __init__(self, data_dir=None, app=None):
        self._owns_dir = data_dir is None
        self.data_dir = data_dir or tempfile.mkdtemp(prefix="pyatcli-engine-")
        os.makedirs(self.data_dir, exist_ok=True)
        self.store = RecordingStore(self.data_dir)
        self.app = dict(app or {"name": "PyATCLI engine"})

    def context(self, emit=None, cancelled=None):
        return OpContext(emit=emit, cancelled=cancelled, store=self.store, data_dir=self.data_dir, app=self.app, wants_progress=emit is not None)

    def handle(self, method, params=None, emit=None, cancelled=None):
        """Runs operation `method` with `params` and returns its JSON-ready result. `emit(event, **fields)` hears progress, warnings, notes and
        files written; `cancelled()` returning True stops the work at its next progress report (ops.Cancelled)."""
        return ops.call(method, params, self.context(emit, cancelled))

    def close(self):
        """Lets go of every recording and, if the session made its own data folder, removes it (best effort: a file a window still has
        mapped cannot be deleted on Windows, and is left for the system's temporary-file cleanup)."""
        for entry_id in self.store.ids():
            self.store.close(entry_id)
        if self._owns_dir:
            shutil.rmtree(self.data_dir, ignore_errors=True)
