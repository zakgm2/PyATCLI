"""
engine/wire.py
--------------
How a recording crosses the process boundary. A recording (the cache dict of recording.py) is a few kilobytes of structure — names,
markers, sampling rate — and tens of megabytes of arrays (a 17-minute, 3-signal TDT recording is 34 MB). The structure travels as JSON;
every array is written once as an `.npy` file in the engine's data folder and replaced, in the structure, by a small reference:

    {"__array__": "0003_signals.normalized.y.npy", "dtype": "float64", "shape": [1055744]}

The receiver maps those files (`numpy.load(..., mmap_mode="c")`: nothing is read until it is touched, and an edit stays in memory, never
in the file), so a window can plot a million samples without them ever passing through a pipe or being parsed. An array two places of the
cache point to (the normalized signal is also `corr`) is one file and, on the other side, one array.
"""

import os
import weakref
from collections import deque

import numpy as np

ARRAY_KEY = "__array__"


class ArrayFiles:
    """The `.npy` files of one data folder: writes each array the first time it is asked for and gives the same name after."""

    def __init__(self, directory):
        self.directory = directory
        self._by_id = {}                                   # id(array) -> (weak reference to the array, its file name)
        self._orphans = deque()                            # files whose array has been freed, to remove when nothing maps them any more
        self._count = 0

    def name_for(self, array, hint=""):
        key = id(array)
        known = self._by_id.get(key)
        if known is not None and known[0]() is array:      # (an id can be reused once its array is freed: the weak reference says whose it is)
            return known[1]
        os.makedirs(self.directory, exist_ok=True)
        self._count += 1
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in hint)[:60].strip("._") or "array"
        name = f"{self._count:04d}_{safe}.npy"
        np.save(os.path.join(self.directory, name), np.ascontiguousarray(array), allow_pickle=False)
        # Weak: the files outlive the arrays only until the next sweep, and an engine that has worked out many splices holds the
        # variants it keeps, not every array it ever wrote.
        self._by_id[key] = (weakref.ref(array, lambda _ref, key=key, name=name: self._freed(key, name)), name)
        return name

    def _freed(self, key, name):
        known = self._by_id.get(key)
        if known is not None and known[1] == name:
            del self._by_id[key]
        self._orphans.append(name)

    def sweep(self):
        """Removes the files of arrays that have been freed. A file a window still maps cannot be removed on Windows: it stays on the list
        for the next sweep."""
        for _ in range(len(self._orphans)):               # (a weak-reference callback may add one from another thread meanwhile)
            name = self._orphans.popleft()
            try:
                os.remove(os.path.join(self.directory, name))
            except FileNotFoundError:
                pass
            except OSError:
                self._orphans.append(name)

    def forget(self):
        """Lets go of the arrays (not the files)."""
        self._by_id.clear()


def encode(obj, files, hint=""):
    """`obj` (a cache or any structure of dicts, lists, numbers, strings and arrays) as JSON-ready values, its arrays written by `files`."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return [encode(v, files, hint) for v in obj.tolist()]
        return {ARRAY_KEY: files.name_for(obj, hint), "dtype": str(obj.dtype), "shape": list(obj.shape)}
    if isinstance(obj, dict):
        return {str(k): encode(v, files, f"{hint}.{k}" if hint else str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [encode(v, files, hint) for v in obj]
    if isinstance(obj, np.generic):
        return encode(obj.item(), files, hint)
    if isinstance(obj, float) and obj != obj:
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def decode(obj, directory, mmap=True, _loaded=None):
    """The structure `encode` made, with its arrays loaded back: mapped (copy-on-write) from the files in `directory`. The same file is
    the same array object."""
    loaded = {} if _loaded is None else _loaded
    if isinstance(obj, dict):
        if ARRAY_KEY in obj:
            name = obj[ARRAY_KEY]
            if name not in loaded:
                loaded[name] = np.load(os.path.join(directory, name), mmap_mode="c" if mmap else None, allow_pickle=False)
            return loaded[name]
        return {k: decode(v, directory, mmap, loaded) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v, directory, mmap, loaded) for v in obj]
    return obj
