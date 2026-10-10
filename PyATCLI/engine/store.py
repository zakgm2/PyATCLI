"""
engine/store.py
---------------
The recordings the engine holds between requests, so a window pays for opening one once (seconds: reading the tank, fitting the dF/F)
and every later request is a lookup.

A recording is held as the recording as recorded (`base`) and the variants of it a list of splices makes — a splice stacks on the ones
before it, each one re-fits the dF/F, and Ctrl+Z, removing a splice and Restore Full Recording are all just *which list* is asked for.
So a request names the recording and the splice list it wants, never "the current one": the engine keeps what it has worked out for each
list (the most recent few), and the same request twice, or after an undo, costs nothing.

    ref = {"id": "r1", "splices": [{"mode": "cut_out", "start": 40, "end": 55}],
           "markers": [...], "signal_labels": {"normalized": "GCaMP"}}

The markers, highlights, text boxes and signal names in a reference are the caller's own state, put on a shallow copy of the variant (the
arrays are shared): they decide what an analysis sees without the engine having to keep a second copy of a window's edits in step.
"""

import json
import os
import shutil
from collections import OrderedDict

from ..splice import MODE_CUT_OUT, splice_cut_out, splice_keep_inside
from ..recording import (HIGHLIGHTS_FILE, TEXT_ANNOTATIONS_FILE, annotation_sidecar_path, apply_signal_labels, apply_splice,
                         legacy_marker_sidecar_path, load_recording, marker_sidecar_path, read_saved_signal_labels, spliced_store_name)
from ..ops import OpError
from .wire import ArrayFiles, encode

MAX_VARIANTS = 6                # spliced variants kept besides the recording as recorded


def splice_key(splices):
    return tuple((s["mode"], float(s["start"]), float(s["end"])) for s in splices)


def _clean(splices):
    return [{"mode": s["mode"], "start": float(s["start"]), "end": float(s["end"])} for s in splices]


def _saved_labels(path, cache):
    """{signal key: name} the recording's saved state gave its signals ({} if none, or the file is unreadable: the load already warned)."""
    signals = cache.get("signals") or {}
    try:
        saved = read_saved_signal_labels(path)
    except Exception:                                        # noqa: BLE001
        return {}
    return {k: v for k, v in saved.items() if k in signals and v}


def _restored(path, cache, splices):
    """What the saved state put on the recording, for a window to tell the person: the number of splices replayed, and how many markers,
    highlights and text boxes were read back (None where nothing was saved, so "restored 0" is never said of a file that is not there)."""
    def exists(p):
        return bool(p) and os.path.exists(p)

    saved_markers = exists(marker_sidecar_path(path)) or exists(legacy_marker_sidecar_path(path))
    out = {"splices": len(splices), "markers": len(cache.get("markers", [])) if saved_markers else None}
    for key, filename in (("highlights", HIGHLIGHTS_FILE), ("text_annotations", TEXT_ANNOTATIONS_FILE)):
        out[key] = len(cache.get(key, [])) if exists(annotation_sidecar_path(path, filename)) else None
    return out


class Entry:
    """One open recording: where it came from, how it was opened, the recording as recorded and the splice variants worked out so far."""

    def __init__(self, entry_id, path, options, loaded, data_dir):
        self.id, self.path, self.options = entry_id, path, options
        self.base = loaded["original_cache"] if loaded["original_cache"] is not None else loaded["cache"]
        self.initial_splices = _clean(loaded["splices"])
        self.warnings = list(loaded["warnings"])
        self.variants = OrderedDict()
        self.variants[splice_key([])] = self.base
        self.variants[splice_key(self.initial_splices)] = loaded["cache"]
        self.files = ArrayFiles(os.path.join(data_dir, entry_id))
        used_saved = options.get("use_saved", True)
        self.saved_labels = _saved_labels(path, loaded["cache"]) if used_saved else {}
        self.restored = _restored(path, loaded["cache"], self.initial_splices) if used_saved else {"splices": 0, "markers": None, "highlights": None, "text_annotations": None}

    @property
    def method(self):
        return self.options.get("regression_method") or "ols"

    def cache_for(self, splices, progress=None):
        """The recording with `splices` applied (stacked, in order): from what is already worked out, or by applying the missing ones to
        the longest prefix that is. ValueError if a splice leaves too little of the recording."""
        splices = _clean(splices)
        key = splice_key(splices)
        if key in self.variants:
            self.variants.move_to_end(key)
            return self.variants[key]
        have = max((n for n in range(len(splices) + 1) if splice_key(splices[:n]) in self.variants), default=0)
        cache = self.variants[splice_key(splices[:have])]
        todo = splices[have:]
        for i, s in enumerate(todo):
            sub = None if progress is None else (lambda f, m="", i=i: progress((i + f) / len(todo), m))
            spliced = apply_splice(cache, s["mode"], s["start"], s["end"], regression_method=self.method, progress=sub)
            if spliced is None:
                raise OpError(f"a {s['mode']} of {s['start']:g}..{s['end']:g} s leaves too little of the recording (fewer than 2 samples)",
                              "invalid_input")
            cache = spliced
            cache['store'] = spliced_store_name(self.base['store'], splices[:have + i + 1])
            self.variants[splice_key(splices[:have + i + 1])] = cache
        self._trim()
        return cache

    def _trim(self):
        keep = {splice_key([]), splice_key(self.initial_splices)}
        extra = [k for k in self.variants if k not in keep]
        for k in extra[:max(0, len(extra) - MAX_VARIANTS)]:
            del self.variants[k]                       # its arrays are freed once nothing else holds them, and their files swept
        self.files.sweep()

    def move_markers(self, markers, splices, start=0):
        """`markers` (dicts with a time) moved through the splices of the list, in order, the way each splice moves the markers of the
        recording it is applied to — from the first splice (`start=0`: the markers are on the recording as recorded) or from splice number
        `start` (they are on the timeline the first `start` splices leave). Returns the moved list: a marker in a stretch cut out is gone."""
        splices = _clean(splices)
        moved = list(markers)
        for i in range(start, len(splices)):
            s = splices[i]
            before = self.cache_for(splices[:i])
            fn = splice_cut_out if s["mode"] == MODE_CUT_OUT else splice_keep_inside
            result = fn(before["x"], before["x"], before["x"], moved, [], s["start"], s["end"])
            if result:
                moved = result["markers"]
        return moved

    def set_signal_labels(self, labels):
        """The names the recording's signals are given, on every variant held (and the recording as recorded)."""
        for cache in {id(c): c for c in self.variants.values()}.values():
            apply_signal_labels(cache, labels)

    def resolve(self, ref=None, progress=None):
        """The recording a reference names, as load_recording's dict: {"cache", "original_cache", "splices", "warnings"}."""
        ref = ref or {}
        splices = _clean(ref["splices"]) if ref.get("splices") is not None else self.initial_splices
        cache = self.cache_for(splices, progress)
        overrides = {k: ref[k] for k in ("markers", "highlights", "text_annotations") if ref.get(k) is not None}
        if overrides or ref.get("signal_labels"):
            cache = dict(cache)
            cache.update(overrides)
            if ref.get("signal_labels"):
                apply_signal_labels(cache, ref["signal_labels"])
        return {"cache": cache, "original_cache": self.base if splices else None, "splices": splices, "warnings": []}

    def loaded(self):
        return self.resolve({})

    def wire(self, splices=None, progress=None):
        """What a window needs to hold this recording: its structure as JSON and its arrays as .npy references (see wire.py)."""
        splices = self.initial_splices if splices is None else _clean(splices)
        cache = self.cache_for(splices, progress)
        return {"id": self.id, "path": self.path, "splices": splices, "warnings": self.warnings,
                "directory": self.files.directory, "cache": encode(cache, self.files)}


    def opened(self, progress=None):
        """wire(), plus what a window needs on opening: the recording as recorded when a saved splice was replayed (what Restore Full
        Recording brings back), the names saved for its signals, and what the saved state restored."""
        result = self.wire(progress=progress)
        result.update(original_cache=encode(self.base, self.files) if self.initial_splices else None,
                      signal_labels=dict(self.saved_labels), restored=dict(self.restored), regression_method=self.method)
        return result


class RecordingStore:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._entries = {}
        self._by_key = {}
        self._count = 0
        self._leftovers = []                    # data folders of closed recordings that could not be removed yet (a window still maps a file)

    def open(self, path, options, progress=None, fresh=False):
        """The entry for `path` opened with `options` — the one already open if there is one, else a fresh load. `fresh` reads the disk
        again even if it is open (a window's Reload: the files may have changed since), letting go of the old entry."""
        self.sweep()
        key = (os.path.normcase(os.path.abspath(path)), json.dumps(options, sort_keys=True))
        existing = self._by_key.get(key)
        if existing is not None and fresh:
            self.close(existing)
            existing = None
        if existing is not None:
            return self._entries[existing]
        loaded = load_recording(path, progress=progress, **options)
        self._count += 1
        entry = Entry(f"r{self._count}", path, dict(options), loaded, self.data_dir)
        self._entries[entry.id] = entry
        self._by_key[key] = entry.id
        return entry

    def get(self, entry_id):
        try:
            return self._entries[entry_id]
        except KeyError:
            raise OpError(f"the engine has no open recording {entry_id!r}", "no_recording",
                          hint="it was closed, or the engine was restarted: open it again") from None

    def close(self, entry_id):
        entry = self.get(entry_id)
        del self._entries[entry_id]
        self._by_key = {k: v for k, v in self._by_key.items() if v != entry_id}
        entry.variants.clear()
        entry.files.forget()
        self._leftovers.append(entry.files.directory)
        self.sweep()

    def ids(self):
        return list(self._entries)

    def sweep(self):
        """Removes the data folders of recordings that were closed. A file a window still maps cannot be removed on Windows: its folder
        stays on the list and goes the next time (or at the end of the session)."""
        left = []
        for directory in self._leftovers:
            shutil.rmtree(directory, ignore_errors=True)
            if os.path.exists(directory):
                left.append(directory)
        self._leftovers = left
        for entry in self._entries.values():
            entry.files.sweep()
