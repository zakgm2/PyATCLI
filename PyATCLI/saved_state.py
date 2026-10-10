"""
saved_state.py
--------------
Writing the state PyAT's Save Changes writes next to a recording, so an edit made in a script is an edit made in the app: the
same files, in the same "JSON saves" folder, which either one reads back when it opens the recording (recording.py holds
where they live and the readers).

    JSON saves/
      markers.json           the markers placed on the plot                  [{"time", "label", "color", "fontsize"}]
      splice.json            every splice stacked on the recording, in order  [{"mode", "start", "end"}]
      signal_labels.json     the names given to its signals                  {signal key: name}
      highlights.json        shaded stretches                                [{"start", "end", "color"}]
      text_annotations.json  text boxes on the plot                          [{"x", "y", "text", "fontsize"}]

Two levels. The writers take what to write — `write_saved_markers(source, markers)` — and check that the app could read it
back. The edits take a recording's path and do what one action in the app does, to what is already saved:

    pl.saved_state.add_marker("D:/rats/Rat7", 312.5, label="Lever")      # Add Marker, then Save Changes
    pl.saved_state.add_splice("D:/rats/Rat7", "cut_out", 40.0, 55.0)     # Splice -> Cut Out, then Save Changes
    pl.saved_state.rename_signals("D:/rats/Rat7", {"main_driver": "GCaMP"})
    pl.saved_state.clear_saved_state("D:/rats/Rat7")                    # Undo All Changes, saved

An edit loads the recording the way the app does (saved splices replayed, so a new splice's times are on the time axis as it
now stands — a splice stacks on the ones before it), changes the one thing, and writes only that thing's file. Nothing is
written outside the recording's JSON saves folder. `python -m PyATCLI saved ...` runs the same functions.
"""

import json
import numbers
import os

from .recording import (
    HIGHLIGHTS_FILE, TEXT_ANNOTATIONS_FILE, annotation_sidecar_path, apply_splice, load_recording, marker_sidecar_path,
    plan_signal_renames, read_saved_annotations, read_saved_markers, read_saved_signal_labels, read_saved_splices, saved_state_dir,
    signal_labels_path, splice_sidecar_path,
)
from .splice import MODE_CUT_OUT, MODE_KEEP_INSIDE

# The five colours the Highlighter offers, and how see-through they are drawn.
HIGHLIGHT_COLORS = {
    "Yellow": "#FFEB3B",
    "Green":  "#8BC34A",
    "Pink":   "#FF4D94",
    "Orange": "#FF9800",
    "Blue":   "#29B6F6",
}
HIGHLIGHT_ALPHA = 0.35
DEFAULT_TEXT_FONT_SIZE = 10                    # the Text tool's dialog opens on this
DEFAULT_MARKER = {"label": "Marker", "color": "green", "fontsize": 8}      # what Add Marker stamps until it is configured
SPLICE_MODES = (MODE_CUT_OUT, MODE_KEEP_INSIDE)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _is_number(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


def _dump(path, document, ensure_ascii=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=ensure_ascii)
    return path


def _remove(path):
    """Takes away a saved file that has nothing left to say (Save Changes never leaves behind an edit that was since undone).
    Returns None."""
    if os.path.exists(path):
        os.remove(path)
    return None


def write_saved_markers(source_path, markers):
    """Writes `markers` (a list of dicts, each with a numeric "time" and a "label") as the recording's saved markers, replacing
    what was saved. Returns the file's path."""
    markers = list(markers)
    for m in markers:
        if not isinstance(m, dict) or not _is_number(m.get("time")):
            raise ValueError(f"a marker is a dict with a numeric 'time' (seconds), not {m!r}")
    return _dump(marker_sidecar_path(source_path), markers)


def _check_splice(s):
    if not (isinstance(s, dict) and s.get("mode") in SPLICE_MODES and _is_number(s.get("start")) and _is_number(s.get("end"))
            and s["start"] < s["end"]):
        raise ValueError(f"a splice is {{'mode': {' or '.join(repr(m) for m in SPLICE_MODES)}, 'start', 'end'}} with start before end, not {s!r}")


def write_saved_splices(source_path, splices):
    """Writes `splices` ([{"mode": "cut_out" or "keep_inside", "start", "end"}], in the order they were applied) as the
    recording's saved splices, replacing what was saved; with none, the file is removed (returns None). Returns the path."""
    splices = list(splices)
    for s in splices:
        _check_splice(s)
    splices = [{"mode": s["mode"], "start": s["start"], "end": s["end"]} for s in splices]
    path = splice_sidecar_path(source_path)
    return _dump(path, splices) if splices else _remove(path)


def write_saved_signal_labels(source_path, labels):
    """Writes `labels` ({signal key: the name it was given}) as the recording's saved signal names, replacing what was saved;
    with none, the file is removed (returns None). Returns the path."""
    labels = {str(k): str(v) for k, v in dict(labels).items() if str(v).strip()}
    path = signal_labels_path(source_path)
    return _dump(path, labels, ensure_ascii=False) if labels else _remove(path)


def write_saved_annotations(source_path, filename, items):
    """Writes a list of dicts to its own file in the saved state (HIGHLIGHTS_FILE, TEXT_ANNOTATIONS_FILE), replacing what
    was saved. Returns the path."""
    items = list(items)
    if not all(isinstance(i, dict) for i in items):
        raise ValueError(f"{filename} holds a list of dicts")
    return _dump(annotation_sidecar_path(source_path, filename), items)


def clear_saved_state(source_path):
    """Undo All Changes, saved: deletes every file in the recording's JSON saves folder (not the folder, and nothing outside
    it) so the recording opens as recorded. Returns the names of the files removed."""
    folder = saved_state_dir(source_path)
    if not os.path.isdir(folder):
        return []
    removed = []
    for name in sorted(os.listdir(folder)):
        entry = os.path.join(folder, name)
        if os.path.isfile(entry):
            os.remove(entry)
            removed.append(name)
    return removed


def saved_state_summary(source_path):
    """What is saved for the recording: {"folder", "markers" [..], "splices" [..], "signal_labels" {..}, "highlights" [..],
    "text_annotations" [..]}, each the list or dict that is saved ([] / {} for nothing). An unreadable file is an error that
    says which."""
    def read(what, fn, *args):
        try:
            return fn(source_path, *args)
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ValueError(f"The saved {what} of {source_path} cannot be read: {e}") from e

    return {"folder": saved_state_dir(source_path),
            "markers": read("markers", read_saved_markers),
            "splices": read("splices", read_saved_splices),
            "signal_labels": read("signal names", read_saved_signal_labels),
            "highlights": read("highlights", read_saved_annotations, HIGHLIGHTS_FILE),
            "text_annotations": read("text annotations", read_saved_annotations, TEXT_ANNOTATIONS_FILE)}


# ---------------------------------------------------------------------------
# What one action in the app makes
# ---------------------------------------------------------------------------

def new_marker(time, label=None, color=None, fontsize=None):
    """A marker as Add Marker stamps it: {"time", "label", "color", "fontsize"}."""
    if not _is_number(time):
        raise ValueError(f"a marker's time is a number of seconds, not {time!r}")
    return {"time": float(time), "label": label or DEFAULT_MARKER["label"], "color": color or DEFAULT_MARKER["color"],
            "fontsize": fontsize if fontsize is not None else DEFAULT_MARKER["fontsize"]}


def new_highlight(start, end, color="Yellow"):
    """A highlight as the Highlighter makes it: {"start", "end", "color"} with start before end. `color` is one of
    HIGHLIGHT_COLORS' names (the app's five) or a colour of your own (a hex string)."""
    if not (_is_number(start) and _is_number(end)) or start == end:
        raise ValueError(f"a highlight runs between two different times in seconds, not {start!r} and {end!r}")
    names = {k.lower(): v for k, v in HIGHLIGHT_COLORS.items()}
    colour = names.get(str(color).lower(), str(color))
    if colour not in HIGHLIGHT_COLORS.values() and not (colour.startswith("#") and len(colour) in (4, 7) and all(c in "0123456789abcdefABCDEF" for c in colour[1:])):
        raise ValueError(f"{color!r} is not a highlight colour: use {', '.join(HIGHLIGHT_COLORS)} or a hex colour like #FFEB3B")
    lo, hi = sorted((float(start), float(end)))
    return {"start": lo, "end": hi, "color": colour}


def new_text_annotation(x, y, text, fontsize=DEFAULT_TEXT_FONT_SIZE):
    """A text box as the Text tool places it: {"x", "y", "text", "fontsize"} (x in seconds, y in the plot's units)."""
    if not (_is_number(x) and _is_number(y)):
        raise ValueError(f"a text box is placed at numbers (x in seconds, y in the signal's units), not {x!r}, {y!r}")
    if not str(text).strip():
        raise ValueError("a text box has to say something")
    if not (isinstance(fontsize, int) and 6 <= fontsize <= 72):
        raise ValueError(f"a text box's font size is a whole number from 6 to 72, not {fontsize!r}")
    return {"x": float(x), "y": float(y), "text": str(text).strip(), "fontsize": fontsize}


def _loaded(source_path, **load):
    """The recording the way the app opens it: its saved splices replayed and its saved markers on top."""
    loaded = load_recording(source_path, **load)
    return loaded["cache"], loaded["splices"]


def add_marker(source_path, time, label=None, color=None, fontsize=None, **load):
    """Add Marker + Save Changes: puts a marker at `time` (seconds on the time axis as the recording now stands, after its saved
    splices) among the saved markers. Returns the marker. `load` is load_recording's own options (regression_method, ...)."""
    marker = new_marker(time, label, color, fontsize)
    cache, _ = _loaded(source_path, **load)
    if not (cache["x"][0] <= marker["time"] <= cache["x"][-1]):
        raise ValueError(f"{time} s is outside the recording, which runs from {cache['x'][0]:.3f} s to {cache['x'][-1]:.3f} s")
    write_saved_markers(source_path, [*cache["markers"], marker])
    return marker


def remove_markers(source_path, time, tolerance=0.05, label=None, **load):
    """Deletes the saved markers within `tolerance` seconds of `time` (and named `label`, if given), and saves. Returns
    the markers removed — [] if there were none, which is not an error here: the caller says what to make of it."""
    cache, _ = _loaded(source_path, **load)
    keep, removed = [], []
    for m in cache["markers"]:
        hit = abs(float(m["time"]) - float(time)) <= tolerance and (label is None or m.get("label") == label)
        (removed if hit else keep).append(m)
    if removed:
        write_saved_markers(source_path, keep)
    return removed


def add_splice(source_path, mode, start, end, **load):
    """Splice + Save Changes: stacks a splice on the recording's saved ones — `mode` "cut_out" (remove start..end and close the
    gap) or "keep_inside" (keep only start..end), `start` and `end` in seconds on the time axis as it now stands. The splice is
    tried first: a range that leaves fewer than 2 samples is an error and nothing is written. Returns the saved splices."""
    _check_splice({"mode": mode, "start": start, "end": end})                                      # before anything is loaded
    cache, applied = _loaded(source_path, **load)
    if apply_splice(cache, mode, float(start), float(end), regression_method=load.get("regression_method", "ols")) is None:
        raise ValueError(f"a {mode} of {start}..{end} s leaves too little of the recording (fewer than 2 samples)")
    splices = [*applied, {"mode": mode, "start": float(start), "end": float(end)}]
    write_saved_splices(source_path, splices)
    return splices


def remove_splice(source_path, number):
    """Takes saved splice `number` (1 = the first applied) out of the stack and saves; the others stay in order and are
    replayed on the full recording the next time it opens. Returns the splices left."""
    splices = read_saved_splices(source_path)
    if not 1 <= number <= len(splices):
        raise ValueError(f"there is no saved splice {number}; the recording has {len(splices)}")
    del splices[number - 1]
    write_saved_splices(source_path, splices)
    return splices


def rename_signals(source_path, names, **load):
    """Plot -> rename a signal + Save Changes: `names` is {signal key: new name}; an empty name takes the rename away (the
    signal goes back to its own). A key the recording does not have, or a name that two signals would share, is an error (the
    latter as in the app). Returns the names saved."""
    cache, _ = _loaded(source_path, **load)
    signals = cache.get("signals") or {}
    unknown = [k for k in names if k not in signals]
    if unknown:
        have = ", ".join(signals) or "none (this recording has no named signals)"
        raise ValueError(f"no signal {', '.join(repr(k) for k in unknown)} in this recording; its signals are: {have}")
    plan_signal_renames(cache, {k: v for k, v in names.items() if str(v).strip()})        # ValueError if two signals would share a name
    saved = read_saved_signal_labels(source_path)
    for key, name in names.items():
        if str(name).strip():
            saved[key] = str(name).strip()
        else:
            saved.pop(key, None)
    write_saved_signal_labels(source_path, saved)
    return saved


def add_highlight(source_path, start, end, color="Yellow"):
    """Highlighter + Save Changes: shades start..end (seconds). Returns the highlight."""
    item = new_highlight(start, end, color)
    write_saved_annotations(source_path, HIGHLIGHTS_FILE, [*read_saved_annotations(source_path, HIGHLIGHTS_FILE), item])
    return item


def add_text(source_path, x, y, text, fontsize=DEFAULT_TEXT_FONT_SIZE):
    """Text tool + Save Changes: puts a text box at (x seconds, y in the signal's units). Returns it."""
    item = new_text_annotation(x, y, text, fontsize)
    write_saved_annotations(source_path, TEXT_ANNOTATIONS_FILE, [*read_saved_annotations(source_path, TEXT_ANNOTATIONS_FILE), item])
    return item
