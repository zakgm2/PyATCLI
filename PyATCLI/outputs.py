"""
outputs.py
----------
Where results go once there is an Output folder: a folder per kind of data, and inside it a folder per recording, so a
recording's tables, figures and reports accumulate together across runs and sessions instead of a growing flat pile of
similar-looking filenames. PyAT's exports land here (Options -> Output folder), and so do the command line's, when it is
given `--output-folder` — the same folders, so a recording analysed from a script and in the app ends up in one place.

    <Output>/
      TDT/<recording name> (Analysis)/...
      Oxysoft/<recording name> (Analysis)/...
      Generic/<recording name> (Analysis)/...
      PT2/<recording name> (Analysis)/...
      Text Field Study/<study folder name> (Analysis)/...
      Group Analysis/<group name>/...

A recording's folder name is its own basename (a file's, without the extension) plus " (Analysis)" — not the recording's
display name, which grows a "[cut ...]" tag every time it is spliced; keying off that would split one recording's results
across a new folder each time it changes. The " (Analysis)" suffix is deliberate: without it, a results folder can end up
named identically to the real recording folder or file sitting right next to it, and it is too easy to mistake one for
the other (and delete or overwrite real data thinking it is disposable output).

Nothing written through here silently replaces an earlier result: unique_path() / unique_stem() give a repeat run
" (2)", " (3)", ... instead of overwriting the last one, and group_dir() does the same for a group name already used.

If a name collision would put two DIFFERENT recordings in one folder — two subjects both called "Day1" in different parent
folders, say, which genuinely happens — the second gets " (2)": a small marker file in each folder (SOURCE_MARKER_FILENAME)
records which absolute source path it belongs to, and is checked before an existing folder is reused.

    folder = recording_dir("D:/Output", "TDT", "D:/rats/Rat7-250101-101010")
    # D:/Output/TDT/Rat7-250101-101010 (Analysis), made if it was not there, the same one next time
"""

import json
import os
from datetime import datetime

SOURCE_MARKER_FILENAME = ".pyat_source.json"
GROUP_TYPE_DIR = "Group Analysis"

_INVALID_CHARS = '<>:"/' + chr(92) + '|?*'
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_MAX_LENGTH = 100


def safe_folder_name(name):
    """`name` as a folder name Windows will accept: characters it disallows become "_", surrounding spaces and trailing
    dots are dropped, and a reserved device name (CON, NUL, ...) gets a "_" in front. Returns "" if nothing usable is
    left."""
    cleaned = "".join("_" if (c in _INVALID_CHARS or ord(c) < 32) else c for c in str(name)).strip()
    cleaned = cleaned[:_MAX_LENGTH].rstrip(". ")
    if not cleaned:
        return ""
    if cleaned.split(".")[0].upper() in _RESERVED:
        cleaned = "_" + cleaned
    return cleaned


def recording_folder_name(source_path):
    """The folder/file's own basename — a file's without its extension, so "MyFile.xlsx" and a same-named "MyFile.csv"
    opened another day still share one folder — plus " (Analysis)", so the results folder never reads as the recording
    itself (see the module docstring)."""
    source_path = os.fspath(source_path)
    stripped = source_path.rstrip("/" + chr(92))
    base = os.path.basename(stripped)
    if os.path.isfile(source_path):
        base = os.path.splitext(base)[0]
    return f"{base} (Analysis)"


def unique_path(path):
    """`path`, or the same name with " (2)", " (3)", ... inserted before the extension if something is already there — so
    a repeat export never silently replaces an earlier one."""
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    n = 2
    while True:
        candidate = f"{root} ({n}){ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def unique_stem(folder, stem, endings):
    """`stem`, or "stem (2)", "stem (3)", ... — the first for which none of the files `stem + ending` is in `folder`. For
    results that are several files sharing one name (a table per file, and a record), so a repeat run is a new set of
    files and never a mixture of old and new."""
    n, candidate = 1, stem
    while any(os.path.exists(os.path.join(folder, candidate + e)) for e in endings):
        n += 1
        candidate = f"{stem} ({n})"
    return candidate


def _normalize(source_path):
    """So the same recording resolves to the same folder however it is re-opened (a different relative path, a different
    case on a case-insensitive filesystem, a trailing slash, ...)."""
    return os.path.normcase(os.path.normpath(os.path.abspath(source_path)))


def _read_marker_source(marker_path):
    try:
        with open(marker_path, "r", encoding="utf-8") as f:
            return json.load(f).get("source_path")
    except (OSError, ValueError):
        return None


def _write_marker(marker_path, data_type, source_path):
    try:
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump({"type": data_type, "source_path": os.fspath(source_path),
                       "created": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    except OSError:
        pass                                    # results still work without it; just no collision protection this time


def recording_dir(base, data_type, source_path):
    """<base>/<data_type>/<name> (Analysis)[ (n)]/ for one recording, auto-disambiguated: creates the folder (and its
    parents) the first time, and reuses the same one on every later call for the same source_path — including across
    separate runs, via SOURCE_MARKER_FILENAME. A different recording that happens to share a name gets "name (2)",
    "name (3)", ... instead of mixing in.

    `base` is the Output folder itself (it is made if it is not there). `data_type` is "TDT", "Oxysoft", "Generic", "PT2"
    or "Text Field Study" (the recording's cache["source"]); it is the top-level subfolder's name, through
    safe_folder_name. `source_path` is the recording's own path."""
    source_path = os.fspath(source_path)
    name = safe_folder_name(recording_folder_name(source_path)) or "Untitled (Analysis)"
    type_dir = os.path.join(os.fspath(base), safe_folder_name(data_type) or "Other")
    norm_source = _normalize(source_path)

    candidate, n = name, 1
    while True:
        folder = os.path.join(type_dir, candidate)
        marker = os.path.join(folder, SOURCE_MARKER_FILENAME)
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
            _write_marker(marker, data_type, source_path)
            return folder
        existing = _read_marker_source(marker)
        if existing is None or _normalize(existing) == norm_source:
            if existing is None:                # a folder that predates the marker (or one made by hand) — claim it
                _write_marker(marker, data_type, source_path)
            return folder
        n += 1
        candidate = f"{name} ({n})"


def group_dir(base, group_name):
    """<base>/Group Analysis/<group name>[ (n)] for a group analysis, or None if the name leaves nothing usable as a folder
    name. Not made: it is a folder with no analysis_settings.json in it yet, so a run can never overwrite an earlier one
    just because the group was named (or renamed back to) the same thing — a name already used for a finished run gets
    " (2)", " (3)", ... instead."""
    from .analysis.group.run import GROUP_SETTINGS_FILENAME

    folder = safe_folder_name(group_name)
    if not folder:
        return None
    type_dir = os.path.join(os.fspath(base), GROUP_TYPE_DIR)
    candidate, n = folder, 2
    while os.path.exists(os.path.join(type_dir, candidate, GROUP_SETTINGS_FILENAME)):
        candidate = f"{folder} ({n})"
        n += 1
    return os.path.join(type_dir, candidate)
