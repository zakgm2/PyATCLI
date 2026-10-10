"""
reproduce.py
------------
Runs an analysis again from its provenance record — the file PyAT (or `python -m PyATCLI run`) writes
beside every export — and says whether the numbers come out the same.

    rep = pl.reproduce_record("Event PETH_L1P_103522.provenance.json", out_dir="again",
                       compare=["Event PETH_L1P_103522.csv"])
    rep.ok            # True if every compared export matches
    rep.warnings      # the recording changed since, a different numpy, ...
    rep.result.tables # the analysis's tables, as run now

What it does, in order: read the record; note whether the software differs from what produced it
(provenance.check_environment); find the recording (the path the record holds, or `recording=`) and check it
is the same one (its fingerprint's digest); rebuild it in exactly the state recorded — motion-correction
method, every splice, store renames — which is not necessarily the state its JSON saves/ folder is in now;
run the analysis with exactly the recorded parameters (for an event analysis that includes the event times
themselves, so the markers no longer matter); write the tables and a new record; and compare any exports
you name, column for column, to a tolerance (and, for an export written with rounded numbers, to the digits it kept).
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import provenance
from .analyses import AnalysisResult, GROUP_KIND, VIEW_KIND, kinds, run_analysis, run_group_from_spec, run_view
from .sources.load import load_dataset_file
from .sources.tables import load_any_file
from .sources.processing_TDT import process_tdt_folder
from .recording import apply_signal_labels, apply_splices, make_generic_cache, make_oxysoft_cache, make_tdt_cache

DEFAULT_TOLERANCE = 1e-4          # |a - b| <= tolerance * (1 + |b|): looser than the digits an export keeps, tighter than any real change


class ReproduceError(RuntimeError):
    """A record cannot be replayed: not an analysis, no recording to run it on, or the wrong recording (strict)."""


@dataclass
class Reproduction:
    record: dict
    result: AnalysisResult
    warnings: list = field(default_factory=list)
    written: list = field(default_factory=list)
    comparisons: list = field(default_factory=list)

    @property
    def ok(self):
        """True if there was nothing compared, or everything compared matched."""
        return all(c["ok"] for c in self.comparisons)


# ---------------------------------------------------------------------------
# The recording
# ---------------------------------------------------------------------------

def _fingerprint_differences(then, now):
    """Which files of a recording differ from what a record saw (by name, size, hash)."""
    old = {f["name"]: f for f in then.get("files", [])}
    new = {f["name"]: f for f in now.get("files", [])}
    out = [f"{n} is missing" for n in sorted(set(old) - set(new))]
    out += [f"{n} is new" for n in sorted(set(new) - set(old))]
    for n in sorted(set(old) & set(new)):
        a, b = old[n], new[n]
        if a["size"] != b["size"] or (a["sha256"] and b["sha256"] and a["sha256"] != b["sha256"]):
            out.append(f"{n} changed")
    return out


def check_recording(record, path):
    """Sentences describing how the recording at `path` differs from the one the record saw ([] = identical)."""
    then = record.get("source")
    if not then or not then.get("digest"):
        return []
    now = provenance.file_fingerprint(path)
    if now.get("kind") == "missing":
        return [f"{path} does not exist"]
    if now["digest"] == then["digest"]:
        return []
    diffs = _fingerprint_differences(then, now)
    return ["The recording is not the one this record describes: " + ("; ".join(diffs[:5]) or "its files differ")]


def load_recording_for_record(record, path, progress=None):
    """Rebuilds the recording a record describes, from its files, in the recorded state. Returns the cache
    (see recording.py). The splices applied and the names the signals had are the record's own, not whatever
    JSON saves/ holds now."""
    state = record.get("recording") or {}
    source_type = state.get("source_type")
    method = state.get("regression_method") or "ols"
    if source_type == "TDT":
        cache = make_tdt_cache(path, process_tdt_folder(path, regression_method=method, progress=progress))
    elif source_type == "Oxysoft":
        cache = make_oxysoft_cache(path, load_dataset_file(path, progress=progress))
    elif source_type == "Generic":
        g = state.get("generic") or {}
        tables = load_any_file(path, progress=progress)
        table = next((t for t in tables if t.name == g.get("table")), None)
        if table is None or g.get("x_column") not in table.headers:
            raise ReproduceError(f"{os.path.basename(path)} no longer holds table {g.get('table')!r} with column {g.get('x_column')!r}")
        missing = [c for c in g.get("y_columns", []) if c not in table.headers]
        if missing:
            raise ReproduceError(f"{os.path.basename(path)}: column(s) {missing} are gone from table {table.name!r}")
        cache = make_generic_cache(path, table, table.headers.index(g["x_column"]), [table.headers.index(c) for c in g["y_columns"]])
    else:
        raise ReproduceError(f"Don't know how to rebuild a {source_type!r} recording")
    splices = state.get("splices") or []
    if splices:
        cache, applied = apply_splices(cache, splices, regression_method=method)
        if len(applied) != len(splices):
            raise ReproduceError(f"{len(splices) - len(applied)} of the record's {len(splices)} splices cannot be applied to this recording")
    return apply_signal_labels(cache, state.get("signal_labels"))


# ---------------------------------------------------------------------------
# Comparing an export
# ---------------------------------------------------------------------------

def _is_blank(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "")


_NUMBER = re.compile(r"\s*[+-]?(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?\s*")


def _rounding_half_width(text):
    """How far the number written as `text` can be from the value it was rounded from: half a unit in its last
    digit ("13.8" -> 0.05, "1.5e-05" -> 5e-07). 0 for a number with no decimal point or exponent ("12"): a whole
    number in an export (a trial number, a count) is exact, not rounded."""
    m = _NUMBER.fullmatch(text)
    if not m or (m.group(2) is None and m.group(3) is None):
        return 0.0
    return 0.5 * 10.0 ** (int(m.group(3) or 0) - len(m.group(2) or ""))


def compare_table(expected_path, table, tolerance=DEFAULT_TOLERANCE, rounded=False):
    """Compares a CSV (an export) with a DataFrame (a result table) column for column. A number matches if it is
    within `tolerance` (see DEFAULT_TOLERANCE); with `rounded=True` also if it is within the rounding its own text
    shows — an export that wrote 13.8 for 13.846 agrees to the one decimal it kept, while a whole number in it
    must still be exact. (`rounded` is for exports whose writer formats numbers to a fixed number of digits, as
    PyAT does; for full-precision CSVs, like this library's own, "13.8" means exactly 13.8, so leave it off.)
    Everything else is compared as text, ignoring case; blank and missing are the same.

    Returns {"file", "ok", "message", "max_abs_diff", "cells"}."""
    out = {"file": os.path.basename(expected_path), "ok": False, "message": "", "max_abs_diff": 0.0, "cells": 0}
    expected = pd.read_csv(expected_path, dtype=object, keep_default_na=False, encoding="utf-8-sig")
    if list(expected.columns) != list(table.columns):
        out["message"] = f"columns differ: {list(expected.columns)} vs {list(table.columns)}"
        return out
    if len(expected) != len(table):
        out["message"] = f"{len(expected)} rows in the export, {len(table)} in the result"
        return out
    worst = 0.0
    for col in expected.columns:
        for i, (a, b) in enumerate(zip(expected[col], table[col])):
            out["cells"] += 1
            if _is_blank(a) and _is_blank(b):
                continue
            try:
                fa, fb = float(a), float(b)
            except (TypeError, ValueError):
                if str(a).strip().lower() != str(b).strip().lower():
                    out["message"] = f"row {i + 1}, {col!r}: {a!r} vs {b!r}"
                    return out
                continue
            diff = abs(fa - fb)
            worst = max(worst, diff)
            allowed = tolerance * (1.0 + abs(fb))
            if rounded:
                allowed = max(allowed, _rounding_half_width(str(a)) * 1.000001)
            if not diff <= allowed:
                out["message"] = f"row {i + 1}, {col!r}: {a} vs {fb:.6g} (difference {diff:.3g})"
                return out
    out.update(ok=True, max_abs_diff=worst, message="matches")
    return out


def _compare(compare, result, tolerance, rounded=False):
    out = []
    for item in compare or ():
        path, table_name = item if isinstance(item, (tuple, list)) else (item, None)
        if table_name is None:
            header = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
            candidates = [n for n, t in result.tables.items() if list(t.columns) == header]
            if not candidates:
                out.append({"file": os.path.basename(path), "ok": False, "max_abs_diff": 0.0, "cells": 0,
                            "message": f"none of the result tables ({', '.join(result.tables)}) has these columns: {header}"})
                continue
            table_name = candidates[0]
        c = compare_table(path, result.tables[table_name], tolerance, rounded)
        c["table"] = table_name
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Reproduce
# ---------------------------------------------------------------------------

def safe_name(text):
    return re.sub(r'[<>:"/\\|?*]+', "_", str(text)).strip() or "result"


def write_result(result, out_dir, stem=None):
    """Writes each table of `result` as <out_dir>/<stem>_<table>.csv (UTF-8, full precision); returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    stem = safe_name(stem or result.kind)
    paths = []
    for name, table in result.tables.items():
        path = os.path.join(out_dir, f"{stem}_{safe_name(name)}.csv")
        table.to_csv(path, index=False, encoding="utf-8")
        paths.append(path)
    return paths


def reproduce(record, recording=None, out_dir=None, compare=(), tolerance=DEFAULT_TOLERANCE, strict=False, progress=None,
              rounded=None):
    """Runs the analysis a provenance record describes. `record` is the record (dict) or the path of its
    .provenance.json. `recording` overrides the location the record holds (moved or renamed data).

    If `out_dir` is given the result tables are written there as CSV, with a new provenance record beside them.
    `compare` lists exports to check the result against: a path, or (path, table name); which table is found
    from the CSV's own header when not named. `rounded` says whether the exports were written with numbers rounded
    to a fixed number of digits (see compare_table): None decides from the record — PyAT's exports are, the
    command line's are not. With `strict=True` a recording that differs from the recorded one (or a software
    difference) raises ReproduceError instead of being reported in `warnings`.

    Returns a Reproduction (`.ok`, `.warnings`, `.result`, `.written`, `.comparisons`)."""
    if isinstance(record, (str, os.PathLike)):
        record = provenance.read_record(os.fspath(record))
    analysis = record.get("analysis") or {}
    kind, params = analysis.get("type", record.get("kind")), dict(analysis.get("parameters") or {})
    warnings = provenance.check_environment(record)
    if rounded is None:
        rounded = ((record.get("software") or {}).get("app") or {}).get("name") == "PyAT"

    if kind == GROUP_KIND:
        result, _ = run_group_from_spec(params, out_dir=out_dir, progress=progress)
        files = list(result.summary.get("files", []))
        return Reproduction(record, result, warnings, files, _compare(compare, result, tolerance, rounded))
    if kind not in kinds() and kind != VIEW_KIND:
        raise ReproduceError(f"A {kind!r} record describes something that is not an analysis, so there is nothing to "
                             f"run again; the analyses are: {', '.join(kinds())} (and a view's numbers)")

    path = recording or (record.get("source") or {}).get("path")
    if not path:
        raise ReproduceError("The record does not say where the recording is (it was written without its path); "
                             "pass the recording's location")
    if not os.path.exists(path):
        raise ReproduceError(f"The recording is not at {path}; pass its current location")
    mismatch = check_recording(record, path)
    if mismatch and strict:
        raise ReproduceError(mismatch[0])
    if warnings and strict:
        raise ReproduceError(warnings[0])
    warnings += mismatch

    cache = load_recording_for_record(record, path, progress=progress)
    state = record.get("recording") or {}
    if kind == VIEW_KIND:                                 # a view's numbers: its data CSV
        result = run_view(cache, params, store_labels=state.get("store_labels"))
    else:
        result = run_analysis(kind, cache, params, store_labels=state.get("store_labels"), progress=progress)

    written = []
    if out_dir:
        stem = safe_name(os.path.basename(path.rstrip("/\\")) + "_" + kind)
        written = write_result(result, out_dir, stem)
        rec = provenance.build_record(
            kind, result.parameters, source=provenance.file_fingerprint(path), recording=state,
            pipeline=provenance.pipeline_parameters(state.get("regression_method") or "ols"),
            outputs=written, notes=[f"reproduced from a record made {record.get('created', '?')}"] + warnings)
        written.append(provenance.write_record(os.path.join(out_dir, f"{stem}.provenance.json"), rec))
    return Reproduction(record, result, warnings, written, _compare(compare, result, tolerance, rounded))
