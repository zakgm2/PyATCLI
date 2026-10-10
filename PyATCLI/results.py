"""
results.py
----------
Reading back what the library, or a program driving it (PyAT), wrote: the numbers of a finished analysis, found from whatever the person names.

    opened = open_results("out/Group Analysis/Sucrose")          # a group analysis's folder: its settings say how it was run
    opened = open_results("out/Rec_event_peth.provenance.json")  # a record: the tables it lists
    opened = open_results("out/Rec_event_peth_trials.csv")       # one table file
    opened = open_results("out")                                 # a folder: its newest record, or else the tables in it

    opened.kind        what made them ("group_analysis", "event_peth", ... or None for a bare table file)
    opened.parameters  what it was asked (a group's settings as a dict)
    opened.tables      {name: DataFrame}; the group's folder is read by the names in analysis/group/report.py's TABLE_FILES
    opened.notes       what the analysis said ([{"level", "measure", "text"}], a group's), opened.software what it ran on
    opened.files       the files the tables were read from

No drawing happens here and no plotting library is needed: this is the data side of looking at a result. What a table is drawn as is the viewer's
business (PyAT's windows draw what this reads). The operation `results.open` is this for the engine.
"""

import os
from dataclasses import dataclass, field

import pandas as pd

from . import provenance
from .analysis.group.report import TABLE_FILES, read_group_notes
from .analysis.group.run import GROUP_SETTINGS_FILENAME, read_group_settings


class NoResults(ValueError):
    """What was named holds nothing that can be read as a result, or is not there."""


@dataclass
class OpenedResults:
    path: str
    directory: str
    kind: object = None
    parameters: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    software: dict = field(default_factory=dict)
    files: list = field(default_factory=list)


def _read_tables(paths):
    tables = {}
    for p in paths:
        if p.lower().endswith(".csv"):
            if not os.path.isfile(p):
                raise NoResults(f"{p} is not there any more")
            # round_trip: the numbers read back exactly as they were written, so a plot of a file is a plot of the result
            tables[os.path.splitext(os.path.basename(p))[0]] = pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip")
    return tables


def _from_record(path):
    record = provenance.read_record(path)
    folder = os.path.dirname(os.path.abspath(path))
    paths = [os.path.join(folder, o["name"]) for o in record.get("outputs", []) if str(o.get("name", "")).lower().endswith(".csv")]
    tables = _read_tables(paths)
    if not tables:
        raise NoResults(f"{os.path.basename(path)} lists no table files (.csv) to plot")
    return OpenedResults(os.path.abspath(path), folder, record.get("kind"), (record.get("analysis") or {}).get("parameters") or {},
                         tables, notes=[dict(n) for n in record.get("notes") or [] if isinstance(n, dict)], files=paths)


def _from_group_folder(folder):
    """A group analysis's folder: its tables by name (trials, subject_means, ..., mean_traces), the settings it was run from, its notes."""
    spec, _doc = read_group_settings(os.path.join(folder, GROUP_SETTINGS_FILENAME))
    paths, tables = [], {}
    for name, filename in TABLE_FILES.items():
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            tables[name] = next(iter(_read_tables([path]).values()))
            paths.append(path)
    if not tables:
        raise NoResults(f"{folder} holds the settings of a group analysis (*{GROUP_SETTINGS_FILENAME}) but none of its tables: it was not run, or its tables were moved")
    notes, software = read_group_notes(folder)
    return OpenedResults(os.path.abspath(folder), os.path.abspath(folder), "group_analysis", spec.to_dict(), tables, notes, software, paths)


def open_results(source):
    """The OpenedResults for a folder, a provenance record, a group's analysis_settings.json or a .csv table. Raises NoResults if there is nothing
    there to read."""
    path = os.path.abspath(os.fspath(source))
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, GROUP_SETTINGS_FILENAME)):      # a group's folder, whatever else was exported into it since
            return _from_group_folder(path)
        names = sorted(os.listdir(path))
        records = sorted((os.path.join(path, n) for n in names if n.endswith(provenance.SIDECAR_SUFFIX)), key=os.path.getmtime)
        if records:
            return _from_record(records[-1])
        paths = [os.path.join(path, n) for n in names if n.lower().endswith(".csv")]
        tables = _read_tables(paths)
        if not tables:                                       # no record, no tables: nothing here to read
            raise NoResults(f"{path} holds no provenance record (*{provenance.SIDECAR_SUFFIX}) or table file (.csv) to plot")
        return OpenedResults(path, path, tables=tables, files=paths)
    if os.path.basename(path) == GROUP_SETTINGS_FILENAME and os.path.isfile(path):
        return open_results(os.path.dirname(path))
    if path.lower().endswith(".json"):
        return _from_record(path)
    if path.lower().endswith(".csv"):
        return OpenedResults(path, os.path.dirname(path), tables=_read_tables([path]), files=[path])
    raise NoResults(f"Don't know how to plot {path}: it can be a .csv table, a .provenance.json record or a folder of them")
