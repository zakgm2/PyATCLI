"""
analysis/group/rebuild.py
-------------------------
A group analysis's GroupResults put back together from its tables: the ones `run_analysis` / run_group_analysis return, or the CSVs written to the
group's folder read back (results.open_results). What a viewer needs to show a result that is no longer in memory: the report is written from
a GroupResults (report.report_text), and so are the figures.
"""

import os

import numpy as np
import pandas as pd

from . import GroupSpec
from .report import CSV_NAMES
from .stats import FRAME_NAMES, GroupResults

_FRAME_BY_STEM = {os.path.splitext(filename)[0]: frame for frame, filename in CSV_NAMES.items()}


def _frames(tables):
    """{frame name: DataFrame} from tables keyed either by frame name (a result in memory) or by the stem of the CSV file it was written to
    (a folder read back)."""
    return {_FRAME_BY_STEM.get(key, key): table for key, table in tables.items()}


def _ordered(values):
    return list(dict.fromkeys(values))


def results_from_tables(tables, parameters=None, notes=None, software=None):
    """A GroupResults rebuilt from a group analysis's tables. `parameters` is the analysis's GroupSpec as a dict (its settings file holds it);
    without it the spec is inferred from the tables (the markers, subjects and measures in them, default windows). `notes` ([{"level",
    "measure", "text"}, ...]) and `software` are what the analysis said and ran on, which no table holds: given, the report of the rebuilt
    results reads as the original's. None if these are not a group analysis's tables (no mean traces and no subject means)."""
    frames = _frames(tables)
    traces_table = frames.get("mean_traces")
    means = frames.get("subject_means")
    if traces_table is None and means is None:
        return None
    if parameters and parameters.get("markers"):
        spec = GroupSpec.from_dict(parameters)
    else:
        source = traces_table if traces_table is not None else means
        spec = GroupSpec(group_name="Group analysis", subjects=[{"subject": s, "folder": ""} for s in _ordered(source["subject"])],
                         markers=_ordered(source["marker"]), metrics=_ordered(means["measure"]) if means is not None else ["auc"])
    traces, grid = {}, None
    if traces_table is not None and len(traces_table):
        grid = np.sort(traces_table["time"].astype(float).unique())
        for (subject, marker), g in traces_table.groupby(["subject", "marker"], sort=False):
            g = g.sort_values("time")
            traces.setdefault(subject, {})[marker] = {"n": int(g["n_trials"].iloc[0]), "mean": g["value"].to_numpy(float)}
    extra = {name: frames.get(name, pd.DataFrame()) for name in FRAME_NAMES}
    results = GroupResults(spec=spec, trials=frames.get("trials", pd.DataFrame()), traces=traces, trace_grid=grid, **extra)
    if notes is not None:
        results.notes = [dict(n) for n in notes]
    if software is not None:
        results.software = dict(software)
    return results
