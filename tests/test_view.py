"""viewplan.py: what the main plot of a recording shows, as data — its lines (under what name, with what numbers), the range, the markers in it — and the
data CSVs and the replay of a view's numbers from its provenance record. PyAT draws it (figures/view.py: colours, widths, the figure; its tests are PyAT's)."""

import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

import PyATCLI as pl
from PyATCLI import ops, viewplan as V

from tests.test_analyses import make_cache
from tests.test_cli import CliCase, run_cli

ROOT = Path(__file__).resolve().parent.parent


def oxysoft_cache(thb=False):
    t = np.arange(600) / 10.0
    rng = np.random.default_rng(1)
    dataset = types.SimpleNamespace(
        metadata={"n_channels": 2, "fit_factor_mean": 91.25}, num_channels=4, signals=rng.normal(size=(4, 600)),
        folder_name="NIRS-1", num_samples=len(t), sample_rate=10.0, events=[{"sample": 100, "label": "A"}])
    cache = pl.make_oxysoft_cache("nirs.txt", dataset)
    if thb:
        cache["thb"] = rng.normal(size=(2, 600))
    return cache


def generic_cache():
    t = np.arange(50.0)
    b = np.cos(t / 5)
    b[10:14] = np.nan
    table = types.SimpleNamespace(headers=["Time (s)", "Left", "Right"], data=np.column_stack([t, np.sin(t / 5), b]), name="Sheet1")
    return pl.make_generic_cache("table.csv", table)


NORMALIZED, MAIN = "Normalized (dF/F)", "Main Driver (x465A)"


class TestPlannedTraces(unittest.TestCase):
    def test_a_tdt_recording_shows_its_normalized_trace_by_default(self):
        traces, title, x_label, y_label = V.view_traces(make_cache())
        (t,) = traces
        self.assertEqual((t.label, t.csv_name, t.key, t.role, t.signal, t.color), (NORMALIZED, NORMALIZED, NORMALIZED, "signal", "normalized", None))
        self.assertEqual((title, x_label, y_label), (f"{NORMALIZED} — Rec", "Time (s)", "Amplitude"))

    def test_a_trace_says_what_it_is_and_never_how_it_looks(self):
        (t,) = V.view_traces(make_cache())[0]
        for style in ("lw", "alpha", "marker", "markersize", "snap", "default_color"):
            self.assertFalse(hasattr(t, style), style)

    def test_a_colour_the_cache_carries_is_passed_on(self):
        cache = make_cache()
        cache["signals"]["normalized"]["color"] = "purple"
        self.assertEqual(V.view_traces(cache)[0][0].color, "purple")

    def test_one_signal_ticked_shows_that_one_and_two_are_overlaid_in_the_recordings_order(self):
        cache = make_cache()
        (t,) = V.view_traces(cache, signals=["main_driver"])[0]
        self.assertEqual((t.label, t.signal), (MAIN, "main_driver"))
        traces, title, _, _ = V.view_traces(cache, signals=["main_driver", "normalized"])
        self.assertEqual([t.label for t in traces], [NORMALIZED, MAIN])
        self.assertEqual(title, "Overlay (2) — Rec")

    def test_a_signal_the_recording_does_not_have_falls_back_to_the_first(self):
        (t,) = V.view_traces(make_cache(), signals=["isosbestic"])[0]
        self.assertEqual(t.label, NORMALIZED)

    def test_the_active_signal_is_the_one_shown_when_nothing_ticked_is_there(self):
        traces, _, _, _ = V.view_traces(make_cache(), signals=["isosbestic", "red"], active="main_driver")
        self.assertEqual([t.label for t in traces], [MAIN])

    def test_a_source_with_no_signals_map_shows_its_corrected_trace(self):
        cache = make_cache()
        del cache["signals"]
        (t,) = V.view_traces(cache)[0]
        self.assertEqual((t.label, t.role), ("dF/F (corrected)", "corrected"))
        assert_array_equal(t.y, cache["corr"])

    def test_an_oxysoft_recording_has_a_line_per_channel_and_a_mean_per_group(self):
        traces, title, x_label, y_label = V.view_traces(oxysoft_cache(thb=True))
        self.assertEqual([t.label for t in traces],
                         ["O2Hb channels", "HHb channels", "_nolegend_", "_nolegend_", "Mean O2Hb  [FF: 91.2%]", "Mean HHb  [FF: 91.2%]", "Mean tHb  [FF: 91.2%]"])
        self.assertEqual([t.role for t in traces], ["o2hb_channel", "hhb_channel"] * 2 + ["o2hb_mean", "hhb_mean", "thb_mean"])
        self.assertEqual([t.key for t in traces[:4]], ["O2Hb channels", "HHb channels"] * 2)               # a group's lines follow the group's name
        self.assertEqual([t.csv_name for t in traces[:4]], ["O2Hb ch1", "HHb ch1", "O2Hb ch2", "HHb ch2"])
        self.assertEqual((title, y_label), ("NIRS — NIRS-1", "Delta Concentration (uM)"))

    def test_a_generic_table_has_each_column_without_its_gaps_and_keeps_them_in_the_data(self):
        traces, title, x_label, y_label = V.view_traces(generic_cache())
        right = next(t for t in traces if t.label == "Right")
        self.assertEqual((len(right.x), len(right.y_full)), (46, 50))
        self.assertEqual([t.role for t in traces], ["column", "column"])
        self.assertEqual((title, x_label, y_label), ("Sheet1", "Time (s)", "Value"))

    def test_the_range_has_a_little_room_and_a_single_point_does_not_break_it(self):
        x = np.arange(0.0, 100.0)
        lo, hi = V.padded_xlim(x)
        self.assertAlmostEqual(lo, -0.02 * 99)
        self.assertAlmostEqual(hi, 99 + 0.02 * 99)
        self.assertEqual(V.padded_xlim(np.array([5.0])), (4.0, 6.0))


class TestTheData(unittest.TestCase):
    def test_it_holds_every_sample_in_view_one_column_per_trace(self):
        cache = make_cache()
        view = V.plan_view(cache, signals=["normalized", "main_driver"], xlim=(10.0, 12.0))
        table = view.data_table()
        inside = (cache["x"] >= 10.0) & (cache["x"] <= 12.0)
        self.assertEqual(list(table.columns), ["time_s", NORMALIZED, MAIN])
        assert_array_equal(table["time_s"].to_numpy(), cache["x"][inside])
        assert_array_equal(table[MAIN].to_numpy(), cache["signals"]["main_driver"]["y"][inside])

    def test_oxysoft_channels_get_names_of_their_own_and_a_generic_table_its_own_x_and_gaps(self):
        columns = list(V.plan_view(oxysoft_cache(), xlim=(0.0, 10.0)).data_table().columns)
        self.assertEqual(columns[:5], ["time_s", "O2Hb ch1", "HHb ch1", "O2Hb ch2", "HHb ch2"])
        self.assertIn("Mean O2Hb  [FF: 91.2%]", columns)
        table = V.plan_view(generic_cache(), xlim=(0.0, 49.0)).data_table()
        self.assertEqual((list(table.columns), len(table), int(table["Right"].isna().sum())), (["Time (s)", "Left", "Right"], 50, 4))

    def test_two_traces_with_one_name_do_not_overwrite_each_other(self):
        view = V.plan_view(make_cache())
        view.traces.append(V.Trace("dup", view.x, view.x * 2, view.traces[0].csv_name, y_full=view.x * 2))
        self.assertEqual(list(view.data_table().columns), ["time_s", NORMALIZED, f"{NORMALIZED} (2)"])

    def test_the_files_are_the_data_and_the_markers_in_view(self):
        cache = make_cache()
        cache["markers"] = [{"time": 11.0, "label": "lever", "color": "green", "fontsize": 8}]
        tmp = self.make_tmp()
        paths = V.plan_view(cache, xlim=(10.0, 12.0)).write_data(os.path.join(tmp, "out"))
        self.assertEqual([os.path.basename(p) for p in paths], ["out_data.csv", "out_markers.csv"])
        self.assertEqual(pd.read_csv(paths[1]).iloc[0].tolist(), [11.0, "lever", "green"])
        paths = V.plan_view(cache, xlim=(20.0, 22.0)).write_data(os.path.join(tmp, "none"))
        self.assertEqual([os.path.basename(p) for p in paths], ["none_data.csv"])

    def make_tmp(self):
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="pyatcli-view-")
        self.addCleanup(shutil.rmtree, tmp, True)
        return tmp

    def test_planning_a_view_needs_no_matplotlib(self):
        code = ("import sys, numpy as np, PyATCLI as pl\n"
                "x = np.arange(100) / 10.0\n"
                "cache = {'source': 'Generic', 'x': x, 'store': 's', 'y_columns': {'a': np.sin(x)}, 'markers': []}\n"
                "r = pl.run_view(cache, {'x_range': [1, 5]})\n"
                "sys.exit(1 if 'matplotlib' in sys.modules or len(r.tables['data']) != 41 else 0)")
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=ROOT).returncode, 0)


class TestReplay(unittest.TestCase):
    def test_a_views_parameters_make_the_same_data_again(self):
        cache = make_cache()
        cache["markers"] = [{"time": 11.0, "label": "lever", "color": "green", "fontsize": 8}]
        attributes = {"title": "T", "leg_entries": [[NORMALIZED, "GCaMP", True]], "line_colors": {NORMALIZED: "#123456"}}          # whatever the viewer drew it with
        view = V.plan_view(cache, signals=["normalized", "main_driver"], xlim=(10.0, 12.0), ylim=(-1.0, 3.0), show_grid=False, attributes=attributes)
        params = json.loads(json.dumps(view.parameters()))                       # what a record stores
        self.assertEqual((params["x_range"], params["y_range"], params["signals"], params["show_grid"]),
                         ([10.0, 12.0], [-1.0, 3.0], ["normalized", "main_driver"], False))
        self.assertEqual(params["attributes"], attributes)                       # carried as it is, never interpreted
        again = pl.run_view(cache, params)
        pd.testing.assert_frame_equal(again.tables["data"], view.data_table())
        pd.testing.assert_frame_equal(again.tables["markers"], view.markers_table())
        self.assertEqual((again.kind, again.parameters), ("view", view.parameters()))

    def test_the_attributes_change_no_number(self):
        cache = make_cache()
        plain = pl.run_view(cache, {"x_range": [10, 12]})
        drawn = pl.run_view(cache, {"x_range": [10, 12], "attributes": {"title": "anything", "anything_else": 1}})
        pd.testing.assert_frame_equal(plain.tables["data"], drawn.tables["data"])

    def test_no_markers_in_view_no_markers_table(self):
        self.assertNotIn("markers", pl.run_view(make_cache(), {"x_range": [10, 12]}).tables)

    def test_a_parameter_a_view_does_not_have_is_named(self):
        with self.assertRaisesRegex(pl.analyses.ParameterError, "no parameter 'xlim'.*x_range"):
            pl.run_view(make_cache(), {"xlim": [1, 2]})

    def test_the_parameters_default_to_all_of_the_recording(self):
        cache = make_cache()
        result = pl.run_view(cache)
        self.assertEqual(len(result.tables["data"]), len(cache["x"]))


class TestAViewRecordTheAppWrites(CliCase):
    """Export View is the app's (the command line never draws); the record it leaves beside the figure's data is the library's to replay."""

    def written(self):
        cache = pl.load_recording(self.oxy)["cache"]
        view = V.plan_view(cache, xlim=(10.0, 30.0))
        paths = view.write_data(os.path.join(self.tmp, "fig"))
        answer = ops.call("provenance.write", {"export": paths[0], "kind": "view", "parameters": view.parameters(), "recording": self.oxy,
                                               "extra_outputs": paths[1:]})
        return paths, answer["record"]

    def test_the_record_replays_and_the_data_csv_matches(self):
        paths, record = self.written()
        rep = pl.reproduce_record(record, compare=[paths[0]])
        self.assertTrue(rep.ok, [c["message"] for c in rep.comparisons])
        self.assertGreater(rep.comparisons[0]["cells"], 100)
        self.assertEqual(rep.result.kind, "view")

    def test_a_replay_through_the_command_line_says_it_matches(self):
        paths, record = self.written()
        status, out, _ = run_cli("reproduce", record, "--compare", paths[0])
        self.assertEqual(status, 0)
        self.assertIn("match", out.lower())

    def test_there_is_no_view_command(self):
        status, _, err = run_cli("view", self.oxy)
        self.assertEqual(status, 2)
        self.assertIn("invalid choice", err)


if __name__ == "__main__":
    unittest.main()
