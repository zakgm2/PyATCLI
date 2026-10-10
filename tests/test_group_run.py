"""Group analysis end to end, headless (analysis/group/scan.py and run.py): scanning what each recording
holds, loading one for analysis, running the whole analysis on stand-in recordings, and the settings file that
makes a run repeatable."""

import datetime
import inspect
import json
import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import tdt

import PyATCLI as pl
from PyATCLI.analysis.group import run as group_run, scan as group_scan
from PyATCLI.analysis.group import GroupSpec
from PyATCLI.analysis.group.report import REPORT_FILENAME


def fake_block(epocs, n=20_000, fs=1000.0, duration=None, blockname="Rat7-250101-101010"):
    """read_block standing in for the tdt SDK. `epocs` = {store: (onsets, offsets)}."""
    t = np.arange(n) / fs
    y = (60 + 5 * np.sin(t)).astype(np.float32)
    heads = tdt.StructType()
    heads.stores = {f"{k}_": types.SimpleNamespace(type_str="epocs", name=f"{k}/") for k in epocs}
    heads.stores["x465A"] = types.SimpleNamespace(type_str="streams", name="x465A")
    data = tdt.StructType()
    data.streams = {"x465A": types.SimpleNamespace(data=y, fs=fs)}
    data.scalars = tdt.StructType()
    data.epocs = tdt.StructType()
    if duration is not None:
        data.info = types.SimpleNamespace(duration=datetime.timedelta(seconds=duration), blockname=blockname,
                                          start_date=datetime.datetime(2025, 12, 22, 21, 4, 8))

    def read_block(path, headers=None, evtype=None, store=None, verbose=None):
        if isinstance(headers, int) and headers == 1:
            return heads
        if store is not None:
            key = store.rstrip("/")
            onsets, offsets = epocs[key]
            block = tdt.StructType()
            block.epocs = tdt.StructType()
            block.epocs[f"{key}_"] = types.SimpleNamespace(name=key, onset=np.asarray(onsets, float),
                                                          offset=np.asarray(offsets, float))
            return block
        if evtype and "streams" not in evtype:
            slim = tdt.StructType()
            slim.streams = tdt.StructType()
            slim.scalars = tdt.StructType()
            slim.epocs = tdt.StructType()
            if hasattr(data, "info"):
                slim.info = data.info
            return slim
        return data

    return read_block


EPOCS = {"L1P": ([3.0, 12.0], [4.0, 13.0])}


def make_tdt_folder(parent, name):
    path = os.path.join(parent, name)
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, name + ".Tbk"), "wb").close()
    return path


class AnalysisResultStub:
    tables = {}
    summary = {"files": []}
    notes = []
    parameters = {}
    kind = "group_analysis"


class Recorder:
    def __init__(self):
        self.fractions = []

    def __call__(self, fraction, message=""):
        self.fractions.append(fraction)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def patch_block(self, **kw):
        patcher = mock.patch.object(tdt, "read_block", fake_block(EPOCS, **kw))
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, folder, name, obj):
        d = os.path.join(folder, "JSON saves")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f)


class TestScan(TempDirCase):
    def setUp(self):
        super().setUp()
        self.patch_block(duration=20.0)
        self.folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")

    def test_a_recording_lists_its_events_under_the_names_the_analyses_use(self):
        out = group_scan.scan_recording(self.folder, self.tmp)
        self.assertIsNone(out["error"])
        self.assertEqual(out["groups"], {"L1P¹": [3.0, 12.0], "L1P⁰": [4.0, 13.0]})
        self.assertEqual(out["n_events"], 4)
        self.assertEqual(out["t_range"], (0.0, 20.0))
        self.assertEqual(out["block_name"], "Rat7-250101-101010")
        self.assertEqual(out["start_time"], "2025-12-22T21:04:08")
        self.assertEqual((out["name"], out["label"]), ("Rat7-250101-101010", "Rat7-250101-101010"))
        self.assertEqual(out["warnings"], [])

    def test_saved_markers_are_counted_with_the_recorded_events(self):
        self.save(self.folder, "markers.json", [{"time": 7.0, "label": "Note", "color": "green", "fontsize": 8}])
        out = group_scan.scan_recording(self.folder, self.tmp)
        self.assertEqual(out["groups"]["Note"], [7.0])
        self.assertEqual((out["n_saved_markers"], out["n_events"]), (1, 5))

    def test_saved_splices_move_the_events_and_shorten_the_recording(self):
        self.save(self.folder, "splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        out = group_scan.scan_recording(self.folder, self.tmp)
        self.assertEqual((out["n_splices_saved"], out["n_splices"]), (1, 1))
        self.assertLess(out["t_range"][1], 19.0)                                       # 3 s shorter than the 20 s recorded
        self.assertLess(out["groups"]["L1P¹"][1], 12.0)                                # the later events slid back

    def test_a_store_rename_changes_the_marker_names(self):
        out = group_scan.scan_recording(self.folder, self.tmp, store_labels={"L1P": "Lever"})
        self.assertEqual(set(out["groups"]), {"Lever⁰", "Lever¹"})

    def test_a_folder_that_is_not_a_block_comes_back_as_an_error_not_an_exception(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        out = group_scan.scan_recording(empty, self.tmp)
        self.assertIn("Invalid Folder", out["error"])
        self.assertEqual(out["groups"], {})

    def test_unreadable_saved_files_are_warnings_and_the_recording_is_still_scanned(self):
        self.save(self.folder, "markers.json", {"not": "a list"})
        out = group_scan.scan_recording(self.folder, self.tmp)
        self.assertIsNone(out["error"])
        self.assertTrue(out["warnings"][0].startswith("Saved markers could not be read"))
        self.assertEqual(out["n_events"], 4)

    def test_scanning_many_goes_in_order_and_one_failure_stops_nothing(self):
        other = make_tdt_folder(self.tmp, "Rat8-250102-101010")
        bad = os.path.join(self.tmp, "bad")
        os.makedirs(bad)
        rec = Recorder()
        out = group_scan.scan_recordings([self.folder, bad, other], self.tmp, progress=rec)
        self.assertEqual([o["name"] for o in out], ["Rat7-250101-101010", "bad", "Rat8-250102-101010"])
        self.assertEqual([o["error"] is None for o in out], [True, False, True])
        self.assertEqual(rec.fractions[-1], 1.0)
        self.assertEqual(rec.fractions, sorted(rec.fractions))

    def test_default_subject_names_are_the_folder_names_unless_two_collide(self):
        a = make_tdt_folder(os.path.join(self.tmp, "cohortA"), "Day1")
        b = make_tdt_folder(os.path.join(self.tmp, "cohortB"), "Day1")
        c = make_tdt_folder(os.path.join(self.tmp, "cohortA"), "Day2")
        self.assertEqual(group_scan.default_subject_names([a, c], self.tmp), ["Day1", "Day2"])
        self.assertEqual(group_scan.default_subject_names([a, b], self.tmp), ["cohortA/Day1", "cohortB/Day1"])


class TestLoadForGroupAnalysis(TempDirCase):
    def setUp(self):
        super().setUp()
        self.patch_block()
        self.folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")

    def test_what_comes_back_is_the_normalized_trace_and_its_events(self):
        out = group_run.load_group_recording(self.folder)
        self.assertEqual(set(out), {"x", "y", "fs", "groups", "n_splices", "warnings"})
        self.assertEqual(len(out["x"]), 20_000)
        self.assertEqual(out["y"].dtype, np.float64)
        self.assertEqual(out["fs"], 1000.0)
        self.assertEqual(out["groups"], {"L1P¹": [3.0, 12.0], "L1P⁰": [4.0, 13.0]})
        self.assertEqual((out["n_splices"], out["warnings"]), (0, []))
        loaded = pl.load_tdt_recording(self.folder)["cache"]
        np.testing.assert_array_equal(out["y"], pl.normalized_signal(loaded)[2])         # the very trace the app analyses

    def test_saved_splices_and_markers_and_renames_are_all_honoured(self):
        self.save(self.folder, "splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        self.save(self.folder, "markers.json", [{"time": 9.0, "label": "Mine", "color": "red", "fontsize": 8}])
        out = group_run.load_group_recording(self.folder, store_labels={"L1P": "Lever"})
        self.assertEqual(out["n_splices"], 1)
        self.assertLess(len(out["x"]), 20_000)
        self.assertEqual(set(out["groups"]), {"Lever⁰", "Lever¹", "Mine"})
        self.assertEqual(out["groups"]["Mine"], [9.0])

    def test_a_folder_that_is_not_a_block_raises(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaisesRegex(ValueError, "Invalid Folder"):
            group_run.load_group_recording(empty)


# --------------------------------------------------------------------------------------------------
# The whole run, on stand-in recordings
# --------------------------------------------------------------------------------------------------

def simulated_loader(n_sub=6, seed=0, fail_on=None, warn_on=None):
    """load(folder, regression_method, store_labels, progress) for stand-in recordings: 'big' events respond
    twice as strongly as 'small' ones. Records what it was asked."""
    rng = np.random.default_rng(seed)
    data, calls = {}, []
    amp = {"big": 2.0, "small": 1.0}
    for s in range(n_sub):
        gain = rng.lognormal(0, 0.3)
        events = {m: [round(float(e), 2) for e in np.sort(rng.uniform(30, 270, 12))] for m in amp}
        x = np.arange(30_000) / 100.0
        y = 0.02 + 0.005 * rng.standard_normal(len(x))
        for m, times in events.items():
            for e in times:
                d = x - e
                y = y + np.where(d >= 0, gain * amp[m] * np.exp(-np.clip(d, 0, None) / 1.5), 0.0)
        data[f"f{s}"] = {"x": x, "y": y, "fs": 100.0, "groups": events, "n_splices": 0, "warnings": []}

    def load(folder, regression_method, store_labels, progress):
        calls.append((folder, regression_method, store_labels))
        if folder == fail_on:
            raise OSError("boom")
        out = dict(data[folder])
        if folder == warn_on:
            out["warnings"] = ["markers.json was unreadable"]
        if progress:
            progress(1.0, "loaded")
        return out

    return load, calls


def simulated_spec(n_sub=6, **kw):
    return GroupSpec(group_name="G", subjects=[{"subject": f"S{i}", "folder": f"f{i}"} for i in range(n_sub)],
                     markers=["big", "small"], metrics=["auc", "peak"], smooth_seconds=0.5, **kw)


class TestRunGroupAnalysis(TempDirCase):
    def test_every_recording_is_loaded_in_order_with_the_specs_own_settings(self):
        load, calls = simulated_loader()
        spec = simulated_spec(regression_method="huber", store_labels={"L1P": "Lever"})
        results = group_run.run_group_analysis(spec, load=load)
        self.assertEqual([c[0] for c in calls], [f"f{i}" for i in range(6)])
        self.assertTrue(all(c[1:] == ("huber", {"L1P": "Lever"}) for c in calls))
        self.assertEqual(results.files, [])                                               # no out_dir, nothing written
        pw = results.pairwise[(results.pairwise["measure"] == "peak") & (results.pairwise["basis"] == "trials")]
        self.assertGreater(pw.iloc[0]["difference"], 0)                                   # 'big' peaks higher than 'small'

    def test_with_an_output_folder_the_tables_and_the_report_are_written(self):
        load, _ = simulated_loader()
        out = os.path.join(self.tmp, "G")
        results = group_run.run_group_analysis(simulated_spec(), out_dir=out, load=load)
        self.assertTrue(results.files)
        self.assertTrue(all(os.path.isfile(p) for p in results.files))
        self.assertIn(REPORT_FILENAME, [os.path.basename(p) for p in results.files])

    def test_extra_writers_run_after_the_tables_and_their_files_are_listed(self):
        load, _ = simulated_loader()
        out = os.path.join(self.tmp, "G")
        seen = []

        def writer(results, out_dir):
            seen.append((results, out_dir))
            path = os.path.join(out_dir, "figure_traces.txt")
            with open(path, "w") as f:
                f.write("stand-in for a figure")
            return [path]

        results = group_run.run_group_analysis(simulated_spec(), out_dir=out, load=load, extra_writers=[writer])
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], results)
        self.assertEqual(seen[0][1], out)
        self.assertEqual(os.path.basename(results.files[-1]), "figure_traces.txt")
        self.assertTrue(all(os.path.isfile(p) for p in results.files))

    def test_extra_writers_do_not_run_when_nothing_is_written(self):
        load, _ = simulated_loader()
        called = []
        group_run.run_group_analysis(simulated_spec(), load=load, extra_writers=[lambda r, d: called.append(1) or []])
        self.assertEqual(called, [])

    def test_a_recording_that_cannot_be_loaded_stops_the_run_and_says_which(self):
        load, _ = simulated_loader(fail_on="f2")
        with self.assertRaises(RuntimeError) as cm:
            group_run.run_group_analysis(simulated_spec(), load=load)
        self.assertEqual(str(cm.exception), "Could not load 'S2' (f2): boom")
        self.assertIsInstance(cm.exception.__cause__, OSError)

    def test_a_recordings_warnings_become_notes_with_its_subject_name(self):
        load, _ = simulated_loader(warn_on="f1")
        results = group_run.run_group_analysis(simulated_spec(), load=load)
        self.assertIn({"level": "warning", "measure": "", "text": "S1: markers.json was unreadable"}, results.notes)

    def test_progress_runs_from_zero_to_a_hundred_percent(self):
        load, _ = simulated_loader()
        rec = Recorder()
        group_run.run_group_analysis(simulated_spec(), load=load, progress=rec)
        self.assertEqual(rec.fractions[-1], 1.0)
        self.assertEqual(rec.fractions, sorted(rec.fractions))

    def test_it_is_exported_and_the_default_loader_is_the_library_one(self):
        self.assertIs(pl.run_group_analysis, group_run.run_group_analysis)
        self.assertIs(pl.load_group_recording, group_run.load_group_recording)
        default = inspect.signature(group_run.run_group_analysis).parameters["load"].default
        self.assertIs(default, group_run.load_group_recording)


# --------------------------------------------------------------------------------------------------
# The settings file
# --------------------------------------------------------------------------------------------------

class TestSettingsFile(TempDirCase):
    def scans(self):
        return [
            {"folder": "f0", "block_name": "B0", "start_time": "2025-12-22T21:04:08", "groups": {"big": [1.0, 2.0], "small": [3.0]},
             "n_splices": 1, "n_saved_markers": 2},
            {"folder": "f1", "block_name": "B1", "start_time": None, "groups": {"big": [1.0]}, "n_splices": 0, "n_saved_markers": 0},
            {"folder": "f-not-in-the-group", "block_name": "X", "start_time": None, "groups": {}, "n_splices": 0, "n_saved_markers": 0},
        ]

    def test_the_file_records_the_spec_the_recordings_and_what_wrote_it(self):
        spec = simulated_spec(n_sub=2, pre=8.0, response=(0.0, 6.0), regression_method="ransac")
        group_dir = os.path.join(self.tmp, "Group Analysis", "G")
        path = group_run.write_group_settings(group_dir, spec, self.scans(), app={"name": "PyAT", "version": "3.2.0"})
        self.assertEqual(path, os.path.join(group_dir, pl.GROUP_SETTINGS_FILENAME))
        doc = read_json(path)
        self.assertEqual(doc["app"], {"name": "PyAT", "version": "3.2.0", "library_version": pl.__version__})
        datetime.datetime.fromisoformat(doc["created"])
        self.assertEqual(doc["spec"], spec.to_dict())
        self.assertEqual([r["subject"] for r in doc["recordings"]], ["S0", "S1"])        # only the recordings in the group
        self.assertEqual(doc["recordings"][0]["events_per_marker"], {"big": 2, "small": 1})
        self.assertEqual((doc["recordings"][0]["saved_splices_applied"], doc["recordings"][0]["saved_markers"]), (1, 2))

    def test_the_settings_file_is_also_a_provenance_record_that_can_be_replayed(self):
        from PyATCLI import provenance as pv, reproduce as R
        spec = simulated_spec(n_sub=2)
        folder = make_tdt_folder(self.tmp, "Rat1-250101")
        scans = [{"folder": folder, "block_name": "B", "start_time": None, "groups": {"big": [1.0]}, "n_splices": 0, "n_saved_markers": 0}]
        spec.subjects = [{"subject": "S0", "folder": folder}, {"subject": "S1", "folder": "f1"}]
        path = group_run.write_group_settings(os.path.join(self.tmp, "G"), spec, scans, app={"name": "PyAT", "version": "3.2.0"})
        rec = pv.read_record(path)                                                        # it is one
        self.assertEqual((rec["kind"], rec["schema"]), ("group_analysis", pv.SCHEMA_VERSION))
        self.assertEqual(rec["analysis"]["parameters"], spec.to_dict())
        self.assertEqual(rec["software"]["app"], {"name": "PyAT", "version": "3.2.0"})
        self.assertEqual(rec["software"]["library"], pl.__version__)
        self.assertEqual(rec["pipeline"]["motion_correction"]["method"], spec.regression_method)
        self.assertEqual(rec["recordings"][0]["digest"], pv.file_fingerprint(folder)["digest"])        # the files it was run on
        read_back, _ = group_run.read_group_settings(path)                               # and the old reader still works
        self.assertEqual(read_back.to_dict(), spec.to_dict())
        fake = AnalysisResultStub()
        with mock.patch.object(R, "run_group_from_spec", return_value=(fake, None)) as run:
            R.reproduce(path)
        self.assertEqual(run.call_args.args[0], spec.to_dict())

    def test_without_an_app_it_names_the_library(self):
        path = group_run.write_group_settings(self.tmp, simulated_spec(n_sub=2), self.scans())
        self.assertEqual(read_json(path)["app"], {"name": "PyATCLI", "library_version": pl.__version__})

    def test_reading_it_back_gives_the_same_spec_from_the_file_or_its_folder(self):
        spec = simulated_spec(n_sub=2, pre=8.0, baseline=(-8.0, -4.0), response=(0.0, 6.0), correction="bonferroni",
                              store_labels={"L1P": "Lever"})
        group_dir = os.path.join(self.tmp, "G")
        path = group_run.write_group_settings(group_dir, spec, self.scans())
        for where in (path, group_dir):
            back, doc = group_run.read_group_settings(where)
            self.assertEqual(back.to_dict(), spec.to_dict())
            self.assertEqual(back.baseline, (-8.0, -4.0))                                  # a tuple again, not a list
            self.assertIn("recordings", doc)

    def test_a_settings_file_is_enough_to_run_the_analysis_again(self):
        load, _ = simulated_loader()
        spec = simulated_spec()
        first = group_run.run_group_analysis(spec, load=load)
        path = group_run.write_group_settings(os.path.join(self.tmp, "G"), spec, [])
        back, _ = group_run.read_group_settings(path)
        again = group_run.run_group_analysis(back, load=simulated_loader()[0])
        np.testing.assert_array_equal(first.pairwise["difference"].to_numpy(), again.pairwise["difference"].to_numpy())


if __name__ == "__main__":
    unittest.main()
