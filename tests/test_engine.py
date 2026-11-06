"""The engine (engine/): recordings held open between requests, their arrays handed over as mapped .npy files, splices as pure state, and the
JSON-lines server a window talks to — in process (a Session, the same code behind the transport), over threads and pipes, and as a real
subprocess."""

import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

import PyATCLI as pl
from PyATCLI import ops
from PyATCLI.engine import wire
from PyATCLI.engine.server import serve
from PyATCLI.engine.session import Session
from PyATCLI.ops import OpError
from PyATCLI.progress import Cancelled

from tests.test_cli import CliCase, make_tdt_folder

ROOT = Path(__file__).resolve().parent.parent


class TestWire(unittest.TestCase):
    def setUp(self):
        import shutil
        import tempfile
        self.dir = tempfile.mkdtemp(prefix="pyatcli-wire-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.files = wire.ArrayFiles(self.dir)

    def roundtrip(self, obj, mmap=True):
        text = json.dumps(wire.encode(obj, self.files))                  # what crosses the pipe is JSON
        return wire.decode(json.loads(text), self.dir, mmap=mmap)

    def test_arrays_become_files_and_come_back_equal_and_mapped(self):
        a = np.linspace(0, 1, 1000)
        back = self.roundtrip({"x": a, "fs": 100.0, "store": "Rec", "markers": [{"time": 1.5, "label": "m"}], "signals": {"n": {"label": "N", "y": a * 2}}})
        assert_array_equal(back["x"], a)
        assert_array_equal(back["signals"]["n"]["y"], a * 2)
        self.assertEqual((back["fs"], back["store"], back["markers"]), (100.0, "Rec", [{"time": 1.5, "label": "m"}]))
        self.assertIsInstance(back["x"], np.memmap)

    def test_the_same_array_in_two_places_is_one_file_and_one_array(self):
        a = np.arange(10.0)
        back = self.roundtrip({"corr": a, "signals": {"normalized": {"y": a}}})
        self.assertIs(back["corr"], back["signals"]["normalized"]["y"])
        self.assertEqual(len([f for f in os.listdir(self.dir) if f.endswith(".npy")]), 1)

    def test_an_edit_in_the_receiver_never_reaches_the_file(self):
        a = np.arange(5.0)
        back = self.roundtrip({"x": a})
        back["x"][0] = 99.0                                                       # copy-on-write: the page is private
        self.assertEqual(float(np.load(os.path.join(self.dir, os.listdir(self.dir)[0]))[0]), 0.0)

    def test_numpy_scalars_nan_and_two_d_arrays(self):
        back = self.roundtrip({"n": np.int64(3), "f": np.float64(2.5), "nan": float("nan"), "o2hb": np.ones((2, 4))})
        self.assertEqual((back["n"], back["f"], back["nan"]), (3, 2.5, None))
        self.assertEqual(back["o2hb"].shape, (2, 4))

    def test_arrays_are_loaded_into_memory_if_asked(self):
        back = self.roundtrip({"x": np.arange(3.0)}, mmap=False)
        self.assertNotIsInstance(back["x"], np.memmap)

    def test_a_whole_recording_survives(self):
        from tests.test_analyses import make_cache
        cache = make_cache()
        back = self.roundtrip(cache)
        self.assertEqual(set(back), set(cache))
        assert_array_equal(back["x"], cache["x"])
        assert_array_equal(back["signals"]["main_driver"]["y"], cache["signals"]["main_driver"]["y"])
        self.assertEqual(back["detected_markers"], cache["detected_markers"])


class SessionCase(CliCase):
    def setUp(self):
        super().setUp()
        self.session = Session()
        self.addCleanup(self.session.close)
        self.events = []

    def call(self, method, params=None, cancelled=None):
        return self.session.handle(method, params, lambda event, **f: self.events.append((event, f)), cancelled)

    def open(self, path=None, **load):
        return self.call("recording.open", {"path": path or self.oxy, "load": load})

    def arrays(self, wired):
        return wire.decode(wired["cache"], wired["directory"])


class TestOpeningARecording(SessionCase):
    def test_it_comes_back_as_an_id_its_structure_and_mapped_arrays(self):
        wired = self.open()
        cache = self.arrays(wired)
        reference = pl.load_recording(self.oxy)["cache"]
        self.assertEqual(wired["id"], "r1")
        self.assertEqual((cache["source"], cache["store"], cache["fs"]), (reference["source"], reference["store"], reference["fs"]))
        assert_array_equal(cache["x"], reference["x"])
        assert_array_equal(cache["o2hb"], reference["o2hb"])
        self.assertEqual(wired["splices"], [])
        self.assertTrue(any(e == "progress" for e, _ in self.events))                # it said how far it was

    def test_the_same_file_opened_again_is_the_same_recording(self):
        self.assertEqual(self.open()["id"], self.open()["id"])
        self.assertEqual(self.session.store.ids(), ["r1"])
        self.assertEqual(self.open(regression_method="huber")["id"], "r2")             # opened another way: another recording

    def test_saved_state_is_applied_on_the_way_in(self):
        pl.saved_state.add_splice(self.oxy, "cut_out", 10.0, 20.0)
        pl.saved_state.add_marker(self.oxy, 5.0, label="Lever")
        wired = self.open()
        self.assertEqual(wired["splices"], [{"mode": "cut_out", "start": 10.0, "end": 20.0}])
        cache = self.arrays(wired)
        self.assertEqual([m["label"] for m in cache["markers"]], ["Lever"])
        self.assertAlmostEqual(len(cache["x"]), 500, delta=2)

    def test_a_tdt_recording_carries_its_motion_correction_inlier_fraction(self):
        folder = make_tdt_folder(self.tmp, "Rat9-250101-101010")
        for method in ("ransac", "ols"):
            wired = self.open(folder, regression_method=method)
            fraction = self.arrays(wired)["motion_correction_inlier_fraction"]
            self.assertTrue(0.0 < fraction <= 1.0, (method, fraction))
            self.assertEqual(wired["regression_method"], method)
        spliced = self.arrays(self.call("recording.view", {"id": wired["id"], "splices": [{"mode": "cut_out", "start": 5.0, "end": 10.0}]}))
        self.assertTrue(0.0 < spliced["motion_correction_inlier_fraction"] <= 1.0)          # of the fit on what is left

    def test_opening_says_what_a_window_needs_to_restore_its_state(self):
        plain = self.open()
        self.assertIsNone(plain["original_cache"])
        self.assertEqual(plain["restored"], {"splices": 0, "markers": None, "highlights": None, "text_annotations": None})
        self.assertEqual(plain["signal_labels"], {})
        folder = make_tdt_folder(self.tmp, "Rat8-250101-101010")
        pl.saved_state.add_splice(folder, "cut_out", 10.0, 20.0)
        pl.saved_state.add_marker(folder, 5.0, label="Lever")
        pl.saved_state.write_saved_signal_labels(folder, {"normalized": "GCaMP", "nonesuch": "x"})
        wired = self.open(folder)
        self.assertEqual(wired["restored"], {"splices": 1, "markers": 1, "highlights": None, "text_annotations": None})
        self.assertEqual(wired["signal_labels"], {"normalized": "GCaMP"})              # only the signals the recording has
        spliced, base = self.arrays(wired), wire.decode(wired["original_cache"], wired["directory"])
        self.assertGreater(len(base["x"]), len(spliced["x"]))                          # what Restore Full Recording brings back
        self.assertEqual(spliced["signals"]["normalized"]["label"], "GCaMP")
        self.assertEqual(base["signals"]["normalized"]["label"], "GCaMP")
        self.assertEqual(wired["regression_method"], "ols")

    def test_fresh_reads_the_disk_again_and_lets_go_of_the_old_one(self):
        first = self.open()
        pl.saved_state.add_marker(self.oxy, 5.0, label="Lever")                        # the files changed since it was opened
        stale = self.open()
        self.assertEqual((stale["id"], self.arrays(stale)["markers"]), (first["id"], []))
        fresh = self.call("recording.open", {"path": self.oxy, "fresh": True})
        self.assertNotEqual(fresh["id"], first["id"])
        self.assertEqual([m["label"] for m in self.arrays(fresh)["markers"]], ["Lever"])
        self.assertEqual(self.session.store.ids(), [fresh["id"]])

    def test_closing_removes_the_files_it_wrote(self):
        wired = self.open()
        self.assertTrue(os.path.isdir(wired["directory"]))
        cache = self.arrays(wired)
        del cache
        self.call("recording.close", {"id": wired["id"]})
        if os.name != "nt":                                   # on Windows a file someone still maps stays until exit
            self.assertFalse(os.path.exists(wired["directory"]))

    def test_the_tables_of_a_file_come_with_their_first_rows(self):
        path = os.path.join(self.tmp, "table.csv")
        pd.DataFrame({"t": np.arange(20) / 10, "a": np.arange(20.0), "b": np.where(np.arange(20) == 1, np.nan, 2.0)}).to_csv(path, index=False)
        table = self.call("file.tables", {"path": path, "preview": 3})["tables"][0]
        self.assertEqual((table["headers"], table["rows"]), (["t", "a", "b"], 20))
        self.assertEqual(table["preview"], [[0.0, 0.0, 2.0], [0.1, 1.0, None], [0.2, 2.0, 2.0]])      # a missing value is null, not NaN
        self.assertEqual(self.call("file.tables", {"path": path, "preview": 0})["tables"][0]["preview"], [])

    def test_a_missing_file_and_a_closed_recording_say_so(self):
        with self.assertRaises(OpError) as caught:
            self.open(os.path.join(self.tmp, "nope.txt"))
        self.assertEqual(caught.exception.code, "not_found")
        wired = self.open()
        self.call("recording.close", {"id": wired["id"]})
        with self.assertRaises(OpError) as caught:
            self.call("recording.view", {"id": wired["id"]})
        self.assertEqual(caught.exception.code, "no_recording")
        self.assertEqual(self.call("recording.list")["open"], [])

    def test_a_tdt_recording(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        cache = self.arrays(self.open(folder))
        self.assertEqual(set(cache["signals"]), {"normalized", "main_driver", "isosbestic"})
        self.assertIs(cache["corr"], cache["signals"]["normalized"]["y"])
        self.assertEqual(cache["x"].dtype, np.float64)


class TestAnalysesOnAnOpenRecording(SessionCase):
    PARAMS = {"center_s": 30.0, "pre_s": 5.0, "post_s": 5.0}

    def test_the_same_tables_as_a_run_on_the_path(self):
        wired = self.open()
        by_id = self.call("run", {"kind": "auc_window", "recording": {"id": wired["id"]}, "parameters": self.PARAMS, "tables": "inline"})
        by_path = self.call("run", {"kind": "auc_window", "recording": self.oxy, "parameters": self.PARAMS, "tables": "inline"})
        self.assertEqual(by_id["tables"], by_path["tables"])
        self.assertEqual(by_id["files"], [])                                          # nothing written unless asked

    def test_an_analysis_that_reports_nothing_of_its_own_still_starts_and_ends_the_bar(self):
        wired = self.open()
        self.events.clear()
        self.call("run", {"kind": "curve_fit", "recording": {"id": wired["id"]}, "parameters": {"t1_s": 20.0, "t2_s": 40.0}})
        fractions = [f["fraction"] for e, f in self.events if e == "progress"]
        self.assertGreaterEqual(len(fractions), 2)
        self.assertEqual(fractions, sorted(fractions))                                # never backwards
        self.assertEqual(fractions[-1], 1.0)                                          # a bar that is watched ends full
        self.assertLess(fractions[0], 0.5)

    def test_reading_from_a_path_is_the_first_part_of_the_bar_and_the_analysis_the_rest(self):
        self.call("run", {"kind": "auc_window", "recording": self.oxy, "parameters": self.PARAMS})
        fractions = [f["fraction"] for e, f in self.events if e == "progress"]
        self.assertEqual(fractions, sorted(fractions))
        self.assertEqual(fractions[-1], 1.0)

    def test_tables_can_be_files_for_the_caller_to_read(self):
        wired = self.open()
        result = self.call("run", {"kind": "zscore_window", "recording": {"id": wired["id"]}, "parameters": self.PARAMS, "tables": "files"})
        path = result["tables"]["window"]["path"]
        self.assertTrue(path.startswith(self.session.data_dir))
        table = pd.read_csv(path, float_precision="round_trip")
        direct = pl.run_analysis("zscore_window", pl.load_recording(self.oxy)["cache"], self.PARAMS).tables["window"]
        pd.testing.assert_frame_equal(table, direct, check_dtype=False)

    def test_the_callers_markers_and_names_are_what_the_analysis_sees(self):
        wired = self.open()
        ref = {"id": wired["id"], "markers": [{"time": 20.0, "label": "mine", "color": "red", "fontsize": 8},
                                              {"time": 40.0, "label": "mine", "color": "red", "fontsize": 8}]}
        result = self.call("run", {"kind": "event_intervals", "recording": ref, "tables": "inline"})
        self.assertEqual(result["tables"]["intervals"]["rows"], 2)                       # the markers given, not the (none) saved ones
        with self.assertRaisesRegex(pl.analyses.AnalysisError, "No markers"):                                  # the saved ones (none) are not the caller's
            self.call("run", {"kind": "event_intervals", "recording": {"id": wired["id"]}})

    def test_signal_names_are_applied_to_what_the_recording_gives_back(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        wired = self.open(folder)
        self.call("recording.labels", {"id": wired["id"], "signal_labels": {"main_driver": "GCaMP"}})
        view = self.call("recording.view", {"id": wired["id"]})
        self.assertEqual(self.arrays(view)["signals"]["main_driver"]["label"], "GCaMP")
        ref = {"id": wired["id"], "signal_labels": {"normalized": "dF/F mine"}}                                   # or a name only for this one request
        table = self.call("run", {"kind": "zscore_window", "recording": ref, "parameters": {"center_s": 10, "pre_s": 3, "post_s": 3}, "tables": "inline"})["tables"]["window"]
        self.assertEqual(set(table["data"]["Channel"]), {"dF/F mine"})
        again = self.call("run", {"kind": "zscore_window", "recording": {"id": wired["id"]}, "parameters": {"center_s": 10, "pre_s": 3, "post_s": 3}, "tables": "inline"})
        self.assertEqual(set(again["tables"]["window"]["data"]["Channel"]), {"Normalized (dF/F)"})              # and the recording itself is as it was

    def test_a_reference_without_an_engine_is_refused(self):
        with self.assertRaises(OpError) as caught:
            ops.call("run", {"kind": "auc_window", "recording": {"id": "r1"}, "parameters": self.PARAMS})
        self.assertEqual(caught.exception.code, "no_recording")


class TestBuildingBlocksForAWindow(SessionCase):
    def setUp(self):
        super().setUp()
        self.folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        self.wired = self.open(self.folder)
        self.ref = {"id": self.wired["id"], "markers": []}

    def reference(self):
        cache = pl.load_recording(self.folder)["cache"]
        key, label, y = pl.normalized_signal(cache)
        return cache, key, label, pl.smooth_signal(y, cache["fs"])

    def test_the_event_peth_matrix_is_the_librarys_to_the_bit(self):
        cache, key, label, clean = self.reference()
        times = [4.0, 9.0, 14.0, 19.0]
        answer = self.call("compute.event_peth", {"recording": self.ref, "event_times": times, "pre": 2.0, "post": 3.0})
        got = wire.decode(answer["result"]["data"], answer["result"]["directory"], mmap=False)
        want = pl.compute_event_zscore_peth(cache["x"], clean, times, 2.0, 3.0)
        self.assertEqual((answer["signal"], answer["label"]), (key, label))
        for name in ("time_axis", "trial_matrix", "mean_trace", "sem_trace"):
            assert_array_equal(got[name], want[name], name)
        self.assertEqual(got["trial_event_times"], want["trial_event_times"])
        self.assertTrue(any(e == "progress" for e, _ in self.events))

    def test_an_event_outside_the_recording_makes_no_trial_and_an_empty_matrix_travels(self):
        answer = self.call("compute.event_peth", {"recording": self.ref, "event_times": [500.0], "pre": 5.0, "post": 5.0})
        got = wire.decode(answer["result"]["data"], answer["result"]["directory"], mmap=False)
        self.assertEqual(got["trial_matrix"].shape, (0, 300))
        self.assertEqual(got["trial_event_times"], [])

    def test_the_markers_in_the_reference_do_not_change_the_signal_it_works_on(self):
        a = self.call("compute.event_peth", {"recording": self.ref, "event_times": [9.0], "pre": 2.0, "post": 3.0})
        b = self.call("compute.event_peth", {"recording": {**self.ref, "markers": [{"time": 1.0}], "splices": []}, "event_times": [9.0], "pre": 2.0, "post": 3.0})
        assert_array_equal(wire.decode(a["result"]["data"], a["result"]["directory"])["trial_matrix"],
                           wire.decode(b["result"]["data"], b["result"]["directory"])["trial_matrix"])

    def test_peaks_of_the_whole_recording(self):
        cache, key, label, clean = self.reference()
        answer = self.call("compute.peaks", {"recording": self.ref, "scope": "whole", "z_threshold": 2.0, "include_troughs": True, "min_distance_s": 0.5})
        want = pl.find_significant_peaks(cache["x"], clean, z_threshold=2.0, min_distance_sec=0.5, include_troughs=True)
        self.assertEqual(answer["peaks"], want)

    def test_the_peak_nearest_each_event(self):
        cache, key, label, clean = self.reference()
        times = [4.0, 9.0, 14.0]
        answer = self.call("compute.peaks", {"recording": self.ref, "scope": "events", "z_threshold": 1.0, "events": {"L1P": times}, "pre": 2.0, "post": 3.0})
        want = pl.find_peak_near_events(cache["x"], clean, times, 2.0, 3.0, z_threshold=1.0, include_troughs=False)
        self.assertEqual(answer["groups"], {"L1P": want})

    def test_every_event_type_scanned_at_once(self):
        answer = self.call("compute.peaks", {"recording": self.ref, "scope": "events", "z_threshold": 1.0, "pre": 2.0, "post": 3.0,
                                              "events": {"A": [4.0, 9.0], "B": [14.0]}})
        self.assertEqual({k: len(v) for k, v in answer["groups"].items()}, {"A": 2, "B": 1})

    def test_the_start_buffer_applies_to_the_whole_recording_search(self):
        cache, key, label, clean = self.reference()
        for buffer_s in (0.0, 5.0, 10.0):
            with self.subTest(buffer_s=buffer_s):
                answer = self.call("compute.peaks", {"recording": self.ref, "scope": "whole", "z_threshold": 2.0, "start_buffer_s": buffer_s})
                self.assertEqual(answer["peaks"], pl.find_significant_peaks(cache["x"], clean, z_threshold=2.0, start_buffer_s=buffer_s))
        default = self.call("compute.peaks", {"recording": self.ref, "scope": "whole", "z_threshold": 2.0})
        self.assertEqual(default["peaks"], pl.find_significant_peaks(cache["x"], clean, z_threshold=2.0))

    def test_the_shuffle_null_comes_with_the_peaks_near_events_when_asked_for(self):
        cache, key, label, clean = self.reference()
        times = [4.0, 9.0, 14.0]
        params = {"recording": self.ref, "scope": "events", "z_threshold": 1.0, "events": {"L1P": times}, "pre": 2.0, "post": 3.0}
        plain = self.call("compute.peaks", params)
        self.assertNotIn("null", plain)
        answer = self.call("compute.peaks", {**params, "n_shuffles": 50, "seed": 3})
        self.assertEqual(answer["groups"], plain["groups"])
        want = pl.shuffle_null_near_events(cache["x"], clean, times, 2.0, 3.0, z_threshold=1.0, n_shuffles=50, seed=3)
        self.assertEqual(answer["null"], {"L1P": {**want, "p_count_holm": want["p_count"]}})        # one event type: nothing to correct for

    def test_every_event_type_gets_its_own_null(self):
        answer = self.call("compute.peaks", {"recording": self.ref, "scope": "events", "z_threshold": 1.0, "pre": 2.0, "post": 3.0,
                                              "events": {"A": [4.0, 9.0], "B": [14.0]}, "n_shuffles": 20})
        self.assertEqual(set(answer["null"]), {"A", "B"})
        self.assertEqual({k: len(v["p_event"]) for k, v in answer["null"].items()}, {"A": 2, "B": 1})
        for null in answer["null"].values():                                                       # Holm across the two types
            self.assertGreaterEqual(null["p_count_holm"], null["p_count"])
            self.assertLessEqual(null["p_count_holm"], 1.0)

    def test_bad_parameters_are_parameter_errors(self):
        for params in ({"scope": "nope", "z_threshold": 1}, {"scope": "events", "z_threshold": 1, "pre": 1, "post": 1},
                       {"scope": "events", "z_threshold": 1, "events": {"A": ["x"]}, "pre": 1, "post": 1},
                       {"scope": "events", "z_threshold": 1, "events": {"A": [1]}, "pre": 0, "post": 1},
                       {"scope": "whole", "z_threshold": 1, "start_buffer_s": -1},
                       {"scope": "events", "z_threshold": 1, "events": {"A": [1]}, "pre": 1, "post": 1, "n_shuffles": -1}):
            with self.subTest(params=params), self.assertRaises(pl.analyses.ParameterError):
                self.call("compute.peaks", {"recording": self.ref, **params})
        with self.assertRaises(pl.analyses.ParameterError):
            self.call("compute.event_peth", {"recording": self.ref, "event_times": ["a"], "pre": 1, "post": 1})

    def test_a_one_shot_run_has_nowhere_to_put_arrays(self):
        with self.assertRaises(OpError) as caught:
            ops.call("compute.event_peth", {"recording": self.folder, "event_times": [9.0], "pre": 2.0, "post": 3.0}, ops.OpContext())
        self.assertEqual(caught.exception.code, "invalid_input")


class TestSplicesArePureState(SessionCase):
    def test_a_splice_gives_the_variant_and_the_markers_moved_through_it(self):
        wired = self.open()
        result = self.call("recording.splice", {"id": wired["id"], "mode": "cut_out", "start": 10.0, "end": 20.0, "splices": [],
                                                "markers": [{"time": 5.0, "label": "a"}, {"time": 15.0, "label": "b"}, {"time": 50.0, "label": "c"}]})
        self.assertEqual(result["splices"], [{"mode": "cut_out", "start": 10.0, "end": 20.0}])
        self.assertAlmostEqual(len(self.arrays(result)["x"]), 500, delta=2)
        self.assertEqual([(m["label"], round(m["time"], 6)) for m in result["markers"]], [("a", 5.0), ("c", 39.9)])             # the gap closes up to one sample
        self.assertIn("[cut", self.arrays(result)["store"])

    def test_splices_stack_and_each_list_is_worked_out_once(self):
        wired = self.open()
        first = self.call("recording.splice", {"id": wired["id"], "mode": "cut_out", "start": 10.0, "end": 20.0, "splices": []})
        second = self.call("recording.splice", {"id": wired["id"], "mode": "keep_inside", "start": 5.0, "end": 30.0, "splices": first["splices"]})
        self.assertEqual([s["mode"] for s in second["splices"]], ["cut_out", "keep_inside"])
        again = self.call("recording.view", {"id": wired["id"], "splices": second["splices"]})
        self.assertEqual(again["cache"]["x"], second["cache"]["x"])                       # the same file: nothing was recomputed
        entry = self.session.store.get(wired["id"])
        self.assertEqual(len(entry.variants), 3)                                          # as recorded (which is as opened), the first, the second

    def test_undo_and_restore_full_recording_are_asking_for_an_earlier_list(self):
        wired = self.open()
        base_x = wired["cache"]["x"]
        spliced = self.call("recording.splice", {"id": wired["id"], "mode": "cut_out", "start": 10.0, "end": 20.0, "splices": []})
        back = self.call("recording.view", {"id": wired["id"], "splices": []})
        self.assertEqual(back["cache"]["x"], base_x)
        self.assertNotEqual(spliced["cache"]["x"], base_x)

    def test_a_splice_that_leaves_too_little_is_refused_and_changes_nothing(self):
        wired = self.open()
        with self.assertRaises(OpError) as caught:
            self.call("recording.splice", {"id": wired["id"], "mode": "keep_inside", "start": 10.0, "end": 10.05, "splices": []})
        self.assertEqual(caught.exception.code, "invalid_input")
        self.assertIn("leaves too little", str(caught.exception))
        self.assertEqual(self.call("recording.view", {"id": wired["id"], "splices": []})["cache"]["x"], wired["cache"]["x"])

    def test_markers_on_the_recording_as_recorded_are_moved_through_a_whole_list_of_splices(self):
        wired = self.open()
        marks = [{"time": 5.0, "label": "a"}, {"time": 15.0, "label": "b"}, {"time": 30.0, "label": "c"}, {"time": 50.0, "label": "d"}]
        splices = [{"mode": "cut_out", "start": 10.0, "end": 20.0}, {"mode": "cut_out", "start": 20.0, "end": 25.0}]
        result = self.call("recording.view", {"id": wired["id"], "splices": splices, "markers": marks})
        reference, applied = pl.apply_splices({**pl.load_recording(self.oxy)["cache"], "markers": marks}, splices)
        self.assertEqual(len(applied), 2)
        self.assertEqual(result["markers"], reference["markers"])                      # what replaying the list on the markers gives
        self.assertEqual([m["label"] for m in result["markers"]], ["a", "c", "d"])      # "b" was in the stretch cut out
        self.assertEqual(self.call("recording.view", {"id": wired["id"], "splices": []})["splices"], [])
        self.assertNotIn("markers", self.call("recording.view", {"id": wired["id"], "splices": splices}))      # only if asked

    def test_a_splice_moves_markers_from_the_timeline_the_earlier_ones_leave(self):
        wired = self.open()
        first = [{"mode": "cut_out", "start": 10.0, "end": 20.0}]
        on_spliced_timeline = self.call("recording.view", {"id": wired["id"], "splices": first, "markers": [{"time": 5.0, "label": "a"}, {"time": 40.0, "label": "d"}]})
        stacked = self.call("recording.splice", {"id": wired["id"], "splices": first, "mode": "cut_out", "start": 20.0, "end": 25.0,
                                                  "markers": on_spliced_timeline["markers"]})
        self.assertEqual(len(stacked["splices"]), 2)
        self.assertAlmostEqual(stacked["markers"][0]["time"], 5.0, places=3)
        self.assertAlmostEqual(stacked["markers"][1]["time"], 25.0, delta=0.5)         # 50 s as recorded, minus 10 s and 5 s cut out

    def test_markers_that_are_not_markers_are_a_parameter_error(self):
        wired = self.open()
        with self.assertRaises(pl.analyses.ParameterError):
            self.call("recording.splice", {"id": wired["id"], "mode": "cut_out", "start": 10, "end": 20, "markers": [5.0]})

    def test_a_bad_mode_is_a_parameter_error(self):
        wired = self.open()
        with self.assertRaises(pl.analyses.ParameterError):
            self.call("recording.splice", {"id": wired["id"], "mode": "trim", "start": 1, "end": 2})

    def test_the_variants_kept_are_bounded(self):
        from PyATCLI.engine import store
        wired = self.open()
        for i in range(store.MAX_VARIANTS + 4):
            self.call("recording.view", {"id": wired["id"], "splices": [{"mode": "cut_out", "start": 1.0 + i, "end": 1.5 + i}]})
        self.assertLessEqual(len(self.session.store.get(wired["id"]).variants), store.MAX_VARIANTS + 1)

    def test_the_analysis_sees_the_splices_asked_for(self):
        wired = self.open()
        spliced = self.call("run", {"kind": "process", "recording": {"id": wired["id"], "splices": [{"mode": "cut_out", "start": 10.0, "end": 20.0}]},
                                    "tables": "shapes"})
        whole = self.call("run", {"kind": "process", "recording": {"id": wired["id"], "splices": []}, "tables": "shapes"})
        self.assertLess(spliced["tables"]["trace"]["rows"], whole["tables"]["trace"]["rows"])

    def test_a_tdt_splice_refits_the_dff(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        wired = self.open(folder)
        before = self.arrays(wired)["corr"]
        after = self.arrays(self.call("recording.splice", {"id": wired["id"], "mode": "cut_out", "start": 5.0, "end": 8.0, "splices": []}))
        self.assertLess(len(after["corr"]), len(before))
        reference = pl.apply_splice(pl.load_recording(folder, use_saved=False)["cache"], "cut_out", 5.0, 8.0)
        np_equal = np.array_equal(after["corr"], reference["corr"])
        self.assertTrue(np_equal)                                                         # the engine's variant is the library's, bit for bit


class TestWhatTheEngineDropsItFrees(SessionCase):
    """An engine that works out many lists of splices must hold the few it keeps, not every array it ever wrote."""

    def cut(self, n):
        return [{"mode": "cut_out", "start": 10.0 + n, "end": 12.0 + n}]

    def test_a_trimmed_variant_is_freed_and_its_files_are_removed(self):
        import gc
        import weakref

        wired = self.open()
        entry = self.session.store.get(wired["id"])
        with mock.patch("PyATCLI.engine.store.MAX_VARIANTS", 1):
            first = self.call("recording.view", {"id": wired["id"], "splices": self.cut(0)})
            first_file = os.path.join(first["directory"], first["cache"]["x"]["__array__"])
            first_x = weakref.ref(entry.variants[next(k for k in entry.variants if len(k) == 1)]["x"])
            self.assertTrue(os.path.isfile(first_file))
            self.call("recording.view", {"id": wired["id"], "splices": self.cut(1)})
            self.call("recording.view", {"id": wired["id"], "splices": self.cut(2)})      # the first is the oldest of two extras: dropped
        del first
        gc.collect()
        self.assertIsNone(first_x())                                                        # freed, not kept for the id's sake
        self.call("recording.list")                                                         # (any later request sweeps)
        entry.files.sweep()
        self.assertFalse(os.path.exists(first_file))

    def test_the_arrays_held_do_not_grow_with_the_splices_worked_out(self):
        wired = self.open()
        entry = self.session.store.get(wired["id"])
        with mock.patch("PyATCLI.engine.store.MAX_VARIANTS", 2):
            for n in range(12):
                self.call("recording.view", {"id": wired["id"], "splices": self.cut(n)})
        import gc
        gc.collect()
        arrays_per_variant = 3                                                               # x, o2hb, hhb
        self.assertLessEqual(len(entry.files._by_id), arrays_per_variant * (2 + 2))          # the two kept, the recording, the first as opened

    def test_the_same_list_asked_for_again_is_the_same_files(self):
        wired = self.open()
        a = self.call("recording.view", {"id": wired["id"], "splices": self.cut(0)})
        b = self.call("recording.view", {"id": wired["id"], "splices": self.cut(0)})
        self.assertEqual(a["cache"], b["cache"])


class TestCancelling(SessionCase):
    def test_a_cancelled_request_stops_at_its_next_progress_report(self):
        with self.assertRaises(Cancelled):
            self.call("recording.open", {"path": make_tdt_folder(self.tmp, "Rat7-250101-101010")}, cancelled=lambda: True)
        self.assertEqual(self.session.store.ids(), [])                                    # nothing half-open is kept

    def test_errors_are_errors_a_caller_can_tell_apart(self):
        self.assertEqual(ops.error_info(OpError("x", "not_found"))["code"], "not_found")
        self.assertEqual(ops.error_info(Cancelled())["code"], "cancelled")
        self.assertEqual(ops.error_info(KeyError("k"))["message"], "missing 'k'")
        self.assertEqual(ops.error_info(RuntimeError("boom"))["code"], "internal_error")
        self.assertTrue(ops.is_expected(ValueError("x")))
        self.assertFalse(ops.is_expected(RuntimeError("x")))


class Feed:
    """A standard input a test types into: iterating it waits for the next line, and ends when it is closed."""

    def __init__(self):
        self.lines = queue.Queue()

    def send(self, message):
        self.lines.put(json.dumps(message) + "\n")

    def raw(self, text):
        self.lines.put(text if text.endswith("\n") else text + "\n")

    def close(self):
        self.lines.put(None)

    def __iter__(self):
        while True:
            line = self.lines.get()
            if line is None:
                return
            yield line


class Bench:
    """An engine running on a thread, with a way to talk to it and to read what it says."""

    def __init__(self, session_dir=None):
        self.stdin, self.out, self.err = Feed(), io.StringIO(), io.StringIO()
        self.heard = queue.Queue()
        outer = self

        class Sink(io.StringIO):
            def write(self, text):
                for line in text.splitlines():
                    if line.strip():
                        outer.heard.put(json.loads(line))
                return len(text)

        self.status = None

        def run():
            self.status = serve(self.stdin, Sink(), self.err, data_dir=session_dir)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def next(self, timeout=20):
        return self.heard.get(timeout=timeout)

    def until(self, wanted, timeout=60):
        """Everything said up to and including the first message `wanted(message)` accepts."""
        said = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            message = self.next(timeout=max(0.1, deadline - time.time()))
            said.append(message)
            if wanted(message):
                return said
        raise AssertionError(f"never heard it; heard {said}")

    def ask(self, rid, method, params=None):
        self.stdin.send({"id": rid, "method": method, "params": params or {}})
        return self.until(lambda m: m.get("id") == rid and m["event"] in ("result", "error"))

    def stop(self):
        self.stdin.send({"method": "shutdown"})
        self.thread.join(20)


class TestTheServer(CliCase):
    def setUp(self):
        super().setUp()
        self.bench = Bench()
        self.addCleanup(self.bench.stop)
        self.ready = self.bench.next()

    def test_it_says_it_is_ready_and_what_it_can_do(self):
        self.assertEqual((self.ready["event"], self.ready["protocol"]), ("ready", 1))
        self.assertIn("recording.open", self.ready["operations"])
        self.assertTrue(os.path.isdir(self.ready["data_dir"]))

    def test_a_request_gets_its_progress_and_one_result_with_its_id(self):
        said = self.bench.ask(7, "recording.open", {"path": self.oxy})
        self.assertTrue(all(m["id"] == 7 for m in said))
        self.assertEqual(said[-1]["event"], "result")
        self.assertTrue(said[-1]["ok"])
        self.assertEqual(said[-1]["result"]["id"], "r1")
        self.assertTrue(any(m["event"] == "progress" for m in said))

    def test_requests_are_answered_in_order_and_recordings_stay_open(self):
        self.bench.ask(1, "recording.open", {"path": self.oxy})
        said = self.bench.ask(2, "run", {"kind": "auc_window", "recording": {"id": "r1"}, "parameters": {"center_s": 30, "pre_s": 5, "post_s": 5}})
        self.assertEqual(said[-1]["result"]["tables"]["auc"]["rows"], 2)
        self.assertEqual(self.bench.ask(3, "recording.list")[-1]["result"]["open"][0]["id"], "r1")

    def test_an_error_is_that_requests_answer_and_the_engine_goes_on(self):
        said = self.bench.ask(1, "nope")
        self.assertEqual((said[-1]["event"], said[-1]["ok"], said[-1]["error"]["code"]), ("error", False, "unknown_operation"))
        said = self.bench.ask(2, "run", {"kind": "auc_window"})
        self.assertEqual(said[-1]["error"]["code"], "bad_parameter")
        said = self.bench.ask(3, "info")
        self.assertEqual(said[-1]["result"]["protocol"], 1)

    def test_a_line_that_is_not_a_request_is_answered_not_fatal(self):
        self.bench.stdin.raw("this is not json")
        said = self.bench.until(lambda m: m["event"] == "error")
        self.assertEqual((said[-1]["id"], said[-1]["error"]["code"]), (None, "invalid_input"))
        self.assertEqual(self.bench.ask(1, "info")[-1]["event"], "result")

    def test_a_bug_in_an_operation_is_reported_as_one_and_the_engine_survives(self):
        def boom(ctx):
            raise ZeroDivisionError("division by zero")
        ops.operation("test.boom", "raises")(boom)
        self.addCleanup(ops._OPERATIONS.pop, "test.boom", None)
        said = self.bench.ask(1, "test.boom")
        self.assertEqual(said[-1]["error"]["code"], "internal_error")
        self.assertIn("ZeroDivisionError", said[-1]["error"]["message"])
        self.assertIn("ZeroDivisionError", self.bench.err.getvalue())                       # the traceback is on stderr, not in the protocol
        self.assertEqual(self.bench.ask(2, "info")[-1]["event"], "result")

    def test_a_request_can_be_cancelled_while_it_runs(self):
        def slow(ctx):
            report = ctx.progress()
            for i in range(500):
                time.sleep(0.01)
                report(i / 500, "working")
            return {"finished": True}
        ops.operation("test.slow", "slow")(slow)
        self.addCleanup(ops._OPERATIONS.pop, "test.slow", None)
        self.bench.stdin.send({"id": 5, "method": "test.slow"})
        self.bench.until(lambda m: m.get("event") == "progress")
        self.bench.stdin.send({"cancel": 5})
        said = self.bench.until(lambda m: m.get("id") == 5 and m["event"] in ("result", "error"))
        self.assertEqual((said[-1]["event"], said[-1]["error"]["code"]), ("error", "cancelled"))
        self.assertEqual(self.bench.ask(6, "info")[-1]["event"], "result")                  # and it is ready for the next

    def test_shutdown_ends_it_politely_and_so_does_closing_its_input(self):
        self.bench.stdin.send({"method": "shutdown"})
        self.assertEqual(self.bench.until(lambda m: m["event"] == "bye")[-1]["event"], "bye")
        self.bench.thread.join(20)
        self.assertEqual(self.bench.status, 0)
        second = Bench()
        second.next()
        second.stdin.close()
        self.assertEqual(second.until(lambda m: m["event"] == "bye")[-1]["event"], "bye")


class _DeadOutput:
    """An output whose reader has gone away."""

    def __init__(self, after=0):
        self.writes, self.after = 0, after

    def write(self, text):
        self.writes += 1
        if self.writes > self.after:
            raise BrokenPipeError(32, "the pipe is being closed")

    def flush(self):
        if self.writes > self.after:
            raise BrokenPipeError(32, "the pipe is being closed")


class TestTheWindowDisappearing(SessionCase):
    """A window that crashes closes its end of the pipes without a goodbye; the engine must end quietly (a frozen windowed program shows an
    unhandled exception in a dialog nobody can see, and waits for it forever)."""

    def run_serve(self, stdout, lines=()):
        stdin = io.BytesIO(b"".join(line.encode() + b"\n" for line in lines))
        errors = io.StringIO()
        status = serve(stdin, stdout, errors)
        return status, errors.getvalue()

    def test_an_output_that_is_gone_before_the_first_word_ends_it_quietly(self):
        status, errors = self.run_serve(_DeadOutput())
        self.assertEqual((status, errors), (0, ""))

    def test_an_output_that_goes_while_a_request_runs_ends_it_at_the_next_report(self):
        reports = []

        def slow(name, params=None, ctx=None):
            for i in range(1000):
                reports.append(i)
                ctx.emit("progress", fraction=i / 1000, message="working")        # raises once nobody can hear
            return {}

        request = json.dumps({"id": 1, "method": "kinds", "params": {}})
        with mock.patch.object(ops, "call", slow):
            status, errors = self.run_serve(_DeadOutput(after=1), [request])
        self.assertEqual((status, errors), (0, ""))
        self.assertLess(len(reports), 5)                                          # it did not go on computing for nobody

    def test_a_data_folder_it_was_given_is_left_alone(self):
        directory = os.path.join(self.tmp, "data")
        stdin = io.BytesIO(b"")
        serve(stdin, _DeadOutput(), io.StringIO(), data_dir=directory)
        self.assertTrue(os.path.isdir(directory))                                  # (a folder it was given is not its to remove)


class TestAsASubprocess(CliCase):
    def spawn(self, *args):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        return subprocess.Popen([sys.executable, "-m", "PyATCLI", "serve", *args], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env)

    def exchange(self, proc, messages, expect):
        """Sends `messages` and reads protocol lines until `expect` of them were results or errors."""
        for m in messages:
            proc.stdin.write((json.dumps(m) + "\n").encode("utf-8"))
        proc.stdin.flush()
        out = []
        done = 0
        while done < expect:
            line = proc.stdout.readline()
            if not line:
                break
            msg = json.loads(line)
            out.append(msg)
            done += msg.get("event") in ("result", "error")
        return out

    def test_a_real_process_speaks_only_protocol_on_stdout_and_exits_cleanly(self):
        proc = self.spawn()
        try:
            ready = json.loads(proc.stdout.readline())
            self.assertEqual(ready["event"], "ready")
            said = self.exchange(proc, [{"id": 1, "method": "recording.open", "params": {"path": self.oxy}},
                                        {"id": 2, "method": "run", "params": {"kind": "auc_window", "recording": {"id": "r1"},
                                                                                "parameters": {"center_s": 30, "pre_s": 5, "post_s": 5}}}], expect=2)
            results = {m["id"]: m for m in said if m["event"] == "result"}
            self.assertEqual(set(results), {1, 2})
            cache = wire.decode(results[1]["result"]["cache"], results[1]["result"]["directory"])
            self.assertEqual(len(cache["x"]), 600)
            self.assertEqual(results[2]["result"]["tables"]["auc"]["rows"], 2)
            proc.stdin.write(b'{"method": "shutdown"}\n')
            proc.stdin.flush()
            self.assertEqual(json.loads(proc.stdout.readline())["event"], "bye")
            self.assertEqual(proc.wait(30), 0)
        finally:
            proc.kill()
            proc.stdin.close()
            proc.stdout.close()
            proc.stderr.close()

    def test_a_library_that_prints_cannot_corrupt_the_protocol(self):
        proc = self.spawn()
        try:
            proc.stdout.readline()
            said = self.exchange(proc, [{"id": 1, "method": "file.tables", "params": {"path": self.oxy}}], expect=1)
            self.assertEqual(said[-1]["event"], "result")
            proc.stdin.close()
            rest = proc.stdout.read().decode("utf-8")
            for line in rest.splitlines():
                json.loads(line)                                                              # every line the engine ever wrote is JSON
            proc.wait(30)
        finally:
            proc.kill()
            proc.stdout.close()
            proc.stderr.close()

    def test_closing_the_input_stops_the_engine(self):
        proc = self.spawn()
        try:
            proc.stdout.readline()
            proc.stdin.close()
            self.assertEqual(proc.wait(30), 0)
        finally:
            proc.kill()
            proc.stdout.close()
            proc.stderr.close()

    def test_a_window_that_vanishes_leaves_no_traceback_and_no_process(self):
        proc = self.spawn()
        try:
            proc.stdout.readline()
            proc.stdout.close()                                    # nobody reads the answers any more ...
            proc.stdin.close()                                     # ... and nobody asks
            self.assertEqual(proc.wait(30), 0)
            self.assertEqual(proc.stderr.read(), b"")              # no "BrokenPipeError" for a frozen windowed program to show in a dialog
        finally:
            proc.kill()
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
