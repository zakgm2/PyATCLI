"""A recording as PyAT holds it (recording.py): the saved state beside it, building the cache from a load, splicing
it (every source), loading a TDT block headlessly with its saved splices and markers, and finding recordings
under a folder."""

import json
import os
import subprocess
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import tdt

import PyATCLI as pl
from PyATCLI.sources import processing_TDT as P
from PyATCLI import recording as rec

TAB, NL = chr(9), chr(10)


def fake_block(n=20_000, fs=1000.0, with_isosbestic=True, seed=3):
    """read_block standing in for the tdt SDK: a 465 stream (and a 415 one), plus one epoc store 'L1P'
    with onsets at 3 s and 12 s and offsets at 4 s and 13 s."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    base = 60 + 30 * np.exp(-t / 8)
    f415 = (0.5 * base * (1 + 0.02 * np.sin(t)) + rng.normal(0, 0.1, n)).astype(np.float32)
    f465 = (base * (1 + 0.02 * np.sin(t)) * (1 + 0.1 * np.exp(-0.5 * ((t - 6) / 0.5) ** 2))
            + rng.normal(0, 0.1, n)).astype(np.float32)
    heads = tdt.StructType()
    heads.stores = {"L1P_": types.SimpleNamespace(type_str="epocs", name="L1P/"),
                    "x465A": types.SimpleNamespace(type_str="streams", name="x465A")}
    data = tdt.StructType()
    data.streams = {"x465A": types.SimpleNamespace(data=f465, fs=fs)}
    if with_isosbestic:
        data.streams["x415A"] = types.SimpleNamespace(data=f415, fs=fs)
    data.scalars = tdt.StructType()
    data.epocs = tdt.StructType()

    def read_block(path, headers=None, evtype=None, store=None, verbose=None):
        if isinstance(headers, int) and headers == 1:
            return heads
        if store is not None:
            block = tdt.StructType()
            block.epocs = tdt.StructType()
            block.epocs["L1P_"] = types.SimpleNamespace(name="L1P", onset=np.array([3.0, 12.0]),
                                                      offset=np.array([4.0, 13.0]))
            return block
        return data

    return read_block


def make_tdt_folder(parent, name):
    """A folder that passes validate_tdt_folder (it only looks for a .Tbk file)."""
    path = os.path.join(parent, name)
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, name + ".Tbk"), "wb").close()
    return path


def user_marker(t, label="Note"):
    return {"time": float(t), "label": label, "color": "green", "fontsize": 8}


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name


# --------------------------------------------------------------------------------------------------
# Saved state
# --------------------------------------------------------------------------------------------------

class TestSavedStatePaths(TempDirCase):
    def test_a_folder_source_keeps_its_saves_inside_it(self):
        d = os.path.join(self.tmp, "Subject1")
        os.makedirs(d)
        saves = os.path.join(d, "JSON saves")
        self.assertEqual(rec.saved_state_dir(d), saves)
        self.assertEqual(rec.marker_sidecar_path(d), os.path.join(saves, "markers.json"))
        self.assertEqual(rec.annotation_sidecar_path(d, "highlights.json"), os.path.join(saves, "highlights.json"))
        self.assertEqual(rec.splice_sidecar_path(d), os.path.join(saves, "splice.json"))
        self.assertEqual(rec.legacy_marker_sidecar_path(d), d + ".markers.json")

    def test_a_file_source_keeps_its_saves_next_to_it_named_after_it(self):
        f = os.path.join(self.tmp, "nirs.txt")
        open(f, "w").close()
        saves = os.path.join(self.tmp, "JSON saves")
        self.assertEqual(rec.saved_state_dir(f), saves)
        self.assertEqual(rec.marker_sidecar_path(f), os.path.join(saves, "nirs.txt.markers.json"))
        self.assertEqual(rec.annotation_sidecar_path(f, "highlights.json"), os.path.join(saves, "nirs.txt.highlights.json"))
        self.assertEqual(rec.splice_sidecar_path(f), os.path.join(saves, "splice.json"))

    def test_the_names_are_exported_by_the_package(self):
        for name in ("SAVES_FOLDER", "saved_state_dir", "marker_sidecar_path", "splice_sidecar_path",
                     "read_saved_splices", "read_saved_markers", "load_tdt_recording", "apply_splice"):
            self.assertTrue(hasattr(pl, name), name)


class TestReadingSavedState(TempDirCase):
    def setUp(self):
        super().setUp()
        self.d = os.path.join(self.tmp, "Rec")
        os.makedirs(os.path.join(self.d, "JSON saves"))

    def write(self, name, obj):
        with open(os.path.join(self.d, "JSON saves", name), "w") as f:
            json.dump(obj, f)

    def test_nothing_saved_reads_as_empty(self):
        self.assertEqual(rec.read_saved_splices(self.d), [])
        self.assertEqual(rec.read_saved_markers(self.d), [])

    def test_splices_come_back_in_order(self):
        s = [{"mode": "cut_out", "start": 1.0, "end": 2.0}, {"mode": "keep_inside", "start": 0.5, "end": 9.0}]
        self.write("splice.json", s)
        self.assertEqual(rec.read_saved_splices(self.d), s)

    def test_the_old_single_splice_format_is_still_read(self):
        self.write("splice.json", {"mode": "cut_out", "start": 1.0, "end": 2.0})
        self.assertEqual(rec.read_saved_splices(self.d), [{"mode": "cut_out", "start": 1.0, "end": 2.0}])

    def test_markers_are_read_from_the_saves_folder(self):
        m = [user_marker(1), user_marker(2, "Other")]
        self.write("markers.json", m)
        self.assertEqual(rec.read_saved_markers(self.d), m)

    def test_markers_saved_before_the_saves_folder_existed_are_still_found(self):
        old = [user_marker(5)]
        with open(self.d + ".markers.json", "w") as f:
            json.dump(old, f)
        self.assertEqual(rec.read_saved_markers(self.d), old)

    def test_the_saves_folder_wins_over_the_old_location(self):
        with open(self.d + ".markers.json", "w") as f:
            json.dump([user_marker(5)], f)
        self.write("markers.json", [user_marker(6)])
        self.assertEqual(rec.read_saved_markers(self.d), [user_marker(6)])

    def test_markers_that_are_not_a_list_are_refused(self):
        self.write("markers.json", {"time": 1})
        with self.assertRaisesRegex(ValueError, "does not hold a list"):
            rec.read_saved_markers(self.d)

    def test_a_file_that_is_not_json_raises_rather_than_returning_nothing(self):
        with open(os.path.join(self.d, "JSON saves", "splice.json"), "w") as f:
            f.write("{not json")
        with self.assertRaises(ValueError):
            rec.read_saved_splices(self.d)


class TestSignalNames(TempDirCase):
    def cache(self):
        return {"signals": {"normalized": {"label": "Normalized (dF/F)", "y": np.zeros(3)},
                            "main_driver": {"label": "Main Driver (_465A)", "y": np.ones(3)}}}

    def test_the_path_is_the_saves_folder_beside_the_recording(self):
        d = os.path.join(self.tmp, "Rat7")
        os.makedirs(d)
        self.assertEqual(pl.signal_labels_path(d), os.path.join(d, "JSON saves", "signal_labels.json"))

    def test_names_are_read_back_and_none_saved_is_empty(self):
        d = os.path.join(self.tmp, "Rat7")
        os.makedirs(os.path.join(d, "JSON saves"))
        self.assertEqual(pl.read_saved_signal_labels(d), {})
        with open(pl.signal_labels_path(d), "w", encoding="utf-8") as f:
            json.dump({"main_driver": "GCaMP (465 nm)"}, f, ensure_ascii=False)
        self.assertEqual(pl.read_saved_signal_labels(d), {"main_driver": "GCaMP (465 nm)"})
        with open(pl.signal_labels_path(d), "w") as f:
            json.dump(["not", "an", "object"], f)
        with self.assertRaisesRegex(ValueError, "does not hold an object"):
            pl.read_saved_signal_labels(d)

    def test_applying_names_changes_the_label_and_nothing_else(self):
        c = self.cache()
        y = c["signals"]["main_driver"]["y"]
        out = pl.apply_signal_labels(c, {"main_driver": "GCaMP (465 nm)"})
        self.assertIs(out, c)
        self.assertEqual(c["signals"]["main_driver"]["label"], "GCaMP (465 nm)")
        self.assertEqual(c["signals"]["normalized"]["label"], "Normalized (dF/F)")
        self.assertIs(c["signals"]["main_driver"]["y"], y)                          # arrays are not copied

    def test_unknown_keys_and_empty_names_are_ignored(self):
        c = self.cache()
        pl.apply_signal_labels(c, {"isosbestic": "x", "main_driver": "", "normalized": None})
        self.assertEqual(pl.current_signal_labels(c), {"normalized": "Normalized (dF/F)", "main_driver": "Main Driver (_465A)"})

    def test_a_recording_with_no_signals_map_is_left_alone(self):
        c = {"y_columns": {"a": np.zeros(2)}}
        self.assertIs(pl.apply_signal_labels(c, {"a": "b"}), c)
        self.assertEqual(pl.current_signal_labels(c), {})

    def test_the_names_in_effect_are_reported(self):
        c = pl.apply_signal_labels(self.cache(), {"main_driver": "GCaMP"})
        self.assertEqual(pl.current_signal_labels(c), {"normalized": "Normalized (dF/F)", "main_driver": "GCaMP"})


# --------------------------------------------------------------------------------------------------
# Building the cache
# --------------------------------------------------------------------------------------------------

def tdt_result(**kw):
    with mock.patch.object(tdt, "read_block", fake_block(**kw)):
        return P.process_tdt_folder("ignored")


class TestBuildingCaches(TempDirCase):
    def test_signal_options_are_the_normalized_trace_plus_each_raw_channel(self):
        result = tdt_result()
        opts = rec.build_signal_options(result)
        self.assertEqual(list(opts), ["normalized", "main_driver", "isosbestic"])
        self.assertEqual(opts["normalized"]["label"], "Normalized (dF/F)")
        assert_array_equal(opts["normalized"]["y"], result["corr"])
        self.assertTrue(all("color" not in o for o in opts.values()))          # colour is the app's business

    def test_a_block_without_an_isosbestic_stream_has_no_such_option(self):
        opts = rec.build_signal_options(tdt_result(with_isosbestic=False))
        self.assertEqual(list(opts), ["normalized", "main_driver"])

    def test_the_tdt_cache_has_what_every_analysis_reads(self):
        result = tdt_result()
        cache = rec.make_tdt_cache("C:/data/Rat7-250101", result)
        self.assertEqual(cache["source"], "TDT")
        self.assertEqual(cache["source_path"], "C:/data/Rat7-250101")
        self.assertEqual(cache["store"], "Rat7-250101")
        assert_array_equal(cache["x"], result["x"])
        assert_array_equal(cache["corr"], result["corr"])
        assert_array_equal(cache["raw"], result["raw"])
        self.assertEqual(cache["fs"], result["fs"])
        self.assertEqual(cache["detected_markers"], result["markers"])
        self.assertEqual(cache["markers"], [])
        self.assertEqual(set(cache["signals"]), {"normalized", "main_driver", "isosbestic"})

    def test_the_store_name_ignores_a_trailing_slash(self):
        self.assertEqual(rec.make_tdt_cache("C:/data/Rat7/", tdt_result())["store"], "Rat7")

    def test_a_caller_can_supply_its_own_signals_map(self):
        mine = {"normalized": {"label": "x", "y": np.zeros(3), "color": "red"}}
        self.assertIs(rec.make_tdt_cache("C:/d/r", tdt_result(), signals=mine)["signals"], mine)

    def write_oxysoft(self, path, n=40, events=None):
        events = events or {}
        lines = [f"Datafile sample rate{TAB}10 Hz", "Legend:", f"1{TAB}sample number", f"2{TAB}Rx1 - Tx1 O2Hb (f)",
                 f"3{TAB}Rx1 - Tx1 HHb (f)", f"4{TAB}Event (Event)", f"5{TAB}Fit Factor", TAB.join("12345"), TAB.join("00000")]
        for i in range(1, n + 1):
            lines.append(TAB.join([str(i), f"{0.5 * i:.3f}", f"{-0.5 * i:.3f}", str(events.get(i, 0)), "90"]))
        with open(path, "w") as f:
            f.write(NL.join(lines) + NL)

    def test_the_oxysoft_cache_places_events_in_seconds(self):
        path = os.path.join(self.tmp, "session.txt")
        self.write_oxysoft(path, events={11: 1, 31: 2})
        ds = pl.load_dataset_file(path)
        cache = rec.make_oxysoft_cache(path, ds)
        self.assertEqual(cache["source"], "Oxysoft")
        self.assertEqual(cache["store"], "session")
        self.assertEqual(cache["fs"], 10.0)
        assert_allclose(cache["x"], np.arange(40) / 10.0)
        self.assertEqual(cache["o2hb"].shape, (1, 40))
        self.assertEqual(cache["hhb"].shape, (1, 40))
        self.assertAlmostEqual(cache["fit_factor_mean"], 90.0)
        self.assertEqual([(m["label"], m["store"], m["color"]) for m in cache["detected_markers"]],
                         [("1", "Events", "black"), ("2", "Events", "black")])
        assert_allclose([m["time"] for m in cache["detected_markers"]], [1.0, 3.0])        # samples 10 and 30 at 10 Hz
        self.assertEqual(cache["markers"], [])
        self.assertNotIn("thb", cache)

    def test_a_total_haemoglobin_array_travels_with_the_cache(self):
        path = os.path.join(self.tmp, "session.txt")
        self.write_oxysoft(path)
        ds = pl.load_dataset_file(path)
        ds.metadata["thb"] = np.ones((1, 40))
        self.assertEqual(rec.make_oxysoft_cache(path, ds)["thb"].shape, (1, 40))

    def generic_table(self):
        data = np.array([[0.0, 1.0, 10.0], [0.1, 2.0, 20.0], [np.nan, 9.0, 90.0], [0.2, 3.0, 30.0], [0.3, 4.0, 40.0]])
        return types.SimpleNamespace(name="Sheet1", headers=["t", "a", "b"], data=data)

    def test_the_generic_cache_drops_rows_with_no_x_and_keys_columns_by_header(self):
        cache = rec.make_generic_cache("C:/d/f.csv", self.generic_table())
        self.assertEqual(cache["source"], "Generic")
        assert_allclose(cache["x"], [0.0, 0.1, 0.2, 0.3])
        self.assertEqual(list(cache["y_columns"]), ["a", "b"])                      # every column but x, by default
        assert_allclose(cache["y_columns"]["b"], [10, 20, 30, 40])
        self.assertEqual((cache["x_label"], cache["store"], cache["markers"]), ("t", "Sheet1", []))
        self.assertAlmostEqual(cache["fs"], 10.0)

    def test_the_generic_cache_takes_the_columns_it_is_told_to(self):
        cache = rec.make_generic_cache("C:/d/f.csv", self.generic_table(), x_index=0, y_indices=[2])
        self.assertEqual(list(cache["y_columns"]), ["b"])

    def test_a_single_x_value_gets_a_sample_rate_of_one(self):
        t = types.SimpleNamespace(name="T", headers=["t", "a"], data=np.array([[0.0, 1.0], [np.nan, 2.0]]))
        self.assertEqual(rec.make_generic_cache("p", t)["fs"], 1.0)


class TestNormalizedSignal(unittest.TestCase):
    def test_it_prefers_the_normalized_entry(self):
        cache = {"signals": {"main_driver": {"label": "MD", "y": np.ones(3)}, "normalized": {"label": "N", "y": np.zeros(3)}}}
        key, label, y = rec.normalized_signal(cache)
        self.assertEqual((key, label), ("normalized", "N"))
        assert_array_equal(y, np.zeros(3))

    def test_without_one_it_takes_the_first(self):
        cache = {"signals": {"main_driver": {"label": "MD", "y": np.ones(3)}}}
        self.assertEqual(rec.normalized_signal(cache)[:2], ("main_driver", "MD"))

    def test_a_source_with_no_signals_map_uses_its_corrected_then_raw_trace(self):
        self.assertEqual(rec.normalized_signal({"corr": np.ones(2), "raw": np.zeros(2)})[1], "dF/F (corrected)")
        assert_array_equal(rec.normalized_signal({"corr": np.ones(2), "raw": np.zeros(2)})[2], np.ones(2))
        assert_array_equal(rec.normalized_signal({"raw": np.zeros(2)})[2], np.zeros(2))


# --------------------------------------------------------------------------------------------------
# Splicing
# --------------------------------------------------------------------------------------------------

class TestApplySpliceTdt(unittest.TestCase):
    def setUp(self):
        self.result = tdt_result()
        self.cache = rec.make_tdt_cache("C:/d/Rec1", self.result)
        self.cache["markers"] = [user_marker(4.0, "a"), user_marker(6.0, "b"), user_marker(15.0, "c")]
        self.fs = float(self.cache["fs"])

    def expected_cut(self, start, end, method="ols"):
        x = self.cache["x"]
        i0, i1 = np.searchsorted(x, start, "left"), np.searchsorted(x, end, "right")
        chans = {c["key"]: np.concatenate([c["y"][:i0], c["y"][i1:]]) for c in self.result["channels"]}
        return i0, i1, chans, P.compute_dff(chans["main_driver"], chans["isosbestic"], self.fs, method)

    def test_cut_out_stitches_the_timeline_and_refits_the_dff_on_what_is_left(self):
        before = {k: np.array(self.cache[k]) for k in ("x", "raw", "corr")}
        s = rec.apply_splice(self.cache, rec.MODE_CUT_OUT, 5.0, 8.0, "ols")
        i0, i1, chans, expected = self.expected_cut(5.0, 8.0)
        self.assertEqual(len(s["x"]), len(self.cache["x"]) - (i1 - i0))
        assert_allclose(np.diff(s["x"]), 1.0 / self.fs, atol=1e-9)                    # contiguous: no gap at the cut
        assert_array_equal(s["corr"], expected["corr"])                              # a fresh fit, not a shorter old one
        assert_array_equal(s["raw"], expected["raw"])
        assert_array_equal(s["signals"]["main_driver"]["y"], chans["main_driver"])
        self.assertIs(s["signals"]["normalized"]["y"], s["corr"])
        for sig in s["signals"].values():
            self.assertEqual(len(sig["y"]), len(s["x"]))
        for k, v in before.items():                                                  # nothing mutated
            assert_array_equal(self.cache[k], v)

    def test_cut_out_drops_the_markers_inside_and_shifts_the_ones_after(self):
        s = rec.apply_splice(self.cache, rec.MODE_CUT_OUT, 5.0, 8.0, "ols")
        shift = float(self.cache["x"][np.searchsorted(self.cache["x"], 8.0, "right")] - self.cache["x"][np.searchsorted(self.cache["x"], 5.0)])
        self.assertEqual([m["label"] for m in s["markers"]], ["a", "c"])               # "b" at 6 s was inside the cut
        assert_allclose([m["time"] for m in s["markers"]], [4.0, 15.0 - shift])
        self.assertEqual([m["label"] for m in s["detected_markers"]][:2], ["L1P", "L1P"])    # TDT's own events too

    def test_keep_inside_trims_and_keeps_absolute_times(self):
        s = rec.apply_splice(self.cache, rec.MODE_KEEP_INSIDE, 2.0, 10.0, "ols")
        self.assertAlmostEqual(s["x"][0], 2.0)
        self.assertAlmostEqual(s["x"][-1], 10.0)
        self.assertEqual([m["time"] for m in s["markers"]], [4.0, 6.0])
        refit = P.compute_dff(s["signals"]["main_driver"]["y"], s["signals"]["isosbestic"]["y"], self.fs, "ols")
        assert_array_equal(s["corr"], refit["corr"])

    def test_the_regression_method_is_the_one_asked_for(self):
        s = rec.apply_splice(self.cache, rec.MODE_CUT_OUT, 5.0, 8.0, "huber")
        assert_array_equal(s["corr"], self.expected_cut(5.0, 8.0, "huber")[3]["corr"])

    def test_a_range_with_no_samples_is_not_usable(self):
        self.assertIsNone(rec.apply_splice(self.cache, rec.MODE_KEEP_INSIDE, 1e6, 1e6 + 1))
        self.assertIsNone(rec.apply_splice(self.cache, rec.MODE_CUT_OUT, 1e6, 1e6 + 1))

    def test_a_cut_that_leaves_under_two_samples_is_not_usable(self):
        self.assertIsNone(rec.apply_splice(self.cache, rec.MODE_CUT_OUT, -1.0, 1e6))

    def test_an_unknown_source_is_not_spliceable(self):
        self.assertIsNone(rec.apply_splice({"source": "Nope", "x": np.arange(5.0), "markers": []}, rec.MODE_CUT_OUT, 1, 2))


class TestApplySpliceOtherSources(TempDirCase):
    def test_oxysoft_arrays_are_cut_along_their_last_axis(self):
        n = 100
        cache = {"source": "Oxysoft", "x": np.arange(n) / 10.0, "o2hb": np.vstack([np.arange(n), 2 * np.arange(n)]).astype(float),
                 "hhb": np.zeros((2, n)), "thb": np.ones((2, n)), "markers": [user_marker(2.0), user_marker(8.0)],
                 "detected_markers": []}
        s = rec.apply_splice(cache, rec.MODE_CUT_OUT, 3.0, 5.0)
        n_left = n - (np.searchsorted(cache["x"], 5.0, "right") - np.searchsorted(cache["x"], 3.0, "left"))
        self.assertEqual((s["o2hb"].shape, s["hhb"].shape, s["thb"].shape, len(s["x"])), ((2, n_left),) * 3 + (n_left,))
        self.assertEqual([m["time"] for m in s["markers"]][0], 2.0)
        self.assertLess(s["markers"][1]["time"], 8.0)                                  # shifted back by the cut
        keep = rec.apply_splice(cache, rec.MODE_KEEP_INSIDE, 3.0, 5.0)
        self.assertEqual(keep["o2hb"].shape[0], 2)
        self.assertEqual(keep["o2hb"].shape[1], len(keep["x"]))
        self.assertNotIn("thb", rec.apply_splice({k: v for k, v in cache.items() if k != "thb"}, rec.MODE_CUT_OUT, 3.0, 5.0))

    def test_generic_columns_are_cut_together(self):
        n = 50
        cache = {"source": "Generic", "x": np.arange(n, dtype=float), "y_columns": {"a": np.arange(n, dtype=float), "b": -np.arange(n, dtype=float)},
                 "markers": []}
        s = rec.apply_splice(cache, rec.MODE_KEEP_INSIDE, 10.0, 19.0)
        self.assertEqual(list(s["y_columns"]), ["a", "b"])
        assert_array_equal(s["y_columns"]["a"], np.arange(10, 20, dtype=float))
        assert_array_equal(s["y_columns"]["b"], -np.arange(10, 20, dtype=float))
        assert_array_equal(s["x"], np.arange(10, 20, dtype=float))


class TestApplySplices(unittest.TestCase):
    def setUp(self):
        self.cache = rec.make_tdt_cache("C:/d/Rec1", tdt_result())
        self.cache["markers"] = [user_marker(15.0, "c")]
        self.cut = {"mode": "cut_out", "start": 5.0, "end": 8.0}
        self.keep = {"mode": "keep_inside", "start": 2.0, "end": 12.0}

    def test_they_stack_each_on_top_of_the_one_before(self):
        out, applied = rec.apply_splices(self.cache, [self.cut, self.keep], "ols")
        step1 = rec.apply_splice(self.cache, "cut_out", 5.0, 8.0, "ols")
        step2 = rec.apply_splice(step1, "keep_inside", 2.0, 12.0, "ols")
        assert_array_equal(out["x"], step2["x"])
        assert_array_equal(out["corr"], step2["corr"])
        self.assertEqual(applied, [self.cut, self.keep])

    def test_the_store_name_lists_every_splice_that_took_effect(self):
        out, applied = rec.apply_splices(self.cache, [self.cut, self.keep])
        self.assertEqual(out["store"], "Rec1 [cut 5.0s-8.0s, kept 2.0s-12.0s]")
        self.assertEqual(out["store"], rec.spliced_store_name("Rec1", applied))

    def test_an_unusable_splice_is_skipped_and_the_rest_still_apply(self):
        bad = {"mode": "keep_inside", "start": 1e6, "end": 1e6 + 1}
        out, applied = rec.apply_splices(self.cache, [bad, self.cut])
        self.assertEqual(applied, [self.cut])
        self.assertEqual(out["store"], "Rec1 [cut 5.0s-8.0s]")

    def test_no_splices_returns_the_cache_untouched(self):
        out, applied = rec.apply_splices(self.cache, [])
        self.assertIs(out, self.cache)
        self.assertEqual(applied, [])

    def test_the_input_cache_is_never_modified(self):
        x, store = np.array(self.cache["x"]), self.cache["store"]
        rec.apply_splices(self.cache, [self.cut, self.keep])
        assert_array_equal(self.cache["x"], x)
        self.assertEqual(self.cache["store"], store)

    def test_the_store_name_for_nothing_is_the_plain_name(self):
        self.assertEqual(rec.spliced_store_name("Rec1", []), "Rec1")


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------

class TestLoadTdtRecording(TempDirCase):
    def setUp(self):
        super().setUp()
        self.folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        patcher = mock.patch.object(tdt, "read_block", fake_block())
        patcher.start()
        self.addCleanup(patcher.stop)

    def save(self, name, obj):
        d = os.path.join(self.folder, "JSON saves")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f)

    def test_with_nothing_saved_it_is_the_recording_as_recorded(self):
        out = rec.load_tdt_recording(self.folder)
        self.assertEqual(set(out), {"cache", "original_cache", "splices", "warnings"})
        self.assertIsNone(out["original_cache"])
        self.assertEqual((out["splices"], out["warnings"], out["cache"]["markers"]), ([], [], []))
        self.assertEqual(len(out["cache"]["x"]), 20_000)

    def test_saved_splices_are_replayed_and_the_unspliced_cache_is_kept(self):
        self.save("splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        out = rec.load_tdt_recording(self.folder)
        self.assertEqual(len(out["splices"]), 1)
        self.assertLess(len(out["cache"]["x"]), 20_000)
        self.assertEqual(len(out["original_cache"]["x"]), 20_000)
        self.assertEqual(out["cache"]["store"], "Rat7-250101-101010 [cut 5.0s-8.0s]")
        self.assertEqual(out["original_cache"]["store"], "Rat7-250101-101010")

    def test_saved_markers_go_on_after_the_splices(self):
        self.save("splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        self.save("markers.json", [user_marker(9.0, "mine")])                        # already on the post-splice timeline
        out = rec.load_tdt_recording(self.folder)
        self.assertEqual(out["cache"]["markers"], [user_marker(9.0, "mine")])        # not shifted a second time

    def test_saved_signal_names_are_applied_to_the_recording_and_to_the_unspliced_copy(self):
        self.save("signal_labels.json", {"main_driver": "GCaMP (465 nm)", "isosbestic": "Isosbestic (415 nm)"})
        self.save("splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        out = rec.load_tdt_recording(self.folder)
        for cache in (out["cache"], out["original_cache"]):
            self.assertEqual(cache["signals"]["main_driver"]["label"], "GCaMP (465 nm)")
            self.assertEqual(cache["signals"]["isosbestic"]["label"], "Isosbestic (415 nm)")
            self.assertEqual(cache["signals"]["normalized"]["label"], "Normalized (dF/F)")

    def test_an_unreadable_names_file_is_a_warning(self):
        os.makedirs(os.path.join(self.folder, "JSON saves"))
        with open(pl.signal_labels_path(self.folder), "w") as f:
            f.write("{broken")
        out = rec.load_tdt_recording(self.folder)
        self.assertTrue(out["warnings"][0].startswith("Saved signal names could not be read"))
        self.assertEqual(out["cache"]["signals"]["main_driver"]["label"], "Main Driver (x465A)")

    def test_use_saved_false_ignores_the_saves_folder(self):
        self.save("splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        self.save("markers.json", [user_marker(9.0)])
        out = rec.load_tdt_recording(self.folder, use_saved=False)
        self.assertEqual(len(out["cache"]["x"]), 20_000)
        self.assertEqual((out["splices"], out["cache"]["markers"], out["warnings"]), ([], [], []))

    def test_unreadable_saves_become_warnings_not_failures(self):
        os.makedirs(os.path.join(self.folder, "JSON saves"))
        with open(os.path.join(self.folder, "JSON saves", "splice.json"), "w") as f:
            f.write("{broken")
        self.save("markers.json", {"not": "a list"})
        out = rec.load_tdt_recording(self.folder)
        self.assertEqual(len(out["cache"]["x"]), 20_000)
        self.assertEqual(len(out["warnings"]), 2)
        self.assertTrue(out["warnings"][0].startswith("Saved splices could not be read"))
        self.assertTrue(out["warnings"][1].startswith("Saved markers could not be read"))

    def test_a_folder_that_is_not_a_tdt_block_is_refused(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        with self.assertRaisesRegex(ValueError, "Invalid Folder"):
            rec.load_tdt_recording(empty)

    def test_progress_runs_to_a_hundred_percent(self):
        self.save("splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        seen = []
        rec.load_tdt_recording(self.folder, progress=lambda f, m: seen.append(f))
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))

    def test_the_regression_method_reaches_both_the_load_and_the_splices(self):
        self.save("splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        ols = rec.load_tdt_recording(self.folder, "ols")["cache"]["corr"]
        huber = rec.load_tdt_recording(self.folder, "huber")["cache"]["corr"]
        self.assertEqual(ols.shape, huber.shape)


class TestFindTdtRecordings(TempDirCase):
    def test_every_block_at_any_depth_is_found_sorted(self):
        b1 = make_tdt_folder(os.path.join(self.tmp, "cohort1"), "Rat1-250101")
        b2 = make_tdt_folder(os.path.join(self.tmp, "cohort2", "sub"), "Rat2-250102")
        b3 = make_tdt_folder(self.tmp, "Rat0-250100")
        os.makedirs(os.path.join(self.tmp, "no-recordings-here", "deeper"))
        open(os.path.join(self.tmp, "notes.txt"), "w").close()
        self.assertEqual(pl.find_tdt_recordings(self.tmp), sorted([b1, b2, b3]))

    def test_a_block_is_not_searched_inside(self):
        outer = make_tdt_folder(self.tmp, "Rat1")
        make_tdt_folder(outer, "inner-looks-like-a-block")
        self.assertEqual(pl.find_tdt_recordings(self.tmp), [outer])

    def test_the_folder_searched_is_not_itself_reported(self):
        root = make_tdt_folder(self.tmp, "Rat1")
        self.assertEqual(pl.find_tdt_recordings(root), [])

    def test_nothing_found_is_an_empty_list(self):
        self.assertEqual(pl.find_tdt_recordings(self.tmp), [])

    @staticmethod
    def remove_link(link):
        """Takes away a symlink (POSIX: os.rmdir refuses one) or an NTFS junction (os.unlink refuses one), never what it points at."""
        if not os.path.lexists(link):
            return
        try:
            os.unlink(link)
        except OSError:
            os.rmdir(link)

    def test_a_folder_that_links_back_to_an_ancestor_does_not_loop(self):
        block = make_tdt_folder(os.path.join(self.tmp, "a"), "Rat1")
        link = os.path.join(self.tmp, "a", "loop")
        try:
            os.symlink(self.tmp, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Windows without symlink rights: an NTFS junction needs none, and it is the case os.walk's own
            # followlinks=False does NOT catch (os.path.islink() is False for a junction).
            if os.name != "nt":
                self.skipTest("this system will not create symlinks for this user")
            subprocess.run(["cmd", "/c", "mklink", "/J", link, self.tmp], capture_output=True, check=False)
            if not os.path.isdir(link):
                self.skipTest("could not create a junction either")
        # Removed before the temp folder is, so cleaning up can never follow it back into the tree.
        self.addCleanup(self.remove_link, link)
        self.assertEqual(pl.find_tdt_recordings(self.tmp), [block])


if __name__ == "__main__":
    unittest.main()
