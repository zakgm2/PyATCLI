"""The analyses registry (analyses.py): each analysis, run from a recording and a parameter dict, gives exactly what
the library calls the app's dialogs make give, in tables laid out like the app's CSV exports; parameters are checked
and defaults filled in; what an analysis looked up is recorded so a record is self-sufficient."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

import PyATCLI as pl
from PyATCLI import analyses as A
from PyATCLI.analysis import (
    compute_auc_from_trace, compute_auc_matrix, compute_auc_window, compute_event_zscore_peth, compute_fft_slice,
    compute_group_stats, compute_peri_event_from_trace, compute_peri_event_matrix, find_fft_peaks, find_peak_near_events,
    find_significant_peaks, fit_model_to_segment, get_zscore_slice, smooth_signal,
)

EVENTS = [40.0, 90.0, 140.0, 200.0, 250.0]


def phase_marker(t, phase, store="L1P"):
    return {"time": float(t), "label": store, "color": "black", "store": store, "phase": phase}


def make_cache(n=30_000, fs=100.0, seed=0):
    """A TDT-shaped recording: a decaying response after each L1P onset, a note at 120 s, two signals."""
    rng = np.random.default_rng(seed)
    x = np.arange(n) / fs
    y = 0.01 * rng.standard_normal(n)
    for e in EVENTS:
        d = x - e
        y = y + np.where(d >= 0, 1.5 * np.exp(-np.clip(d, 0, None) / 2.0), 0.0)
    return {
        "source": "TDT", "source_path": "C:/x/Rec", "store": "Rec", "x": x, "fs": fs, "raw": y, "corr": y,
        "detected_markers": [phase_marker(e, "high") for e in EVENTS] + [phase_marker(e + 0.5, "low") for e in EVENTS],
        "markers": [{"time": 120.0, "label": "MyNote", "color": "green", "fontsize": 8}],
        "signals": {"normalized": {"label": "Normalized (dF/F)", "y": y},
                    "main_driver": {"label": "Main Driver (x465A)", "y": 2 * y + 10}},
    }


class TestParameters(unittest.TestCase):
    def test_defaults_are_filled_in(self):
        p = A.normalize_parameters("zscore_window", {"center_s": 100})
        self.assertEqual(p, {"center_s": 100.0, "pre_s": 15.0, "post_s": 15.0, "signal": "normalized"})

    def test_a_missing_required_parameter_is_named(self):
        with self.assertRaisesRegex(A.ParameterError, "center_s is required"):
            A.normalize_parameters("zscore_window", {})

    def test_an_unknown_parameter_lists_the_valid_ones(self):
        with self.assertRaisesRegex(A.ParameterError, r"unknown parameter\(s\) center.*valid: center_s"):
            A.normalize_parameters("zscore_window", {"center": 5})

    def test_an_unknown_analysis_lists_the_analyses(self):
        with self.assertRaisesRegex(A.ParameterError, "Unknown analysis 'nope'.*event_peth"):
            A.normalize_parameters("nope")

    def test_values_are_coerced_or_refused_naming_the_parameter(self):
        self.assertEqual(A.normalize_parameters("zscore_window", {"center_s": "12.5"})["center_s"], 12.5)
        for bad in ("abc", float("nan"), float("inf"), None):
            if bad is None:
                continue
            with self.subTest(bad), self.assertRaisesRegex(A.ParameterError, "center_s"):
                A.normalize_parameters("zscore_window", {"center_s": bad})

    def test_lists_come_from_a_list_or_a_comma_separated_string(self):
        self.assertEqual(A.normalize_parameters("event_peth", {"event": "x", "selected_trials": "1, 3 5"})["selected_trials"], [1, 3, 5])
        self.assertEqual(A.normalize_parameters("event_peth", {"event": "x", "event_times": [1, 2.5]})["event_times"], [1.0, 2.5])
        self.assertEqual(A.normalize_parameters("event_peth", {"event": "x", "event_times": "1.5,2.5"})["event_times"], [1.5, 2.5])

    def test_booleans_accept_words(self):
        for text, want in (("true", True), ("yes", True), ("0", False), ("off", False)):
            self.assertIs(A.normalize_parameters("peak_finder", {"include_troughs": text})["include_troughs"], want)
        with self.assertRaises(A.ParameterError):
            A.normalize_parameters("peak_finder", {"include_troughs": "maybe"})

    def test_a_choice_must_be_one_of_its_choices(self):
        with self.assertRaisesRegex(A.ParameterError, "scope.*whole"):
            A.normalize_parameters("peak_finder", {"scope": "everywhere"})

    def test_a_whole_number_parameter_refuses_a_fraction(self):
        with self.assertRaisesRegex(A.ParameterError, "not a whole number"):
            A._coerce(A.Param("n", "int"), 1.5)
        self.assertEqual(A._coerce(A.Param("n", "int"), 2.0), 2)

    def test_kinds_and_their_descriptions(self):
        self.assertEqual(A.kinds(), ("zscore_window", "auc_window", "fft_window", "curve_fit", "event_peth", "peak_finder",
                                     "event_intervals", "process"))
        info = A.describe_kinds()
        self.assertEqual(set(info), set(A.kinds()))
        for kind, d in info.items():
            self.assertTrue(d["summary"], kind)
            self.assertTrue(all(p["doc"] for p in d["parameters"]), kind)
        names = [p["name"] for p in info["event_peth"]["parameters"]]
        self.assertEqual(names, ["event", "pre_s", "post_s", "bin_s", "selected_trials", "event_times"])
        self.assertTrue(next(p for p in info["event_peth"]["parameters"] if p["name"] == "event")["required"])


class TestWindowAnalyses(unittest.TestCase):
    def setUp(self):
        self.c = make_cache()
        self.x, self.fs = self.c["x"], self.c["fs"]
        self.y = self.c["signals"]["normalized"]["y"]

    def test_zscore_window_is_the_smoothed_signal_z_scored_against_its_pre_event_baseline(self):
        r = A.run_analysis("zscore_window", self.c, {"center_s": 90.0, "pre_s": 10, "post_s": 20})
        sx, z = get_zscore_slice(self.x, smooth_signal(self.y, self.fs), 90.0, pre=10, post=20)
        w = r.tables["window"]
        self.assertEqual(list(w.columns), ["Channel", "time_from_center_s", "z_score"])
        assert_allclose(w["time_from_center_s"], sx - 90.0)
        assert_array_equal(w["z_score"].to_numpy(), z)
        self.assertEqual(set(w["Channel"]), {"Normalized (dF/F)"})
        peak = r.tables["peaks"].iloc[0]
        self.assertEqual(peak["Peak Z-score"], z[np.argmax(np.abs(z))])
        self.assertEqual(r.parameters, {"center_s": 90.0, "pre_s": 10.0, "post_s": 20.0, "signal": "normalized"})

    def test_a_different_signal_is_a_different_channel_with_its_own_label(self):
        r = A.run_analysis("zscore_window", self.c, {"center_s": 90.0, "signal": "main_driver"})
        self.assertEqual(set(r.tables["window"]["Channel"]), {"Main Driver (x465A)"})

    def test_a_window_outside_the_recording_is_an_error_not_an_empty_table(self):
        with self.assertRaisesRegex(A.AnalysisError, "Not enough data"):
            A.run_analysis("zscore_window", self.c, {"center_s": 5000.0})

    def test_auc_window_is_compute_auc_window(self):
        r = A.run_analysis("auc_window", self.c, {"center_s": 140.0, "pre_s": 5, "post_s": 10})
        want = compute_auc_window(self.x, self.y, 140.0, 5, 10)
        row = r.tables["auc"].iloc[0]
        self.assertEqual(list(r.tables["auc"].columns), ["Channel", "Total AUC", "Positive AUC", "Negative AUC", "Window Pre (s)", "Window Post (s)"])
        self.assertEqual((row["Total AUC"], row["Positive AUC"], row["Negative AUC"]), (want["total_auc"], want["positive_auc"], want["negative_auc"]))
        self.assertEqual((row["Window Pre (s)"], row["Window Post (s)"]), (5.0, 10.0))
        assert_array_equal(r.tables["window"]["value"].to_numpy(), want["seg_y"])

    def test_fft_window_is_compute_fft_slice_and_its_peaks(self):
        t = np.arange(20_000) / 100.0
        c = {**self.c, "x": t, "signals": {"normalized": {"label": "N", "y": np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.sin(2 * np.pi * 3 * t)}}}
        r = A.run_analysis("fft_window", c, {"center_s": 100.0, "pre_s": 30, "post_s": 30})
        f, p, _, _ = compute_fft_slice(t, c["signals"]["normalized"]["y"], 100.0, 100.0, pre=30, post=30)
        assert_array_equal(r.tables["spectrum"]["frequency_hz"].to_numpy(), f)
        assert_array_equal(r.tables["spectrum"]["power"].to_numpy(), p)
        peaks = r.tables["peaks"]
        self.assertEqual(list(peaks.columns), ["Channel", "Peak Rank", "Frequency (Hz)", "BPM", "Power"])
        self.assertAlmostEqual(peaks.iloc[0]["Frequency (Hz)"], 1.2, delta=0.05)
        self.assertAlmostEqual(peaks.iloc[0]["BPM"], peaks.iloc[0]["Frequency (Hz)"] * 60)
        self.assertEqual(list(peaks["Peak Rank"]), list(range(1, len(peaks) + 1)))
        self.assertEqual(peaks.iloc[0]["Frequency (Hz)"], find_fft_peaks(f, p)[0]["freq_hz"])

    def test_oxysoft_runs_every_channel(self):
        n = 6000
        x = np.arange(n) / 10.0
        c = {"source": "Oxysoft", "x": x, "fs": 10.0, "o2hb": np.vstack([np.sin(x), np.cos(x)]), "hhb": np.vstack([-np.sin(x), -np.cos(x)]),
             "thb": np.ones((2, n)), "markers": []}
        r = A.run_analysis("auc_window", c, {"center_s": 300.0, "pre_s": 5, "post_s": 5})
        self.assertEqual(list(r.tables["auc"]["Channel"]), ["Mean O2Hb", "Mean HHb", "Mean tHb"])
        no_thb = {k: v for k, v in c.items() if k != "thb"}
        self.assertEqual(list(A.run_analysis("auc_window", no_thb, {"center_s": 300.0}).tables["auc"]["Channel"]), ["Mean O2Hb", "Mean HHb"])

    def test_a_generic_table_uses_its_first_column(self):
        x = np.arange(500.0)
        c = {"source": "Generic", "x": x, "fs": 1.0, "y_columns": {"a": np.sin(x / 20), "b": x}, "markers": []}
        r = A.run_analysis("auc_window", c, {"center_s": 250.0, "pre_s": 20, "post_s": 20})
        self.assertEqual(list(r.tables["auc"]["Channel"]), ["Signal"])

    def test_a_recording_with_only_a_corrected_trace(self):
        c = {"source": "TDT", "x": self.x, "fs": self.fs, "corr": self.y, "markers": []}
        r = A.run_analysis("auc_window", c, {"center_s": 140.0})
        self.assertEqual(list(r.tables["auc"]["Channel"]), ["dF/F (corrected)"])


class TestCurveFit(unittest.TestCase):
    def setUp(self):
        x = np.arange(0, 100, 0.1)
        self.c = {"source": "TDT", "x": x, "fs": 10.0, "markers": [],
                  "signals": {"normalized": {"label": "N", "y": 3.0 * x + 2.0}}}

    def test_a_linear_fit_recovers_the_line_between_the_two_times(self):
        r = A.run_analysis("curve_fit", self.c, {"t1_s": 10.0, "t2_s": 30.0})
        fit = r.tables["fit"]
        self.assertEqual(list(fit.columns), ["Channel", "Model", "Parameter", "Value", "R2", "t1 (s)", "t2 (s)", "dt (s)"])
        self.assertEqual(list(fit["Parameter"]), ["m (slope)", "b (intercept)"])
        assert_allclose(fit["Value"], [3.0, 2.0], rtol=1e-9)
        self.assertAlmostEqual(fit.iloc[0]["R2"], 1.0)
        self.assertEqual((fit.iloc[0]["t1 (s)"], fit.iloc[0]["t2 (s)"], fit.iloc[0]["dt (s)"]), (10.0, 30.0, 20.0))
        self.assertEqual(r.summary["model"], "Linear  (y = mx + b)")

    def test_it_is_fit_model_to_segment_on_those_samples(self):
        model, p0, _ = pl.CURVE_FIT_MODELS["Linear  (y = mx + b)"]
        x, y = self.c["x"], self.c["signals"]["normalized"]["y"]
        i1, i2 = 100, 300
        want = fit_model_to_segment(x[i1:i2 + 1], y[i1:i2 + 1], model, p0)
        r = A.run_analysis("curve_fit", self.c, {"t1_s": x[i1], "t2_s": x[i2]})
        assert_array_equal(r.tables["fit"]["Value"].to_numpy(), want["popt"])
        assert_array_equal(r.tables["curve"]["fit"].to_numpy(), want["y_fit"])

    def test_the_two_times_snap_to_the_nearest_samples_in_either_order(self):
        a = A.run_analysis("curve_fit", self.c, {"t1_s": 10.02, "t2_s": 29.98})
        b = A.run_analysis("curve_fit", self.c, {"t1_s": 29.98, "t2_s": 10.02})
        self.assertEqual((a.tables["fit"].iloc[0]["t1 (s)"], a.tables["fit"].iloc[0]["t2 (s)"]), (10.0, 30.0))
        assert_array_equal(a.tables["fit"]["Value"].to_numpy(), b.tables["fit"]["Value"].to_numpy())

    def test_models_are_named_by_registry_name_or_short_id(self):
        names = {"linear": "Linear  (y = mx + b)", "exponential_decay": "Exponential Decay  (a*e^(-bx) + c)",
                 "double_exponential": "Double Exponential  (a*e^(-bx) + c*e^(-dx) + k)"}
        for short, full in names.items():
            self.assertEqual(A.resolve_curve_model(short), full)
            self.assertEqual(A.resolve_curve_model(full), full)
        self.assertEqual(A.resolve_curve_model("Exponential Decay"), names["exponential_decay"])
        self.assertEqual(A.resolve_curve_model("exponential-decay"), names["exponential_decay"])
        self.assertEqual(set(A.curve_model_ids()), {"linear", "exponential_decay", "exponential_rise", "gaussian", "sinusoidal", "double_exponential"})
        with self.assertRaisesRegex(A.ParameterError, "unknown curve-fit model"):
            A.resolve_curve_model("cubic")

    def test_an_exponential_decay_is_recovered(self):
        x = np.arange(0, 20, 0.1)
        c = {**self.c, "x": x, "signals": {"normalized": {"label": "N", "y": 4 * np.exp(-0.3 * x) + 1}}}
        r = A.run_analysis("curve_fit", c, {"t1_s": 0.0, "t2_s": 19.0, "model": "exponential_decay"})
        assert_allclose(r.tables["fit"]["Value"], [4.0, 0.3, 1.0], rtol=1e-4)

    def test_too_short_a_stretch_is_a_note_not_a_crash(self):
        r = A.run_analysis("curve_fit", self.c, {"t1_s": 10.0, "t2_s": 10.1})
        self.assertEqual(len(r.tables["fit"]), 0)
        self.assertTrue(any("fewer than 4 samples" in n for n in r.notes))


class TestEventPeth(unittest.TestCase):
    def setUp(self):
        self.c = make_cache()
        self.y = self.c["signals"]["normalized"]["y"]

    def expected(self, event_times=EVENTS, pre=10.0, post=20.0, bin_s=1.0, selected=None):
        res = compute_event_zscore_peth(self.c["x"], smooth_signal(self.y, self.c["fs"]), event_times, pre, post)
        m = res["trial_matrix"]
        sel = list(range(m.shape[0])) if selected is None else selected
        mean, sem = compute_group_stats(m[sel])
        return res, m, sel, mean, sem

    def test_it_is_compute_event_zscore_peth_and_its_group_statistics(self):
        r = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 10, "post_s": 20, "bin_s": 2})
        res, m, sel, mean, sem = self.expected(bin_s=2.0)
        assert_array_equal(r.tables["zscore"]["z_score"].to_numpy(), mean)
        assert_array_equal(r.tables["zscore"]["time_from_event_s"].to_numpy(), res["time_axis"])
        assert_array_equal(r.tables["mean_sem"]["sem"].to_numpy(), sem)
        self.assertEqual(r.summary["n_trials"], 5)
        self.assertEqual(r.summary["n_selected"], 5)
        auc = compute_auc_matrix(res["time_axis"], m)
        a = r.tables["auc"]
        self.assertEqual(list(a.columns), ["Trial", "Trial Event Time (s)", "Total AUC", "Positive AUC", "Negative AUC", "Checked"])
        self.assertEqual(list(a["Trial"][:5]), [1, 2, 3, 4, 5])                  # 1-based
        self.assertEqual(a.iloc[-1]["Trial"], "GROUP (checked only)")
        self.assertEqual(a.iloc[0]["Total AUC"], auc["per_trial"][0]["total_auc"])
        self.assertEqual(a.iloc[-1]["Total AUC"], compute_auc_from_trace(res["time_axis"], mean)["total_auc"])
        peri = compute_peri_event_matrix(res["time_axis"], m, 20.0, 2.0)
        pk = r.tables["peak"]
        self.assertEqual(list(pk.columns), ["Trial", "Trial Event Time (s)", "Peak Amplitude", "Time-to-Peak (s)", "Checked"])
        self.assertEqual(pk.iloc[0]["Peak Amplitude"], peri["per_trial"][0]["peak_amplitude"])
        self.assertEqual(pk.iloc[-1]["Peak Amplitude"], compute_peri_event_from_trace(res["time_axis"], mean, 20.0, 2.0)["peak_amplitude"])
        self.assertEqual(r.summary["group_peak"], pk.iloc[-1]["Peak Amplitude"])

    def test_trial_numbers_are_one_based_with_each_trials_own_event_time(self):
        r = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
        t = r.tables["trials"]
        self.assertEqual(list(t.columns), ["Trial", "Trial Event Time (s)", "time_from_event_s", "z_score"])
        self.assertEqual(sorted(set(t["Trial"])), [1, 2, 3, 4, 5])
        self.assertEqual(sorted(set(t["Trial Event Time (s)"])), EVENTS)
        self.assertEqual(len(t), 5 * 300)

    def test_bin_columns_are_named_by_their_edges(self):
        r = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "post_s": 6, "pre_s": 5, "bin_s": 2})
        self.assertEqual(list(r.tables["bins"].columns), ["Trial", "Trial Event Time (s)", "Bin 0.0-2.0s Mean", "Bin 2.0-4.0s Mean",
                                                          "Bin 4.0-6.0s Mean", "Checked"])

    def test_selecting_trials_changes_the_group_and_the_checked_column_but_not_the_per_trial_numbers(self):
        everything = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
        some = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 10, "post_s": 20, "selected_trials": [2, 4]})
        _, m, sel, mean, sem = self.expected(selected=[1, 3])
        assert_array_equal(some.tables["zscore"]["z_score"].to_numpy(), mean)
        assert_array_equal(some.tables["mean_sem"]["sem"].to_numpy(), sem)
        self.assertEqual(sorted(set(some.tables["trials"]["Trial"])), [2, 4])                 # only the checked traces
        self.assertEqual(list(some.tables["auc"]["Checked"][:5]), [False, True, False, True, False])
        self.assertTrue(pd.isna(some.tables["auc"].iloc[-1]["Checked"]))
        assert_array_equal(some.tables["auc"].iloc[:5].drop(columns="Checked").to_numpy(),
                           everything.tables["auc"].iloc[:5].drop(columns="Checked").to_numpy())
        self.assertEqual((some.summary["n_trials"], some.summary["n_selected"]), (5, 2))

    def test_no_trials_selected_gives_a_zero_group(self):
        r = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 10, "post_s": 20, "selected_trials": []})
        self.assertEqual(r.summary["n_selected"], 0)
        self.assertTrue((r.tables["zscore"]["z_score"] == 0).all())
        self.assertEqual(len(r.tables["trials"]), 0)

    def test_a_trial_that_does_not_exist_is_refused(self):
        with self.assertRaisesRegex(A.ParameterError, r"trial\(s\) \[9\] do not exist; there are 5"):
            A.run_analysis("event_peth", self.c, {"event": "L1P¹", "selected_trials": [1, 9]})

    def test_events_are_found_by_the_name_the_app_shows_including_renames(self):
        a = A.run_analysis("event_peth", self.c, {"event": "Lever¹", "pre_s": 10, "post_s": 20}, store_labels={"L1P": "Lever"})
        b = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
        assert_array_equal(a.tables["zscore"]["z_score"].to_numpy(), b.tables["zscore"]["z_score"].to_numpy())
        with self.assertRaisesRegex(A.ParameterError, "no event named 'L1P¹'"):
            A.run_analysis("event_peth", self.c, {"event": "L1P¹"}, store_labels={"L1P": "Lever"})

    def test_a_note_placed_by_hand_is_an_event_too(self):
        r = A.run_analysis("event_peth", self.c, {"event": "MyNote", "pre_s": 10, "post_s": 20})
        self.assertEqual(r.summary["n_trials"], 1)

    def test_an_unknown_event_lists_what_there_is(self):
        with self.assertRaisesRegex(A.ParameterError, "no event named 'Nope'.*L1P¹"):
            A.run_analysis("event_peth", self.c, {"event": "Nope"})

    def test_the_event_times_used_are_recorded_in_the_parameters_and_replay_exactly(self):
        r = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
        self.assertEqual(r.parameters["event_times"], EVENTS)
        again = A.run_analysis("event_peth", {**self.c, "markers": [], "detected_markers": []}, r.parameters)    # no markers needed
        for name in r.tables:
            pd.testing.assert_frame_equal(r.tables[name], again.tables[name])

    def test_an_event_with_no_samples_in_its_window_is_skipped_with_a_note(self):
        c = make_cache()
        c["detected_markers"].append(phase_marker(400.0, "high"))                  # the recording is 300 s long
        r = A.run_analysis("event_peth", c, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
        self.assertEqual((r.summary["n_events"], r.summary["n_trials"]), (6, 5))
        self.assertTrue(any("1 of 6" in n for n in r.notes))

    def test_an_event_with_no_usable_trial_is_an_error(self):
        with self.assertRaisesRegex(A.AnalysisError, "No usable trials"):
            A.run_analysis("event_peth", self.c, {"event": "x", "event_times": [5000.0], "pre_s": 10, "post_s": 20})

    def test_the_window_must_be_positive(self):
        with self.assertRaisesRegex(A.ParameterError, "must be positive"):
            A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 0})

    def test_it_always_analyses_the_normalized_signal(self):
        a = A.run_analysis("event_peth", self.c, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
        c = {**self.c, "signals": {"main_driver": self.c["signals"]["main_driver"], "normalized": self.c["signals"]["normalized"]}}
        b = A.run_analysis("event_peth", c, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
        assert_array_equal(a.tables["zscore"]["z_score"].to_numpy(), b.tables["zscore"]["z_score"].to_numpy())


class TestPeakFinder(unittest.TestCase):
    def setUp(self):
        self.c = make_cache()
        self.y = self.c["signals"]["normalized"]["y"]
        self.clean = smooth_signal(self.y, self.c["fs"])

    def test_whole_recording_finds_the_same_peaks_as_the_library(self):
        r = A.run_analysis("peak_finder", self.c, {"scope": "whole", "z_threshold": 2.0, "min_distance_s": 5.0})
        want = find_significant_peaks(self.c["x"], self.clean, z_threshold=2.0, min_distance_sec=5.0)
        self.assertEqual(list(r.tables["peaks"].columns), ["time_s", "z_score", "kind"])
        assert_array_equal(r.tables["peaks"]["time_s"].to_numpy(), [q["time"] for q in want])
        self.assertEqual(r.summary["n_peaks"], len(want))
        self.assertGreaterEqual(len(want), 1)

    def test_all_events_of_one_name(self):
        r = A.run_analysis("peak_finder", self.c, {"scope": "all_same_name", "event": "L1P¹", "z_threshold": 2.0, "pre_s": 5, "post_s": 10})
        want = find_peak_near_events(self.c["x"], self.clean, EVENTS, 5, 10, z_threshold=2.0)
        self.assertEqual(list(r.tables["alignment"].columns), ["Event Time (s)", "Found", "Latency (s)", "Z-score"])
        self.assertEqual(list(r.tables["alignment"]["Found"]), [w["found"] for w in want])
        self.assertEqual(r.summary["n_found"], sum(w["found"] for w in want))
        self.assertEqual(r.parameters["event_times"], EVENTS)

    def test_one_event(self):
        r = A.run_analysis("peak_finder", self.c, {"scope": "one_event", "event_time": 90.0, "z_threshold": 2.0, "pre_s": 5, "post_s": 10})
        self.assertEqual(len(r.tables["alignment"]), 1)
        self.assertEqual(r.tables["alignment"].iloc[0]["Event Time (s)"], 90.0)

    def test_not_found_rows_have_blank_latency_and_z(self):
        r = A.run_analysis("peak_finder", self.c, {"scope": "one_event", "event_time": 20.0, "z_threshold": 50.0, "pre_s": 5, "post_s": 10})
        row = r.tables["alignment"].iloc[0]
        self.assertFalse(row["Found"])
        self.assertTrue(pd.isna(row["Latency (s)"]) and pd.isna(row["Z-score"]))

    def test_scan_all_types_is_sorted_by_hit_rate(self):
        r = A.run_analysis("peak_finder", self.c, {"scope": "scan_all", "z_threshold": 3.0, "pre_s": 5, "post_s": 10})
        scan = r.tables["scan"]
        self.assertEqual(list(scan.columns), ["Event", "Occurrences", "Found", "Hit Rate", "Avg Z (found only)"])
        self.assertEqual(set(scan["Event"]), {"L1P¹", "L1P⁰", "MyNote"})
        self.assertEqual(list(scan["Hit Rate"]), sorted(scan["Hit Rate"], reverse=True))
        self.assertEqual(set(r.tables["alignment"]["Event"]), set(scan["Event"]))

    def test_scan_all_records_the_groups_it_used_and_replays_without_the_markers(self):
        r = A.run_analysis("peak_finder", self.c, {"scope": "scan_all", "z_threshold": 3.0, "pre_s": 5, "post_s": 10})
        self.assertEqual(set(r.parameters["event_groups"]), {"L1P¹", "L1P⁰", "MyNote"})
        self.assertEqual(r.parameters["event_groups"]["L1P¹"], EVENTS)
        again = A.run_analysis("peak_finder", {**self.c, "markers": [], "detected_markers": []}, r.parameters)
        for name in r.tables:
            pd.testing.assert_frame_equal(r.tables[name], again.tables[name])

    def test_event_groups_are_checked(self):
        p = A.normalize_parameters("peak_finder", {"scope": "scan_all", "event_groups": {"a": ["1", 2]}})
        self.assertEqual(p["event_groups"], {"a": [1.0, 2.0]})
        with self.assertRaises(A.ParameterError):
            A.normalize_parameters("peak_finder", {"scope": "scan_all", "event_groups": {"a": ["x"]}})

    def test_the_scope_needs_what_it_needs(self):
        with self.assertRaisesRegex(A.ParameterError, "one_event needs event_time"):
            A.run_analysis("peak_finder", self.c, {"scope": "one_event"})
        with self.assertRaisesRegex(A.ParameterError, "all_same_name needs event"):
            A.run_analysis("peak_finder", self.c, {"scope": "all_same_name"})


class TestEventIntervalsAndProcess(unittest.TestCase):
    def test_intervals_are_the_librarys_with_the_exports_column_names(self):
        c = make_cache()
        c["markers"] = c["detected_markers"][:6]
        r = A.run_analysis("event_intervals", c, {})
        t = r.tables["intervals"]
        self.assertEqual(list(t.columns), ["time_s", "store", "label", "phase", "dt_since_prev_same_store_s", "dt_since_prev_any_s"])
        self.assertEqual(len(t), 6)
        self.assertTrue(pd.isna(t.iloc[0]["dt_since_prev_any_s"]))
        self.assertEqual(r.summary["n_markers"], 6)

    def test_the_markers_measured_are_recorded_and_replay_without_the_recordings_markers(self):
        c = make_cache()
        c["markers"] = c["detected_markers"][:6]
        r = A.run_analysis("event_intervals", c, {})
        self.assertEqual(len(r.parameters["markers"]), 6)
        again = A.run_analysis("event_intervals", {**c, "markers": []}, r.parameters)
        pd.testing.assert_frame_equal(r.tables["intervals"], again.tables["intervals"])

    def test_nothing_to_measure_is_an_error(self):
        with self.assertRaisesRegex(A.AnalysisError, "No markers"):
            A.run_analysis("event_intervals", {**make_cache(), "markers": []}, {})

    def test_process_returns_every_signal_the_recording_offers(self):
        c = make_cache()
        t = A.run_analysis("process", c, {}).tables["trace"]
        self.assertEqual(list(t.columns), ["time_s", "normalized", "main_driver"])
        self.assertEqual(len(t), len(c["x"]))
        self.assertTrue(all(t[col].dtype == np.float64 for col in t.columns))

    def test_process_can_be_limited_and_refuses_an_unknown_signal(self):
        c = make_cache()
        self.assertEqual(list(A.run_analysis("process", c, {"signals": "main_driver"}).tables["trace"].columns), ["time_s", "main_driver"])
        with self.assertRaisesRegex(A.ParameterError, "no signal"):
            A.run_analysis("process", c, {"signals": "isosbestic"})

    def test_process_on_oxysoft_and_generic(self):
        n = 50
        ox = {"source": "Oxysoft", "x": np.arange(n) / 10.0, "fs": 10.0, "o2hb": np.ones((2, n)), "hhb": np.zeros((2, n)), "markers": []}
        self.assertEqual(list(A.run_analysis("process", ox, {}).tables["trace"].columns), ["time_s", "O2Hb 1", "O2Hb 2", "HHb 1", "HHb 2"])
        gen = {"source": "Generic", "x": np.arange(n, dtype=float), "fs": 1.0, "y_columns": {"a": np.ones(n), "b": np.zeros(n)}, "markers": []}
        self.assertEqual(list(A.run_analysis("process", gen, {}).tables["trace"].columns), ["time_s", "a", "b"])


class TestGroupFromSpec(unittest.TestCase):
    def spec(self):
        return A.GroupSpec(group_name="G", subjects=[{"subject": "S1", "folder": "f1"}, {"subject": "S2", "folder": "f2"}],
                           markers=["a"], metrics=["auc"])

    def test_a_spec_is_checked_before_anything_runs(self):
        bad = A.GroupSpec(group_name="G", subjects=[], markers=[])
        with mock.patch.object(A, "run_group_analysis") as run, self.assertRaisesRegex(A.ParameterError, "not usable"):
            A.run_group_from_spec(bad)
        run.assert_not_called()

    def test_a_dict_or_a_spec_are_both_accepted_and_the_tables_come_back(self):
        fake = mock.Mock()
        fake.frames.return_value = {"descriptives": pd.DataFrame({"a": [1]}), "empty": pd.DataFrame()}
        fake.notes = [{"text": "careful"}]
        fake.files = ["x.csv"]
        for spec in (self.spec(), self.spec().to_dict()):
            with mock.patch.object(A, "run_group_analysis", return_value=fake) as run:
                result, raw = A.run_group_from_spec(spec, out_dir="out")
            run.assert_called_once()
            self.assertEqual(run.call_args.kwargs["out_dir"], "out")
            self.assertEqual(list(result.tables), ["descriptives"])                  # empty tables are left out
            self.assertEqual((result.kind, result.notes, result.summary["files"]), (A.GROUP_KIND, ["careful"], ["x.csv"]))
            self.assertIs(raw, fake)
            self.assertEqual(result.parameters, self.spec().to_dict())


class TestDeterminism(unittest.TestCase):
    def test_running_twice_gives_identical_tables(self):
        c = make_cache()
        for kind, params in (("event_peth", {"event": "L1P¹", "pre_s": 10, "post_s": 20}), ("zscore_window", {"center_s": 90.0}),
                             ("fft_window", {"center_s": 100.0}), ("auc_window", {"center_s": 100.0}),
                             ("curve_fit", {"t1_s": 10.0, "t2_s": 20.0})):
            a, b = A.run_analysis(kind, c, params), A.run_analysis(kind, c, params)
            for name in a.tables:
                with self.subTest(kind, table=name):
                    pd.testing.assert_frame_equal(a.tables[name], b.tables[name])

    def test_float32_input_gives_the_same_tables(self):
        c = make_cache()
        c32 = {**c, "x": c["x"].astype(np.float32), "signals": {k: {**v, "y": v["y"].astype(np.float32)} for k, v in c["signals"].items()}}
        c64 = {**c32, "x": c32["x"].astype(np.float64), "signals": {k: {**v, "y": v["y"].astype(np.float64)} for k, v in c32["signals"].items()}}
        a, b = A.run_analysis("zscore_window", c32, {"center_s": 90.0}), A.run_analysis("zscore_window", c64, {"center_s": 90.0})
        pd.testing.assert_frame_equal(a.tables["window"], b.tables["window"])
        self.assertTrue(all(t[col].dtype != np.float32 for t in a.tables.values() for col in t.columns))


if __name__ == "__main__":
    unittest.main()
