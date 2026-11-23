"""The pipeline's options (pipeline.py): the registry, and each option of the TDT processing and the event-locked analyses.

The rule they all keep: every option's default is what the pipeline did before the option existed, bit for bit. tests/legacy_pipeline.py is a frozen copy of the old
functions; the default options must give the same arrays as it. Beyond that each option is tested for what it says it does (against an independent computation),
and the options are followed from the recording's saved state through loading, a splice, the engine, the analyses and the provenance record that replays them."""

import json
import os
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import butter, sosfiltfilt

import tdt

import PyATCLI as pl
from PyATCLI import pipeline, saved_state
from PyATCLI import provenance as pv
from PyATCLI.analysis import shared, zscore_peth
from PyATCLI.engine import wire
from PyATCLI.engine.session import Session
from PyATCLI.analysis.event_peth import compute_event_zscore_peth
from PyATCLI.pipeline import PipelineOptions
from PyATCLI.recording import apply_splice, load_recording, pipeline_sidecar_path, read_saved_pipeline, resolve_pipeline
from PyATCLI.sources import processing_TDT as P

from tests import legacy_pipeline as legacy
from tests.test_cli import fake_block, make_tdt_folder
from tests.test_motion_correction import FS, _photometry


def same(a, b):
    """Bit for bit: the same array, not merely close."""
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class TestTheRegistry(unittest.TestCase):
    def test_every_option_of_every_step_is_declared_with_a_default_that_is_allowed_and_words_for_a_person(self):
        options = pipeline.all_options()
        self.assertEqual({o.step for o in options}, {"motion", "dff", "denoise", "smoothing", "zscore"})
        self.assertEqual(len(options), 14)
        for o in options:
            self.assertEqual(o.check(o.default), o.default, o.name)
            self.assertTrue(o.label and o.doc, f"{o.name} needs a label and an explanation: the app's dialog shows them")
        self.assertEqual([s.name for s in pipeline.steps()], ["motion", "dff", "denoise", "splice", "smoothing", "zscore"])    # in the order of the pipeline (splice has no options)

    def test_the_defaults_are_what_the_code_had_before(self):
        d = pipeline.defaults()
        self.assertEqual((d["denoise.enabled"], d["denoise.cutoff_hz"], d["denoise.order"]), (True, float(P.DENOISE_CUTOFF_HZ), P.DENOISE_ORDER))
        self.assertEqual((d["dff.f0_method"], d["smoothing.window_s"]), ("double_exponential", shared.SMOOTH_WINDOW_SEC))
        self.assertEqual((d["zscore.baseline_start_s"], d["zscore.baseline_end_s"], d["zscore.statistic"]), (None, 0.0, "mean_sd"))
        self.assertTrue(all(d[k] is None for k in ("motion.fit_start_s", "motion.fit_end_s", "motion.exclude_start_s", "motion.exclude_end_s")))
        self.assertTrue(PipelineOptions().is_default())
        self.assertEqual(PipelineOptions().changed(), {})

    def test_a_value_is_checked_and_the_error_says_what_is_wrong(self):
        for values, words in (({"denoise.order": 9}, "at most 8"), ({"denoise.order": 2.5}, "whole number"), ({"denoise.cutoff_hz": 0}, "at least"),
                              ({"denoise.cutoff_hz": "five"}, "is a number"), ({"denoise.enabled": 1}, "true or false"),
                              ({"dff.f0_method": "cubic"}, "must be one of"), ({"dff.f0_percentile": 101}, "at most 100"),
                              ({"smoothing.window_s": -1}, "at least"), ({"smoothing.window_s": float("nan")}, "is a number"),
                              ({"denoise.cutoff_hz": None}, "needs a number")):
            with self.subTest(values):
                with self.assertRaisesRegex(ValueError, words):
                    PipelineOptions(values)

    def test_an_unknown_option_is_refused_with_the_list_of_the_real_ones(self):
        with self.assertRaisesRegex(ValueError, r"unknown pipeline option denoise\.cutoff.*denoise\.cutoff_hz"):
            PipelineOptions({"denoise.cutoff": 5})

    def test_options_that_disagree_with_each_other_are_refused(self):
        for values, words in (({"motion.fit_start_s": 60, "motion.fit_end_s": 30}, "must be before"), ({"motion.exclude_start_s": 10}, "together"),
                              ({"motion.exclude_start_s": 30, "motion.exclude_end_s": 10}, "must be before"),
                              ({"zscore.baseline_start_s": 0.0}, "must be before"), ({"zscore.baseline_start_s": -3, "zscore.baseline_end_s": -5}, "must be before")):
            with self.subTest(values):
                with self.assertRaisesRegex(ValueError, words):
                    PipelineOptions(values)

    def test_a_value_is_held_as_the_kind_it_is_and_the_set_can_be_listed_changed_and_copied(self):
        o = PipelineOptions({"denoise.cutoff_hz": 8, "denoise.order": 3.0})
        self.assertEqual((o["denoise.cutoff_hz"], type(o["denoise.cutoff_hz"]), o["denoise.order"], type(o["denoise.order"])), (8.0, float, 3, int))
        self.assertEqual(o.changed(), {"denoise.cutoff_hz": 8.0, "denoise.order": 3})
        self.assertEqual(len(o.to_dict()), 14)
        json.dumps(o.to_dict())
        again = o.with_values({"dff.f0_method": "linear"})
        self.assertEqual(again.changed(), {"denoise.cutoff_hz": 8.0, "denoise.order": 3, "dff.f0_method": "linear"})
        self.assertEqual(o.changed(), {"denoise.cutoff_hz": 8.0, "denoise.order": 3})                  # (the first is not changed)
        self.assertEqual(PipelineOptions(o.to_dict()), o)
        self.assertEqual(o.step_values("denoise"), {"enabled": True, "cutoff_hz": 8.0, "order": 3})
        self.assertIs(PipelineOptions.from_dict(o), o)
        self.assertIs(PipelineOptions.from_dict(None), PipelineOptions.from_dict(None))                # (the defaults are made once)
        with self.assertRaises(KeyError):
            o["denoise.cutoff"]

    def test_text_typed_in_a_box_or_a_flag_is_parsed_by_the_option(self):
        o = pipeline.OPTIONS
        self.assertEqual(o["denoise.cutoff_hz"].parse(" 7.5 "), 7.5)
        self.assertEqual((o["denoise.enabled"].parse("No"), o["denoise.enabled"].parse("on")), (False, True))
        self.assertEqual(o["denoise.order"].parse("3"), 3)
        self.assertEqual(o["dff.f0_method"].parse("linear"), "linear")
        self.assertEqual((o["motion.fit_start_s"].parse("none"), o["motion.fit_start_s"].parse(""), o["motion.fit_start_s"].parse("12")), (None, None, 12.0))
        for option, text in (("denoise.cutoff_hz", "fast"), ("denoise.enabled", "maybe"), ("denoise.order", "2.5"), ("dff.f0_method", "cubic"), ("denoise.cutoff_hz", "none")):
            with self.assertRaises(ValueError, msg=(option, text)):
                o[option].parse(text)

    def test_a_step_can_be_declared_again_and_cannot_hold_an_option_of_another_step(self):
        before = dict(pipeline.OPTIONS), dict(pipeline.STEPS)
        self.addCleanup(lambda: (pipeline.OPTIONS.clear(), pipeline.OPTIONS.update(before[0]), pipeline.STEPS.clear(), pipeline.STEPS.update(before[1])))
        extra = pipeline.Option("extra", "gain", "float", 1.0, "Gain", "A made-up option.", minimum=0.0)
        pipeline.register_step("extra", "Extra", 90, [extra])
        self.assertEqual(PipelineOptions({"extra.gain": 2})["extra.gain"], 2.0)                         # (adding an option is declaring it: nothing else changes)
        pipeline.register_step("extra", "Extra", 90, [pipeline.Option("extra", "gain", "float", 3.0, "Gain", "Again.")])
        self.assertEqual(PipelineOptions()["extra.gain"], 3.0)
        with self.assertRaisesRegex(ValueError, "declared in step"):
            pipeline.register_step("other", "Other", 91, [extra])
        with self.assertRaisesRegex(ValueError, "kind must be"):
            pipeline.Option("x", "y", "complex", 1, "Y")
        with self.assertRaisesRegex(ValueError, "not one of its choices"):
            pipeline.Option("x", "y", "choice", "c", "Y", choices=("a", "b"))

    def test_the_option_names_of_a_cache_without_options_are_the_defaults(self):
        self.assertTrue(pipeline.options_of({}).is_default())
        self.assertTrue(pipeline.options_of(None).is_default())
        self.assertEqual(pipeline.options_of({"pipeline": {"smoothing.window_s": 2}})["smoothing.window_s"], 2.0)


# ---------------------------------------------------------------------------
# The defaults are what the pipeline did before: bit for bit
# ---------------------------------------------------------------------------

class TestDefaultsReproduceTheOldPipelineExactly(unittest.TestCase):
    def test_dff_with_a_reference_for_every_regression_method_and_with_none(self):
        t, f465, f415, _ = _photometry(seconds=200)
        f465, f415 = f465.astype(np.float32), f415.astype(np.float32)                                  # (TDT hands over float32)
        for method in P.REGRESSION_METHODS:
            for reference in (f415, None):
                with self.subTest(method=method, reference=reference is not None):
                    old = legacy.compute_dff(f465, reference, FS, regression_method=method)
                    for options in (None, {}, PipelineOptions(), pipeline.defaults()):
                        new = P.compute_dff(f465, reference, FS, regression_method=method, options=options, x=t)
                        for key in ("raw", "corr", "dff", "f0"):
                            self.assertTrue(same(old[key], new[key]), f"{key} differs from the old pipeline")
                        self.assertEqual(old["motion_correction_inlier_fraction"], new["motion_correction_inlier_fraction"])

    def test_a_recording_too_short_to_fit_a_baseline_is_as_it_was_the_floor_and_all(self):
        short = np.linspace(10, 11, 150)
        reference = np.linspace(5, 5.5, 150)
        for y415 in (None, reference):
            old = legacy.compute_dff(short, y415, 50.0)
            new = P.compute_dff(short, y415, 50.0)
            for key in ("raw", "corr", "dff", "f0"):
                self.assertTrue(same(old[key], new[key]), key)
            self.assertTrue(np.all(new["f0"] == 1e-6))

    def test_the_low_pass_filter_function_is_the_old_one_at_its_old_settings(self):
        rng = np.random.default_rng(2)
        y = rng.normal(size=4000)
        self.assertTrue(same(legacy.denoise_signal(y, 100.0), P.denoise_signal(y, 100.0)))
        self.assertTrue(same(legacy.denoise_signal(y, 20.0, cutoff=5, order=2), P.denoise_signal(y, 20.0, cutoff=5.0, order=2)))

    def test_smoothing_and_the_z_score_of_a_window(self):
        rng = np.random.default_rng(3)
        fs = 100.0
        t = np.arange(8000) / fs
        y = np.cumsum(rng.normal(0, 0.01, t.size)) + 0.2 * np.sin(t / 3)
        self.assertTrue(same(legacy.smooth_signal(y, fs), shared.smooth_signal(y, fs)))
        self.assertTrue(same(legacy.smooth_signal(y, fs), shared.smooth_for({"fs": fs}, y)))
        flat = np.ones_like(y)
        for centre in (0.0, 1.0, 10.0, 40.0, 33.37, 79.5, 90.0):                                        # (edges, an event past the end, a flat baseline)
            for signal in (y, flat):
                for pre, post in ((5, 8), (0.5, 0.5), (20, 20)):
                    xa, za = legacy.get_zscore_slice(t, signal, centre, pre=pre, post=post)
                    for options in (None, {}):
                        xb, zb = zscore_peth.get_zscore_slice(t, signal, centre, pre=pre, post=post, options=options)
                        self.assertTrue(same(xa, xb) and same(za, zb), (centre, pre, post))
        xa, za = legacy.get_zscore_slice(t, y, 40.0, window=12)
        xb, zb = zscore_peth.get_zscore_slice(t, y, 40.0, window=12)
        self.assertTrue(same(za, zb))

    def test_a_trial_matrix(self):
        rng = np.random.default_rng(4)
        fs = 50.0
        t = np.arange(10_000) / fs
        y = np.cumsum(rng.normal(0, 0.02, t.size))
        events = [20.0, 55.5, 90.0, 140.0, 199.0]
        new = compute_event_zscore_peth(t, y, events, 5, 10)
        with mock.patch("PyATCLI.analysis.event_peth.get_zscore_slice", lambda *a, options=None, **k: legacy.get_zscore_slice(*a, **k)):
            old = compute_event_zscore_peth(t, y, events, 5, 10)
        self.assertTrue(same(old["trial_matrix"], new["trial_matrix"]))
        self.assertEqual(old["trial_event_times"], new["trial_event_times"])

    def test_a_whole_tdt_recording_is_processed_to_the_same_bits(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tdt, "read_block", fake_block()):
            folder = make_tdt_folder(tmp, "Rec")
            plain = P.process_tdt_folder(folder)
            explicit = P.process_tdt_folder(folder, options=PipelineOptions().to_dict())
            data = fake_block()(folder)
            f465, f415, fs = data.streams["x465A"].data, data.streams["x415A"].data, data.streams["x465A"].fs
            old = legacy.compute_dff(f465, f415, fs)
        for key in ("raw", "corr", "dff", "f0"):
            self.assertTrue(same(plain[key], old[key]) and same(explicit[key], old[key]), key)


# ---------------------------------------------------------------------------
# The processing options
# ---------------------------------------------------------------------------

class TestTheBaselineF0(unittest.TestCase):
    def setUp(self):
        self.t, self.f465, self.f415, self.true = _photometry(seconds=300)

    def dff(self, **options):
        return P.compute_dff(self.f465, self.f415, FS, options=options, x=self.t)

    def test_every_method_gives_a_dff_that_finds_the_known_transients(self):
        for method in P.F0_METHODS:
            with self.subTest(method):
                result = self.dff(**{"dff.f0_method": method})
                self.assertEqual(len(result["dff"]), len(self.t))
                self.assertTrue(np.all(np.isfinite(result["dff"])))
                at = int(100 * FS)
                around = result["dff"][at - 20:at + 20].max() - np.median(result["dff"])
                self.assertGreater(around, 0.02, f"{method}: the transient of 5% at 100 s is not there")

    def test_the_methods_differ_from_each_other_and_from_the_default_except_where_they_agree(self):
        results = {m: self.dff(**{"dff.f0_method": m})["f0"] for m in P.F0_METHODS}
        for a in P.F0_METHODS:
            for b in P.F0_METHODS:
                if a < b and {a, b} != {"double_exponential"} and "fitted_isosbestic" not in (a, b):
                    self.assertFalse(np.array_equal(results[a], results[b]), (a, b))

    def test_the_fitted_isosbestic_is_gu_ppys_465_minus_fitted_415_over_fitted_415(self):
        result = self.dff(**{"dff.f0_method": "fitted_isosbestic", "denoise.enabled": False})
        a, b, _ = P._robust_linear_fit(self.f415, self.f465, method="ols")
        fitted = a * self.f415 + b
        assert_allclose(result["f0"], np.maximum(fitted, 1e-6), rtol=1e-12)
        assert_allclose(result["dff"], (self.f465 - fitted) / np.maximum(fitted, 1e-6), rtol=1e-12, atol=1e-12)

    def test_a_line_is_a_line_a_single_exponential_decays_and_the_running_ones_follow_the_signal(self):
        line = self.dff(**{"dff.f0_method": "linear"})["f0"]
        self.assertLess(np.abs(np.diff(line, 2)).max(), 1e-9)
        single = self.dff(**{"dff.f0_method": "single_exponential"})["f0"]
        self.assertTrue(np.all(np.diff(single) <= 1e-9) and single[0] > single[-1])
        self.assertLess(np.abs(single - self.dff()["f0"]).mean() / self.dff()["f0"].mean(), 0.05)       # (close to the double exponential of this bleaching)
        median = self.dff(**{"dff.f0_method": "running_median", "dff.f0_window_s": 60.0})["f0"]
        assert_allclose(median[:5], np.median(self.f465[:int(30 * FS) + 1]), rtol=0.05)               # (the first window is cut short at the start)

    def test_the_running_window_and_percentile_are_options_of_those_methods_only(self):
        base = self.dff(**{"dff.f0_method": "running_median"})["f0"]
        self.assertFalse(np.array_equal(base, self.dff(**{"dff.f0_method": "running_median", "dff.f0_window_s": 20.0})["f0"]))
        low = self.dff(**{"dff.f0_method": "running_percentile", "dff.f0_percentile": 10.0})["f0"]
        high = self.dff(**{"dff.f0_method": "running_percentile", "dff.f0_percentile": 90.0})["f0"]
        self.assertTrue(np.all(low <= high + 1e-9) and np.any(low < high))
        self.assertTrue(same(self.dff(**{"dff.f0_window_s": 5.0, "dff.f0_percentile": 80.0})["f0"], self.dff()["f0"]))      # (not used by the default method)

    def test_the_running_percentile_is_the_percentile_of_the_window_at_its_centre(self):
        y = np.random.default_rng(5).normal(size=2000)
        trend = P.running_percentile_baseline(y, 10.0, 20.0, 25.0)                                      # 20 s = 200 samples, every 2 s
        for centre in (0, 100, 1000, 1500, 1999):
            window = y[max(0, centre - 100):min(2000, centre + 101)]
            if centre % 20 == 0 or centre == 1999:
                self.assertAlmostEqual(trend[centre], np.percentile(window, 25), places=10)
        self.assertEqual(len(trend), 2000)
        self.assertEqual(len(P.running_percentile_baseline(np.array([]), 10.0, 20.0, 50.0)), 0)

    def test_a_recording_with_no_reference_uses_the_method_on_its_own_trace_and_fitted_isosbestic_falls_back_with_a_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = P.compute_dff(self.f465, None, FS, options={"dff.f0_method": "linear"})
        assert_allclose(result["dff"], P.denoise_signal((self.f465 - result["f0"]) / result["f0"], FS), rtol=1e-12, atol=1e-12)
        with self.assertWarnsRegex(RuntimeWarning, "needs a 415 reference"):
            fallback = P.compute_dff(self.f465, None, FS, options={"dff.f0_method": "fitted_isosbestic"})
        self.assertTrue(same(fallback["dff"], P.compute_dff(self.f465, None, FS)["dff"]))

    def test_too_few_samples_to_fit_gives_the_floor_as_it_always_did(self):
        short = np.linspace(10, 11, 150)
        for method in ("single_exponential", "linear"):
            result = P.compute_dff(short, None, 50.0, options={"dff.f0_method": method})
            self.assertTrue(np.all(result["f0"] == 1e-6), method)


class TestTheLowPassFilter(unittest.TestCase):
    def setUp(self):
        t, self.f465, self.f415, _ = _photometry(seconds=200)
        self.t = t

    def run_dff(self, **options):
        return P.compute_dff(self.f465, self.f415, FS, options=options, x=self.t)

    def test_switched_off_the_dff_is_the_unfiltered_ratio(self):
        result = self.run_dff(**{"denoise.enabled": False})
        a, b, _ = P._robust_linear_fit(self.f415, self.f465, method="ols")
        assert_allclose(result["dff"], (self.f465 - (a * self.f415 + b)) / result["f0"], rtol=1e-12)
        self.assertFalse(np.array_equal(result["dff"], self.run_dff()["dff"]))

    def test_any_other_cutoff_or_order_is_a_zero_phase_butterworth_in_second_order_sections(self):
        unfiltered = self.run_dff(**{"denoise.enabled": False})["dff"]
        for cutoff, order in ((2.0, 2), (5.0, 4), (8.0, 3), (1.0, 6)):
            with self.subTest(cutoff=cutoff, order=order):
                got = self.run_dff(**{"denoise.cutoff_hz": cutoff, "denoise.order": order})["dff"]
                expected = sosfiltfilt(butter(order, cutoff / (FS / 2), btype="low", output="sos"), unfiltered)
                self.assertTrue(same(got, expected))

    def test_a_lower_cutoff_leaves_less_high_frequency_and_a_higher_one_more(self):
        def high(cutoff):
            y = self.run_dff(**{"denoise.cutoff_hz": cutoff})["dff"]
            return np.abs(np.fft.rfft(y - y.mean()))[int(len(y) * 6 / FS):].sum()
        self.assertLess(high(1.0), high(3.0))
        self.assertLess(high(3.0), high(8.0))

    def test_a_cutoff_not_below_half_the_sampling_rate_says_so(self):
        with self.assertRaisesRegex(ValueError, r"below half the sampling rate \(10 Hz\)"):
            self.run_dff(**{"denoise.cutoff_hz": 10.0})
        with self.assertRaisesRegex(ValueError, "below half the sampling rate"):
            P.denoise_signal(np.zeros(100), 20.0, cutoff=0, order=2)

    def test_the_old_settings_given_explicitly_are_the_old_filter(self):
        got = self.run_dff(**{"denoise.cutoff_hz": 5.0, "denoise.order": 2})["dff"]
        self.assertTrue(same(got, self.run_dff()["dff"]))


class TestTheRegressionFitWindow(unittest.TestCase):
    def setUp(self):
        self.t, self.f465, self.f415, _ = _photometry(seconds=300, artifact=True)

    def expected(self, keep):
        a, b, inliers = P._robust_linear_fit(self.f415[keep], self.f465[keep], method="ols")
        fit = a * self.f415 + b
        return self.f465 - fit, inliers

    def test_an_epoch_restricts_the_fit_and_the_fitted_line_is_applied_to_every_sample(self):
        keep = (self.t >= 20) & (self.t <= 200)
        result = P.compute_dff(self.f465, self.f415, FS, options={"motion.fit_start_s": 20, "motion.fit_end_s": 200, "denoise.enabled": False}, x=self.t)
        residual, _ = self.expected(keep)
        assert_allclose(result["raw"], residual, rtol=1e-12, atol=1e-12)
        self.assertEqual(len(result["raw"]), len(self.t))

    def test_a_region_can_be_left_out_so_an_artifact_in_it_does_not_drag_the_fit(self):
        keep = ~((self.t >= 245) & (self.t <= 260))
        plain = P.compute_dff(self.f465, self.f415, FS, options={"denoise.enabled": False}, x=self.t)
        left_out = P.compute_dff(self.f465, self.f415, FS, options={"motion.exclude_start_s": 245, "motion.exclude_end_s": 260, "denoise.enabled": False}, x=self.t)
        residual, _ = self.expected(keep)
        assert_allclose(left_out["raw"], residual, rtol=1e-12, atol=1e-12)
        self.assertFalse(np.allclose(plain["raw"], left_out["raw"]))

    def test_an_epoch_and_a_region_together(self):
        keep = (self.t >= 10) & ~((self.t >= 100) & (self.t <= 120))
        got = P.compute_dff(self.f465, self.f415, FS, options={"motion.fit_start_s": 10, "motion.exclude_start_s": 100, "motion.exclude_end_s": 120,
                                                               "denoise.enabled": False}, x=self.t)
        assert_allclose(got["raw"], self.expected(keep)[0], rtol=1e-12, atol=1e-12)

    def test_the_inlier_fraction_is_of_the_samples_fitted(self):
        keep = self.t <= 200
        got = P.compute_dff(self.f465, self.f415, FS, regression_method="huber", options={"motion.fit_end_s": 200}, x=self.t)
        a, b, inliers = P._robust_linear_fit(self.f415[keep], self.f465[keep], method="huber")
        self.assertEqual(got["motion_correction_inlier_fraction"], inliers)

    def test_without_a_time_axis_the_samples_are_counted_from_the_first_at_the_sampling_rate(self):
        a = P.compute_dff(self.f465, self.f415, FS, options={"motion.fit_end_s": 150})
        b = P.compute_dff(self.f465, self.f415, FS, options={"motion.fit_end_s": 150}, x=np.arange(len(self.f465)) / FS)
        self.assertTrue(same(a["dff"], b["dff"]))
        shifted = P.compute_dff(self.f465, self.f415, FS, options={"motion.fit_end_s": 150}, x=np.arange(len(self.f465)) / FS + 100.0)     # (a spliced time axis)
        self.assertFalse(same(a["dff"], shifted["dff"]))

    def test_a_window_with_too_few_samples_says_so(self):
        with self.assertRaisesRegex(ValueError, "leaves 21 samples; it needs at least 100"):
            P.compute_dff(self.f465, self.f415, FS, options={"motion.fit_start_s": 10, "motion.fit_end_s": 11}, x=self.t)

    def test_a_recording_with_no_reference_has_nothing_to_fit_so_the_window_is_not_an_error(self):
        got = P.compute_dff(self.f465, None, FS, options={"motion.fit_start_s": 10, "motion.fit_end_s": 11}, x=self.t)
        self.assertTrue(same(got["dff"], P.compute_dff(self.f465, None, FS)["dff"]))


# ---------------------------------------------------------------------------
# The analysis options
# ---------------------------------------------------------------------------

class TestSmoothing(unittest.TestCase):
    def test_the_window_is_the_recordings_own_option(self):
        y = np.random.default_rng(6).normal(size=3000)
        cache = {"fs": 100.0, "pipeline": {"smoothing.window_s": 1.0}}
        self.assertTrue(same(shared.smooth_for(cache, y), shared.smooth_signal(y, 100.0, window_sec=1.0)))
        self.assertFalse(same(shared.smooth_for(cache, y), shared.smooth_for({"fs": 100.0}, y)))
        self.assertTrue(same(shared.smooth_for({"fs": 100.0, "pipeline": {"smoothing.window_s": 0.0}}, y), y))        # no smoothing


class TestTheZScoreBaseline(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.fs = 100.0
        self.t = np.arange(6000) / self.fs
        self.y = rng.normal(0, 1, self.t.size)
        self.y[(self.t > 27) & (self.t < 29)] += 6.0                                                 # something in the window that is not baseline

    def z(self, **options):
        return zscore_peth.get_zscore_slice(self.t, self.y, 30.0, pre=10, post=10, options=options)

    def test_the_baseline_can_start_and_end_where_it_is_asked_to_relative_to_the_event(self):
        x, z = self.z(**{"zscore.baseline_start_s": -6.0, "zscore.baseline_end_s": -2.0})
        keep = (x >= 24.0) & (x < 28.0)
        mu, sd = zscore_peth._baseline_mean_std(self.y[(self.t >= 20) & (self.t < 40)][keep])
        assert_allclose(z, (self.y[(self.t >= 20) & (self.t < 40)] - mu) / sd, rtol=1e-12)

    def test_it_can_end_after_the_event_and_start_at_the_start_of_the_window(self):
        x, z = self.z(**{"zscore.baseline_end_s": 3.0})
        seg = self.y[(self.t >= 20) & (self.t < 40)]
        mu, sd = zscore_peth._baseline_mean_std(seg[x < 33.0])
        assert_allclose(z, (seg - mu) / sd, rtol=1e-12)

    def test_median_and_mad_is_the_median_over_1_4826_times_the_mad(self):
        x, z = self.z(**{"zscore.statistic": "median_mad"})
        seg = self.y[(self.t >= 20) & (self.t < 40)]
        base = seg[x < 30.0]
        med = np.median(base)
        assert_allclose(z, (seg - med) / (1.4826 * np.median(np.abs(base - med))), rtol=1e-12)
        self.assertFalse(np.allclose(z, self.z()[1]))

    def test_a_baseline_that_holds_nothing_is_the_whole_window_as_it_was(self):
        a = self.z(**{"zscore.baseline_start_s": -50.0, "zscore.baseline_end_s": -40.0})[1]
        self.assertTrue(np.all(np.isfinite(a)))
        x, z = self.z(**{"zscore.baseline_start_s": -50.0, "zscore.baseline_end_s": -40.0, "zscore.statistic": "median_mad"})
        seg = self.y[(self.t >= 20) & (self.t < 40)]
        med = np.median(seg)
        assert_allclose(z, (seg - med) / (1.4826 * np.median(np.abs(seg - med))), rtol=1e-12)

    def test_a_flat_baseline_still_gives_zeros(self):
        flat = np.ones_like(self.y)
        for statistic in ("mean_sd", "median_mad"):
            _, z = zscore_peth.get_zscore_slice(self.t, flat, 30.0, pre=5, post=5, options={"zscore.statistic": statistic})
            self.assertTrue(np.all(z == 0))

    def test_the_trial_matrix_and_the_peak_finder_take_the_options(self):
        opts = PipelineOptions({"zscore.statistic": "median_mad"})
        plain = compute_event_zscore_peth(self.t, self.y, [30.0, 40.0], 5, 5)["trial_matrix"]
        median = compute_event_zscore_peth(self.t, self.y, [30.0, 40.0], 5, 5, options=opts)["trial_matrix"]
        self.assertFalse(np.allclose(plain, median))
        from PyATCLI.analysis.peak_finder import find_peak_near_events, shuffle_null_near_events
        found = find_peak_near_events(self.t, self.y, [30.0], 8, 8, z_threshold=3.0, options=opts)
        manual = zscore_peth.get_zscore_slice(self.t, self.y, 30.0, pre=8, post=8, options=opts)[1]
        self.assertAlmostEqual(abs(found[0]["z_score"]), np.abs(manual).max(), places=9)
        null = shuffle_null_near_events(self.t, self.y, [30.0, 40.0, 45.0], 5, 5, z_threshold=3.0, n_shuffles=5, seed=1, options=opts)
        self.assertIn("available", null)


# ---------------------------------------------------------------------------
# Where the options live: a recording, its saved state, a splice, the engine, the analyses, the record
# ---------------------------------------------------------------------------

class RecordingCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        patcher = mock.patch.object(tdt, "read_block", fake_block())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = make_tdt_folder(self.tmp, "Rat7")

    def load(self, **kw):
        return load_recording(self.folder, **kw)

    def saved_bytes(self):
        with open(pipeline_sidecar_path(self.folder), "rb") as f:
            return f.read()


class TestLoadingAndSavedState(RecordingCase):
    def test_a_recording_with_no_saved_choices_is_processed_with_the_defaults_and_says_so(self):
        loaded = self.load()
        self.assertEqual(loaded["cache"]["pipeline"], pipeline.defaults())
        self.assertEqual(loaded["warnings"], [])
        self.assertFalse(os.path.exists(pipeline_sidecar_path(self.folder)))

    def test_options_given_to_the_load_are_what_the_recording_is_processed_with(self):
        plain = self.load()["cache"]
        loaded = self.load(pipeline={"denoise.cutoff_hz": 2.0, "dff.f0_method": "linear"})["cache"]
        self.assertEqual(loaded["pipeline"]["denoise.cutoff_hz"], 2.0)
        data = fake_block()(self.folder)
        expected = P.compute_dff(data.streams["x465A"].data, data.streams["x415A"].data, 1000.0,
                                 options={"denoise.cutoff_hz": 2.0, "dff.f0_method": "linear"}, x=plain["x"])
        self.assertTrue(same(loaded["corr"], expected["corr"]))
        self.assertFalse(same(loaded["corr"], plain["corr"]))
        self.assertTrue(same(loaded["signals"]["normalized"]["y"], expected["corr"]))

    def test_saved_choices_are_applied_when_the_recording_opens_and_explicit_ones_go_on_top(self):
        saved_state.set_pipeline(self.folder, {"denoise.cutoff_hz": 3.0, "smoothing.window_s": 1.0})
        cache = self.load()["cache"]
        self.assertEqual((cache["pipeline"]["denoise.cutoff_hz"], cache["pipeline"]["smoothing.window_s"]), (3.0, 1.0))
        on_top = self.load(pipeline={"denoise.cutoff_hz": 4.0})["cache"]["pipeline"]
        self.assertEqual((on_top["denoise.cutoff_hz"], on_top["smoothing.window_s"]), (4.0, 1.0))
        skipped = self.load(use_saved=False)["cache"]["pipeline"]
        self.assertEqual(skipped, pipeline.defaults())                                                # (the recording exactly as recorded)

    def test_what_is_saved_is_every_option_in_a_file_that_is_there_only_while_something_is_not_the_default(self):
        path = saved_state.write_saved_pipeline(self.folder, {"zscore.statistic": "median_mad"})
        self.assertEqual(path, pipeline_sidecar_path(self.folder))
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["schema"], 1)
        self.assertEqual(saved["options"], PipelineOptions({"zscore.statistic": "median_mad"}).to_dict())          # all of them, so a later default changes nothing
        self.assertEqual(read_saved_pipeline(self.folder), saved["options"])
        self.assertIsNone(saved_state.write_saved_pipeline(self.folder, PipelineOptions()))                         # all defaults: nothing left saved
        self.assertFalse(os.path.exists(path))
        self.assertEqual(read_saved_pipeline(self.folder), {})

    def test_set_pipeline_adds_to_what_is_saved_and_refuses_what_is_not_allowed_without_writing(self):
        saved_state.set_pipeline(self.folder, {"denoise.order": 3})
        now = saved_state.set_pipeline(self.folder, {"smoothing.window_s": 2.0})
        self.assertEqual((now["denoise.order"], now["smoothing.window_s"]), (3, 2.0))
        before = self.saved_bytes()
        with self.assertRaisesRegex(ValueError, "at most 8"):
            saved_state.set_pipeline(self.folder, {"denoise.order": 12})
        with self.assertRaisesRegex(ValueError, "unknown pipeline option"):
            saved_state.set_pipeline(self.folder, {"denoise.nope": 1})
        self.assertEqual(self.saved_bytes(), before)

    def test_a_saved_file_that_cannot_be_read_is_a_warning_and_the_defaults_not_a_failed_load(self):
        path = pipeline_sidecar_path(self.folder)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for text in ("not json", json.dumps({"schema": 1, "options": {"denoise.cutoff": 5}}), json.dumps({"schema": 1, "options": {"denoise.order": 99}}),
                     json.dumps([1, 2]), json.dumps({"schema": 7, "options": {}})):
            with self.subTest(text):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                loaded = self.load()
                self.assertEqual(loaded["cache"]["pipeline"], pipeline.defaults())
                self.assertTrue(any("Saved pipeline options could not be read" in w for w in loaded["warnings"]), loaded["warnings"])
        options, warnings_ = resolve_pipeline(self.folder, {"denoise.order": 3})                                   # (an explicit choice still goes on top of the defaults)
        self.assertEqual(options["denoise.order"], 3)

    def test_an_explicit_value_that_is_not_allowed_is_an_error_not_a_warning(self):
        with self.assertRaisesRegex(ValueError, "at most 8"):
            self.load(pipeline={"denoise.order": 12})

    def test_the_options_of_an_oxysoft_and_a_generic_recording_are_kept_too(self):
        from tests.test_cli import write_oxysoft
        path = os.path.join(self.tmp, "session.txt")
        write_oxysoft(path)
        cache = load_recording(path, pipeline={"smoothing.window_s": 2.0})["cache"]
        self.assertEqual(cache["pipeline"]["smoothing.window_s"], 2.0)
        saved_state.set_pipeline(path, {"zscore.statistic": "median_mad"})
        self.assertEqual(load_recording(path)["cache"]["pipeline"]["zscore.statistic"], "median_mad")
        self.assertEqual(load_recording(path, use_saved=False)["cache"]["pipeline"], pipeline.defaults())


class TestASpliceAppliesTheRecordingsOptions(RecordingCase):
    def test_the_dff_of_what_is_left_is_made_with_the_same_options_and_a_fit_window_follows_the_spliced_time_axis(self):
        options = {"denoise.enabled": False, "dff.f0_method": "linear"}
        cache = self.load(pipeline=options)["cache"]
        spliced = apply_splice(cache, "cut_out", 5.0, 8.0)
        self.assertEqual(spliced["pipeline"], cache["pipeline"])
        data = fake_block()(self.folder)
        keep = (cache["x"] < 5.0) | (cache["x"] > 8.0)
        expected = P.compute_dff(data.streams["x465A"].data[keep], data.streams["x415A"].data[keep], 1000.0, options=options, x=spliced["x"])
        self.assertTrue(same(spliced["corr"], expected["corr"]))
        plain = apply_splice(self.load()["cache"], "cut_out", 5.0, 8.0)
        self.assertFalse(same(spliced["corr"], plain["corr"]))
        windowed = apply_splice(self.load(pipeline={"motion.fit_end_s": 10.0})["cache"], "cut_out", 5.0, 8.0)
        keep_fit = spliced["x"] <= 10.0
        self.assertEqual(int(keep_fit.sum()), int((windowed["x"] <= 10.0).sum()))                      # (counted on the stitched axis, as the sample times now are)

    def test_a_saved_splice_is_replayed_with_the_saved_options(self):
        saved_state.set_pipeline(self.folder, {"denoise.enabled": False})
        saved_state.add_splice(self.folder, "cut_out", 5.0, 8.0)
        cache = self.load()["cache"]
        data = fake_block()(self.folder)
        t = np.arange(len(data.streams["x465A"].data)) / 1000.0
        keep = (t < 5.0) | (t > 8.0)
        expected = P.compute_dff(data.streams["x465A"].data[keep], data.streams["x415A"].data[keep], 1000.0, options={"denoise.enabled": False}, x=cache["x"])
        self.assertTrue(same(cache["corr"], expected["corr"]))


class TestTheAnalysesAndTheRecord(RecordingCase):
    def test_the_event_peth_analysis_uses_the_recordings_z_score_baseline_and_smoothing(self):
        plain = self.load()["cache"]
        tweaked = self.load(pipeline={"zscore.statistic": "median_mad", "smoothing.window_s": 1.0})["cache"]
        a = pl.run_analysis("event_peth", plain, {"event": "L1P¹", "pre_s": 2.0, "post_s": 3.0})
        b = pl.run_analysis("event_peth", tweaked, {"event": "L1P¹", "pre_s": 2.0, "post_s": 3.0})
        self.assertFalse(np.allclose(a.tables["trials"]["z_score"].to_numpy(), b.tables["trials"]["z_score"].to_numpy()))
        again = pl.run_analysis("event_peth", self.load()["cache"], {"event": "L1P¹", "pre_s": 2.0, "post_s": 3.0})
        self.assertTrue(same(a.tables["trials"]["z_score"].to_numpy(), again.tables["trials"]["z_score"].to_numpy()))

    def test_the_click_z_score_window_follows_the_options_too(self):
        plain = pl.run_analysis("zscore_window", self.load()["cache"], {"center_s": 9.0, "pre_s": 3.0, "post_s": 3.0})
        other = pl.run_analysis("zscore_window", self.load(pipeline={"zscore.baseline_start_s": -2.0, "zscore.baseline_end_s": -1.0})["cache"],
                                {"center_s": 9.0, "pre_s": 3.0, "post_s": 3.0})
        self.assertFalse(np.allclose(plain.tables["window"]["z_score"].to_numpy(), other.tables["window"]["z_score"].to_numpy()))

    def test_the_provenance_record_names_the_options_and_a_replay_uses_them(self):
        options = {"denoise.cutoff_hz": 2.0, "dff.f0_method": "running_median", "zscore.statistic": "median_mad", "smoothing.window_s": 1.0}
        cache = self.load(pipeline=options)["cache"]
        state = pv.recording_state(cache, regression_method="ols")
        self.assertEqual(state["pipeline"], PipelineOptions(options).to_dict())
        record = pv.build_record("event_peth", {"event": "L1P¹", "pre_s": 2.0, "post_s": 3.0}, source=self.folder, recording=state,
                                 pipeline=pv.pipeline_parameters("ols", state["pipeline"]))
        self.assertEqual(record["pipeline"]["options"], state["pipeline"])
        self.assertEqual(record["pipeline"]["photobleaching"]["model"], "running median of the 465 stream")
        from PyATCLI import reproduce
        rebuilt = reproduce.load_recording_for_record(record, self.folder)
        self.assertEqual(rebuilt["pipeline"], state["pipeline"])
        self.assertTrue(same(rebuilt["corr"], cache["corr"]))

    def test_a_record_from_before_the_options_replays_with_the_defaults_even_if_options_are_saved_now(self):
        old = self.load()["cache"]
        record = pv.build_record("event_peth", {"event": "L1P¹"}, source=self.folder, recording={"source_type": "TDT", "regression_method": "ols",
                                                                                              "splices": [], "n_samples": 20000})
        self.assertNotIn("pipeline", record["recording"])
        saved_state.set_pipeline(self.folder, {"denoise.enabled": False})                                          # chosen after the record was written
        from PyATCLI import reproduce
        rebuilt = reproduce.load_recording_for_record(record, self.folder)
        self.assertTrue(same(rebuilt["corr"], old["corr"]))
        self.assertEqual(rebuilt["pipeline"], pipeline.defaults())


class TestTheEngine(RecordingCase):
    def setUp(self):
        super().setUp()
        self.session = Session()
        self.addCleanup(self.session.close)

    def call(self, method, params=None):
        return self.session.handle(method, params, lambda event, **fields: None, None)

    def test_an_open_recording_says_the_options_it_was_processed_with_and_a_different_choice_is_a_different_recording(self):
        plain = self.call("recording.open", {"path": self.folder})
        self.assertEqual(plain["pipeline"], pipeline.defaults())
        other = self.call("recording.open", {"path": self.folder, "load": {"pipeline": {"denoise.cutoff_hz": 2.0}}})
        self.assertEqual(other["pipeline"]["denoise.cutoff_hz"], 2.0)
        self.assertNotEqual(other["id"], plain["id"])
        again = self.call("recording.open", {"path": self.folder, "load": {"pipeline": {"denoise.cutoff_hz": 2.0}}})
        self.assertEqual(again["id"], other["id"])                                                          # (the same choice is the recording already open)

    def test_a_splice_in_the_engine_uses_the_options_of_the_recording(self):
        opened = self.call("recording.open", {"path": self.folder, "load": {"pipeline": {"denoise.enabled": False}}})
        view = self.call("recording.view", {"id": opened["id"], "splices": [{"mode": "cut_out", "start": 5.0, "end": 8.0}]})
        spliced = wire.decode(view["cache"], view["directory"])
        data = fake_block()(self.folder)
        keep = (np.arange(len(data.streams["x465A"].data)) / 1000.0 < 5.0) | (np.arange(len(data.streams["x465A"].data)) / 1000.0 > 8.0)
        expected = P.compute_dff(data.streams["x465A"].data[keep], data.streams["x415A"].data[keep], 1000.0, options={"denoise.enabled": False}, x=spliced["x"])
        self.assertTrue(same(spliced["corr"], expected["corr"]))

    def test_a_choice_that_is_not_allowed_is_refused_before_anything_is_read(self):
        with self.assertRaisesRegex(Exception, "at most 8"):
            self.call("recording.open", {"path": self.folder, "load": {"pipeline": {"denoise.order": 50}}})
        self.assertEqual(self.call("recording.list")["open"], [])

    def test_the_analyses_the_engine_runs_use_the_options_too(self):
        opened = self.call("recording.open", {"path": self.folder, "load": {"pipeline": {"zscore.statistic": "median_mad"}}})
        plain = self.call("recording.open", {"path": self.folder})
        params = {"event_times": [4.0, 9.0], "pre": 2.0, "post": 3.0}
        a = self.call("compute.event_peth", {"recording": {"id": opened["id"]}, **params})
        b = self.call("compute.event_peth", {"recording": {"id": plain["id"]}, **params})
        ma = wire.decode(a["result"]["data"], a["result"]["directory"], mmap=False)["trial_matrix"]
        mb = wire.decode(b["result"]["data"], b["result"]["directory"], mmap=False)["trial_matrix"]
        self.assertFalse(np.allclose(ma, mb))


class TestTheCommandLine(RecordingCase):
    def run_cli(self, *argv):
        from tests.test_cli import run_cli
        return run_cli(*argv)

    def test_the_pipeline_command_lists_every_option_with_its_default(self):
        status, out, _ = self.run_cli("pipeline")
        self.assertEqual(status, 0)
        for option in pipeline.all_options():
            self.assertIn(option.name, out)
        self.assertIn("denoise.cutoff_hz", out)
        status, out, _ = self.run_cli("pipeline", "--json")
        data = json.loads(out)["result"]
        self.assertEqual(data["defaults"], pipeline.defaults())
        self.assertEqual([s["name"] for s in data["steps"]], [s.name for s in pipeline.steps()])

    def test_describe_is_the_registry_as_data_for_a_dialog(self):
        described = pipeline.describe()
        option = next(o for s in described["steps"] for o in s["options"] if o["name"] == "dff.f0_method")
        self.assertEqual((option["kind"], option["default"], option["choices"][0], option["label"]), ("choice", "double_exponential", "double_exponential", "Baseline F0"))
        json.dumps(described)

    def test_describe_says_which_kinds_of_recording_a_step_applies_to(self):
        sources = {s["name"]: s["sources"] for s in pipeline.describe()["steps"]}
        self.assertEqual({name: sources[name] for name in ("motion", "dff", "denoise")}, {"motion": ["TDT"], "dff": ["TDT"], "denoise": ["TDT"]})
        self.assertEqual({name: sources[name] for name in ("splice", "smoothing", "zscore")}, {"splice": None, "smoothing": None, "zscore": None})

    def test_a_flag_gives_a_run_an_option_and_a_wrong_one_is_an_error_that_says_what_is_wrong(self):
        out_dir = os.path.join(self.tmp, "out")
        status, out, err = self.run_cli("run", "zscore_window", self.folder, "-p", "center_s=9", "--out", out_dir, "--pipeline", "zscore.statistic=median_mad",
                                        "--pipeline", "smoothing.window_s=1")
        self.assertEqual(status, 0, err)
        records = [f for f in os.listdir(out_dir) if f.endswith(".provenance.json")]
        with open(os.path.join(out_dir, records[0]), encoding="utf-8") as f:
            record = json.load(f)
        self.assertEqual(record["recording"]["pipeline"]["zscore.statistic"], "median_mad")
        self.assertEqual(record["pipeline"]["zscore"]["statistic"], "median and MAD (1.4826 x MAD for the SD)")
        for flags, words in ((["--pipeline", "denoise.order"], "not name=value"), (["--pipeline", "denoise.nope=1"], "unknown pipeline option"),
                             (["--pipeline", "denoise.order=99"], "at most 8")):
            status, _, err = self.run_cli("run", "zscore_window", self.folder, "-p", "center_s=9", "--out", out_dir, *flags)
            self.assertEqual(status, 1, flags)                                                              # (as any other parameter that is not allowed)
            self.assertIn(words, err)

    def test_saved_pipeline_chooses_options_for_a_recording_and_saved_show_lists_them(self):
        status, out, err = self.run_cli("saved", "pipeline", self.folder, "denoise.cutoff_hz=8", "dff.f0_method=linear")
        self.assertEqual(status, 0, err)
        self.assertIn("denoise.cutoff_hz = 8.0", out)
        self.assertEqual(self.load()["cache"]["pipeline"]["dff.f0_method"], "linear")
        status, out, _ = self.run_cli("saved", "show", self.folder)
        self.assertIn("pipeline    denoise.cutoff_hz = 8.0", out)
        self.assertNotIn("denoise.order", out)                                                              # (only what is not the default is listed)
        status, _, err = self.run_cli("saved", "pipeline", self.folder)                                     # nothing to set
        self.assertEqual(status, 1)
        self.assertIn("give at least one NAME=VALUE", err)
        status, out, _ = self.run_cli("saved", "pipeline", self.folder, "--reset")
        self.assertEqual(status, 0)
        self.assertFalse(os.path.exists(pipeline_sidecar_path(self.folder)))

    def test_undo_all_changes_takes_the_options_away_with_the_rest_of_the_saved_state(self):
        saved_state.set_pipeline(self.folder, {"denoise.order": 3})
        self.assertIn("pipeline.json", saved_state.clear_saved_state(self.folder))
        self.assertEqual(self.load()["cache"]["pipeline"], pipeline.defaults())

    def test_save_changes_writes_the_options_whole_and_nothing_for_the_defaults(self):
        from PyATCLI.engine.session import Session
        session = Session()
        self.addCleanup(session.close)
        files = session.handle("saved.write", {"recording": self.folder, "pipeline": {"denoise.cutoff_hz": 6.0}}, None, None)["files"]
        self.assertEqual(files, [pipeline_sidecar_path(self.folder)])
        self.assertEqual(read_saved_pipeline(self.folder)["denoise.cutoff_hz"], 6.0)
        self.assertEqual(session.handle("saved.write", {"recording": self.folder, "pipeline": pipeline.defaults()}, None, None)["files"], [])
        self.assertFalse(os.path.exists(pipeline_sidecar_path(self.folder)))
        self.assertEqual(session.handle("saved.show", {"recording": self.folder}, None, None)["pipeline"], {})


if __name__ == "__main__":
    unittest.main()
