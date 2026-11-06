"""Z-score windows, PETH stacking/statistics, curve fitting and peak finding."""

import inspect
import unittest

import numpy as np
from numpy.testing import assert_allclose

from PyATCLI import models
from PyATCLI.analysis.auc import compute_auc_from_trace
from PyATCLI.analysis.curve_fit import compute_slope_segment, fit_model_to_segment
from PyATCLI.analysis.event_peth import (
    compute_auc_matrix, compute_event_zscore_peth, compute_group_stats,
    compute_peri_event_from_trace, compute_peri_event_matrix,
)
from PyATCLI.analysis.peak_finder import (
    DEFAULT_START_BUFFER_S, find_peak_near_events, find_significant_peaks, holm_adjust, shuffle_null_near_events,
)
from PyATCLI.analysis.zscore_peth import bin_for_heatmap, get_zscore_slice


def _trace_with_events(fs=20.0, seconds=400, event_times=(100, 200, 300), amp=1.0, latency=2.0,
                       noise=0.05, seed=0):
    """A noisy baseline plus an identical Gaussian response `latency` s after each event."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fs)
    y = rng.normal(0, noise, t.size)
    for ev in event_times:
        y += amp * np.exp(-0.5 * ((t - (ev + latency)) / 0.7) ** 2)
    return t, y


class TestZScoreSlice(unittest.TestCase):
    def test_baseline_is_standardised_using_only_the_pre_event_part(self):
        rng = np.random.default_rng(1)
        t = np.arange(0, 100, 0.1)
        y = rng.normal(1.0, 0.3, t.size)
        y[t >= 50] += 2.0                                     # a step right at the event
        x, z = get_zscore_slice(t, y, 50.0, pre=10, post=10)
        pre = z[x < 50]
        self.assertAlmostEqual(pre.mean(), 0.0, places=10)    # exactly standardised over the baseline
        self.assertAlmostEqual(pre.std(), 1.0, places=10)
        self.assertGreater(z[x >= 50].mean(), 5.0)            # step of 2 units / sd 0.3 ~ 6.7 sd

    def test_window_is_split_evenly_but_pre_and_post_override_it(self):
        t = np.arange(0, 100, 0.1)
        y = np.sin(t)
        x1, _ = get_zscore_slice(t, y, 50.0, window=20)
        x2, _ = get_zscore_slice(t, y, 50.0, pre=10, post=10)
        assert_allclose(x1, x2)
        x3, _ = get_zscore_slice(t, y, 50.0, window=20, pre=5, post=15)
        self.assertAlmostEqual(x3[0], 45.0, places=6)
        self.assertLess(x3[-1], 65.0)

    def test_a_flat_baseline_returns_zeros_not_nan_or_inf(self):
        t = np.arange(0, 100, 0.1)
        _, z = get_zscore_slice(t, np.full_like(t, 3.0), 50.0, pre=10, post=10)
        self.assertTrue(np.all(z == 0))

    def test_zscore_is_invariant_to_offset_at_fractional_scale(self):
        rng = np.random.default_rng(2)
        t = np.arange(0, 100, 0.1)
        y = rng.normal(0, 0.02, t.size)                       # dF/F-like magnitudes
        _, z1 = get_zscore_slice(t, y, 50.0, pre=10, post=10)
        _, z2 = get_zscore_slice(t, y + 1.0, 50.0, pre=10, post=10)
        assert_allclose(z1, z2, atol=1e-9)

    def test_zscore_is_invariant_to_the_units_of_the_signal(self):
        """A z-score must not depend on units. (get_zscore_slice used to hard-clip the signal to
        +/-5 in RAW signal units before z-scoring: the same recording as a fraction, in percent,
        or x1000 gave peak z of ~6.0, ~1.7 and ~0.9. It also treated any baseline SD below an
        absolute 1e-6 as flat, so a signal in tiny units came back as all zeros.)"""
        rng = np.random.default_rng(3)
        t = np.arange(0, 100, 0.1)
        y = rng.normal(0, 0.03, t.size)
        y += 0.15 * np.exp(-0.5 * ((t - 55) / 1.0) ** 2)      # a transient ~5 sd tall
        peaks = []
        for k in (1, 100, 1000, 1e-3, 1e-9, 1e9):
            _, z = get_zscore_slice(t, k * y, 50.0, pre=10, post=10)
            peaks.append(z.max())
        assert_allclose(peaks, peaks[0], rtol=1e-6)
        self.assertGreater(peaks[0], 4.0)

    def test_a_signal_with_a_large_baseline_value_still_gets_a_real_zscore(self):
        """Any signal whose values sit above 5 -- a temperature column of 37 +/- 0.5 in a
        generic CSV, say -- used to be clipped to a constant 5, its baseline standard deviation
        became 0, and every z-score silently came back as 0."""
        rng = np.random.default_rng(7)
        t = np.arange(0, 100, 0.1)
        y = 37.0 + rng.normal(0, 0.5, t.size)
        y[t >= 50] += 3.0
        _, z = get_zscore_slice(t, y, 50.0, pre=10, post=10)
        self.assertGreater(np.abs(z).max(), 2.0)

    def test_a_large_real_transient_is_not_truncated(self):
        """A large true response is reported at its real height (it used to be flattened by the
        +/-5 raw-unit clip)."""
        rng = np.random.default_rng(4)
        t = np.arange(0, 100, 0.1)
        y = rng.normal(0, 1.0, t.size)                        # signal already in "sd" units
        y += 12.0 * np.exp(-0.5 * ((t - 55) / 1.0) ** 2)      # a true 12-sd response
        x, z = get_zscore_slice(t, y, 50.0, pre=10, post=10)
        self.assertGreater(z.max(), 9.0)

    def test_the_scored_samples_are_never_altered(self):
        """The z-score is an exact affine function of the signal over the whole window: nothing
        is clipped, however large a sample is (here a 500-unit artefact after the event)."""
        rng = np.random.default_rng(8)
        t = np.arange(0, 100, 0.1)
        y = rng.normal(0, 1.0, t.size)
        y[(t > 53) & (t < 53.5)] += 500.0
        x, z = get_zscore_slice(t, y, 50.0, pre=10, post=10)
        seg = y[np.searchsorted(t, 40.0):np.searchsorted(t, 60.0)]
        slope, intercept = np.polyfit(seg, z, 1)
        assert_allclose(z, slope * seg + intercept, atol=1e-9)
        self.assertGreater(z.max(), 400)

    def test_a_burst_of_artefact_inside_the_baseline_does_not_shrink_the_zscores(self):
        """Three huge samples in the baseline (a motion spike) must not inflate the baseline SD and
        flatten the trial's real response: an unprotected SD would be ~35 here and the response
        would come out at z ~0.2 instead of ~6."""
        rng = np.random.default_rng(9)
        t = np.arange(0, 100, 0.1)
        y = rng.normal(0, 1.0, t.size)
        y[[430, 431, 432]] += 200.0                           # t = 43.0-43.2, inside the 10 s baseline
        y += 8.0 * np.exp(-0.5 * ((t - 55) / 1.0) ** 2)
        x, z = get_zscore_slice(t, y, 50.0, pre=10, post=10)
        self.assertGreater(z[(x > 53) & (x < 57)].max(), 4.0)

    def test_a_flat_baseline_is_flat_in_any_units(self):
        t = np.arange(0, 100, 0.1)
        for level in (3.0, 3e-9, 3e9):
            with self.subTest(level=level):
                _, z = get_zscore_slice(t, np.full_like(t, level), 50.0, pre=10, post=10)
                self.assertTrue(np.all(z == 0))


    def test_a_window_with_no_samples_gives_empty_arrays_and_no_numpy_warnings(self):
        """An event past the end of the recording has nothing to z-score; that must not shout RuntimeWarnings
        ("Mean of empty slice") on the way to the empty result."""
        import warnings
        x = np.arange(0, 100, 0.1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            seg_x, z = get_zscore_slice(x, np.sin(x), 500.0, pre=10, post=20)
        self.assertEqual((len(seg_x), len(z)), (0, 0))

class TestHeatmapBinning(unittest.TestCase):
    def test_bins_are_equal_width_means(self):
        assert_allclose(bin_for_heatmap(np.arange(12, dtype=float), num_bins=4), [1.0, 4.0, 7.0, 10.0])

    def test_empty_input_gives_zeros(self):
        assert_allclose(bin_for_heatmap(np.array([]), num_bins=5), np.zeros(5))
        assert_allclose(bin_for_heatmap(None, num_bins=5), np.zeros(5))


class TestEventPETH(unittest.TestCase):
    def test_trial_matrix_shape_time_axis_and_alignment(self):
        t, y = _trace_with_events(amp=3.0, latency=2.0)
        res = compute_event_zscore_peth(t, y, [100, 200, 300], pre=10, post=10, num_bins=200)
        self.assertEqual(res["trial_matrix"].shape, (3, 200))
        assert_allclose(res["time_axis"][[0, -1]], [-10, 10])
        self.assertEqual(res["trial_event_times"], [100, 200, 300])
        # the response sits 2 s after each event, so the trial-average peaks at ~ +2 s
        peak_time = res["time_axis"][int(np.argmax(res["mean_trace"]))]
        self.assertAlmostEqual(peak_time, 2.0, delta=0.4)
        self.assertGreater(res["mean_trace"].max(), 8.0)

    def test_events_past_the_end_of_the_recording_are_skipped(self):
        t, y = _trace_with_events()
        res = compute_event_zscore_peth(t, y, [100, 10_000], pre=10, post=10)
        self.assertEqual(res["trial_event_times"], [100])
        self.assertEqual(res["trial_matrix"].shape[0], 1)

    def test_no_usable_events_gives_empty_result(self):
        t, y = _trace_with_events()
        res = compute_event_zscore_peth(t, y, [10_000], pre=10, post=10, num_bins=50)
        self.assertEqual(res["trial_matrix"].shape, (0, 50))
        assert_allclose(res["mean_trace"], 0.0)

    def test_group_stats_are_mean_and_sem(self):
        m = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        mean, sem = compute_group_stats(m)
        assert_allclose(mean, [3.0, 6.0])
        assert_allclose(sem, np.array([2.0, 4.0]) / np.sqrt(3))       # sd(ddof=1) / sqrt(n)

    def test_group_stats_edge_cases(self):
        mean, sem = compute_group_stats(np.array([[1.0, 2.0]]))
        assert_allclose(mean, [1.0, 2.0])
        assert_allclose(sem, 0.0)                                      # SEM undefined for one trial
        mean, sem = compute_group_stats(np.zeros((0, 3)))
        assert_allclose(mean, 0.0)
        assert_allclose(sem, 0.0)

    def test_auc_matrix_matches_per_row_integration(self):
        x = np.linspace(-1, 3, 401)
        m = np.vstack([np.full_like(x, 2.0), np.sin(x), -np.ones_like(x)])
        res = compute_auc_matrix(x, m)
        for row, got in zip(m, res["per_trial"]):
            self.assertEqual(got, compute_auc_from_trace(x, row))
        self.assertEqual(res["group"], compute_auc_from_trace(x, m.mean(axis=0)))
        self.assertEqual(compute_auc_matrix(x, np.zeros((0, 401)))["group"]["total_auc"], 0.0)


class TestPeriEventStats(unittest.TestCase):
    def test_peak_is_the_signed_largest_magnitude_after_the_event(self):
        rel = np.linspace(-5, 5, 1001)
        y = np.zeros_like(rel)
        y[rel < 0] = 100.0                                            # a huge PRE-event blip must not count
        y += -4.0 * np.exp(-0.5 * ((rel - 1.5) / 0.3) ** 2)           # inhibitory response at +1.5 s
        out = compute_peri_event_from_trace(rel, y, post=5.0)
        self.assertAlmostEqual(out["peak_amplitude"], -4.0, delta=0.05)     # negative, not abs()
        self.assertAlmostEqual(out["latency"], 1.5, delta=0.02)

    def test_mean_bins_partition_the_post_window_with_a_truncated_last_bin(self):
        rel = np.linspace(0, 2.5, 251)
        out = compute_peri_event_from_trace(rel, np.ones_like(rel), post=2.5, bin_width=1.0)
        edges = [(b["bin_start"], b["bin_end"]) for b in out["mean_bins"]]
        self.assertEqual(edges, [(0.0, 1.0), (1.0, 2.0), (2.0, 2.5)])
        assert_allclose([b["mean"] for b in out["mean_bins"]], 1.0)

    def test_no_post_event_samples_gives_zero_peak_and_latency(self):
        rel = np.linspace(-5, -1, 50)
        out = compute_peri_event_from_trace(rel, np.ones_like(rel), post=5.0)
        self.assertEqual((out["peak_amplitude"], out["latency"]), (0.0, 0.0))

    def test_matrix_version_has_one_entry_per_trial_plus_a_group(self):
        x = np.linspace(-2, 4, 601)
        m = np.vstack([np.exp(-0.5 * ((x - 1) / 0.3) ** 2), 2 * np.exp(-0.5 * ((x - 2) / 0.3) ** 2)])
        res = compute_peri_event_matrix(x, m, post=4.0, bin_width=2.0)
        self.assertEqual(len(res["per_trial"]), 2)
        self.assertAlmostEqual(res["per_trial"][1]["peak_amplitude"], 2.0, delta=0.02)
        self.assertAlmostEqual(res["per_trial"][1]["latency"], 2.0, delta=0.02)
        self.assertEqual(len(res["group"]["mean_bins"]), 2)
        empty = compute_peri_event_matrix(x, np.zeros((0, 601)), post=4.0)
        self.assertEqual(empty["group"], {"peak_amplitude": 0.0, "latency": 0.0, "mean_bins": []})


class TestCurveFitting(unittest.TestCase):
    def test_slope_of_an_exact_line(self):
        x = np.linspace(0, 10, 101)
        r = compute_slope_segment(x, 2.0 * x + 1.0, 20, 60)
        self.assertAlmostEqual(r["slope"], 2.0, places=9)
        self.assertAlmostEqual(r["intercept"], 1.0, places=9)
        self.assertAlmostEqual(r["x1"], x[20])
        self.assertAlmostEqual(r["y2"], 2.0 * x[60] + 1.0)

    def test_index_order_does_not_matter(self):
        x = np.linspace(0, 10, 101)
        a = compute_slope_segment(x, 3.0 * x, 20, 60)
        b = compute_slope_segment(x, 3.0 * x, 60, 20)
        self.assertEqual(a["slope"], b["slope"])

    def test_context_crop_stays_within_the_data(self):
        x = np.linspace(0, 10, 101)
        r = compute_slope_segment(x, x, 2, 98)
        self.assertEqual(len(r["crop_x"]), 101)

    def test_single_point_segment_gives_zero_slope(self):
        x = np.linspace(0, 10, 101)
        self.assertEqual(compute_slope_segment(x, x, 30, 30)["slope"], 0.0)

    def test_recovers_exponential_decay_parameters(self):
        rng = np.random.default_rng(5)
        x = np.linspace(0, 10, 400)
        y = models.single_exponential_model(x, 3.0, 0.7, 1.0) + rng.normal(0, 0.01, x.size)
        r = fit_model_to_segment(x, y, models.single_exponential_model, lambda xs, ys: [2.0, 0.5, 0.5])
        self.assertTrue(r["success"])
        assert_allclose(r["popt"], [3.0, 0.7, 1.0], rtol=0.03)
        self.assertGreater(r["r2"], 0.999)

    def test_recovers_gaussian_parameters(self):
        x = np.linspace(-5, 5, 400)
        y = models.gaussian_model(x, 2.5, 0.8, 1.2)
        r = fit_model_to_segment(x, y, models.gaussian_model, lambda xs, ys: [2.0, 0.0, 1.0])
        assert_allclose(r["popt"], [2.5, 0.8, 1.2], rtol=1e-4)
        self.assertAlmostEqual(r["r2"], 1.0, places=8)

    def test_recovers_sinusoid_parameters(self):
        x = np.linspace(0, 10, 1000)
        y = models.sinusoidal_model(x, 1.5, 0.4, 0.6, 2.0)
        r = fit_model_to_segment(x, y, models.sinusoidal_model, lambda xs, ys: [1.0, 0.41, 0.5, 1.5])
        assert_allclose(r["popt"], [1.5, 0.4, 0.6, 2.0], rtol=1e-3)

    def test_failures_are_reported_not_raised(self):
        def bad_p0(x, y):
            raise ValueError("no initial guess possible")
        r = fit_model_to_segment(np.arange(10.0), np.arange(10.0), models.linear_model, bad_p0)
        self.assertFalse(r["success"])
        self.assertIn("no initial guess", r["error"])
        self.assertEqual(r["r2"], 0.0)

    def test_a_constant_segment_gives_r2_zero_not_a_division_error(self):
        x = np.arange(20.0)
        r = fit_model_to_segment(x, np.full(20, 4.0), models.linear_model, lambda xs, ys: [0.0, 0.0])
        self.assertTrue(r["success"])
        self.assertEqual(r["r2"], 0.0)


class TestPeakFinding(unittest.TestCase):
    FS = 20.0

    def _signal(self, bumps, noise=0.1, seed=6, seconds=600):
        rng = np.random.default_rng(seed)
        t = np.arange(0, seconds, 1 / self.FS)
        y = rng.normal(0, noise, t.size)
        for centre, amp in bumps:
            y += amp * np.exp(-0.5 * ((t - centre) / 1.0) ** 2)
        return t, y

    def test_injected_peaks_are_found_at_the_right_times(self):
        t, y = self._signal([(100, 5), (250, 5), (400, 5)])
        found = find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0)
        self.assertEqual(len(found), 3)
        for got, want in zip(found, (100, 250, 400)):
            self.assertAlmostEqual(got["time"], want, delta=1.0)
            self.assertEqual(got["kind"], "peak")
            self.assertGreater(got["z_score"], 2.5)
        self.assertEqual([f["time"] for f in found], sorted(f["time"] for f in found))

    def test_troughs_are_only_reported_when_asked_for(self):
        t, y = self._signal([(150, 5), (350, -5)])
        without = find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0)
        with_troughs = find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0, include_troughs=True)
        self.assertEqual([f["kind"] for f in without], ["peak"])
        kinds = {f["kind"]: f for f in with_troughs}
        self.assertEqual(set(kinds), {"peak", "trough"})
        self.assertLess(kinds["trough"]["z_score"], -2.5)
        self.assertAlmostEqual(kinds["trough"]["time"], 350, delta=1.0)

    def test_two_nearby_bumps_collapse_to_the_larger_one(self):
        t, y = self._signal([(200, 5), (205, 4)])
        found = find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]["time"], 200, delta=1.0)

    def test_a_flat_signal_has_no_peaks(self):
        t = np.arange(0, 100, 1 / self.FS)
        self.assertEqual(find_significant_peaks(t, np.full_like(t, 2.0)), [])

    def test_the_default_threshold_is_five_sigma(self):
        for fn in (find_significant_peaks, find_peak_near_events):
            self.assertEqual(inspect.signature(fn).parameters["z_threshold"].default, 5.0)

    def test_significant_peaks_default_finds_large_peaks_and_ignores_weak_ones(self):
        t, y = self._signal([(100, 5), (250, 5), (400, 5)])   # ~50 sd tall
        self.assertEqual(len(find_significant_peaks(t, y, min_distance_sec=10.0)), 3)
        t, y = self._signal([(100, 0.3)])                     # ~3 sd: real, but below five sigma
        self.assertEqual(find_significant_peaks(t, y, min_distance_sec=10.0), [])
        self.assertGreaterEqual(len(find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0)), 1)

    def test_the_default_threshold_rarely_finds_a_response_in_pure_noise(self):
        """The old default of 2.5 called ~87% of windows of pure white noise a 'response' (on a
        real recording, searched the way PyAT does it, ~58% of random pseudo-events), because
        the search takes the largest z over every sample in the window. At five sigma chance
        findings in white noise are rare."""
        rng = np.random.default_rng(12)
        t = np.arange(0, 2000, 1 / self.FS)
        y = rng.normal(0, 0.1, t.size)
        events = list(rng.uniform(20, 1980, 400))

        def found_fraction(**kw):
            return np.mean([r["found"] for r in find_peak_near_events(t, y, events, pre=5, post=10, **kw)])

        self.assertLess(found_fraction(), 0.02)
        self.assertGreater(found_fraction(z_threshold=2.5), 0.7)

    def test_latency_of_a_tall_response_is_not_biased_early(self):
        """A response taller than 5 units used to get a flat top from get_zscore_slice's
        +/-5 raw-unit clip, and argmax then reported the FIRST sample of that plateau, so the
        latency came out early (1.05 s here instead of the true 2.0 s)."""
        t, y = self._signal([(102, 8)])
        res = find_peak_near_events(t, y, [100], pre=5, post=10, z_threshold=5.0)
        self.assertTrue(res[0]["found"])
        self.assertAlmostEqual(res[0]["latency"], 2.0, delta=0.15)

    def test_response_search_around_events_reports_latency_and_misses(self):
        t, y = self._signal([(102, 4), (202, 4)])             # responses 2 s after events at 100 and 200
        # default threshold (5): at the old default of 2.5 a z >= 2.5 sample turns up by chance
        # in most 10 s windows of pure noise, so the event at 300 would have "found" a response too
        res = find_peak_near_events(t, y, [100, 200, 300], pre=5, post=10)
        self.assertEqual([r["found"] for r in res], [True, True, False])
        for r in res[:2]:
            self.assertAlmostEqual(r["latency"], 2.0, delta=0.3)
            self.assertEqual(r["kind"], "peak")
            self.assertGreater(r["z_score"], 5.0)
        self.assertIsNone(res[2]["peak_time"])
        self.assertIsNone(res[2]["latency"])

    def test_peaks_inside_the_start_buffer_are_not_reported(self):
        t, y = self._signal([(5, 5), (100, 5), (250, 5)])
        every = find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0, start_buffer_s=0)
        buffered = find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0, start_buffer_s=10)
        self.assertEqual(len(every), 3)
        self.assertEqual(len(buffered), 2)
        for got, want in zip(buffered, (100, 250)):
            self.assertAlmostEqual(got["time"], want, delta=1.0)

    def test_the_buffer_does_not_change_the_z_score_of_a_peak_that_remains(self):
        """The mean and SD still come from the whole recording: cutting the start out instead shrinks the SD and
        raises every z (and, with motion in the data, the number of false peaks)."""
        t, y = self._signal([(5, 8), (100, 5), (250, 5)])
        every = find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0, start_buffer_s=0)
        for buffer_s in (10, 60, 200):
            with self.subTest(buffer_s=buffer_s):
                self.assertEqual(find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0, start_buffer_s=buffer_s),
                                 [f for f in every if f["time"] >= buffer_s])

    def test_the_buffer_counts_from_the_first_sample_not_from_time_zero(self):
        t, y = self._signal([(5, 5), (100, 5)], seconds=200)
        late = t + 5000.0                                             # the same recording on a clock that starts at 5000 s
        a = find_significant_peaks(t, y, z_threshold=2.5, min_distance_sec=10.0, start_buffer_s=10)
        b = find_significant_peaks(late, y, z_threshold=2.5, min_distance_sec=10.0, start_buffer_s=10)
        self.assertEqual(len(a), 1)
        self.assertEqual([f["time"] + 5000.0 for f in a], [f["time"] for f in b])

    def test_the_buffer_cannot_be_negative_and_can_cover_everything(self):
        t, y = self._signal([(100, 5)], seconds=200)
        with self.assertRaises(ValueError):
            find_significant_peaks(t, y, start_buffer_s=-1)
        self.assertEqual(find_significant_peaks(t, y, z_threshold=2.5, start_buffer_s=500), [])

    def test_the_default_buffer_is_ten_seconds(self):
        self.assertEqual(DEFAULT_START_BUFFER_S, 10.0)
        self.assertEqual(inspect.signature(find_significant_peaks).parameters["start_buffer_s"].default, DEFAULT_START_BUFFER_S)


class TestShuffleNull(unittest.TestCase):
    FS = 20.0
    LOCKED = [60.0, 180.0, 300.0, 420.0]

    def _noise(self, seed=3, seconds=600, noise=0.1):
        rng = np.random.default_rng(seed)
        t = np.arange(0, seconds, 1 / self.FS)
        return t, rng.normal(0, noise, t.size)

    def test_events_with_a_response_beat_chance_and_events_without_one_do_not(self):
        t, y = _trace_with_events(seconds=600, event_times=self.LOCKED, amp=0.8, noise=0.1, seed=1)
        locked = shuffle_null_near_events(t, y, self.LOCKED, pre=5, post=10, n_shuffles=499, seed=0)
        self.assertTrue(locked["available"])
        self.assertEqual((locked["n_events"], locked["n_tested"], locked["found"]), (4, 4, 4))
        self.assertLessEqual(locked["p_count"], 0.05)
        self.assertTrue(all(p <= 0.1 for p in locked["p_event"]))
        quiet = shuffle_null_near_events(t, y, [90.0, 150.0, 240.0, 360.0], pre=5, post=10, n_shuffles=499, seed=0)
        self.assertEqual(quiet["found"], 0)
        self.assertEqual(quiet["p_count"], 1.0)                      # nothing to beat: every shuffle has at least 0
        self.assertTrue(all(p > 0.05 for p in quiet["p_event"]))

    def test_the_chance_rate_is_the_rate_the_search_gives_random_events(self):
        t, y = self._noise(seed=5)
        events = list(np.random.default_rng(6).uniform(20, 580, 400))
        direct = np.mean([r["found"] for r in find_peak_near_events(t, y, events, pre=5, post=10, z_threshold=2.5)])
        null = shuffle_null_near_events(t, y, events[:40], pre=5, post=10, z_threshold=2.5, n_shuffles=200, seed=0)
        self.assertGreater(direct, 0.7)                              # (the point of the null: this is most windows of pure noise)
        self.assertAlmostEqual(null["chance_hit_rate"], direct, delta=0.05)

    def test_events_that_are_chance_findings_are_not_significant(self):
        t, y = self._noise(seed=7)
        events = list(np.sort(np.random.default_rng(8).uniform(20, 580, 30)))
        null = shuffle_null_near_events(t, y, events, pre=5, post=10, z_threshold=2.5, n_shuffles=499, seed=0)
        self.assertGreater(null["found"], 15)                        # "found" in most windows ...
        self.assertGreater(null["p_count"], 0.05)                    # ... and still no better than shifted copies of the same events
        self.assertAlmostEqual(null["expected_by_chance"], null["chance_hit_rate"] * null["n_tested"], delta=1.5)

    def test_it_counts_the_events_find_peak_near_events_finds(self):
        t, y = _trace_with_events(seconds=600, event_times=(100, 200, 300), amp=0.8, noise=0.1)
        events = [100.0, 200.0, 300.0, 400.0]
        direct = find_peak_near_events(t, y, events, 5, 10)
        null = shuffle_null_near_events(t, y, events, 5, 10, n_shuffles=50)
        self.assertEqual(null["found"], sum(r["found"] for r in direct))

    def test_troughs_count_only_when_asked_for(self):
        t, y = _trace_with_events(seconds=600, event_times=self.LOCKED, amp=-0.8, noise=0.1, seed=1)
        off = shuffle_null_near_events(t, y, self.LOCKED, 5, 10, n_shuffles=99)
        on = shuffle_null_near_events(t, y, self.LOCKED, 5, 10, include_troughs=True, n_shuffles=99)
        self.assertEqual(off["found"], 0)
        self.assertEqual(on["found"], 4)

    def test_the_same_seed_gives_the_same_answer_and_another_seed_another_one(self):
        t, y = self._noise(seed=9)
        events = [100.0, 200.0, 300.0]
        a = shuffle_null_near_events(t, y, events, 5, 10, z_threshold=2.0, n_shuffles=100, seed=1)
        self.assertEqual(a, shuffle_null_near_events(t, y, events, 5, 10, z_threshold=2.0, n_shuffles=100, seed=1))
        self.assertNotEqual(a["p_event"], shuffle_null_near_events(t, y, events, 5, 10, z_threshold=2.0, n_shuffles=100, seed=2)["p_event"])
        self.assertEqual(a["seed"], 1)

    def test_every_p_is_a_probability_and_cannot_be_zero(self):
        t, y = _trace_with_events(seconds=600, event_times=self.LOCKED, amp=3.0, noise=0.1, seed=1)
        null = shuffle_null_near_events(t, y, self.LOCKED, 5, 10, n_shuffles=99)
        floor = 1 / 100
        self.assertTrue(all(floor <= p <= 1 for p in null["p_event"] + [null["p_count"]]))

    def test_an_event_whose_window_is_cut_stays_put_and_one_past_the_end_has_no_p(self):
        t, y = self._noise()
        null = shuffle_null_near_events(t, y, [2.0, 300.0, 5000.0], 5, 10, n_shuffles=50)
        self.assertEqual((null["n_events"], null["n_tested"]), (3, 1))
        self.assertIsNotNone(null["p_event"][0])
        self.assertIsNotNone(null["p_event"][1])
        self.assertIsNone(null["p_event"][2])

    def test_when_nothing_can_be_shifted_it_says_why(self):
        t, y = self._noise(seconds=10)
        short = shuffle_null_near_events(t, y, [5.0], pre=5, post=10)
        self.assertFalse(short["available"])
        self.assertIn("shorter than the window", short["reason"])
        self.assertEqual(short["p_event"], [None])
        t, y = self._noise()
        edge = shuffle_null_near_events(t, y, [1.0, 2.0], pre=5, post=10)
        self.assertFalse(edge["available"])
        self.assertIn("complete window", edge["reason"])

    def test_holm_adjust_is_statsmodels_holm_and_leaves_untested_entries_alone(self):
        from statsmodels.stats.multitest import multipletests
        rng = np.random.default_rng(0)
        for n in (1, 2, 5, 12):
            p = list(rng.uniform(0, 1, n) ** 3)
            np.testing.assert_allclose(holm_adjust(p), multipletests(p, method="holm")[1], rtol=1e-12)
        self.assertEqual(holm_adjust([0.01, None, 0.04]), [0.02, None, 0.04])
        self.assertEqual(holm_adjust([None, None]), [None, None])
        self.assertEqual(holm_adjust([]), [])

    def test_the_number_of_shuffles_must_be_at_least_one(self):
        t, y = self._noise()
        with self.assertRaises(ValueError):
            shuffle_null_near_events(t, y, [100.0], 5, 10, n_shuffles=0)


if __name__ == "__main__":
    unittest.main()
