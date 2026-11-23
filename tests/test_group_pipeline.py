"""A group analysis uses the pipeline options: they are part of its spec (and so of its settings file), every recording is processed with them, a trial's baseline
is taken the way zscore.statistic says, the report's methods paragraph says what was chosen, and `group setup --pipeline` puts them in the settings."""

import json
import os
import unittest

import numpy as np

import PyATCLI as pl
from PyATCLI import pipeline
from PyATCLI.analysis.group import GroupSpec, DEFAULT_BASELINE
from PyATCLI.analysis.group import report as group_report
from PyATCLI.analysis.group import run as group_run
from PyATCLI.analysis.group import trials as gt
from PyATCLI.analysis.zscore_peth import _baseline_mean_std, baseline_level_and_spread

from tests.test_group_run import TempDirCase, make_tdt_folder, simulated_loader, simulated_spec
from tests.test_group_setup_cli import SetupCase


class TestTheSpecCarriesThePipeline(unittest.TestCase):
    def test_a_spec_has_none_of_its_own_unless_given(self):
        self.assertEqual(GroupSpec().pipeline, {})
        self.assertFalse([p for p in GroupSpec().problems() if "ipeline" in p])

    def test_it_goes_through_the_settings_file_and_an_old_file_without_it_still_reads(self):
        spec = GroupSpec(group_name="G", pipeline={"denoise.cutoff_hz": 8.0, "dff.f0_method": "linear"})
        again = GroupSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        self.assertEqual(again.pipeline, {"denoise.cutoff_hz": 8.0, "dff.f0_method": "linear"})
        old = spec.to_dict()
        del old["pipeline"]
        self.assertEqual(GroupSpec.from_dict(old).pipeline, {})

    def test_what_the_library_cannot_use_is_a_problem_in_words(self):
        for options, words in (({"nope.x": 1}, "unknown pipeline option nope.x"), ({"denoise.order": 99}, "at most 8"),
                               ({"motion.exclude_start_s": 3.0}, "given together")):
            with self.subTest(options):
                found = [p for p in GroupSpec(pipeline=options).problems() if p.startswith("Pipeline options:")]
                self.assertEqual(len(found), 1)
                self.assertIn(words, found[0])
        self.assertTrue([p for p in GroupSpec(pipeline=["denoise.order"]).problems() if "{option name: value}" in p])

    def test_a_spec_made_from_options_keeps_only_the_ones_that_are_not_defaults(self):
        spec = GroupSpec.from_pipeline({"denoise.cutoff_hz": 5.0, "denoise.order": 3})
        self.assertEqual(spec.pipeline, {"denoise.order": 3})

    def test_the_groups_smoothing_and_baseline_start_from_the_options_where_they_say_something(self):
        spec = GroupSpec.from_pipeline({"smoothing.window_s": 1.0, "zscore.baseline_end_s": -2.0}, group_name="G")
        self.assertEqual((spec.smooth_seconds, spec.baseline), (1.0, (DEFAULT_BASELINE[0], -2.0)))
        spec = GroupSpec.from_pipeline({"zscore.baseline_start_s": -8.0})
        self.assertEqual(spec.baseline, (-8.0, DEFAULT_BASELINE[1]))

    def test_what_the_options_leave_alone_is_the_specs_default_and_what_is_given_is_used_as_it_is(self):
        plain = GroupSpec.from_pipeline({"denoise.order": 3})
        self.assertEqual((plain.smooth_seconds, plain.baseline), (GroupSpec().smooth_seconds, GroupSpec().baseline))
        given = GroupSpec.from_pipeline({"smoothing.window_s": 1.0}, smooth_seconds=0.2, baseline=(-9.0, -3.0))
        self.assertEqual((given.smooth_seconds, given.baseline), (0.2, (-9.0, -3.0)))

    def test_options_the_library_does_not_accept_are_refused(self):
        with self.assertRaisesRegex(ValueError, "at most 8"):
            GroupSpec.from_pipeline({"denoise.order": 99})


class TestEveryRecordingIsProcessedWithThem(TempDirCase):
    def setUp(self):
        super().setUp()
        self.patch_block()
        self.folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")

    def test_the_dff_is_the_one_a_script_gets_with_the_same_options(self):
        options = {"denoise.cutoff_hz": 2.0, "dff.f0_method": "linear"}
        out = group_run.load_group_recording(self.folder, pipeline=options)
        reference = pl.normalized_signal(pl.load_tdt_recording(self.folder, pipeline=options)["cache"])[2]
        np.testing.assert_array_equal(out["y"], reference)
        self.assertFalse(np.array_equal(out["y"], group_run.load_group_recording(self.folder)["y"]))
        self.assertEqual(out["y"].dtype, np.float64)
        self.assertEqual(out["warnings"], [])

    def test_without_options_it_is_what_it_always_was(self):
        reference = pl.normalized_signal(pl.load_tdt_recording(self.folder)["cache"])[2]
        np.testing.assert_array_equal(group_run.load_group_recording(self.folder)["y"], reference)

    def test_a_choice_the_recording_saved_for_itself_is_used_and_said(self):
        pl.saved_state.set_pipeline(self.folder, {"denoise.cutoff_hz": 2.0})
        out = group_run.load_group_recording(self.folder)
        self.assertEqual(out["warnings"], ["processed with the pipeline options saved for this recording: denoise.cutoff_hz = 2.0"])
        chosen = group_run.load_group_recording(self.folder, pipeline={"denoise.cutoff_hz": 2.0})
        self.assertEqual(chosen["warnings"], [])                                    # (what the group asked for is not news)
        np.testing.assert_array_equal(out["y"], chosen["y"])

    def test_the_analysis_hands_the_loader_the_specs_options_and_only_when_it_has_some(self):
        seen = {}
        stand_in, _ = simulated_loader(n_sub=3)

        def load(folder, regression_method, store_labels, progress, **extra):
            seen.setdefault("kwargs", []).append(extra)
            return stand_in(folder, regression_method, store_labels, progress)

        group_run.run_group_analysis(simulated_spec(n_sub=3), load=load)
        group_run.run_group_analysis(simulated_spec(n_sub=3, pipeline={"denoise.cutoff_hz": 8.0}), load=load)
        self.assertEqual(seen["kwargs"], [{}, {}, {}, {"pipeline": {"denoise.cutoff_hz": 8.0}}, {"pipeline": {"denoise.cutoff_hz": 8.0}}, {"pipeline": {"denoise.cutoff_hz": 8.0}}])

    def test_a_loader_that_does_not_know_about_options_still_works_when_there_are_none(self):
        load, calls = simulated_loader(n_sub=3)
        group_run.run_group_analysis(simulated_spec(n_sub=3), load=load)
        self.assertEqual(len(calls), 3)


class TestHowATrialsBaselineIsTaken(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.x = np.arange(0, 100, 0.1)
        self.y = 0.05 * rng.standard_normal(len(self.x))
        self.y[(self.x >= 41.0) & (self.x <= 41.5)] += 3.0                         # a brief artefact inside the baseline of the trial at 50 s
        self.rel = self.x - 50.0
        self.base = self.y[(self.rel >= -10.0) & (self.rel <= -6.0)]

    def trial(self, **spec_fields):
        spec = GroupSpec(group_name="G", markers=["A"], pre=10.0, post=10.0, baseline=(-10.0, -6.0), response=(0.0, 10.0), smooth_seconds=0.0, **spec_fields)
        return gt.extract_group_trials(self.x, self.y, {"A": [50.0]}, spec, subject="S", recording="f")["trials"].iloc[0]

    def test_by_default_it_is_the_mean_and_sd_with_the_artefact_pulled_in_as_ever(self):
        row = self.trial(signal="zscore")
        mu, sd = _baseline_mean_std(self.base)
        self.assertEqual((row["baseline_mean"], row["baseline_sd"]), (float(mu), float(sd)))
        same = self.trial(signal="zscore", pipeline={"zscore.statistic": "mean_sd"})
        self.assertEqual((same["baseline_mean"], same["baseline_sd"]), (row["baseline_mean"], row["baseline_sd"]))

    def test_with_median_and_mad_it_is_those(self):
        row = self.trial(signal="zscore", pipeline={"zscore.statistic": "median_mad"})
        mu = np.median(self.base)
        self.assertEqual((row["baseline_mean"], row["baseline_sd"]), (float(mu), float(1.4826 * np.median(np.abs(self.base - mu)))))
        self.assertTrue(row["valid"])

    def test_the_two_ways_of_taking_a_baseline_are_one_function(self):
        self.assertEqual(baseline_level_and_spread(self.base, "mean_sd"), _baseline_mean_std(self.base))
        mu = np.median(self.base)
        self.assertEqual(baseline_level_and_spread(self.base, "median_mad"), (mu, 1.4826 * np.median(np.abs(self.base - mu))))

    def test_the_dff_signal_has_the_baselines_level_subtracted(self):
        by_mean = self.trial(signal="dff")
        by_median = self.trial(signal="dff", pipeline={"zscore.statistic": "median_mad"})
        self.assertNotEqual(by_mean["baseline_mean"], by_median["baseline_mean"])
        self.assertEqual(by_median["baseline_mean"], float(np.median(self.base)))


class TestTheRecordsSayWhatWasChosen(TempDirCase):
    def results(self, **fields):
        load, _ = simulated_loader(n_sub=4)

        def stand_in(folder, regression_method, store_labels, progress, **extra):
            return load(folder, regression_method, store_labels, progress)

        return group_run.run_group_analysis(simulated_spec(n_sub=4, **fields), load=stand_in)

    def test_the_methods_paragraph_says_how_the_dff_was_made_from_the_options(self):
        default = group_report._methods(self.results())
        self.assertIn("dF/F was computed by OLS regression of the 415 nm channel onto the 465 nm channel", default)
        self.assertIn("double-exponential photobleaching baseline", default)
        self.assertIn("5 Hz", default)
        chosen = group_report._methods(self.results(regression_method="huber", pipeline={"denoise.cutoff_hz": 8.0, "dff.f0_method": "linear"}))
        self.assertIn("Huber (robust) regression", chosen)
        self.assertIn("linear baseline", chosen)
        self.assertIn("8 Hz", chosen)
        self.assertNotIn("double-exponential", chosen)

    def test_the_baseline_sentence_follows_the_statistic(self):
        default = group_report._methods(self.results(signal="zscore"))
        self.assertIn("baseline-corrected with the mean of -10 to -6 s and divided by the baseline SD", default)
        self.assertIn("winsorised at 5 robust SDs", default)
        robust = group_report._methods(self.results(signal="zscore", pipeline={"zscore.statistic": "median_mad"}))
        self.assertIn("baseline-corrected with the median of -10 to -6 s and divided by 1.4826 x the baseline's median absolute deviation", robust)
        self.assertNotIn("winsorised", robust)

    def test_the_settings_file_is_a_record_that_has_the_options_in_it_twice_over(self):
        spec = simulated_spec(n_sub=2, pipeline={"denoise.cutoff_hz": 8.0})
        path = group_run.write_group_settings(os.path.join(self.tmp, "G"), spec, [])
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["spec"]["pipeline"], {"denoise.cutoff_hz": 8.0})
        self.assertEqual(doc["pipeline"]["options"]["denoise.cutoff_hz"], 8.0)
        self.assertEqual(doc["pipeline"]["options"]["denoise.order"], 2)              # (every option, the others at their defaults)
        read, _ = pl.read_group_settings(path)
        self.assertEqual(read.pipeline, {"denoise.cutoff_hz": 8.0})


class TestGroupSetupTakesThem(SetupCase):
    def test_the_options_go_into_the_settings_it_writes(self):
        status, out, err = self.setup("--marker", "L1P¹", "--output-folder", self.root, "--pipeline", "denoise.cutoff_hz=8", "--pipeline", "zscore.statistic=median_mad")
        self.assertEqual(status, 0, err)
        found = [os.path.join(d, pl.GROUP_SETTINGS_FILENAME) for d, _, files in os.walk(self.root) if pl.GROUP_SETTINGS_FILENAME in files]
        spec, doc = pl.read_group_settings(found[0])
        self.assertEqual(spec.pipeline, {"denoise.cutoff_hz": 8.0, "zscore.statistic": "median_mad"})
        self.assertEqual(doc["pipeline"]["options"]["denoise.cutoff_hz"], 8.0)

    def test_an_option_that_is_not_allowed_is_an_error_that_says_what_is_wrong(self):
        status, _, err = self.setup("--marker", "L1P¹", "--output-folder", self.root, "--pipeline", "denoise.order=99")
        self.assertEqual(status, 1)
        self.assertIn("at most 8", err)

    def test_without_the_flag_the_settings_have_none(self):
        status, _, err = self.setup("--marker", "L1P¹", "--output-folder", self.root)
        self.assertEqual(status, 0, err)
        found = [os.path.join(d, pl.GROUP_SETTINGS_FILENAME) for d, _, files in os.walk(self.root) if pl.GROUP_SETTINGS_FILENAME in files]
        self.assertEqual(pl.read_group_settings(found[0])[0].pipeline, {})

    def test_the_default_options_are_not_written_as_choices(self):
        status, _, err = self.setup("--marker", "L1P¹", "--output-folder", self.root, "--pipeline", f"denoise.cutoff_hz={pipeline.defaults()['denoise.cutoff_hz']}")
        self.assertEqual(status, 0, err)
        found = [os.path.join(d, pl.GROUP_SETTINGS_FILENAME) for d, _, files in os.walk(self.root) if pl.GROUP_SETTINGS_FILENAME in files]
        self.assertEqual(pl.read_group_settings(found[0])[0].pipeline, {})


if __name__ == "__main__":
    unittest.main()
