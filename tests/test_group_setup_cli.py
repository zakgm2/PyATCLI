"""`group setup` on the command line: the Hypothesis Testing wizard's steps — find the recordings, list the markers they share, take the
choices, check the design, write analysis_settings.json — and the settings it writes run (`group`) and reproduce."""

import json
import os
import types
import unittest
from pathlib import Path
from unittest import mock

import PyATCLI as pl
from PyATCLI import cli
from PyATCLI.ops import group as ops_group

from tests.test_cli import run_cli
from tests.test_group_run import TempDirCase, make_tdt_folder


class SetupCase(TempDirCase):
    def setUp(self):
        super().setUp()
        self.patch_block(duration=20.0)                                   # every recording: L1P pressed at 3 s and 12 s, 20 s long
        self.cohort = os.path.join(self.tmp, "cohort")
        self.rec = [make_tdt_folder(os.path.join(self.cohort, "sub"), f"Rat{i}-250101-10101{i}") for i in range(1, 4)]
        self.root = os.path.join(self.tmp, "Output")

    def setup(self, *argv):
        return run_cli("group", "setup", self.cohort, "--name", "Sucrose", *argv)

    def settings(self, folder):
        spec, doc = pl.read_group_settings(folder)
        return spec, doc


class TestListingTheMarkers(SetupCase):
    def test_without_a_marker_it_lists_what_the_recordings_have_and_writes_nothing(self):
        status, out, err = self.setup("--output-folder", self.root)
        self.assertEqual(status, 2)
        self.assertIn("3 recordings under", out)
        self.assertIn("* L1P¹", out)                                         # in every recording
        self.assertIn("3 of 3", out)
        self.assertIn("say which markers to compare with --marker", err)
        self.assertFalse(os.path.exists(self.root))

    def test_a_marker_none_of_them_has_lists_the_ones_there_are(self):
        status, _, err = self.setup("--marker", "Nope", "--output-folder", self.root)
        self.assertEqual(status, 1)
        self.assertIn("no recording has the marker 'Nope'; the markers are: L1P¹, L1P⁰", err)

    def test_a_marker_that_is_only_in_some_is_listed_without_the_star(self):
        self.save(self.rec[0], "markers.json", [{"time": 7.0, "label": "Note", "color": "green", "fontsize": 8}])
        status, out, _ = self.setup("--output-folder", self.root)
        self.assertIn("  Note", out)
        self.assertNotIn("* Note", out)
        self.assertIn("1 of 3", out)


class TestWritingTheSettings(SetupCase):
    def test_the_wizards_defaults_and_the_choices_made_are_in_the_file_it_writes(self):
        status, out, err = self.setup("--marker", "L1P¹", "--marker", "L1P⁰", "--output-folder", self.root)
        self.assertEqual((status, err), (0, ""), err)
        folder = os.path.join(self.root, "Group Analysis", "Sucrose")
        spec, doc = self.settings(folder)
        self.assertEqual(spec.group_name, "Sucrose")
        self.assertEqual([s["subject"] for s in spec.subjects], ["Rat1-250101-101011", "Rat2-250101-101012", "Rat3-250101-101013"])
        self.assertEqual(spec.markers, ["L1P¹", "L1P⁰"])
        self.assertEqual((spec.pre, spec.post, spec.baseline, spec.response), (10.0, 10.0, (-10.0, -6.0), (0.0, 10.0)))
        self.assertEqual((spec.signal, spec.smooth_seconds, spec.metrics, spec.correction, spec.regression_method),
                         ("dff", 0.5, list(pl.analysis.group.METRICS), "holm", "ols"))
        self.assertEqual(doc["app"]["name"], "PyATCLI command line")
        self.assertEqual([r["folder"] for r in doc["recordings"]], [os.path.abspath(f) for f in self.rec])
        self.assertIn("run it with: python -m PyATCLI group", out)

    def test_every_choice_the_wizard_has_is_an_option(self):
        status, _, err = self.setup("--marker", "Lever¹", "--out", os.path.join(self.tmp, "G"), "--pre", "6", "--post", "8", "--baseline", "-6", "-2",
                                    "--response", "0", "5", "--signal", "zscore", "--smooth", "0", "--measure", "auc", "--measure", "latency",
                                    "--peak-direction", "positive", "--decay-fraction", "0.25", "--correction", "fdr_bh", "--alpha", "0.01",
                                    "--regression", "huber", "--store-label", "L1P=Lever", "-q")
        self.assertEqual((status, err), (0, ""), err)
        spec, _ = self.settings(os.path.join(self.tmp, "G"))
        self.assertEqual((spec.pre, spec.post, spec.baseline, spec.response), (6.0, 8.0, (-6.0, -2.0), (0.0, 5.0)))
        self.assertEqual((spec.signal, spec.smooth_seconds, spec.metrics, spec.peak_direction), ("zscore", 0.0, ["auc", "latency"], "positive"))
        self.assertEqual((spec.decay_fraction, spec.correction, spec.alpha, spec.regression_method), (0.25, "fdr_bh", 0.01, "huber"))
        self.assertEqual(spec.store_labels, {"L1P": "Lever"})

    def test_a_recording_can_be_left_out_and_one_that_cannot_be_read_is_skipped_and_said(self):
        os.makedirs(os.path.join(self.cohort, "sub", "not-a-block"))
        make_tdt_folder(os.path.join(self.cohort, "sub"), "Rat9-250101-101019")
        status, _, err = self.setup("--marker", "L1P¹", "--exclude", "Rat9-250101-101019", "--out", os.path.join(self.tmp, "G"), "-q")
        self.assertEqual((status, err), (0, ""), err)
        spec, _ = self.settings(os.path.join(self.tmp, "G"))
        self.assertEqual(len(spec.subjects), 3)
        status, _, err = self.setup("--marker", "L1P¹", "--exclude", "Nobody", "--out", os.path.join(self.tmp, "H"))
        self.assertEqual(status, 1)
        self.assertIn("no recording named 'Nobody'", err)

    def test_a_design_that_cannot_be_analysed_says_why_and_writes_nothing(self):
        status, _, err = self.setup("--marker", "L1P¹", "--baseline", "-3", "-12", "--out", os.path.join(self.tmp, "G"))
        self.assertEqual(status, 1)
        self.assertIn("The baseline window must run from an earlier to a later time.", err)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "G")))
        status, _, err = run_cli("group", "setup", self.rec[0], "--name", "One", "--marker", "L1P¹", "--out", os.path.join(self.tmp, "G"))
        self.assertEqual(status, 1)
        self.assertIn("Pick at least two recordings", err)

    def test_the_design_warnings_the_wizard_shows_are_said(self):
        status, out, _ = self.setup("--marker", "L1P¹", "--out", os.path.join(self.tmp, "G"))
        self.assertIn("warning: Only 3 subjects", out)

    def test_it_needs_to_be_told_where_to_write(self):
        status, _, err = self.setup("--marker", "L1P¹")
        self.assertEqual(status, 1)
        self.assertIn("--out DIR, or --output-folder DIR", err)

    def test_a_group_name_already_used_gets_a_number_like_in_the_app(self):
        for expected in ("Sucrose", "Sucrose (2)"):
            status, _, err = self.setup("--marker", "L1P¹", "--output-folder", self.root, "-q")
            self.assertEqual((status, err), (0, ""), err)
            self.assertTrue(os.path.isfile(os.path.join(self.root, "Group Analysis", expected, pl.GROUP_SETTINGS_FILENAME)))


class TestTheSettingsRun(SetupCase):
    def test_run_goes_on_to_run_the_analysis_into_the_same_folder(self):
        with mock.patch.object(ops_group, "run_group_from_spec") as run:
            run.return_value = (mock.Mock(notes=["a note"], summary={"files": ["a", "b"]}, tables={}), types.SimpleNamespace(notes=[], software={}))
            status, out, err = self.setup("--marker", "L1P¹", "--marker", "L1P⁰", "--output-folder", self.root, "--run")
        self.assertEqual((status, err), (0, ""), err)
        folder = os.path.join(self.root, "Group Analysis", "Sucrose")
        self.assertEqual(run.call_args.kwargs["out_dir"], folder)
        self.assertEqual(run.call_args.args[0].markers, ["L1P¹", "L1P⁰"])
        self.assertIn(f"wrote 2 files to {folder}", out)
        self.assertIn("note: a note", out)

    def test_the_file_it_writes_is_what_group_runs_and_reproduce_replays(self):
        self.setup("--marker", "L1P¹", "--marker", "L1P⁰", "--out", os.path.join(self.tmp, "G"), "-q")
        spec, doc = self.settings(os.path.join(self.tmp, "G"))
        self.assertEqual(doc["kind"], "group_analysis")                    # a provenance record, as the wizard's file is
        self.assertEqual(doc["analysis"]["parameters"], spec.to_dict())
        for r in doc["recordings"]:
            self.assertTrue(r["digest"])                                    # fingerprinted when it was set up
        with mock.patch.object(ops_group, "run_group_from_spec") as run:
            run.return_value = (mock.Mock(notes=[], summary={"files": []}, tables={}), types.SimpleNamespace(notes=[], software={}))
            status, _, _ = run_cli("group", "run", os.path.join(self.tmp, "G"), "-q")
        self.assertEqual(status, 0)
        self.assertEqual(run.call_args.args[0].to_dict(), spec.to_dict())


if __name__ == "__main__":
    unittest.main()
