"""The methods paragraph (methods.py): it states the values that were actually chosen, for every pipeline option, in the order of the pipeline, and then the software.

The words of each step are registered next to the step's code (pipeline.sentence), so these tests are of the whole: the paragraph for the defaults, one change per
option, a test that every option the registry holds is covered here (an option added without a sentence fails it), a step added to the registry showing up in the
paragraph untouched, and every kind of input — a recording, a provenance record, a cache, a set of options — through the library, the engine and the command line."""

import json
import os
import tempfile
import unittest
from unittest import mock

import tdt

import PyATCLI as pl
from PyATCLI import pipeline, saved_state
from PyATCLI import provenance as pv
from PyATCLI.engine.session import Session
from PyATCLI.methods import facts_of, methods_text, software_sentence
from PyATCLI.pipeline import PipelineOptions
from PyATCLI.recording import load_recording

from tests.test_cli import fake_block, make_tdt_folder, run_cli, write_oxysoft

DEFAULT_FIRST_SENTENCE = ("dF/F was computed by OLS regression of the 415 nm channel onto the 465 nm channel, with the fitted 415 subtracted, then divided by a "
                          "double-exponential photobleaching baseline fitted to the 465 nm channel, then low-pass filtered (Butterworth, 2nd order, 5 Hz, zero-phase).")

# One change to each option of the registry, with what the paragraph must then say. (An option that is not in this table fails test_every_option_is_covered.)
CHANGES = {
    "motion.fit_start_s": ({"motion.fit_start_s": 20.0}, ["fitted on the samples from 20 s to the end", "the fitted line was applied to every sample"]),
    "motion.fit_end_s": ({"motion.fit_end_s": 200.0}, ["fitted on the samples from the start to 200 s"]),
    "motion.exclude_start_s": ({"motion.exclude_start_s": 100.0, "motion.exclude_end_s": 120.0}, ["leaving out 100-120 s"]),
    "motion.exclude_end_s": ({"motion.exclude_start_s": 100.0, "motion.exclude_end_s": 125.5}, ["leaving out 100-125.5 s"]),
    "dff.f0_method": ({"dff.f0_method": "single_exponential"}, ["then divided by a single-exponential photobleaching baseline fitted to the 465 nm channel"]),
    "dff.f0_window_s": ({"dff.f0_method": "running_median", "dff.f0_window_s": 30.0}, ["then divided by a running median of the 465 nm channel (30 s window)"]),
    "dff.f0_percentile": ({"dff.f0_method": "running_percentile", "dff.f0_percentile": 8.0}, ["a running 8th percentile of the 465 nm channel (60 s window)"]),
    "denoise.enabled": ({"denoise.enabled": False}, ["and was not low-pass filtered"]),
    "denoise.cutoff_hz": ({"denoise.cutoff_hz": 8.0}, ["(Butterworth, 2nd order, 8 Hz, zero-phase)"]),
    "denoise.order": ({"denoise.order": 3}, ["(Butterworth, 3rd order, 5 Hz, zero-phase)"]),
    "smoothing.window_s": ({"smoothing.window_s": 1.5}, ["smoothed with a 1.5 s moving average"]),
    "zscore.baseline_start_s": ({"zscore.baseline_start_s": -6.0}, ["the samples from 6 s before the event until the event"]),
    "zscore.baseline_end_s": ({"zscore.baseline_end_s": -1.0}, ["the samples from the start of the window until 1 s before the event"]),
    "zscore.statistic": ({"zscore.statistic": "median_mad"}, ["the median and the median absolute deviation (scaled by 1.4826) of that baseline"]),
}


class TestTheParagraphForTheDefaults(unittest.TestCase):
    def test_it_opens_in_the_style_of_a_methods_section_and_says_what_each_step_did(self):
        text = methods_text(software=False)
        self.assertTrue(text.startswith(DEFAULT_FIRST_SENTENCE), text)
        self.assertIn("For event-locked analyses the dF/F was smoothed with a 0.5 s moving average.", text)
        self.assertIn("Each trial was z-scored against its own baseline, the samples before the event, using the mean and standard deviation of that baseline, "
                      "after samples more than 5 robust standard deviations (1.4826 x the median absolute deviation) from its median were pulled in to that limit.", text)
        self.assertNotIn("spliced", text)                                                          # (nothing was spliced)
        self.assertFalse(text.startswith(" ") or text.endswith(" "))

    def test_the_software_ends_it_with_the_versions_that_ran(self):
        text = methods_text()
        self.assertIn(f"Analyses were run with PyATCLI {pl.library_version()} on Python ", text)
        sw = pv.software_versions()
        for name in ("numpy", "scipy", "pandas"):
            self.assertIn(f"{name} {sw['packages'][name]}", text)
        self.assertTrue(text.endswith("."))
        self.assertNotIn("scikit-learn", text)                                                     # (OLS needs none of it)
        self.assertNotIn("Analyses were run", methods_text(software=False))

    def test_the_whole_paragraph_is_made_of_whole_sentences_one_after_another(self):
        text = methods_text(software=False)
        self.assertEqual(text.count(". "), 2)                                                      # three sentences: how the dF/F was made, the smoothing, the z-score
        self.assertEqual(text.count(".."), 0)


class TestEachOptionIsStated(unittest.TestCase):
    def test_every_option_is_covered(self):
        self.assertEqual(set(CHANGES), {o.name for o in pipeline.all_options()},
                         "an option was added (or removed) in the registry: add what the paragraph says about it to CHANGES, and its sentence next to its step's code")

    def test_changing_an_option_changes_the_paragraph_and_the_paragraph_says_the_value_chosen(self):
        plain = methods_text(software=False)
        for name, (values, expected) in CHANGES.items():
            with self.subTest(name):
                text = methods_text(values, software=False)
                self.assertNotEqual(text, plain, f"{name} is not in the paragraph")
                for phrase in expected:
                    self.assertIn(phrase, text)

    def test_the_regression_method_is_named(self):
        for method, words in (("ols", "OLS regression"), ("huber", "Huber (robust) regression"), ("ransac", "RANSAC (robust) regression")):
            self.assertIn(f"dF/F was computed by {words} of the 415 nm channel", methods_text(regression_method=method, software=False))

    def test_every_baseline_is_described(self):
        said = {"double_exponential": "a double-exponential photobleaching baseline fitted to the 465 nm channel",
                "single_exponential": "a single-exponential photobleaching baseline fitted to the 465 nm channel",
                "linear": "a linear baseline fitted to the 465 nm channel",
                "fitted_isosbestic": "the fitted 415 nm channel (dF/F = (465 - fitted 415) / fitted 415)",
                "running_median": "a running median of the 465 nm channel (60 s window)",
                "running_percentile": "a running 10th percentile of the 465 nm channel (60 s window)"}
        self.assertEqual(set(said), set(pipeline.OPTIONS["dff.f0_method"].choices))
        for method, phrase in said.items():
            self.assertIn(phrase, methods_text({"dff.f0_method": method}, software=False), method)

    def test_several_changes_are_all_in_the_paragraph_in_the_order_of_the_pipeline(self):
        text = methods_text({"motion.fit_end_s": 300.0, "dff.f0_method": "linear", "denoise.cutoff_hz": 3.0, "smoothing.window_s": 1.0, "zscore.statistic": "median_mad"},
                            regression_method="huber", software=False)
        at = [text.index(p) for p in ("Huber", "fitted on the samples", "a linear baseline", "3 Hz", "1 s moving average", "median absolute deviation")]
        self.assertEqual(at, sorted(at))

    def test_a_window_with_values_after_the_event_is_said_so(self):
        text = methods_text({"zscore.baseline_start_s": -4.0, "zscore.baseline_end_s": 2.0}, software=False)
        self.assertIn("the samples from 4 s before the event until 2 s after the event", text)
        self.assertIn("from 3 s after the event", methods_text({"zscore.baseline_start_s": 3.0, "zscore.baseline_end_s": 8.0}, software=False))

    def test_no_smoothing_says_so(self):
        self.assertIn("The dF/F was not smoothed before event-locked analyses.", methods_text({"smoothing.window_s": 0.0}, software=False))

    def test_the_ordinals(self):
        self.assertEqual([pipeline.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 101, 111)],
                         ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "101st", "111th"])

    def test_an_option_that_is_not_allowed_is_refused_not_written_about(self):
        with self.assertRaisesRegex(ValueError, "at most 8"):
            methods_text({"denoise.order": 12})
        with self.assertRaisesRegex(ValueError, "unknown pipeline option"):
            methods_text({"denoise.cutoff": 5})


class TestTheStepsAreTheRegistrys(unittest.TestCase):
    def test_a_step_declared_with_a_sentence_appears_in_the_paragraph_without_methods_py_being_touched(self):
        before = dict(pipeline.OPTIONS), dict(pipeline.STEPS)
        self.addCleanup(lambda: (pipeline.OPTIONS.clear(), pipeline.OPTIONS.update(before[0]), pipeline.STEPS.clear(), pipeline.STEPS.update(before[1])))
        pipeline.register_step("baseline_window", "Baseline window", 45, [pipeline.Option("baseline_window", "gain", "float", 1.0, "Gain", "A made-up option.", minimum=0.0)])

        @pipeline.sentence("baseline_window")
        def say(options, state):
            return f"The made-up gain was {options['baseline_window.gain']:g}" if options["baseline_window.gain"] != 1.0 else None

        self.assertNotIn("made-up", methods_text(software=False))                                  # (nothing to say for the default)
        text = methods_text({"baseline_window.gain": 2.5}, software=False)
        self.assertIn("The made-up gain was 2.5.", text)
        self.assertLess(text.index("moving average"), text.index("made-up"))                        # (after the smoothing, before the z-score: its place in the pipeline)
        self.assertLess(text.index("made-up"), text.index("z-scored"))

    def test_a_step_with_no_sentence_says_nothing_and_a_clause_joins_the_dff_sentence(self):
        before = dict(pipeline.OPTIONS), dict(pipeline.STEPS)
        self.addCleanup(lambda: (pipeline.OPTIONS.clear(), pipeline.OPTIONS.update(before[0]), pipeline.STEPS.clear(), pipeline.STEPS.update(before[1])))
        pipeline.register_step("silent", "Silent", 5, [pipeline.Option("silent", "x", "float", 0.0, "X", "Nothing.")])
        pipeline.register_step("gain", "Gain", 25, [])

        @pipeline.sentence("gain", clause=True)
        def say(options, state):
            return "then scaled"

        text = methods_text(software=False)
        self.assertIn("divided by a double-exponential photobleaching baseline fitted to the 465 nm channel, then scaled, then low-pass filtered", text)


class TestWhatIsSaidAboutTheRecording(unittest.TestCase):
    def test_a_recording_with_no_reference_stream_is_computed_from_the_465_alone(self):
        state = {"source_type": "TDT", "regression_method": "ols", "signal_labels": {"normalized": "Normalized (dF/F)", "main_driver": "Main Driver (x465A)"}}
        text = methods_text(state, software=False)
        self.assertTrue(text.startswith("dF/F was computed from the 465 nm channel alone (the recording has no 415 nm reference stream), as (F - F0) / F0, F0 being "), text)
        self.assertNotIn("415 nm channel onto", text)
        self.assertIn("as (F - F0) / F0, F0 being a running median", methods_text({**state, "pipeline": {"dff.f0_method": "running_median"}}, software=False))
        self.assertIn("F0 being a double-exponential photobleaching baseline", methods_text({**state, "pipeline": {"dff.f0_method": "fitted_isosbestic"}}, software=False))

    def test_a_recording_that_is_not_tdt_has_no_dff_pipeline_to_describe_but_its_analyses_are_described(self):
        for source in ("Oxysoft", "Generic"):
            text = methods_text({"source_type": source}, software=False)
            self.assertNotIn("dF/F was computed", text)
            self.assertNotIn("low-pass", text)
            self.assertTrue(text.startswith("For event-locked analyses the dF/F was smoothed"), text)
            self.assertIn("z-scored", text)

    def test_splices_are_said_in_order_and_the_dff_was_computed_again_for_tdt(self):
        splices = [{"mode": "cut_out", "start": 40.0, "end": 55.0}, {"mode": "keep_inside", "start": 10.0, "end": 500.5}]
        text = methods_text({"source_type": "TDT", "splices": splices}, software=False)
        self.assertIn("The recording was spliced: 40-55 s was cut out, then only 10-500.5 s was kept and the dF/F was computed again on the samples that remained.", text)
        self.assertLess(text.index("low-pass filtered"), text.index("spliced"))
        self.assertLess(text.index("spliced"), text.index("moving average"))
        oxy = methods_text({"source_type": "Oxysoft", "splices": splices[:1]}, software=False)
        self.assertIn("The recording was spliced: 40-55 s was cut out.", oxy)

    def test_the_software_of_a_record_is_the_software_that_made_it_not_what_runs_now(self):
        software = {"library": "2026.9.1", "python": "3.12.1", "packages": {"numpy": "1.26.4", "scipy": "1.11.0", "pandas": "2.1.0", "scikit-learn": "1.4.0", "tdt": "0.6.4"},
                    "app": {"name": "PyAT", "version": "3.0.1"}}
        record = pv.build_record("event_peth", {}, recording={"source_type": "TDT", "regression_method": "huber", "splices": []}, pipeline={"options": {"denoise.order": 4}})
        record["software"] = software
        text = methods_text(record)
        self.assertIn("Analyses were run with PyAT 3.0.1 (library: PyATCLI 2026.9.1) on Python 3.12.1, numpy 1.26.4, scipy 1.11.0, pandas 2.1.0, scikit-learn 1.4.0, tdt 0.6.4.", text)
        self.assertIn("(Butterworth, 4th order, 5 Hz, zero-phase)", text)                         # (the options the record names)
        self.assertIn("Huber (robust) regression", text)
        self.assertNotIn(f"PyATCLI {pl.library_version()}", text)                                  # (not the library running now)

    def test_the_software_sentence_names_what_matters_to_the_recording(self):
        sw = {"library": "2026.11.5", "python": "3.14.0", "packages": {"numpy": "2.0", "scipy": "1.15", "pandas": "3.0", "scikit-learn": "1.6", "tdt": "0.7"}}
        self.assertEqual(software_sentence(sw, "ols", "TDT"), "Analyses were run with PyATCLI 2026.11.5 on Python 3.14.0, numpy 2.0, scipy 1.15, pandas 3.0, tdt 0.7")
        self.assertIn("scikit-learn 1.6", software_sentence(sw, "ransac", "TDT"))
        self.assertNotIn("tdt", software_sentence(sw, "ols", "Oxysoft"))
        self.assertIn("PyAT 4.1.1 (library: PyATCLI 2026.11.5)", software_sentence({**sw, "app": {"name": "PyAT", "version": "4.1.1"}}))
        self.assertIn("with PyATCLI 2026.11.5 on", software_sentence({**sw, "app": {"name": "PyATCLI engine"}}))
        self.assertEqual(software_sentence({}), "Analyses were run with PyATCLI on Python")


class RecordingCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        patcher = mock.patch.object(tdt, "read_block", fake_block())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = make_tdt_folder(self.tmp, "Rat7")


class TestEveryKindOfInput(RecordingCase):
    def test_a_recording_says_what_is_saved_for_it(self):
        saved_state.set_pipeline(self.folder, {"denoise.cutoff_hz": 3.0, "dff.f0_method": "linear", "zscore.statistic": "median_mad"})
        saved_state.add_splice(self.folder, "cut_out", 5.0, 8.0)
        text = methods_text(self.folder, software=False)
        for phrase in ("3 Hz", "a linear baseline", "median absolute deviation", "5-8 s was cut out", "computed again"):
            self.assertIn(phrase, text)
        self.assertIn("OLS regression", text)
        self.assertIn("Huber", methods_text(self.folder, regression_method="huber", software=False))

    def test_a_recording_with_nothing_saved_is_the_default_paragraph(self):
        self.assertEqual(methods_text(self.folder), methods_text())

    def test_a_cache_a_state_and_a_set_of_options_say_the_same_as_the_recording(self):
        saved_state.set_pipeline(self.folder, {"denoise.order": 3})
        loaded = load_recording(self.folder)
        by_cache = methods_text(loaded["cache"], regression_method="ols", splices=loaded["splices"])
        by_state = methods_text(pv.recording_state(loaded["cache"], "ols", loaded["splices"]))
        self.assertEqual(by_cache, by_state)
        self.assertEqual(by_state, methods_text(self.folder))
        self.assertEqual(methods_text(PipelineOptions({"denoise.order": 3})), methods_text({"denoise.order": 3}))

    def test_a_provenance_record_by_value_or_by_path_says_what_produced_the_export(self):
        saved_state.set_pipeline(self.folder, {"smoothing.window_s": 2.0})
        out = os.path.join(self.tmp, "out")
        status, _, err = run_cli("run", "zscore_window", self.folder, "-p", "center_s=9", "--out", out)
        self.assertEqual(status, 0, err)
        path = os.path.join(out, next(f for f in os.listdir(out) if f.endswith(".provenance.json")))
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        self.assertIn("smoothed with a 2 s moving average", methods_text(path))
        self.assertEqual(methods_text(path), methods_text(record))
        self.assertIn(f"PyATCLI {record['software']['library']}", methods_text(path))

    def test_facts_hold_the_options_whole_and_the_software(self):
        facts = facts_of({"denoise.order": 4})
        self.assertEqual(facts["pipeline"]["denoise.order"], 4)
        self.assertEqual(len(facts["pipeline"]), len(pipeline.all_options()))
        self.assertIn("python", facts["software"])
        self.assertEqual(facts_of(None, regression_method="huber")["regression_method"], "huber")


class TestTheEngineAndTheCommandLine(RecordingCase):
    def setUp(self):
        super().setUp()
        self.session = Session()
        self.addCleanup(self.session.close)

    def call(self, method, params=None):
        return self.session.handle(method, params, lambda event, **fields: None, None)

    def test_the_engine_writes_the_paragraph_for_an_open_recording_with_its_options_and_splices(self):
        opened = self.call("recording.open", {"path": self.folder, "load": {"regression_method": "huber", "pipeline": {"denoise.cutoff_hz": 2.0}}})
        view = self.call("recording.view", {"id": opened["id"], "splices": [{"mode": "cut_out", "start": 5.0, "end": 8.0}]})
        text = self.call("methods.text", {"recording": {"id": opened["id"], "splices": view["splices"]}, "software": False})
        self.assertIn("Huber (robust) regression", text["text"])
        self.assertIn("(Butterworth, 2nd order, 2 Hz, zero-phase)", text["text"])
        self.assertIn("5-8 s was cut out", text["text"])
        self.assertEqual(text["options"]["denoise.cutoff_hz"], 2.0)
        self.assertNotIn("Analyses were run", text["text"])

    def test_the_engine_names_the_program_it_is_working_for_and_writes_the_defaults_for_no_recording(self):
        text = self.call("methods.text", {"app": {"name": "PyAT", "version": "9.9.9"}})
        self.assertIn("PyAT 9.9.9 (library: PyATCLI ", text["text"])
        self.assertEqual(text["options"], pipeline.defaults())
        by_path = self.call("methods.text", {"recording": self.folder, "load": {"pipeline": {"zscore.statistic": "median_mad"}}})
        self.assertIn("median absolute deviation", by_path["text"])

    def test_the_methods_command_prints_the_paragraph_for_a_recording_and_for_a_record(self):
        status, out, err = run_cli("methods", self.folder, "--pipeline", "denoise.cutoff_hz=8", "--no-software")
        self.assertEqual(status, 0, err)
        self.assertIn("(Butterworth, 2nd order, 8 Hz, zero-phase)", out)
        self.assertNotIn("Analyses were run", out)
        status, out, _ = run_cli("methods", self.folder, "--json")
        self.assertEqual(json.loads(out)["result"]["text"], methods_text(self.folder))
        record_dir = os.path.join(self.tmp, "out")
        run_cli("run", "zscore_window", self.folder, "-p", "center_s=9", "--out", record_dir, "--pipeline", "smoothing.window_s=2")
        record = os.path.join(record_dir, next(f for f in os.listdir(record_dir) if f.endswith(".provenance.json")))
        status, out, _ = run_cli("methods", record)
        self.assertEqual(status, 0)
        self.assertIn("a 2 s moving average", out)

    def test_an_oxysoft_export_is_described_without_a_dff_sentence(self):
        path = os.path.join(self.tmp, "session.txt")
        write_oxysoft(path)
        status, out, _ = run_cli("methods", path, "--no-software")
        self.assertEqual(status, 0)
        self.assertNotIn("dF/F was computed", out)
        self.assertIn("z-scored", out)


if __name__ == "__main__":
    unittest.main()
