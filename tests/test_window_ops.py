"""The operations a window needs beyond `run`: a group analysis whose results come back as files to read (the engine draws nothing; the window draws
them), and the text field study's pipelines — through the engine's session, the same code behind the JSON-lines server."""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import PyATCLI as pl
from PyATCLI import analyses
from PyATCLI.analysis.group.run import run_group_analysis
from PyATCLI.engine.session import Session
from PyATCLI.analysis.group.rebuild import results_from_tables
from PyATCLI.progress import Cancelled

import tdt

from tests.test_group_run import EPOCS, fake_block, make_tdt_folder, simulated_loader, simulated_spec
from tests.test_progress import _StubModel, _write_study


class SessionCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.session = Session()
        self.addCleanup(self.session.close)
        self.events = []

    def call(self, method, params=None):
        return self.session.handle(method, params, lambda event, **f: self.events.append((event, f)))


class TestGroupRunForAWindow(SessionCase):
    def setUp(self):
        super().setUp()
        loader, _calls = simulated_loader()
        real = analyses.run_group_analysis
        patcher = mock.patch.object(analyses, "run_group_analysis", lambda spec, out_dir=None, progress=None, **k: real(spec, out_dir=out_dir, progress=progress, load=loader, **k))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.spec = simulated_spec()
        self.out = os.path.join(self.tmp, "G")

    def test_the_engine_draws_nothing_and_has_no_option_to(self):
        result = self.call("group.run", {"settings": self.spec.to_dict(), "out": self.out})
        self.assertFalse(any(os.path.basename(p).startswith("figure_") for p in result["files"]))
        self.assertFalse(any(p.lower().endswith((".png", ".pdf", ".svg", ".tif", ".tiff")) for p in result["files"]))
        with self.assertRaises(analyses.ParameterError):
            self.call("group.run", {"settings": self.spec.to_dict(), "out": self.out, "figures": True})

    def test_the_results_can_be_rebuilt_from_the_files_and_read_as_the_original_does(self):
        result = self.call("group.run", {"settings": self.spec.to_dict(), "out": self.out, "tables": "files"})
        frames = {name: pd.read_csv(t["path"]) for name, t in result["table_data"].items()}
        self.assertIn("mean_traces", frames)
        rebuilt = results_from_tables(frames, result["spec"], notes=result["note_records"], software=result["software"])
        original = run_group_analysis(self.spec, load=simulated_loader()[0])
        self.assertEqual(pl.group_report_text(rebuilt), pl.group_report_text(original))
        self.assertEqual(rebuilt.spec.to_dict(), original.spec.to_dict())
        self.assertEqual([n["text"] for n in rebuilt.notes], [n["text"] for n in original.notes])

    def test_the_note_records_carry_the_level_a_window_colours_by(self):
        result = self.call("group.run", {"settings": self.spec.to_dict(), "out": self.out})
        self.assertTrue(all({"level", "text"} <= set(n) for n in result["note_records"]))
        self.assertEqual(result["notes"], [n["text"] for n in result["note_records"]])
        self.assertEqual(result["spec"], self.spec.to_dict())

    def test_a_cancel_stops_it(self):
        with self.assertRaises(Cancelled):
            self.session.handle("group.run", {"settings": self.spec.to_dict(), "out": self.out}, lambda *a, **k: None, lambda: True)
        self.assertEqual(self.events, [])


class TestWriteSettingsWithTheRun(SessionCase):
    def setUp(self):
        super().setUp()
        loader, _calls = simulated_loader()
        real = analyses.run_group_analysis
        patcher = mock.patch.object(analyses, "run_group_analysis", lambda spec, out_dir=None, progress=None, **k: real(spec, out_dir=out_dir, progress=progress, load=loader, **k))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.spec = simulated_spec()
        self.out = os.path.join(self.tmp, "G")

    def test_the_settings_are_saved_in_the_results_folder_with_the_scans_and_the_app(self):
        scans = [{"folder": "f0", "block_name": "Rat0", "start_time": "2025-01-01T10:00:00", "groups": {"big": [1.0, 2.0]}, "n_splices": 0, "n_saved_markers": 0}]
        answer = self.call("group.run", {"settings": self.spec.to_dict(), "out": self.out, "scans": scans, "write_settings": True})
        path = os.path.join(self.out, pl.GROUP_SETTINGS_FILENAME)
        self.assertEqual(answer["settings"], path)
        spec, doc = pl.read_group_settings(path)
        self.assertEqual(spec.to_dict(), self.spec.to_dict())
        self.assertEqual(doc["app"]["name"], "PyATCLI engine")
        self.assertEqual(doc["recordings"][0]["events_per_marker"], {"big": 2})              # the scans it was set up from

    def test_they_are_not_written_unless_asked(self):
        answer = self.call("group.run", {"settings": self.spec.to_dict(), "out": self.out})
        self.assertNotIn("settings", answer)
        self.assertFalse(os.path.exists(os.path.join(self.out, pl.GROUP_SETTINGS_FILENAME)))

    def test_a_run_that_fails_still_leaves_them(self):
        with mock.patch.object(analyses, "run_group_analysis", side_effect=RuntimeError("could not load S0")):
            with self.assertRaises(RuntimeError):
                self.call("group.run", {"settings": self.spec.to_dict(), "out": self.out, "write_settings": True})
        self.assertTrue(os.path.isfile(os.path.join(self.out, pl.GROUP_SETTINGS_FILENAME)))


class TestInspectingPaths(SessionCase):
    def test_a_tdt_block_is_itself_the_recording(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        answer = self.call("files.inspect", {"path": folder})
        self.assertEqual((answer["kind"], answer["valid"], answer["recordings"]), ("tdt", True, [folder]))

    def test_a_cohort_folder_lists_the_blocks_at_any_depth(self):
        cohort = os.path.join(self.tmp, "cohort")
        a = make_tdt_folder(os.path.join(cohort, "A"), "Rat1-250101-101010")
        b = make_tdt_folder(cohort, "Rat2-250101-101011")
        answer = self.call("files.inspect", {"path": cohort})
        self.assertEqual((answer["kind"], answer["valid"]), ("folder", False))
        self.assertEqual(sorted(answer["recordings"]), sorted([a, b]))
        self.assertIn("No TDT block", answer["message"])

    def test_the_walk_can_be_skipped(self):
        cohort = os.path.join(self.tmp, "cohort")
        make_tdt_folder(cohort, "Rat1-250101-101010")
        self.assertEqual(self.call("files.inspect", {"path": cohort, "search": False})["recordings"], [])

    def test_an_empty_folder_has_no_recordings(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        answer = self.call("files.inspect", {"path": empty})
        self.assertEqual((answer["kind"], answer["recordings"]), ("folder", []))

    def test_files_are_told_apart(self):
        from tests.test_cli import write_oxysoft
        oxy, csv = os.path.join(self.tmp, "s.txt"), os.path.join(self.tmp, "t.csv")
        write_oxysoft(oxy)
        pd.DataFrame({"t": [0, 1], "a": [1, 2]}).to_csv(csv, index=False)
        self.assertEqual(self.call("files.inspect", {"path": oxy})["kind"], "oxysoft")
        self.assertEqual(self.call("files.inspect", {"path": csv})["kind"], "table")

    def test_a_path_that_is_not_there_says_so(self):
        from PyATCLI.ops import OpError
        with self.assertRaises(OpError) as caught:
            self.call("files.inspect", {"path": os.path.join(self.tmp, "nope")})
        self.assertEqual(caught.exception.code, "not_found")


class TestPt2(SessionCase):
    def test_the_image_comes_back_as_a_file_to_map(self):
        from PyATCLI.engine import wire
        img = np.arange(256, dtype=np.float32).reshape(16, 16) + 1.0
        path = os.path.join(self.tmp, "scan.pt2")
        with open(path, "wb") as fh:
            fh.write(b"header-bytes" + b"LAER" + img.astype("<f4").tobytes())
        answer = self.call("pt2.load", {"path": path})
        got = wire.decode(answer["image"]["data"], answer["image"]["directory"])["image"]
        np.testing.assert_array_equal(got, pl.load_pt2(path))
        self.assertEqual(answer["shape"], [16, 16])

    def test_a_file_that_is_not_one_is_a_message(self):
        path = os.path.join(self.tmp, "nope.pt2")
        with open(path, "wb") as fh:
            fh.write(b"not an image")
        with self.assertRaises(ValueError):
            self.call("pt2.load", {"path": path})


class TestProvenanceBesideAnExport(SessionCase):
    def setUp(self):
        super().setUp()
        self.folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        patcher = mock.patch.object(tdt, "read_block", fake_block(EPOCS, duration=20.0))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.wired = self.call("recording.open", {"path": self.folder})
        self.export = os.path.join(self.tmp, "Out.csv")
        with open(self.export, "w") as f:
            f.write("a,b" + chr(10) + "1,2" + chr(10))

    def write(self, **params):
        return self.call("provenance.write", {"export": self.export, **params})

    def test_the_record_sits_beside_the_export_and_describes_it(self):
        answer = self.write(kind="zscore_window", parameters={"center_s": 10.0, "pre_s": 3.0, "post_s": 3.0, "signal": "normalized"},
                            recording={"id": self.wired["id"], "splices": [], "markers": []}, regression_method="ols")
        self.assertEqual(answer["record"], os.path.join(self.tmp, "Out.provenance.json"))
        record = pl.read_record(answer["record"])
        self.assertEqual(record["kind"], "zscore_window")
        self.assertEqual(record["source"]["name"], "Rat7-250101-101010")
        self.assertEqual(record["analysis"]["parameters"]["center_s"], 10.0)
        self.assertEqual(record["recording"]["source_type"], "TDT")
        self.assertEqual(record["outputs"][0]["name"], "Out.csv")
        self.assertEqual(record["software"]["app"]["name"], "PyATCLI engine")

    def test_it_is_the_record_a_script_builds(self):
        loaded = pl.load_recording(self.folder)
        ref = {"id": self.wired["id"], "splices": [], "markers": []}
        answer = self.write(kind="zscore_window", parameters={"center_s": 10.0, "signal": "normalized"}, recording=ref, regression_method="ols")
        engine = pl.read_record(answer["record"])
        script = pl.build_record("zscore_window", {"center_s": 10.0, "signal": "normalized"}, source=self.folder,
                                 recording=pl.recording_state(loaded["cache"], "ols", loaded["splices"], signal="normalized"),
                                 pipeline=pl.pipeline_parameters("ols"), app={"name": "PyATCLI engine"}, outputs=[self.export])
        for key in ("source", "recording", "pipeline"):
            self.assertEqual(engine[key], script[key], key)

    def test_the_markers_the_window_holds_are_what_the_record_counts(self):
        marker = {"time": 5.0, "label": "mine", "color": "red", "fontsize": 8}
        a = pl.read_record(self.write(kind="table", recording={"id": self.wired["id"], "markers": []})["record"])
        b = pl.read_record(self.write(kind="table", recording={"id": self.wired["id"], "markers": [marker]})["record"])
        self.assertNotEqual(a["recording"]["markers"], b["recording"]["markers"])

    def test_the_kind_defaults_to_figure_or_table_by_extension(self):
        png = os.path.join(self.tmp, "Fig.png")
        with open(png, "wb") as f:
            f.write(b"png")
        self.assertEqual(pl.read_record(self.write()["record"])["kind"], "table")
        self.assertEqual(pl.read_record(self.call("provenance.write", {"export": png})["record"])["kind"], "figure")

    def test_extra_outputs_are_listed(self):
        extra = os.path.join(self.tmp, "Out_data.csv")
        with open(extra, "w") as f:
            f.write("x")
        record = pl.read_record(self.write(extra_outputs=[extra])["record"])
        self.assertEqual({o["name"] for o in record["outputs"]}, {"Out.csv", "Out_data.csv"})

    def test_an_export_that_has_no_recording_gets_the_source_and_the_software_only(self):
        study = os.path.join(self.tmp, "study")
        os.makedirs(study)
        with open(os.path.join(study, "P-01.json"), "w") as f:
            f.write("{}")
        record = pl.read_record(self.write(kind="table", source_path=study)["record"])
        self.assertEqual(record["source"]["name"], "study")
        self.assertNotIn("recording", record)
        self.assertNotIn("pipeline", record)

    def test_the_fingerprint_is_remembered_until_the_folder_changes(self):
        from PyATCLI.ops import records
        with mock.patch.object(records.provenance, "file_fingerprint", wraps=records.provenance.file_fingerprint) as spy:
            records._fingerprints.clear()
            self.write(source_path=self.folder)
            self.write(source_path=self.folder)
            self.assertEqual(spy.call_count, 1)
            with open(os.path.join(self.folder, "new.bin"), "wb") as f:
                f.write(b"1")
            os.utime(self.folder, None)
            self.write(source_path=self.folder)
            self.assertEqual(spy.call_count, 2)

    def test_remembering_the_latest_result_is_gone_and_a_record_is_always_written(self):
        from PyATCLI.ops import ParameterError
        with self.assertRaises(ParameterError):
            self.call("provenance.write", {"export": self.export, "remember": True})
        with self.assertRaises(ParameterError):
            self.call("provenance.write", {"export": self.export, "record": False})
        self.assertIn("record", self.call("provenance.write", {"export": self.export, "kind": "table", "parameters": {}}))

    def test_a_missing_export_is_not_found(self):
        from PyATCLI.ops import OpError
        with self.assertRaises(OpError) as caught:
            self.call("provenance.write", {"export": os.path.join(self.tmp, "gone.csv")})
        self.assertEqual(caught.exception.code, "not_found")


class TestGroupScanForAWindow(SessionCase):
    def test_the_scan_the_wizard_shows(self):
        folders = [make_tdt_folder(self.tmp, f"Rat{i}-250101-10101{i}") for i in range(2)]
        with mock.patch.object(tdt, "read_block", fake_block(EPOCS, duration=20.0)):
            answer = self.call("group.scan", {"folders": folders, "store_labels": {"L1P¹": "Left lever"}})
        self.assertEqual(len(answer["scans"]), 2)
        self.assertEqual({s["subject"] for s in answer["scans"]}, {"Rat0-250101-101010", "Rat1-250101-101011"})
        self.assertTrue(all(s["error"] is None for s in answer["scans"]))
        self.assertEqual(answer["scans"][0]["groups"], {"L1P¹": [3.0, 12.0], "L1P⁰": [4.0, 13.0]})


class TestTextFieldStudy(SessionCase):
    def setUp(self):
        super().setUp()
        self.folder = os.path.join(self.tmp, "study")
        os.makedirs(self.folder)
        _write_study(self.folder)
        patcher = mock.patch.dict(sys.modules, {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=_StubModel)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_fields_of_a_study(self):
        self.assertEqual(self.call("fieldstudy.fields", {"folder": self.folder})["fields"], ["participant_id", "q1", "q2", "q3"])

    def test_no_files_is_an_error_that_says_so(self):
        os.makedirs(os.path.join(self.tmp, "empty"))
        with self.assertRaises(ValueError):
            self.call("fieldstudy.fields", {"folder": os.path.join(self.tmp, "empty")})

    def test_the_study_comes_back_as_a_table_to_read(self):
        answer = self.call("fieldstudy.run", {"folder": self.folder, "text_fields": ["q1", "q2", "q3"], "delta_pair": ["q1", "q3"],
                                                "paired_fields": [["q1", "q2", "pair12"]], "tables": "files"})
        self.assertEqual(answer["rows"], 14)
        table = pd.read_csv(answer["tables"]["study"]["path"])
        self.assertEqual(len(table), 14)
        self.assertIn("sim_pair12", table.columns)
        self.assertIn("delta_magnitude", table.columns)
        self.assertTrue(any(e == "progress" for e, _ in self.events))

    def test_validation_has_a_row_per_pair(self):
        answer = self.call("fieldstudy.validate", {"folder": self.folder, "text_fields": ["q1", "q2"],
                                                     "paired_fields": [["q1", "q2", "pair12"]], "tables": "files"})
        table = pd.read_csv(answer["tables"]["validation"]["path"])
        self.assertEqual(len(table), 1)
        self.assertIn("p_value_fdr", table.columns)

    def test_validation_needs_a_pair(self):
        with self.assertRaises(analyses.ParameterError):
            self.call("fieldstudy.validate", {"folder": self.folder, "text_fields": ["q1"], "paired_fields": []})
        with self.assertRaises(analyses.ParameterError):
            self.call("fieldstudy.run", {"folder": self.folder, "text_fields": []})


if __name__ == "__main__":
    unittest.main()
