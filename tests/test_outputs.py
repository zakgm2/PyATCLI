"""outputs.py: where results go once there is an Output folder — PyAT's layout (<Output>/<data type>/<recording> (Analysis)/ and
<Output>/Group Analysis/<group>), which the command line writes into too with --output-folder."""

import json
import os
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import PyATCLI as pl
from PyATCLI import outputs
from PyATCLI.ops import group as ops_group
from PyATCLI.analyses import AnalysisResult

from tests.test_cli import CliCase, make_tdt_folder, run_cli


class TestFolderNames(unittest.TestCase):
    def test_characters_a_folder_cannot_have_become_underscores(self):
        self.assertEqual(outputs.safe_folder_name('a<b>c:d"e/f' + chr(92) + 'g|h?i*j'), "a_b_c_d_e_f_g_h_i_j")
        self.assertEqual(outputs.safe_folder_name("  Day 1. "), "Day 1")

    def test_a_reserved_device_name_is_made_safe_and_nothing_usable_is_nothing(self):
        self.assertEqual(outputs.safe_folder_name("con"), "_con")
        self.assertEqual(outputs.safe_folder_name("NUL.txt"), "_NUL.txt")
        self.assertEqual(outputs.safe_folder_name(" . "), "")

    def test_a_very_long_name_is_cut(self):
        self.assertEqual(len(outputs.safe_folder_name("x" * 300)), 100)

    def test_a_recordings_folder_is_its_own_name_and_never_reads_as_the_recording(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        table = os.path.join(folder, "MyFile.xlsx")
        Path(table).write_bytes(b"x")
        self.assertEqual(outputs.recording_folder_name(table), "MyFile (Analysis)")          # a file's extension is dropped
        self.assertEqual(outputs.recording_folder_name(folder + "/"), os.path.basename(folder) + " (Analysis)")


class TestNotOverwriting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def touch(self, *names):
        for n in names:
            Path(self.tmp, n).write_text("x", encoding="utf-8")

    def test_a_path_that_is_taken_gets_a_number_before_its_extension(self):
        path = os.path.join(self.tmp, "a.csv")
        self.assertEqual(outputs.unique_path(path), path)
        self.touch("a.csv")
        self.assertEqual(outputs.unique_path(path), os.path.join(self.tmp, "a (2).csv"))
        self.touch("a (2).csv")
        self.assertEqual(outputs.unique_path(path), os.path.join(self.tmp, "a (3).csv"))

    def test_a_set_of_files_that_share_a_name_moves_to_a_new_name_together(self):
        endings = ["_window.csv", "_peaks.csv", ".provenance.json"]
        self.assertEqual(outputs.unique_stem(self.tmp, "Rec_zscore", endings), "Rec_zscore")
        self.touch("Rec_zscore_peaks.csv")                                                    # even one of them is enough
        self.assertEqual(outputs.unique_stem(self.tmp, "Rec_zscore", endings), "Rec_zscore (2)")
        self.touch("Rec_zscore (2).provenance.json")
        self.assertEqual(outputs.unique_stem(self.tmp, "Rec_zscore", endings), "Rec_zscore (3)")


class TestRecordingDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.base = os.path.join(self.tmp, "Output")

    def source(self, *parts):
        path = os.path.join(self.tmp, "data", *parts)
        os.makedirs(path)
        return path

    def test_a_recording_gets_a_folder_under_its_data_type_that_is_made_on_the_way(self):
        rec = self.source("Rat7-250101")
        folder = outputs.recording_dir(self.base, "TDT", rec)
        self.assertEqual(folder, os.path.join(self.base, "TDT", "Rat7-250101 (Analysis)"))
        self.assertTrue(os.path.isdir(folder))
        marker = json.loads(Path(folder, outputs.SOURCE_MARKER_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual((marker["type"], marker["source_path"]), ("TDT", rec))

    def test_the_same_recording_is_the_same_folder_however_it_is_spelt_and_whenever_it_is_opened(self):
        rec = self.source("Rat7-250101")
        first = outputs.recording_dir(self.base, "TDT", rec)
        self.assertEqual(outputs.recording_dir(self.base, "TDT", rec + os.sep), first)
        self.assertEqual(outputs.recording_dir(self.base, "TDT", os.path.join(rec, "..", "Rat7-250101")), first)

    def test_a_different_recording_with_the_same_name_gets_its_own_folder(self):
        a, b, c = self.source("subjectA", "Day1"), self.source("subjectB", "Day1"), self.source("subjectC", "Day1")
        folders = [outputs.recording_dir(self.base, "TDT", p) for p in (a, b, c)]
        self.assertEqual([os.path.basename(f) for f in folders], ["Day1 (Analysis)", "Day1 (Analysis) (2)", "Day1 (Analysis) (3)"])
        self.assertEqual(outputs.recording_dir(self.base, "TDT", b), folders[1])             # and each finds its own again

    def test_a_folder_with_no_marker_is_claimed_not_skipped(self):
        rec = self.source("Rat7")
        os.makedirs(os.path.join(self.base, "TDT", "Rat7 (Analysis)"))                       # made by hand, or before the marker existed
        folder = outputs.recording_dir(self.base, "TDT", rec)
        self.assertEqual(os.path.basename(folder), "Rat7 (Analysis)")
        self.assertTrue(os.path.isfile(os.path.join(folder, outputs.SOURCE_MARKER_FILENAME)))

    def test_a_data_type_that_is_not_a_folder_name_is_made_one(self):
        folder = outputs.recording_dir(self.base, "Text/Field", self.source("x"))
        self.assertEqual(Path(folder).parent.name, "Text_Field")
        self.assertEqual(Path(outputs.recording_dir(self.base, "", self.source("y"))).parent.name, "Other")


class TestGroupDir(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.base, True)

    def test_a_group_folder_is_not_made_and_a_finished_run_is_never_reused(self):
        first = outputs.group_dir(self.base, "My group")
        self.assertEqual(first, os.path.join(self.base, "Group Analysis", "My group"))
        self.assertFalse(os.path.exists(first))
        os.makedirs(first)
        self.assertEqual(outputs.group_dir(self.base, "My group"), first)                    # no settings in it yet: a run that never finished
        Path(first, pl.GROUP_SETTINGS_FILENAME).write_text("{}", encoding="utf-8")
        second = outputs.group_dir(self.base, "My group")
        self.assertEqual(second, os.path.join(self.base, "Group Analysis", "My group (2)"))
        os.makedirs(second)
        Path(second, pl.GROUP_SETTINGS_FILENAME).write_text("{}", encoding="utf-8")
        self.assertEqual(outputs.group_dir(self.base, "My group"), os.path.join(self.base, "Group Analysis", "My group (3)"))

    def test_a_name_with_nothing_usable_has_no_folder(self):
        self.assertIsNone(outputs.group_dir(self.base, " . "))


class TestTheCommandLineUsesTheLayout(CliCase):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.tmp, "Output")

    def test_run_writes_under_the_data_type_and_recording(self):
        status, out, err = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "-p", "pre_s=5", "-p", "post_s=5", "--output-folder", self.root)
        self.assertEqual((status, err), (0, ""), err)
        folder = os.path.join(self.root, "Oxysoft", "session (Analysis)")
        names = sorted(n for n in os.listdir(folder) if not n.startswith("."))
        self.assertEqual(names, ["session.txt_auc_window.provenance.json", "session.txt_auc_window_auc.csv", "session.txt_auc_window_window.csv"])
        self.assertIn(f"wrote 3 files to {folder}", out)

    def test_a_tdt_recording_goes_under_tdt(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        status, _, err = run_cli("run", "zscore_window", folder, "-p", "center_s=9", "-p", "pre_s=4", "-p", "post_s=4", "--output-folder", self.root, "-q")
        self.assertEqual((status, err), (0, ""), err)
        self.assertTrue(os.path.isdir(os.path.join(self.root, "TDT", "Rat7-250101-101010 (Analysis)")))

    def test_a_repeat_run_is_a_new_set_of_files_and_the_first_is_untouched(self):
        args = ("run", "auc_window", self.oxy, "-p", "center_s=30", "-p", "pre_s=5", "-p", "post_s=5", "--output-folder", self.root, "-q")
        run_cli(*args)
        folder = os.path.join(self.root, "Oxysoft", "session (Analysis)")
        first = {n: Path(folder, n).read_bytes() for n in os.listdir(folder) if not n.startswith(".")}
        run_cli(*args)
        names = sorted(n for n in os.listdir(folder) if not n.startswith("."))
        self.assertEqual(len(names), 6)
        self.assertTrue(all(Path(folder, n).read_bytes() == data for n, data in first.items()))
        self.assertIn("session.txt_auc_window (2).provenance.json", names)
        record = pl.read_record(os.path.join(folder, "session.txt_auc_window (2).provenance.json"))
        self.assertEqual(sorted(o["name"] for o in record["outputs"]), ["session.txt_auc_window (2)_auc.csv", "session.txt_auc_window (2)_window.csv"])

    def test_the_result_is_read_back_from_the_folder_it_landed_in(self):
        run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "-p", "pre_s=5", "-p", "post_s=5", "--output-folder", self.root, "-q")
        folder = os.path.join(self.root, "Oxysoft", "session (Analysis)")
        opened = pl.open_results(folder)
        self.assertEqual(opened.kind, "auc_window")
        auc = next(table for name, table in opened.tables.items() if name.endswith("_auc"))
        self.assertEqual(len(auc), 2)                                                        # the two channels of the Oxysoft file, from the nested files

    def test_out_and_output_folder_are_not_both_allowed(self):
        status, _, err = run_cli("run", "auc_window", self.oxy, "--out", self.out, "--output-folder", self.root)
        self.assertEqual(status, 2)
        self.assertIn("not allowed with argument", err)

    def test_a_failed_run_leaves_no_folder_behind(self):
        status, _, _ = run_cli("run", "auc_window", os.path.join(self.tmp, "nope.txt"), "--output-folder", self.root)
        self.assertEqual(status, 1)
        self.assertFalse(os.path.exists(self.root))

    def test_batch_gives_each_recording_its_own_folder_and_the_combined_tables_theirs(self):
        jobs = os.path.join(self.tmp, "jobs.json")
        Path(jobs).write_text(json.dumps({"jobs": [{"kind": "auc_window", "recording": self.oxy, "parameters": {"center_s": 30, "pre_s": 5, "post_s": 5}}]}), encoding="utf-8")
        status, _, err = run_cli("batch", jobs, "--output-folder", self.root, "-q")
        self.assertEqual((status, err), (0, ""), err)
        self.assertTrue(os.path.isdir(os.path.join(self.root, "Oxysoft", "session (Analysis)")))
        self.assertEqual(sorted(os.listdir(os.path.join(self.root, "Batch", "01_auc_window"))), ["auc_window_auc_all.csv", "auc_window_window_all.csv"])

    def test_a_group_goes_to_group_analysis_and_a_name_used_before_gets_a_number(self):
        spec = pl.GroupSpec(group_name="G", subjects=[{"subject": "S1", "folder": "f1"}], markers=["a"], metrics=["auc"])
        settings = pl.write_group_settings(os.path.join(self.tmp, "settings"), spec, [])
        fake = AnalysisResult("group_analysis", spec.to_dict(), {"descriptives": pd.DataFrame({"a": [1]})}, {"files": ["a"]}, [])
        with mock.patch.object(ops_group, "run_group_from_spec", return_value=(fake, types.SimpleNamespace(notes=[], software={}))) as run:
            for expected in ("G", "G (2)"):
                status, out, err = run_cli("group", "run", settings, "--output-folder", self.root)
                self.assertEqual((status, err), (0, ""), err)
                folder = os.path.join(self.root, "Group Analysis", expected)
                self.assertEqual(run.call_args.kwargs["out_dir"], folder)
                self.assertTrue(os.path.isfile(os.path.join(folder, pl.GROUP_SETTINGS_FILENAME)))     # the finished run's mark: the next name is a new one


if __name__ == "__main__":
    unittest.main()
