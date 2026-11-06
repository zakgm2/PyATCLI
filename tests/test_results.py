"""Reading a finished analysis back (results.py, and the engine's `results.open`): the numbers a window or a script shows come from the files the
library wrote, found from a folder, a provenance record or a table. Nothing here draws."""

import os
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import PyATCLI as pl
from PyATCLI import ops
from PyATCLI.analysis.group.report import NOTES_FILENAME, REPORT_FILENAME, TABLE_FILES
from PyATCLI.analysis.group.run import GROUP_SETTINGS_FILENAME, run_group_analysis, write_group_settings
from PyATCLI.results import NoResults, _read_tables, open_results

from tests.test_cli import CliCase
from tests.test_group_run import TempDirCase, simulated_loader, simulated_spec


def run_group(folder):
    """A simulated group analysis written to `folder` the way group.run writes it: its settings first, then the tables, report and notes."""
    load, _ = simulated_loader()
    spec = simulated_spec()
    write_group_settings(folder, spec, [])
    return spec, run_group_analysis(spec, out_dir=folder, load=load)


class TestAGroupFolder(TempDirCase):
    def setUp(self):
        super().setUp()
        self.folder = os.path.join(self.tmp, "G")
        self.spec, self.results = run_group(self.folder)

    def test_it_opens_as_the_group_analysis_it_was(self):
        opened = open_results(self.folder)
        self.assertEqual(opened.kind, "group_analysis")
        self.assertEqual(opened.parameters, self.spec.to_dict())
        self.assertEqual(opened.directory, os.path.abspath(self.folder))

    def test_its_tables_are_named_as_the_results_name_them_and_read_back_exactly(self):
        opened = open_results(self.folder)
        made = {name for name, frame in self.results.frames().items() if len(frame)} | {"mean_traces"}
        self.assertEqual(set(opened.tables), made)
        for name in ("subject_means", "descriptives", "pairwise"):
            with self.subTest(table=name):
                pd.testing.assert_frame_equal(opened.tables[name], getattr(self.results, name), check_dtype=False)

    def test_what_the_analysis_said_and_ran_on_comes_back_with_it(self):
        opened = open_results(self.folder)
        self.assertTrue(opened.notes)
        self.assertEqual(opened.notes, self.results.notes)
        self.assertEqual(opened.software, self.results.software)

    def test_the_settings_file_opens_the_same_folder(self):
        self.assertEqual(open_results(os.path.join(self.folder, GROUP_SETTINGS_FILENAME)).tables.keys(), open_results(self.folder).tables.keys())

    def test_a_folder_from_before_the_notes_were_kept_opens_without_them(self):
        os.remove(os.path.join(self.folder, NOTES_FILENAME))
        opened = open_results(self.folder)
        self.assertEqual((opened.notes, opened.software), ([], {}))
        self.assertIn("subject_means", opened.tables)

    def test_a_figure_exported_into_the_folder_does_not_change_what_it_opens_as(self):
        picture = os.path.join(self.folder, "figure_traces_101010.png")
        with open(picture, "wb") as f:
            f.write(b"png")
        ops.call("provenance.write", {"export": picture, "kind": "figure"})                    # a record newer than the group's own
        self.assertEqual(open_results(self.folder).kind, "group_analysis")

    def test_settings_without_tables_say_so(self):
        os.remove(os.path.join(self.folder, TABLE_FILES["subject_means"]))
        for name in TABLE_FILES.values():
            path = os.path.join(self.folder, name)
            if os.path.exists(path):
                os.remove(path)
        with self.assertRaisesRegex(NoResults, "none of its tables"):
            open_results(self.folder)

    def test_the_report_and_notes_are_in_the_files_it_listed(self):
        names = {os.path.basename(p) for p in self.results.files}
        self.assertLessEqual({REPORT_FILENAME, NOTES_FILENAME}, names)


class TestARecordOrATable(CliCase):
    def setUp(self):
        super().setUp()
        self.answer = ops.call("run", {"kind": "auc_window", "recording": self.oxy, "out": self.out,
                                       "parameters": {"center_s": 30, "pre_s": 5, "post_s": 5}})
        self.record = next(os.path.join(self.out, n) for n in self.files() if n.endswith(".provenance.json"))

    def test_a_record_opens_with_the_tables_it_lists_and_what_it_was_asked(self):
        opened = open_results(self.record)
        self.assertEqual(opened.kind, "auc_window")
        self.assertEqual(opened.parameters["center_s"], 30)
        self.assertEqual(set(opened.tables), {os.path.splitext(n)[0] for n in self.files() if n.endswith(".csv")})
        self.assertTrue(all(os.path.isfile(p) for p in opened.files))

    def test_a_folder_opens_as_its_newest_record(self):
        self.assertEqual(open_results(self.out).kind, "auc_window")
        self.assertEqual(open_results(self.out).path, os.path.abspath(self.record))

    def test_a_table_file_is_just_its_table(self):
        csv = next(os.path.join(self.out, n) for n in self.files() if n.endswith("_auc.csv"))
        opened = open_results(csv)
        self.assertIsNone(opened.kind)
        self.assertEqual(list(opened.tables), [os.path.splitext(os.path.basename(csv))[0]])
        self.assertIn("Total AUC", next(iter(opened.tables.values())).columns)

    def test_what_is_not_there_or_is_not_a_result_says_so(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        note = os.path.join(self.tmp, "note.txt")
        with open(note, "w", encoding="utf-8") as f:
            f.write("hello")
        for what, pattern in ((os.path.join(self.tmp, "nope.csv"), "not there any more"), (empty, "holds no provenance record"), (note, "Don't know how to")):
            with self.subTest(what=os.path.basename(what)):
                with self.assertRaisesRegex(NoResults, pattern):
                    open_results(what)


class TestTheOperation(TempDirCase):
    def setUp(self):
        super().setUp()
        self.folder = os.path.join(self.tmp, "G")
        self.spec, self.results = run_group(self.folder)

    def test_it_answers_with_what_the_folder_holds(self):
        answer = ops.call("results.open", {"path": self.folder})
        self.assertEqual(answer["kind"], "group_analysis")
        self.assertEqual(answer["parameters"], self.spec.to_dict())
        self.assertEqual(answer["notes"], self.results.notes)
        self.assertIn("subject_means", answer["tables"])
        self.assertEqual(answer["tables"]["subject_means"]["rows"], len(self.results.subject_means))
        self.assertTrue(all(os.path.isfile(p) for p in answer["files"]))

    def test_the_tables_can_travel_with_their_values(self):
        answer = ops.call("results.open", {"path": self.folder, "tables": "inline"})
        self.assertEqual(answer["tables"]["subject_means"]["data"]["subject"], self.results.subject_means["subject"].tolist())

    def test_nothing_to_open_is_an_error_a_person_can_act_on(self):
        with self.assertRaises(ops.OpError) as caught:
            ops.call("results.open", {"path": os.path.join(self.tmp, "nope")})
        self.assertEqual(caught.exception.code, "invalid_input")

    def test_reading_results_needs_no_plotting_library(self):
        code = ("import sys\n"
                "from PyATCLI import ops\n"
                "ops.call('results.open', {'path': sys.argv[1]})\n"
                "loaded = [m for m in ('matplotlib', 'PyQt6', 'PySide6', 'tkinter', 'pyqtgraph') if m in sys.modules]\n"
                "print(loaded)\n"
                "sys.exit(1 if loaded else 0)\n")
        done = subprocess.run([sys.executable, "-W", "ignore", "-c", code, self.folder], capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, f"it imported a plotting library: {done.stdout} {done.stderr[-400:]}")


class TestNumbersComeBackAsWritten(unittest.TestCase):
    def test_a_table_file_reads_back_exactly_and_without_a_pandas_warning(self):
        """Exact on every pandas (the default converter before 3.1 is not, so the argument is passed there) and silent on 3.1 and later
        (which deprecated the argument, so it is not passed there)."""
        rng = np.random.default_rng(1)
        values = np.concatenate([rng.normal(size=20000), rng.normal(size=20000) * 10.0 ** rng.integers(-12, 12, size=20000)])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "numbers.csv")
            pd.DataFrame({"a": values}).to_csv(path, index=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                tables = _read_tables([path])
        np.testing.assert_array_equal(tables["numbers"]["a"].to_numpy(), values)
        self.assertEqual([str(w.message) for w in caught if "float_precision" in str(w.message)], [])


if __name__ == "__main__":
    unittest.main()
