"""analysis/group/rebuild.py: a group analysis's GroupResults put back together from its tables — the ones the run returned, or the CSVs written to its
folder read back — so a viewer can show a result that is no longer in memory (the report is written from a GroupResults, and so are PyAT's figures)."""

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from PyATCLI import analyses as A
from PyATCLI.analysis.group import run as group_run
from PyATCLI.analysis.group.rebuild import results_from_tables
from PyATCLI.analysis.group.report import CSV_NAMES, report_text

from tests.test_group_run import simulated_loader, simulated_spec


class RebuildCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="pyatcli-rebuild-")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        load, _ = simulated_loader()
        runner = lambda spec, out_dir=None, progress=None: group_run.run_group_analysis(spec, out_dir=out_dir, progress=progress, load=load)
        with mock.patch.object(A, "run_group_analysis", side_effect=runner):
            self.result, self.raw = A.run_group_from_spec(simulated_spec())
        self.tables, self.parameters = self.result.tables, self.result.parameters


class TestRebuildingTheResults(RebuildCase):
    def test_the_traces_the_grid_and_the_spec_come_back_as_they_were(self):
        rebuilt = results_from_tables(self.tables, self.parameters)
        assert_array_equal(rebuilt.trace_grid, self.raw.trace_grid)
        self.assertEqual(set(rebuilt.traces), set(self.raw.traces))
        for subject, markers in self.raw.traces.items():
            for marker, trace in markers.items():
                self.assertEqual(rebuilt.traces[subject][marker]["n"], trace["n"])
                assert_allclose(rebuilt.traces[subject][marker]["mean"], trace["mean"], rtol=0, atol=1e-12)
        self.assertEqual(rebuilt.spec.to_dict(), self.raw.spec.to_dict())

    def test_without_the_spec_it_is_inferred_from_the_tables(self):
        spec = results_from_tables(self.tables).spec
        self.assertEqual((spec.group_name, spec.markers, [s["subject"] for s in spec.subjects]),
                         ("Group analysis", ["big", "small"], [f"S{i}" for i in range(6)]))

    def test_tables_keyed_by_the_csv_they_were_written_to_are_the_same_tables(self):
        by_file = {os.path.splitext(CSV_NAMES.get(name, name))[0]: table for name, table in self.tables.items()}
        self.assertIn("trials_long", by_file)                                                # the files' names, not the frames'
        rebuilt = results_from_tables(by_file, self.parameters)
        self.assertEqual(len(rebuilt.trials), len(self.raw.trials))
        self.assertEqual(len(rebuilt.diagnostics), len(self.raw.diagnostics))

    def test_tables_that_are_not_a_group_analysis_are_not_one(self):
        self.assertIsNone(results_from_tables({"x": pd.DataFrame({"a": [1]})}))

    def test_what_the_analysis_said_and_ran_on_make_the_report_read_as_the_original(self):
        rebuilt = results_from_tables(self.tables, self.parameters, notes=self.raw.notes, software=self.raw.software)
        self.assertEqual(report_text(rebuilt), report_text(self.raw))


if __name__ == "__main__":
    unittest.main()
