"""The command line (cli.py / `python -m PyATCLI`): each command through main(), its exit statuses, what it writes,
and one real child process."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import tdt

import PyATCLI as pl
from PyATCLI import cli
from PyATCLI.ops import group as ops_group
from PyATCLI import provenance as pv
from PyATCLI.analyses import AnalysisResult

ROOT = Path(__file__).resolve().parent.parent
TAB, NL = chr(9), chr(10)


def fake_block(n=20_000, fs=1000.0, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    base = 60 + 30 * np.exp(-t / 8)
    f415 = (0.5 * base + rng.normal(0, 0.1, n)).astype(np.float32)
    f465 = (base * (1 + 0.1 * np.exp(-0.5 * ((t - 9) / 0.5) ** 2)) + rng.normal(0, 0.1, n)).astype(np.float32)
    heads = tdt.StructType()
    heads.stores = {"L1P_": types.SimpleNamespace(type_str="epocs", name="L1P/"), "x465A": types.SimpleNamespace(type_str="streams", name="x465A")}
    data = tdt.StructType()
    data.streams = {"x465A": types.SimpleNamespace(data=f465, fs=fs), "x415A": types.SimpleNamespace(data=f415, fs=fs)}
    data.scalars = tdt.StructType()
    data.epocs = tdt.StructType()

    def read_block(path, headers=None, evtype=None, store=None, verbose=None):
        if "broken" in str(path):
            raise OSError("the block cannot be read")
        if isinstance(headers, int) and headers == 1:
            return heads
        if store is not None:
            block = tdt.StructType()
            block.epocs = tdt.StructType()
            block.epocs["L1P_"] = types.SimpleNamespace(name="L1P", onset=np.array([4.0, 9.0, 14.0]), offset=np.array([4.5, 9.5, 14.5]))
            return block
        return data

    return read_block


def make_tdt_folder(parent, name):
    path = os.path.join(parent, name)
    os.makedirs(path)
    with open(os.path.join(path, name + ".Tbk"), "wb") as f:
        f.write(b"block " + name.encode())
    return path


def write_oxysoft(path, n=600):
    lines = [f"Datafile sample rate{TAB}10 Hz", "Legend:", f"1{TAB}sample number", f"2{TAB}Rx1 - Tx1 O2Hb (f)", f"3{TAB}Rx1 - Tx1 HHb (f)",
             TAB.join("123"), TAB.join("000")]
    rng = np.random.default_rng(1)
    for i in range(1, n + 1):
        v = np.round(rng.normal(0, 1, 2) + np.sin(i / 20.0), 6)
        lines.append(TAB.join([str(i)] + [str(x) for x in v]))
    with open(path, "w") as f:
        f.write(NL.join(lines) + NL)


def run_cli(*argv):
    """(exit status, stdout, stderr) of cli.main(argv)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            status = cli.main([str(a) for a in argv])
        except SystemExit as e:                          # argparse's own usage errors
            status = e.code
    return status, out.getvalue(), err.getvalue()


class CliCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.out = os.path.join(self.tmp, "out")
        patcher = mock.patch.object(tdt, "read_block", fake_block())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.oxy = os.path.join(self.tmp, "session.txt")
        write_oxysoft(self.oxy)

    def files(self, folder=None):
        return sorted(os.listdir(folder or self.out))


class TestSmallCommands(CliCase):
    def test_info_names_the_library_python_and_the_packages(self):
        status, out, _ = run_cli("info")
        self.assertEqual(status, 0)
        self.assertIn(f"PyATCLI {pl.__version__}", out)
        self.assertIn(f"numpy {np.__version__}", out)

    def test_version(self):
        status, out, _ = run_cli("--version")
        self.assertEqual((status, out.strip()), (0, f"PyATCLI {pl.__version__}"))

    def test_kinds_lists_every_analysis_and_one_can_be_described(self):
        status, out, _ = run_cli("kinds")
        self.assertEqual(status, 0)
        for kind in pl.analyses.kinds() + ("group_analysis",):
            self.assertIn(kind, out)
        status, out, _ = run_cli("kinds", "curve_fit")
        self.assertIn("t1_s", out)
        self.assertIn("required", out)
        self.assertNotIn("event_peth", out)

    def test_an_unknown_kind_to_describe_is_an_error(self):
        status, _, err = run_cli("kinds", "nope")
        self.assertEqual(status, 1)
        self.assertIn("unknown analysis 'nope'", err)

    def test_no_command_and_an_unknown_analysis_are_usage_errors(self):
        self.assertEqual(run_cli("run", "nope", self.oxy)[0], 2)
        self.assertEqual(run_cli("run")[0], 2)
        self.assertEqual(run_cli("frobnicate")[0], 2)

    def test_values_are_json_when_they_can_be_and_text_otherwise(self):
        self.assertEqual([cli.parse_value(v) for v in ("10", "2.5", "true", "[1, 2]", "L1P¹", "1,2,3")], [10, 2.5, True, [1, 2], "L1P¹", "1,2,3"])

    def test_parameters_come_from_a_file_then_the_flags(self):
        f = os.path.join(self.tmp, "p.json")
        with open(f, "w") as fh:
            json.dump({"parameters": {"center_s": 10, "pre_s": 3}}, fh)
        self.assertEqual(cli.collect_parameters(["pre_s=7", "signal=x"], f), {"center_s": 10, "pre_s": 7, "signal": "x"})
        with self.assertRaisesRegex(pl.analyses.ParameterError, "NAME=VALUE"):
            cli.collect_parameters(["oops"])


class TestRun(CliCase):
    def test_it_writes_the_tables_and_a_record_beside_them(self):
        status, out, _ = run_cli("run", "auc_window", self.oxy, "--param", "center_s=30", "--param", "pre_s=5", "--param", "post_s=5", "--out", self.out)
        self.assertEqual(status, 0)
        names = self.files()
        self.assertTrue(any(n.endswith("_auc.csv") for n in names))
        self.assertTrue(any(n.endswith(".provenance.json") for n in names))
        self.assertIn("auc: 2 rows x 6 columns", out)
        self.assertIn("wrote 3 files", out)
        rec = pv.read_record(os.path.join(self.out, next(n for n in names if n.endswith(".provenance.json"))))
        self.assertEqual(rec["kind"], "auc_window")
        self.assertEqual(rec["analysis"]["parameters"]["center_s"], 30.0)
        self.assertEqual(rec["recording"]["source_type"], "Oxysoft")
        self.assertEqual(rec["software"]["app"]["name"], "PyATCLI command line")
        self.assertEqual(rec["source"]["kind"], "file")
        self.assertEqual(len(rec["outputs"]), 2)

    def test_json_output_is_machine_readable(self):
        status, out, _ = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "--out", self.out, "--json")
        doc = json.loads(out)                                                  # the whole of stdout is one JSON document
        self.assertEqual((status, doc["ok"], doc["schema"], doc["command"]), (0, True, 1, "run"))
        result = doc["result"]
        self.assertEqual((result["kind"], result["parameters"]["center_s"]), ("auc_window", 30.0))
        self.assertEqual(len(result["files"]), 3)
        self.assertEqual(result["tables"]["auc"]["rows"], 2)
        self.assertIn("Total AUC", result["tables"]["auc"]["columns"])

    def test_quiet_prints_nothing(self):
        status, out, err = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "--out", self.out, "-q")
        self.assertEqual((status, out, err), (0, "", ""))

    def test_a_bad_parameter_fails_before_anything_is_loaded_or_written(self):
        with mock.patch("PyATCLI.recording.load_recording") as load:
            status, _, err = run_cli("run", "auc_window", self.oxy, "-p", "centre=30", "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("unknown parameter(s) centre", err)
        load.assert_not_called()
        self.assertFalse(os.path.exists(self.out))

    def test_a_missing_required_parameter(self):
        status, _, err = run_cli("run", "auc_window", self.oxy, "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("center_s is required", err)

    def test_a_recording_that_does_not_exist(self):
        status, _, err = run_cli("run", "auc_window", os.path.join(self.tmp, "nope.txt"), "-p", "center_s=1", "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("does not exist", err)

    def test_an_analysis_that_cannot_run_on_this_recording_says_why(self):
        status, _, err = run_cli("run", "event_peth", self.oxy, "-p", "event=Nope", "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("no event named 'Nope'", err)

    def test_debug_shows_the_traceback(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(pl.analyses.ParameterError):
            cli.main(["--debug", "run", "auc_window", self.oxy, "--out", self.out])

    def test_a_tdt_folder_uses_its_saved_splices_unless_told_not_to(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101")
        os.makedirs(os.path.join(folder, "JSON saves"))
        with open(os.path.join(folder, "JSON saves", "splice.json"), "w") as f:
            json.dump([{"mode": "cut_out", "start": 5.0, "end": 8.0}], f)
        run_cli("run", "process", folder, "--out", os.path.join(self.tmp, "saved"), "-q")
        run_cli("run", "process", folder, "--out", os.path.join(self.tmp, "plain"), "-q", "--no-saved")
        saved = pd.read_csv(os.path.join(self.tmp, "saved", "Rat7-250101_process_trace.csv"))
        plain = pd.read_csv(os.path.join(self.tmp, "plain", "Rat7-250101_process_trace.csv"))
        self.assertEqual((len(plain), len(saved)), (20_000, 20_000 - 3001))
        rec = pv.read_record(os.path.join(self.tmp, "saved", "Rat7-250101_process.provenance.json"))
        self.assertEqual(rec["recording"]["splices"], [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        self.assertEqual(rec["source"]["kind"], "folder")

    def test_the_regression_method_is_used_and_recorded(self):
        folder = make_tdt_folder(self.tmp, "Rat8-250102")
        run_cli("run", "process", folder, "--out", os.path.join(self.tmp, "o"), "-q", "--regression", "huber")
        rec = pv.read_record(os.path.join(self.tmp, "o", "Rat8-250102_process.provenance.json"))
        self.assertEqual(rec["recording"]["regression_method"], "huber")
        self.assertEqual(rec["pipeline"]["motion_correction"]["method"], "huber")

    def test_a_generic_file_needs_its_table_chosen_when_there_are_several(self):
        import openpyxl
        path = os.path.join(self.tmp, "two.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        for i, row in enumerate([("t", "a"), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]):
            for j, v in enumerate(row):
                ws.cell(row=i + 1, column=j + 1, value=v)
        for i, row in enumerate([("x", "y"), (0, 10), (1, 11), (2, 12), (3, 13), (4, 14)]):
            for j, v in enumerate(row):
                ws.cell(row=i + 1, column=j + 5, value=v)                                    # a second block, two empty columns away
        wb.save(path)
        status, _, err = run_cli("run", "process", path, "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("holds 2 tables (Sheet1 | cols A-B, Sheet1 | cols E-F); say which one to open", err)
        status, _, err = run_cli("run", "process", path, "--table", "Sheet9", "--out", self.out)
        self.assertEqual((status, "has no table named 'Sheet9'" in err), (1, True))
        status, _, _ = run_cli("run", "process", path, "--table", "Sheet1 | cols E-F", "--out", self.out, "-q")
        self.assertEqual(status, 0)
        trace = pd.read_csv(os.path.join(self.out, "two.xlsx_process_trace.csv"))
        self.assertEqual(list(trace.columns), ["time_s", "y"])                                # x is the time axis, y the trace
        rec = pv.read_record(os.path.join(self.out, "two.xlsx_process.provenance.json"))
        self.assertEqual(rec["recording"]["generic"], {"table": "Sheet1 | cols E-F", "x_column": "x", "y_columns": ["y"]})
        status, out, _ = run_cli("reproduce", os.path.join(self.out, "two.xlsx_process.provenance.json"),
                                 "--compare", os.path.join(self.out, "two.xlsx_process_trace.csv"))
        self.assertEqual(status, 0)
        self.assertIn("MATCH", out)

    def test_a_generic_file_with_x_and_y_columns(self):
        path = os.path.join(self.tmp, "data.csv")
        x = np.arange(0, 60, 0.5)
        pd.DataFrame({"time": x, "left": np.sin(x / 5), "right": np.cos(x / 5)}).to_csv(path, index=False)
        status, _, _ = run_cli("run", "auc_window", path, "-p", "center_s=30", "-p", "pre_s=10", "-p", "post_s=10",
                               "--x", "time", "--y", "right", "--out", self.out, "-q")
        self.assertEqual(status, 0)
        rec = pv.read_record(os.path.join(self.out, "data.csv_auc_window.provenance.json"))
        self.assertEqual(rec["recording"]["generic"]["y_columns"], ["right"])
        status, _, err = run_cli("run", "auc_window", path, "-p", "center_s=30", "--x", "nope", "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("column(s) ['nope'] are not in table", err)


class TestReproduceCommand(CliCase):
    def setUp(self):
        super().setUp()
        run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "-p", "pre_s=5", "-p", "post_s=5", "--out", self.out, "-q")
        self.record = next(os.path.join(self.out, n) for n in self.files() if n.endswith(".provenance.json"))
        self.csv = next(os.path.join(self.out, n) for n in self.files() if n.endswith("_auc.csv"))

    def test_a_match_exits_zero_and_says_so(self):
        status, out, err = run_cli("reproduce", self.record, "--compare", self.csv)
        self.assertEqual((status, err), (0, ""))
        self.assertIn("MATCH", out)
        self.assertIn("largest difference 0", out)

    def test_rounded_and_exact_are_exclusive_flags(self):
        self.assertEqual(run_cli("reproduce", self.record, "--rounded", "--exact")[0], 2)
        df = pd.read_csv(self.csv)
        df["Total AUC"] = df["Total AUC"].map(lambda v: f"{v:.2f}")
        df.to_csv(self.csv, index=False)
        self.assertEqual(run_cli("reproduce", self.record, "--compare", self.csv, "--exact")[0], 3)
        self.assertEqual(run_cli("reproduce", self.record, "--compare", self.csv, "--rounded")[0], 0)

    def test_a_mismatch_exits_three(self):
        df = pd.read_csv(self.csv)
        df.loc[0, "Total AUC"] += 1.0
        df.to_csv(self.csv, index=False)
        status, out, _ = run_cli("reproduce", self.record, "--compare", self.csv)
        self.assertEqual(status, 3)
        self.assertIn("MISMATCH", out)
        self.assertIn("Total AUC", out)

    def test_a_changed_recording_is_a_warning_on_stderr_and_an_error_with_strict(self):
        with open(self.oxy, "a") as f:
            f.write(TAB.join(["601", "0.1", "0.2"]) + NL)
        status, _, err = run_cli("reproduce", self.record, "--compare", self.csv)
        self.assertIn("warning: The recording is not the one this record describes", err)
        status, _, err = run_cli("reproduce", self.record, "--strict")
        self.assertEqual(status, 1)
        self.assertIn("not the one this record describes", err)

    def test_it_can_write_the_result_somewhere(self):
        again = os.path.join(self.tmp, "again")
        status, out, _ = run_cli("reproduce", self.record, "--out", again)
        self.assertEqual(status, 0)
        self.assertIn("wrote 3 files", out)
        self.assertEqual(len(os.listdir(again)), 3)

    def test_a_record_that_is_not_one(self):
        bad = os.path.join(self.tmp, "x.json")
        with open(bad, "w") as f:
            json.dump({"hello": 1}, f)
        status, _, err = run_cli("reproduce", bad)
        self.assertEqual(status, 1)
        self.assertIn("not a provenance record", err)


class TestBatch(CliCase):
    def jobs(self, jobs, **defaults):
        path = os.path.join(self.tmp, "jobs.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"defaults": defaults, "jobs": jobs}, f)
        return path

    def test_a_folder_of_blocks_runs_for_each_and_the_tables_are_combined(self):
        cohort = os.path.join(self.tmp, "cohort")
        os.makedirs(os.path.join(cohort, "A"))
        make_tdt_folder(os.path.join(cohort, "A"), "Rat1-250101")
        make_tdt_folder(cohort, "Rat2-250102")
        jobs = self.jobs([{"kind": "event_peth", "recording": cohort, "parameters": {"event": "L1P¹", "pre_s": 2, "post_s": 5}}],
                         regression_method="ols")
        status, out, err = run_cli("batch", jobs, "--out", self.out)
        self.assertEqual((status, err), (0, ""))
        job_dir = os.path.join(self.out, "01_event_peth")
        self.assertEqual(sorted(d for d in os.listdir(job_dir) if os.path.isdir(os.path.join(job_dir, d))), ["Rat1-250101", "Rat2-250102"])
        combined = pd.read_csv(os.path.join(job_dir, "event_peth_zscore_all.csv"))
        self.assertEqual(sorted(combined["recording"].unique()), ["Rat1-250101", "Rat2-250102"])
        self.assertEqual(len(combined), 2 * 300)                                              # 2 recordings x 300 time bins
        for rec_dir in ("Rat1-250101", "Rat2-250102"):
            self.assertTrue(any(n.endswith(".provenance.json") for n in os.listdir(os.path.join(job_dir, rec_dir))))

    def test_one_recording_failing_does_not_stop_the_others(self):
        cohort = os.path.join(self.tmp, "cohort")
        os.makedirs(cohort)
        make_tdt_folder(cohort, "Rat1-250101")
        make_tdt_folder(cohort, "broken-rat")
        make_tdt_folder(cohort, "Rat3-250103")
        jobs = self.jobs([{"kind": "process", "recording": cohort}])
        status, out, err = run_cli("batch", jobs, "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("job 1 (process) on broken-rat", err)
        combined = pd.read_csv(os.path.join(self.out, "01_process", "process_trace_all.csv"))
        self.assertEqual(sorted(combined["recording"].unique()), ["Rat1-250101", "Rat3-250103"])

    def test_several_jobs_get_their_own_numbered_folders(self):
        jobs = self.jobs([{"kind": "auc_window", "recording": self.oxy, "parameters": {"center_s": 30}},
                          {"kind": "fft_window", "recording": self.oxy, "parameters": {"center_s": 30, "pre_s": 10, "post_s": 10}}])
        self.assertEqual(run_cli("batch", jobs, "--out", self.out, "-q")[0], 0)
        self.assertEqual(self.files(), ["01_auc_window", "02_fft_window"])

    def test_defaults_apply_to_every_job_and_a_jobs_own_parameters_win(self):
        jobs = self.jobs([{"kind": "auc_window", "recording": self.oxy, "parameters": {"center_s": 30, "pre_s": 2}}],
                         parameters={"center_s": 99, "post_s": 3})
        run_cli("batch", jobs, "--out", self.out, "-q")
        rec = pv.read_record(next(Path(self.out).rglob("*.provenance.json")))
        p = rec["analysis"]["parameters"]
        self.assertEqual((p["center_s"], p["pre_s"], p["post_s"]), (30.0, 2.0, 3.0))

    def test_a_folder_with_no_recordings_is_an_error(self):
        empty = os.path.join(self.tmp, "empty")
        os.makedirs(empty)
        status, _, err = run_cli("batch", self.jobs([{"kind": "process", "recording": empty}]), "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("no TDT recordings found", err)


class TestGroupCommand(CliCase):
    def test_it_runs_the_group_analysis_from_the_settings_file_or_its_folder(self):
        spec = pl.GroupSpec(group_name="G", subjects=[{"subject": "S1", "folder": "f1"}, {"subject": "S2", "folder": "f2"}],
                            markers=["a"], metrics=["auc"])
        group_dir = os.path.join(self.tmp, "G")
        path = pl.write_group_settings(group_dir, spec, [])
        fake = AnalysisResult("group_analysis", spec.to_dict(), {"descriptives": pd.DataFrame({"a": [1]})}, {"files": ["a", "b"]}, ["a note"])
        for where in (path, group_dir):
            with mock.patch.object(ops_group, "run_group_from_spec", return_value=(fake, types.SimpleNamespace(notes=[], software={}))) as run:
                status, out, _ = run_cli("group", "run", where, "--out", os.path.join(self.tmp, "res"))
            self.assertEqual(status, 0)
            self.assertEqual(run.call_args.args[0].to_dict(), spec.to_dict())
            self.assertIn("1 tables", out)
            self.assertIn("note: a note", out)
            self.assertIn("wrote 2 files", out)
            copied = os.path.join(self.tmp, "res", pl.GROUP_SETTINGS_FILENAME)              # the settings stay with the tables
            self.assertEqual(Path(copied).read_bytes(), Path(path).read_bytes())

    def test_running_again_into_the_folder_the_settings_are_in_leaves_them_alone(self):
        spec = pl.GroupSpec(group_name="G", subjects=[{"subject": "S1", "folder": "f1"}], markers=["a"], metrics=["auc"])
        group_dir = os.path.join(self.tmp, "G")
        path = pl.write_group_settings(group_dir, spec, [])
        before = Path(path).read_bytes()
        fake = AnalysisResult("group_analysis", spec.to_dict(), {"descriptives": pd.DataFrame({"a": [1]})}, {"files": ["a"]}, [])
        with mock.patch.object(ops_group, "run_group_from_spec", return_value=(fake, types.SimpleNamespace(notes=[], software={}))):
            status, _, _ = run_cli("group", "run", path, "--out", group_dir)
        self.assertEqual(status, 0)
        self.assertEqual(Path(path).read_bytes(), before)

    def test_a_missing_settings_file(self):
        status, _, err = run_cli("group", "run", os.path.join(self.tmp, "nope"))
        self.assertEqual(status, 1)


class TestRealProcess(unittest.TestCase):
    def test_python_dash_m_pyatcli_works_as_a_command(self):
        r = subprocess.run([sys.executable, "-m", "PyATCLI", "info"], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"PyATCLI {pl.__version__}", r.stdout)
        r = subprocess.run([sys.executable, "-m", "PyATCLI", "run", "nope", "x"], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
        self.assertEqual(r.returncode, 2)
        self.assertIn("invalid choice", r.stderr)


if __name__ == "__main__":
    unittest.main()
