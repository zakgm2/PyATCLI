"""Reproducing an analysis from its provenance record (reproduce.py): the same numbers come back from the record
alone — not from whatever the recording's saved splices and markers say now — and a changed recording, a changed
export or a different environment is reported rather than silently accepted."""

import json
import os
import shutil
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import tdt

import PyATCLI as pl
from PyATCLI import provenance as pv
from PyATCLI import analyses as A
from PyATCLI import reproduce as R
from PyATCLI.analyses import AnalysisResult, run_analysis

TAB, NL = chr(9), chr(10)


def fake_block(n=20_000, fs=1000.0, seed=3):
    """read_block standing in for the tdt SDK: 465 + 415 streams and one epoc store 'L1P' (onsets 4, 9, 14 s)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    base = 60 + 30 * np.exp(-t / 8)
    f415 = (0.5 * base * (1 + 0.02 * np.sin(t)) + rng.normal(0, 0.1, n)).astype(np.float32)
    f465 = (base * (1 + 0.02 * np.sin(t)) * (1 + 0.1 * np.exp(-0.5 * ((t - 9) / 0.5) ** 2)) + rng.normal(0, 0.1, n)).astype(np.float32)
    heads = tdt.StructType()
    heads.stores = {"L1P_": types.SimpleNamespace(type_str="epocs", name="L1P/"), "x465A": types.SimpleNamespace(type_str="streams", name="x465A")}
    data = tdt.StructType()
    data.streams = {"x465A": types.SimpleNamespace(data=f465, fs=fs), "x415A": types.SimpleNamespace(data=f415, fs=fs)}
    data.scalars = tdt.StructType()
    data.epocs = tdt.StructType()

    def read_block(path, headers=None, evtype=None, store=None, verbose=None):
        if isinstance(headers, int) and headers == 1:
            return heads
        if store is not None:
            block = tdt.StructType()
            block.epocs = tdt.StructType()
            block.epocs["L1P_"] = types.SimpleNamespace(name="L1P", onset=np.array([4.0, 9.0, 14.0]), offset=np.array([4.5, 9.5, 14.5]))
            return block
        return data

    return read_block


def make_tdt_folder(parent, name="Rat7-250101-101010"):
    path = os.path.join(parent, name)
    os.makedirs(path)
    with open(os.path.join(path, name + ".Tbk"), "wb") as f:
        f.write(b"block header")
    return path


def write_oxysoft(path, n=600):
    lines = [f"Datafile sample rate{TAB}10 Hz", "Legend:", f"1{TAB}sample number", f"2{TAB}Rx1 - Tx1 O2Hb (f)", f"3{TAB}Rx1 - Tx1 HHb (f)",
             f"4{TAB}Rx1 - Tx2 O2Hb (f)", f"5{TAB}Rx1 - Tx2 HHb (f)", TAB.join("12345"), TAB.join("00000")]
    rng = np.random.default_rng(1)
    for i in range(1, n + 1):
        v = np.round(rng.normal(0, 1, 4) + np.sin(i / 20.0), 6)
        lines.append(TAB.join([str(i)] + [str(x) for x in v]))
    with open(path, "w") as f:
        f.write(NL.join(lines) + NL)


def export(folder, out_dir, kind, params, regression="ols", use_saved=True, **load_kw):
    """What `run` / the app's export does: load, run, write the tables and the record. Returns (record path, result, csv paths)."""
    loaded = pl.load_recording(folder, regression_method=regression, use_saved=use_saved, **load_kw)
    cache = loaded["cache"]
    result = run_analysis(kind, cache, params)
    paths = R.write_result(result, out_dir, "export")
    record = pv.build_record(kind, result.parameters, source=pv.file_fingerprint(folder),
                             recording=pv.recording_state(cache, regression, loaded["splices"], signal=result.parameters.get("signal")),
                             pipeline=pv.pipeline_parameters(regression), outputs=paths)
    return pv.write_record(os.path.join(out_dir, "export.provenance.json"), record), result, paths


class DiskCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        patcher = mock.patch.object(tdt, "read_block", fake_block())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.folder = make_tdt_folder(self.tmp)
        self.out = os.path.join(self.tmp, "out")

    def save(self, name, obj):
        d = os.path.join(self.folder, "JSON saves")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f)

    def csv(self, paths, table):
        return next(p for p in paths if p.endswith(f"export_{table}.csv"))


PETH = {"event": "L1P¹", "pre_s": 2.0, "post_s": 5.0, "bin_s": 1.0}


class TestRoundTrip(DiskCase):
    def test_the_numbers_come_back_from_the_record_alone(self):
        rec_path, result, paths = export(self.folder, self.out, "event_peth", PETH)
        rep = R.reproduce(rec_path, compare=[self.csv(paths, t) for t in ("zscore", "trials", "auc", "peak", "bins", "mean_sem")])
        self.assertTrue(rep.ok)
        self.assertEqual(rep.warnings, [])
        self.assertEqual(len(rep.comparisons), 6)
        self.assertTrue(all(c["max_abs_diff"] == 0 for c in rep.comparisons))              # bit for bit, not just close
        for name, table in result.tables.items():
            pd.testing.assert_frame_equal(table, rep.result.tables[name])

    def test_a_record_may_be_given_as_a_dict_or_a_path(self):
        rec_path, result, _ = export(self.folder, self.out, "zscore_window", {"center_s": 9.0, "pre_s": 3, "post_s": 3})
        by_path = R.reproduce(rec_path)
        by_dict = R.reproduce(pv.read_record(rec_path))
        pd.testing.assert_frame_equal(by_path.result.tables["window"], by_dict.result.tables["window"])

    def test_with_an_output_folder_the_tables_and_a_new_record_are_written_and_that_reproduces_too(self):
        rec_path, _, _ = export(self.folder, self.out, "event_peth", PETH)
        again = os.path.join(self.tmp, "again")
        rep = R.reproduce(rec_path, out_dir=again)
        names = sorted(os.path.basename(p) for p in rep.written)
        self.assertEqual(sum(n.endswith(".csv") for n in names), 6)
        self.assertEqual(sum(n.endswith(".provenance.json") for n in names), 1)
        new = pv.read_record(next(p for p in rep.written if p.endswith(".provenance.json")))
        self.assertEqual(new["kind"], "event_peth")
        self.assertTrue(any("reproduced from a record made" in n for n in new["notes"]))
        self.assertEqual({o["name"] for o in new["outputs"]}, {n for n in names if n.endswith(".csv")})
        chain = R.reproduce(next(p for p in rep.written if p.endswith(".provenance.json")))           # a reproduction is itself reproducible
        pd.testing.assert_frame_equal(chain.result.tables["zscore"], rep.result.tables["zscore"])

    def test_the_regression_method_in_the_record_is_the_one_used(self):
        ols_path, ols, _ = export(self.folder, os.path.join(self.tmp, "ols"), "auc_window", {"center_s": 9.0}, regression="ols")
        hub_path, hub, _ = export(self.folder, os.path.join(self.tmp, "hub"), "auc_window", {"center_s": 9.0}, regression="huber")
        self.assertFalse(ols.tables["auc"].equals(hub.tables["auc"]))
        pd.testing.assert_frame_equal(R.reproduce(hub_path).result.tables["auc"], hub.tables["auc"])
        pd.testing.assert_frame_equal(R.reproduce(ols_path).result.tables["auc"], ols.tables["auc"])


class TestTheRecordNotTheSavedState(DiskCase):
    def test_the_splices_of_the_record_are_applied_not_the_ones_on_disk_now(self):
        self.save("splice.json", [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        rec_path, result, _ = export(self.folder, self.out, "event_peth", {**PETH, "event": "L1P¹"})
        self.assertEqual(pv.read_record(rec_path)["recording"]["splices"], [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        os.remove(os.path.join(self.folder, "JSON saves", "splice.json"))                  # the person undid it afterwards
        again = R.reproduce(rec_path)
        pd.testing.assert_frame_equal(result.tables["zscore"], again.result.tables["zscore"])
        cache = R.load_recording_for_record(pv.read_record(rec_path), self.folder)
        self.assertEqual(len(cache["x"]), 20_000 - 3001)                                    # samples 5.000 s to 8.000 s inclusive are gone
        self.assertEqual(cache["store"], "Rat7-250101-101010 [cut 5.0s-8.0s]")

    def test_markers_edited_after_the_export_change_nothing(self):
        self.save("markers.json", [{"time": 11.0, "label": "mine", "color": "green", "fontsize": 8}])
        rec_path, result, _ = export(self.folder, self.out, "event_peth", {"event": "mine", "pre_s": 2.0, "post_s": 5.0})
        self.save("markers.json", [{"time": 3.0, "label": "mine", "color": "green", "fontsize": 8}])
        pd.testing.assert_frame_equal(result.tables["trials"], R.reproduce(rec_path).result.tables["trials"])

    def test_the_signal_names_in_the_record_are_the_ones_a_replay_uses(self):
        self.save("signal_labels.json", {"main_driver": "GCaMP (465 nm)"})
        params = {"center_s": 9.0, "pre_s": 3, "post_s": 3, "signal": "main_driver"}
        rec_path, result, paths = export(self.folder, self.out, "zscore_window", params)
        self.assertEqual(set(result.tables["window"]["Channel"]), {"GCaMP (465 nm)"})
        self.assertEqual(pv.read_record(rec_path)["recording"]["signal_labels"]["main_driver"], "GCaMP (465 nm)")
        os.remove(os.path.join(self.folder, "JSON saves", "signal_labels.json"))             # renamed back afterwards
        rep = R.reproduce(rec_path, compare=[self.csv(paths, "window")])
        self.assertTrue(rep.ok)
        self.assertEqual(set(rep.result.tables["window"]["Channel"]), {"GCaMP (465 nm)"})

    def test_the_markers_summary_is_in_the_record_but_not_the_markers_themselves_beyond_what_was_analysed(self):
        self.save("markers.json", [{"time": 11.0, "label": "mine", "color": "green", "fontsize": 8}])
        rec_path, _, _ = export(self.folder, self.out, "event_peth", {"event": "mine", "pre_s": 2.0, "post_s": 5.0})
        rec = pv.read_record(rec_path)
        self.assertEqual(rec["recording"]["markers"]["n_user"], 1)
        self.assertEqual(rec["analysis"]["parameters"]["event_times"], [11.0])


class TestWhenTheRecordingIsNotTheSame(DiskCase):
    def setUp(self):
        super().setUp()
        self.rec_path, _, self.paths = export(self.folder, self.out, "zscore_window", {"center_s": 9.0, "pre_s": 3, "post_s": 3})

    def test_a_changed_file_is_a_warning_naming_it(self):
        with open(os.path.join(self.folder, "Rat7-250101-101010.Tbk"), "wb") as f:
            f.write(b"a different block header")
        rep = R.reproduce(self.rec_path)
        self.assertEqual(len(rep.warnings), 1)
        self.assertIn("not the one this record describes", rep.warnings[0])
        self.assertIn("Tbk changed", rep.warnings[0])

    def test_a_missing_or_extra_file_is_named(self):
        open(os.path.join(self.folder, "extra.sev"), "wb").close()
        self.assertIn("extra.sev is new", R.reproduce(self.rec_path).warnings[0])

    def test_strict_refuses_a_different_recording(self):
        with open(os.path.join(self.folder, "Rat7-250101-101010.Tbk"), "wb") as f:
            f.write(b"x")
        with self.assertRaisesRegex(R.ReproduceError, "not the one this record describes"):
            R.reproduce(self.rec_path, strict=True)

    def test_a_moved_recording_is_found_with_recording_and_is_still_the_same_one(self):
        new = os.path.join(self.tmp, "moved")
        os.makedirs(new)
        moved = os.path.join(new, "Rat7-250101-101010")
        shutil.move(self.folder, moved)
        with self.assertRaisesRegex(R.ReproduceError, "is not at"):
            R.reproduce(self.rec_path)
        rep = R.reproduce(self.rec_path, recording=moved, strict=True)
        self.assertEqual(rep.warnings, [])

    def test_a_record_without_a_path_needs_one(self):
        rec = pv.read_record(self.rec_path)
        rec["source"] = pv.file_fingerprint(self.folder, portable=True)
        with self.assertRaisesRegex(R.ReproduceError, "does not say where"):
            R.reproduce(rec)
        self.assertTrue(R.reproduce(rec, recording=self.folder).ok)

    def test_a_different_environment_is_a_warning_or_with_strict_an_error(self):
        rec = pv.read_record(self.rec_path)
        rec["software"]["library"] = "1999.1.1"
        rep = R.reproduce(rec)
        self.assertTrue(any("1999.1.1" in w for w in rep.warnings))
        with self.assertRaisesRegex(R.ReproduceError, "1999.1.1"):
            R.reproduce(rec, strict=True)

    def test_something_that_is_not_an_analysis_cannot_be_run_again(self):
        rec = pv.read_record(self.rec_path)
        rec["kind"] = "figure"
        rec["analysis"] = {"type": "figure", "parameters": {}}
        with self.assertRaisesRegex(R.ReproduceError, "not an analysis.*zscore_window"):
            R.reproduce(rec)


class TestComparingExports(DiskCase):
    def setUp(self):
        super().setUp()
        self.rec_path, self.result, self.paths = export(self.folder, self.out, "event_peth", PETH)

    def test_a_tampered_export_is_a_mismatch_naming_the_cell(self):
        path = self.csv(self.paths, "zscore")
        df = pd.read_csv(path)
        df.loc[10, "z_score"] += 0.5
        df.to_csv(path, index=False)
        rep = R.reproduce(self.rec_path, compare=[path])
        self.assertFalse(rep.ok)
        self.assertIn("row 11, 'z_score'", rep.comparisons[0]["message"])
        self.assertEqual(rep.comparisons[0]["table"], "zscore")

    def test_an_export_rounded_the_way_the_app_rounds_still_matches(self):
        path = self.csv(self.paths, "trials")
        df = pd.read_csv(path)
        df["time_from_event_s"] = df["time_from_event_s"].map(lambda v: f"{v:.4f}")        # what the app writes
        df["z_score"] = df["z_score"].map(lambda v: f"{v:.6g}")
        df.to_csv(path, index=False)
        self.assertTrue(R.reproduce(self.rec_path, compare=[path]).ok)

    def test_the_table_is_found_from_the_exports_own_header_or_can_be_named(self):
        path = self.csv(self.paths, "auc")
        self.assertEqual(R.reproduce(self.rec_path, compare=[path]).comparisons[0]["table"], "auc")
        named = R.reproduce(self.rec_path, compare=[(self.csv(self.paths, "zscore"), "zscore")])
        self.assertTrue(named.ok)

    def test_a_csv_that_matches_no_table_says_so(self):
        other = os.path.join(self.tmp, "other.csv")
        pd.DataFrame({"a": [1], "b": [2]}).to_csv(other, index=False)
        rep = R.reproduce(self.rec_path, compare=[other])
        self.assertFalse(rep.ok)
        self.assertIn("none of the result tables", rep.comparisons[0]["message"])

    def test_rounding_is_honoured_for_pyats_exports_and_not_for_the_command_lines_unless_told(self):
        path = self.csv(self.paths, "zscore")
        df = pd.read_csv(path)
        df["z_score"] = df["z_score"].map(lambda v: f"{v:.2f}")                                  # a coarse, formatted export
        df.to_csv(path, index=False)
        rec = pv.read_record(self.rec_path)
        self.assertFalse(R.reproduce(rec, compare=[path]).ok)                                      # written by the command line: exact-ish
        self.assertTrue(R.reproduce(rec, compare=[path], rounded=True).ok)
        rec["software"]["app"] = {"name": "PyAT", "version": "3.2.0"}
        self.assertTrue(R.reproduce(rec, compare=[path]).ok)                                       # written by PyAT: its formats round
        self.assertFalse(R.reproduce(rec, compare=[path], rounded=False).ok)

    def test_nothing_compared_is_ok(self):
        self.assertTrue(R.reproduce(self.rec_path).ok)


class TestCompareTable(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "e.csv")

    def write(self, df, **kw):
        df.to_csv(self.path, index=False, **kw)

    def test_numbers_within_tolerance_match_beyond_it_do_not(self):
        self.write(pd.DataFrame({"t": [0.0, 1.0], "v": [10.0, 20.0]}))
        self.assertTrue(R.compare_table(self.path, pd.DataFrame({"t": [0.00005, 1.0], "v": [10.0, 20.0]}))["ok"])      # 5e-5 on 0
        bad = R.compare_table(self.path, pd.DataFrame({"t": [0.0002, 1.0], "v": [10.0, 20.0]}))
        self.assertFalse(bad["ok"])
        self.assertIn("row 1, 't'", bad["message"])

    def test_a_number_matches_to_the_rounding_the_export_shows_only_when_asked(self):
        self.write(pd.DataFrame({"bpm": ["13.8", "72.0"], "p": ["1.5e-05", "0.0001234"]}))
        table = pd.DataFrame({"bpm": [13.846, 72.04], "p": [1.54e-05, 0.00012344]})
        self.assertFalse(R.compare_table(self.path, table)["ok"])                              # a full-precision CSV: 13.8 means 13.8
        self.assertTrue(R.compare_table(self.path, table, rounded=True)["ok"])                  # a formatted export: 13.8 is 13.846 to 1 dp
        bad = R.compare_table(self.path, pd.DataFrame({"bpm": [13.9, 72.0], "p": [1.5e-05, 0.0001234]}), rounded=True)
        self.assertFalse(bad["ok"])                                                              # 13.9 would have been written 13.9
        self.assertIn("'bpm'", bad["message"])

    def test_a_whole_number_in_an_export_must_be_exact_even_when_rounding_is_honoured(self):
        self.write(pd.DataFrame({"Trial": [1, 2, 3]}))
        self.assertTrue(R.compare_table(self.path, pd.DataFrame({"Trial": [1.0, 2.0, 3.0]}), rounded=True)["ok"])
        self.assertFalse(R.compare_table(self.path, pd.DataFrame({"Trial": [1.0, 2.0, 3.4]}), rounded=True)["ok"])

    def test_the_rounding_a_text_implies(self):
        for text, half in (("13.8", 0.05), ("0.1234", 5e-5), ("1.5e-05", 5e-7), ("2E3", 0.5 * 10 ** 3), ("12", 0.0), ("-3.25", 0.005),
                           ("abc", 0.0), ("", 0.0)):
            self.assertAlmostEqual(R._rounding_half_width(text), half, places=12, msg=text)

    def test_the_tolerance_is_relative_for_large_values(self):
        self.write(pd.DataFrame({"v": [1000.0]}))
        self.assertTrue(R.compare_table(self.path, pd.DataFrame({"v": [1000.05]}))["ok"])                # 1e-4 * 1001
        self.assertFalse(R.compare_table(self.path, pd.DataFrame({"v": [1000.5]}))["ok"])
        self.assertTrue(R.compare_table(self.path, pd.DataFrame({"v": [1000.5]}), tolerance=1e-3)["ok"])

    def test_blank_and_missing_are_the_same_and_text_is_compared_exactly_ignoring_case(self):
        self.write(pd.DataFrame({"a": ["GROUP (checked only)", "x"], "n": [None, 5.0], "ok": [True, False]}))
        table = pd.DataFrame({"a": ["GROUP (checked only)", "x"], "n": [np.nan, 5.0], "ok": [True, False]})
        self.assertTrue(R.compare_table(self.path, table)["ok"])
        table["a"] = ["GROUP (checked only)", "y"]
        self.assertIn("'x' vs 'y'", R.compare_table(self.path, table)["message"])

    def test_columns_and_row_counts_must_agree(self):
        self.write(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        self.assertIn("columns differ", R.compare_table(self.path, pd.DataFrame({"a": [1, 2], "c": [3, 4]}))["message"])
        self.assertIn("2 rows in the export, 3 in the result", R.compare_table(self.path, pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]}))["message"])

    def test_an_excel_friendly_bom_is_no_problem(self):
        self.write(pd.DataFrame({"a": [1.5]}), encoding="utf-8-sig")
        self.assertTrue(R.compare_table(self.path, pd.DataFrame({"a": [1.5]}))["ok"])

    def test_it_reports_how_many_cells_and_the_largest_difference(self):
        self.write(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        r = R.compare_table(self.path, pd.DataFrame({"a": [1.00001, 2.0], "b": [3.0, 4.00002]}))
        self.assertEqual((r["ok"], r["cells"]), (True, 4))
        self.assertAlmostEqual(r["max_abs_diff"], 2e-5, places=9)


class TestOtherSources(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def test_an_oxysoft_file_round_trips(self):
        path = os.path.join(self.tmp, "session.txt")
        write_oxysoft(path)
        rec_path, result, paths = export(path, os.path.join(self.tmp, "out"), "auc_window", {"center_s": 30.0, "pre_s": 5, "post_s": 5})
        self.assertEqual(pv.read_record(rec_path)["recording"]["source_type"], "Oxysoft")
        rep = R.reproduce(rec_path, compare=[next(p for p in paths if p.endswith("export_auc.csv"))])
        self.assertTrue(rep.ok)
        self.assertEqual(list(rep.result.tables["auc"]["Channel"]), ["Mean O2Hb", "Mean HHb"])

    def test_an_oxysoft_splice_in_the_record_is_applied(self):
        path = os.path.join(self.tmp, "session.txt")
        write_oxysoft(path)
        splice_dir = os.path.join(self.tmp, "JSON saves")
        os.makedirs(splice_dir)
        with open(os.path.join(splice_dir, "splice.json"), "w") as f:
            json.dump([{"mode": "cut_out", "start": 10.0, "end": 20.0}], f)
        rec_path, result, _ = export(path, os.path.join(self.tmp, "out"), "auc_window", {"center_s": 40.0, "pre_s": 5, "post_s": 5})
        shutil.rmtree(splice_dir)
        pd.testing.assert_frame_equal(R.reproduce(rec_path).result.tables["auc"], result.tables["auc"])

    def write_csv(self, name="data.csv"):
        path = os.path.join(self.tmp, name)
        x = np.arange(0, 60, 0.5)
        pd.DataFrame({"time": x, "left": np.sin(x / 5), "right": np.cos(x / 5)}).to_csv(path, index=False)
        return path

    def test_a_generic_table_round_trips_with_its_columns(self):
        path = self.write_csv()
        rec_path, result, paths = export(path, os.path.join(self.tmp, "out"), "auc_window", {"center_s": 30.0, "pre_s": 10, "post_s": 10},
                                         table=None, x_column="time", y_columns=["right"])
        g = pv.read_record(rec_path)["recording"]["generic"]
        self.assertEqual((g["x_column"], g["y_columns"]), ("time", ["right"]))
        pd.testing.assert_frame_equal(R.reproduce(rec_path).result.tables["auc"], result.tables["auc"])

    def test_a_generic_file_that_lost_a_column_says_which(self):
        path = self.write_csv()
        rec_path, _, _ = export(path, os.path.join(self.tmp, "out"), "auc_window", {"center_s": 30.0}, x_column="time", y_columns=["right"])
        pd.DataFrame({"time": [0.0, 1.0], "left": [1.0, 2.0]}).to_csv(path, index=False)
        with self.assertRaisesRegex(R.ReproduceError, r"column\(s\) \['right'\] are gone"):
            R.reproduce(rec_path)

    def test_a_group_record_runs_the_group_analysis_from_its_spec(self):
        spec = {"group_name": "G", "subjects": [{"subject": "S1", "folder": "f1"}, {"subject": "S2", "folder": "f2"}], "markers": ["a"], "metrics": ["auc"]}
        record = pv.build_record("group_analysis", spec)
        fake = AnalysisResult("group_analysis", spec, {"descriptives": pd.DataFrame({"a": [1]})}, {"files": ["x.csv"]}, [])
        with mock.patch.object(R, "run_group_from_spec", return_value=(fake, None)) as run:
            rep = R.reproduce(record, out_dir="out")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], spec)
        self.assertEqual((rep.result, rep.written), (fake, ["x.csv"]))


class TestWriteResult(unittest.TestCase):
    def test_each_table_is_a_csv_named_after_the_stem_and_the_table(self):
        result = AnalysisResult("k", {}, {"one": pd.DataFrame({"a": [1.5, 2.5]}), "two": pd.DataFrame({"b": ["x¹"]})})
        with tempfile.TemporaryDirectory() as d:
            paths = R.write_result(result, os.path.join(d, "deep"), stem='Rat 7: "x"/y')
            self.assertEqual([os.path.basename(p) for p in paths], ["Rat 7_ _x_y_one.csv", "Rat 7_ _x_y_two.csv"])
            with open(paths[1], encoding="utf-8") as f:
                self.assertEqual(f.read().splitlines(), ["b", "x¹"])
            self.assertEqual(pd.read_csv(paths[0])["a"].tolist(), [1.5, 2.5])                     # no index column

    def test_safe_names(self):
        self.assertEqual(R.safe_name("a<b>c"), "a_b_c")
        self.assertEqual(R.safe_name("   "), "result")


class TestScriptingApiIsAtTheTopLevel(unittest.TestCase):
    """What the README and API reference tell a script to use must be importable from `PyATCLI` itself."""

    def test_the_documented_names(self):
        for name in ("run_analysis", "kinds", "describe_kinds", "normalize_parameters", "AnalysisResult", "ParameterError",
                     "AnalysisError", "reproduce_record", "compare_table", "Reproduction", "ReproduceError", "load_recording",
                     "build_record", "read_record", "write_record", "recording_state", "file_fingerprint", "check_environment",
                     "read_saved_signal_labels", "apply_signal_labels", "current_signal_labels", "signal_labels_path"):
            self.assertTrue(hasattr(pl, name), name)

    def test_reproduce_is_still_the_module_and_its_function_is_reproduce_record(self):
        self.assertIs(pl.reproduce_record, R.reproduce)
        self.assertTrue(hasattr(R, "compare_table"))
        self.assertIs(pl.kinds, A.kinds)


if __name__ == "__main__":
    unittest.main()
