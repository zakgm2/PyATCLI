"""What the engine's callers can count on (engine/contract.py): the operations and their parameters, the analyses, the tables they return and the
columns that are always there. ENGINE_CONTRACT.json is this, generated; tests/contracts/protocol_N.json is the contract as it was when protocol N
began. A change that removes or renames something a caller could rely on fails here until ops.PROTOCOL_VERSION is raised and the new baseline added.

    python -m PyATCLI docs --format contract --out ENGINE_CONTRACT.json
"""

import copy
import json
import os
import unittest
from unittest import mock
from pathlib import Path

import PyATCLI as pl
from PyATCLI import ops
from PyATCLI.analyses import describe_kinds, describe_outputs, kinds
from PyATCLI.analysis.group.report import NOTES_FILENAME, REPORT_FILENAME, TABLE_FILES, trace_frame
from PyATCLI.analysis.group.run import GROUP_SETTINGS_FILENAME
from PyATCLI.engine import contract
from PyATCLI.engine.session import Session
from PyATCLI.markers import group_markers_by_name
from PyATCLI.recording import CACHE_KEYS

from tests.test_analyses import make_cache
from tests.test_cli import CliCase, make_tdt_folder
from tests.test_group_run import TempDirCase, simulated_loader, simulated_spec

ROOT = Path(__file__).resolve().parent.parent
BASELINES = Path(__file__).resolve().parent / "contracts"


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestTheContractFile(unittest.TestCase):
    def test_it_is_up_to_date(self):
        self.assertEqual(read(ROOT / "ENGINE_CONTRACT.json"), json.loads(json.dumps(contract.build(), ensure_ascii=False)),
                         "ENGINE_CONTRACT.json is out of date: python -m PyATCLI docs --format contract --out ENGINE_CONTRACT.json")

    def test_the_engine_still_honours_the_baseline_of_its_protocol(self):
        baseline = BASELINES / f"protocol_{ops.PROTOCOL_VERSION}.json"
        self.assertTrue(baseline.exists(), f"add {baseline.name} to tests/contracts: a copy of ENGINE_CONTRACT.json as it is when protocol "
                                           f"{ops.PROTOCOL_VERSION} begins")
        problems = contract.breaking_changes(read(baseline), contract.build())
        self.assertEqual(problems, [], "a caller written against this protocol would break. If that is meant, raise ops.PROTOCOL_VERSION and add the "
                                       "new baseline (and tell PyAT: backend/contract.py lists the protocols it speaks); otherwise put it back")

    def test_the_baseline_says_which_protocol_it_is_of(self):
        for path in BASELINES.glob("protocol_*.json"):
            self.assertEqual(read(path)["protocol"], int(path.stem.split("_")[1]), path.name)


class TestWhatCountsAsBreaking(unittest.TestCase):
    def setUp(self):
        self.old = contract.build()

    def changed(self, edit):
        new = copy.deepcopy(self.old)
        edit(new)
        return contract.breaking_changes(self.old, new)

    def test_nothing_changed_is_nothing_broken(self):
        self.assertEqual(contract.breaking_changes(self.old, copy.deepcopy(self.old)), [])

    def test_adding_is_never_breaking(self):
        def add(c):
            c["operations"]["brand.new"] = {"summary": "x", "parameters": [{"name": "a", "required": True, "default": None}]}
            c["operations"]["run"]["parameters"].append({"name": "extra", "required": False, "default": 1})
            c["analyses"]["zscore_window"]["parameters"].append({"name": "more", "type": "float", "required": False, "default": 1, "choices": []})
            c["analyses"]["zscore_window"]["tables"]["extra_table"] = ["a"]
            c["analyses"]["zscore_window"]["tables"]["window"].append("another_column")
            c["protocol"] += 1
        self.assertEqual(self.changed(add), [])

    def test_removing_or_renaming_an_operation_or_a_parameter_is_breaking(self):
        self.assertIn("operation 'run' is gone", self.changed(lambda c: c["operations"].pop("run")))
        self.assertTrue(any("parameter" in p and "is gone" in p for p in self.changed(lambda c: c["operations"]["run"]["parameters"].pop(0))))

    def test_a_parameter_that_must_now_be_given_is_breaking(self):
        def require(c):
            optional = next(p for p in c["operations"]["run"]["parameters"] if not p["required"])
            optional["required"] = True
        self.assertTrue(any("must now be given" in p for p in self.changed(require)))
        self.assertTrue(any("must now be given" in p for p in self.changed(
            lambda c: c["operations"]["run"]["parameters"].append({"name": "new_one", "required": True, "default": None}))))

    def test_a_table_or_a_column_that_is_gone_is_breaking(self):
        self.assertTrue(any("table 'window' is gone" in p for p in self.changed(lambda c: c["analyses"]["zscore_window"]["tables"].pop("window"))))
        self.assertTrue(any("no column 'z_score'" in p for p in self.changed(lambda c: c["analyses"]["zscore_window"]["tables"]["window"].remove("z_score"))))

    def test_an_output_table_column_or_file_that_is_gone_or_renamed_is_breaking(self):
        outputs = lambda c: c["outputs"]["group_analysis"]
        self.assertIn("output 'view' is gone", self.changed(lambda c: c["outputs"].pop("view")))
        self.assertTrue(any("table 'trials' is gone" in p for p in self.changed(lambda c: outputs(c)["tables"].pop("trials"))))
        self.assertTrue(any("no column 'auc'" in p for p in self.changed(lambda c: outputs(c)["tables"]["trials"].remove("auc"))))
        self.assertTrue(any("no longer written to 'subject_means.csv'" in p for p in self.changed(lambda c: outputs(c)["files"].update(subject_means="means.csv"))))
        self.assertTrue(any("'settings' is no longer written" in p for p in self.changed(lambda c: outputs(c)["files"].pop("settings"))))

    def test_adding_an_output_table_column_or_file_is_not(self):
        def add(c):
            c["outputs"]["group_analysis"]["tables"]["new_table"] = ["a"]
            c["outputs"]["group_analysis"]["tables"]["trials"].append("another_column")
            c["outputs"]["group_analysis"]["files"]["new_table"] = "new_table.csv"
            c["outputs"]["something_new"] = {"summary": "x", "tables": {"t": ["a"]}, "files": {}}
        self.assertEqual(self.changed(add), [])

    def test_a_removed_choice_or_a_changed_type_is_breaking(self):
        self.assertTrue(any("no longer accepts" in p for p in self.changed(
            lambda c: next(p for p in c["analyses"]["peak_finder"]["parameters"] if p["name"] == "scope")["choices"].remove("whole"))))
        self.assertTrue(any("changed type" in p for p in self.changed(
            lambda c: c["analyses"]["zscore_window"]["parameters"][0].update(type="str"))))

    def test_a_result_key_that_was_always_there_and_is_not_any_more_is_breaking(self):
        self.assertTrue(any("no longer always has 'cache'" in p for p in self.changed(lambda c: c["operations"]["recording.open"]["result"].remove("cache"))))
        self.assertEqual(self.changed(lambda c: c["operations"]["recording.open"]["result"].append("something_new")), [])

    def test_a_recording_key_that_is_gone_is_breaking(self):
        self.assertTrue(any("no longer always has 'x'" in p for p in self.changed(lambda c: c["recording_cache"]["always"].remove("x"))))
        self.assertTrue(any("'TDT'" in p for p in self.changed(lambda c: c["recording_cache"].pop("TDT"))))
        self.assertEqual(self.changed(lambda c: c["recording_cache"]["always"].append("extra")), [])

    def test_a_lower_protocol_is_breaking(self):
        self.assertTrue(any("protocol went down" in p for p in self.changed(lambda c: c.update(protocol=self.old["protocol"] - 1))))


class TestTheDeclaredTablesAreWhatTheAnalysesMake(unittest.TestCase):
    """The columns an analysis says are always there are there, in every table it makes, for every way of asking for it."""

    @staticmethod
    def runs(cache):
        groups = group_markers_by_name(cache, {})
        name = next(iter(groups))
        yield "zscore_window", {"center_s": 30.0}
        yield "auc_window", {"center_s": 30.0}
        yield "fft_window", {"center_s": 30.0}
        yield "curve_fit", {"t1_s": 20.0, "t2_s": 40.0}
        yield "event_peth", {"event": name}
        yield "peak_finder", {}
        yield "peak_finder", {"scope": "all_same_name", "event": name}
        yield "peak_finder", {"scope": "scan_all"}
        yield "peak_finder", {"scope": "one_event", "event": name, "event_time": groups[name][0]}
        yield "event_intervals", {}
        yield "process", {}

    def test_every_analysis_declares_its_tables(self):
        declared = describe_kinds()
        self.assertEqual(set(declared), set(kinds()))
        for kind, spec in declared.items():
            self.assertTrue(spec["tables"], f"{kind} declares no tables (analyses.TABLES)")

    def test_what_is_made_has_what_was_declared_and_nothing_undeclared(self):
        declared = describe_kinds()
        seen = {(kind, table) for kind, spec in declared.items() for table in spec["tables"]}
        made = set()
        cache = make_cache()
        for kind, params in self.runs(cache):
            result = pl.run_analysis(kind, cache, params)
            for table, frame in result.tables.items():
                with self.subTest(kind=kind, params=params, table=table):
                    self.assertIn(table, declared[kind]["tables"], f"{kind} makes a table {table!r} that analyses.TABLES does not declare")
                    missing = [c for c in declared[kind]["tables"][table] if c not in frame.columns]
                    self.assertEqual(missing, [], f"{kind}.{table} lacks declared columns")
                made.add((kind, table))
        self.assertEqual(sorted(seen - made), [], "declared tables that no run made: a typo, or a table that is gone")


class TestTheDeclaredOutputsAreWhatIsWritten(TempDirCase):
    """A group analysis's folder and an Export View's tables: the files and columns declared are there, and every table made is declared."""

    def test_a_group_analysis_writes_the_tables_files_and_columns_it_declares(self):
        from PyATCLI.analysis.group.run import run_group_analysis, write_group_settings

        declared = describe_outputs()["group_analysis"]
        folder = os.path.join(self.tmp, "G")
        spec = simulated_spec()
        write_group_settings(folder, spec, [])
        load, _ = simulated_loader()
        results = run_group_analysis(spec, out_dir=folder, load=load)
        self.assertEqual(set(declared["tables"]), set(TABLE_FILES), "a table with columns declared and no file, or a file with no columns declared")
        for name, columns in declared["tables"].items():
            with self.subTest(table=name):
                frame = results.frames().get(name) if name != "mean_traces" else trace_frame(results)
                self.assertGreater(len(frame), 0, f"the simulated analysis makes no {name}: the declaration cannot be checked")
                self.assertEqual([c for c in columns if c not in frame.columns], [], f"{name} lacks declared columns")
                self.assertEqual(os.path.basename(os.path.join(folder, declared["files"][name])), TABLE_FILES[name])
                self.assertTrue(os.path.isfile(os.path.join(folder, declared["files"][name])))
        for role, filename in (("settings", GROUP_SETTINGS_FILENAME), ("report", REPORT_FILENAME), ("notes", NOTES_FILENAME)):
            with self.subTest(file=role):
                self.assertEqual(declared["files"][role], filename)
                self.assertTrue(os.path.isfile(os.path.join(folder, filename)))

    def test_an_export_view_makes_the_tables_it_declares(self):
        declared = describe_outputs()["view"]["tables"]
        result = pl.run_view(make_cache())
        self.assertEqual(set(result.tables), set(declared))
        for name, frame in result.tables.items():
            self.assertEqual([c for c in declared[name] if c not in frame.columns], [], f"view.{name} lacks declared columns")

    def test_they_are_in_the_contract(self):
        built = contract.build()["outputs"]
        self.assertEqual(set(built), {"group_analysis", "view"})
        self.assertEqual(built["group_analysis"]["tables"], describe_outputs()["group_analysis"]["tables"])


class TestTheLoadedRecordingHasWhatIsPromised(CliCase):
    def csv(self):
        path = os.path.join(self.tmp, "table.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("t,a\n" + "\n".join(f"{i / 10},{i}" for i in range(60)) + "\n")
        return path

    def test_every_kind_of_recording_has_its_keys(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        for source, path in (("TDT", folder), ("Oxysoft", self.oxy), ("Generic", self.csv())):
            cache = pl.load_recording(path)["cache"]
            with self.subTest(source=source):
                self.assertEqual(cache["source"], source)
                promised = CACHE_KEYS["always"] + CACHE_KEYS[source]
                self.assertEqual([k for k in promised if k not in cache], [])

    def test_it_names_the_sources_there_are(self):
        self.assertEqual(set(CACHE_KEYS), {"always", "TDT", "Oxysoft", "Generic"})

    def test_it_is_in_the_contract(self):
        self.assertEqual(contract.build()["recording_cache"], CACHE_KEYS)


class TestWhatTheOperationsReturnHasWhatIsPromised(CliCase):
    """Every operation a test runs is checked against its declared result keys (tests/__init__.py turns that on); these two are run nowhere else."""

    def setUp(self):
        super().setUp()
        self.session = Session()
        self.addCleanup(self.session.close)

    def call(self, method, params=None):
        return self.session.handle(method, params, lambda event, **fields: None)

    def test_the_checking_is_on_and_a_missing_key_is_loud(self):
        self.assertTrue(os.environ.get("PYATCLI_CHECK_RESULTS"))
        with mock.patch.dict(ops._OPERATIONS["info"].__dict__, {"result": ("python", "no_such_key")}):
            with self.assertRaises(AssertionError) as caught:
                self.call("info")
        self.assertIn("no_such_key", str(caught.exception))

    def test_saving_what_a_window_holds(self):
        answer = self.call("saved.write", {"recording": self.oxy, "markers": []})
        self.assertIn("files", answer)

    def test_the_operations_say_what_they_are(self):
        answer = self.call("operations")
        self.assertEqual(answer["protocol"], ops.PROTOCOL_VERSION)
        self.assertIn("recording.open", answer["operations"])
        self.assertTrue(answer["operations"]["recording.open"]["result"])                   # (and what each promises to return)


if __name__ == "__main__":
    unittest.main()
