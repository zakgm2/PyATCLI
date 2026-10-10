"""The command line against the conventions of a well-behaved CLI: help that helps, stdout that is only ever the result (so `--json | jq` always
works), errors on stderr in words, exit codes that mean something, the same global flags on every command in any order, `-` for standard
input and output, nothing asked when told not to, and Ctrl-C that stops."""

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import PyATCLI as pl
from PyATCLI import cli, ops
from PyATCLI.cli import render

from tests.test_cli import CliCase, make_tdt_folder, run_cli


def run_binary(*argv, stdin=None):
    """(status, stdout bytes, stderr text) of cli.main(argv) with a stdout that can take bytes."""
    out_bytes, err = io.BytesIO(), io.StringIO()
    out = io.TextIOWrapper(out_bytes, encoding="utf-8", newline="\n", write_through=True)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        if stdin is not None:
            with mock.patch.object(sys, "stdin", io.StringIO(stdin)):
                status = cli.main([str(a) for a in argv])
        else:
            status = cli.main([str(a) for a in argv])
    out.flush()
    return status, out_bytes.getvalue(), err.getvalue()


def json_of(text):
    """The one JSON document a --json run prints; fails the test if stdout is anything else."""
    return json.loads(text)


class TestHelp(CliCase):
    def test_no_arguments_shows_concise_help_on_stdout_and_is_not_an_error(self):
        status, out, err = run_cli()
        self.assertEqual((status, err), (0, ""))
        self.assertIn("usage: python -m PyATCLI <command>", out)
        self.assertIn("examples:", out)
        self.assertLess(len(out.splitlines()), 40)                                           # concise: it fits a screen
        self.assertIn("--help", out)

    def test_h_and_help_show_everything_and_ignore_the_rest(self):
        for flag in ("-h", "--help"):
            status, out, err = run_cli(flag, "--nonsense", "x")
            self.assertEqual((status, err), (0, ""))
            self.assertIn("global options", out)
            self.assertIn("exit status:", out)
        status, out, _ = run_cli("run", "--help", "--no-such-flag", "ignored")
        self.assertEqual(status, 0)
        self.assertIn("--param NAME=VALUE", out)

    def test_every_command_has_examples_in_its_help(self):
        for argv in (["info"], ["kinds"], ["run"], ["reproduce"], ["batch"], ["group"], ["group", "setup"], ["group", "run"], ["saved"],
                     ["saved", "marker"]):
            with self.subTest(command=" ".join(argv)):
                status, out, _ = run_cli(*argv, "--help")
                self.assertEqual(status, 0)
                self.assertIn("examples:", out)

    def test_the_version_has_a_short_flag_too(self):
        for flag in ("--version", "-V"):
            status, out, _ = run_cli(flag)
            self.assertEqual((status, out.strip()), (0, f"PyATCLI {pl.__version__}"))

    def test_a_command_that_has_subcommands_and_is_given_none_says_so(self):
        status, out, _ = run_cli("saved")
        self.assertEqual(status, 2)
        self.assertIn("<action>", out)


class TestStdoutIsOnlyTheResult(CliCase):
    def noisy_load(self, noise="Found Synapse note file: x"):
        real = pl.recording.load_recording

        def load(*a, **kw):
            print(noise)                                                                     # what the tdt SDK does for every block
            return real(*a, **kw)
        return mock.patch("PyATCLI.recording.load_recording", side_effect=load)

    def test_a_library_that_prints_does_not_break_json(self):
        with self.noisy_load():
            status, out, err = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "--out", self.out, "--json")
        self.assertEqual(status, 0)
        self.assertEqual(json_of(out)["command"], "run")
        self.assertNotIn("Found Synapse", out)
        self.assertNotIn("Found Synapse", err)                                               # dropped, not moved

    def test_verbose_shows_what_the_library_printed_on_stderr(self):
        with self.noisy_load():
            status, out, err = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "--out", self.out, "--json", "-v")
        self.assertEqual(status, 0)
        json_of(out)
        self.assertIn("Found Synapse", err)

    def test_a_library_that_prints_does_not_leak_into_the_human_output_either(self):
        with self.noisy_load():
            status, out, _ = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "--out", self.out)
        self.assertNotIn("Found Synapse", out)

    def test_warnings_go_to_stderr_and_into_the_envelope(self):
        pl.saved_state.write_saved_markers(self.oxy, [{"time": 5.0, "label": "x"}])
        Path(pl.saved_state.splice_sidecar_path(self.oxy)).write_text("{not json", encoding="utf-8")       # unreadable: a warning, not a failure
        status, out, err = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "--out", self.out, "--json")
        self.assertEqual(status, 0)
        self.assertIn("warning: Saved splices could not be read", err)
        self.assertTrue(any("Saved splices could not be read" in w for w in json_of(out)["warnings"]))


class TestJsonOnEveryCommand(CliCase):
    def check(self, *argv, command=None):
        status, out, err = run_cli(*argv, "--json")
        self.assertEqual(status, 0, err)
        doc = json_of(out)
        self.assertEqual((doc["ok"], doc["schema"]), (True, 1))
        self.assertIn("result", doc)
        if command:
            self.assertEqual(doc["command"], command)
        return doc["result"]

    def test_info_and_kinds(self):
        info = self.check("info", command="info")
        self.assertEqual((info["library"], info["protocol"]), (pl.__version__, ops.PROTOCOL_VERSION))
        kinds = self.check("kinds", command="kinds")
        self.assertIn("event_peth", kinds["analyses"])
        self.assertEqual(list(self.check("kinds", "curve_fit")["analyses"]), ["curve_fit"])

    def test_run_reproduce_and_batch(self):
        run = self.check("run", "auc_window", self.oxy, "-p", "center_s=30", "-p", "pre_s=5", "-p", "post_s=5", "--out", self.out, command="run")
        record = next(f for f in run["files"] if f.endswith(".provenance.json"))
        csv = next(f for f in run["files"] if f.endswith("_window.csv"))
        rep = self.check("reproduce", record, "--compare", csv, command="reproduce")
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["comparisons"][0]["table"], "window")
        jobs = os.path.join(self.tmp, "jobs.json")
        Path(jobs).write_text(json.dumps({"jobs": [{"kind": "auc_window", "recording": self.oxy, "parameters": {"center_s": 30, "pre_s": 5, "post_s": 5}}]}), encoding="utf-8")
        batch = self.check("batch", jobs, "--out", os.path.join(self.tmp, "b"), command="batch")
        self.assertEqual((batch["ok"], batch["failed"], len(batch["jobs"])), (True, [], 1))

    def test_saved_commands(self):
        self.assertEqual(self.check("saved", "marker", self.oxy, "--time", "12.5", "--label", "Lever", command="saved.marker")["marker"]["label"], "Lever")
        self.assertEqual(self.check("saved", "splice", self.oxy, "--mode", "cut_out", "--start", "30", "--end", "40")["splices"][0]["end"], 40.0)
        self.assertEqual(len(self.check("saved", "show", self.oxy)["markers"]), 1)
        self.check("saved", "highlight", self.oxy, "--start", "1", "--end", "2")
        self.check("saved", "text", self.oxy, "--x", "5", "--y", "0.5", "--text", "peak")
        self.assertEqual(len(self.check("saved", "unmark", self.oxy, "--time", "12.5")["removed"]), 1)
        self.assertEqual(self.check("saved", "unsplice", self.oxy, "1")["splices"], [])
        self.assertEqual(len(self.check("saved", "clear", self.oxy, "--yes")["removed"]), 3)

    def test_group_setup_and_run(self):
        from tests.test_group_run import fake_block, make_tdt_folder as make_block, EPOCS
        import tdt
        cohort = os.path.join(self.tmp, "cohort")
        for i in (1, 2):
            make_block(cohort, f"Rat{i}-250101-10101{i}")
        with mock.patch.object(tdt, "read_block", fake_block(EPOCS, duration=20.0)):
            listing = run_cli("group", "setup", cohort, "--name", "G", "--json")                # no place to write needed just to list
            self.assertEqual(listing[0], 2)                                                  # no markers chosen: say so, but the listing is the result
            doc = json_of(listing[1])
            self.assertEqual(doc["result"]["stage"], "markers")
            self.assertEqual([m["name"] for m in doc["result"]["markers"]], ["L1P¹", "L1P⁰"])
            setup = self.check("group", "setup", cohort, "--name", "G", "--marker", "L1P¹", "--out", os.path.join(self.tmp, "G"), command="group.setup")
        self.assertEqual((setup["stage"], setup["subjects"], setup["ran"]), ("settings", 2, False))


class TestPlain(CliCase):
    def test_rows_and_nothing_else(self):
        status, out, _ = run_cli("kinds", "curve_fit", "--plain")
        self.assertEqual(status, 0)
        rows = [line.split("\t") for line in out.splitlines()]
        self.assertTrue(all(len(r) == 5 for r in rows), rows[:2])
        self.assertEqual(rows[0][:2], ["curve_fit", "t1_s"])
        status, out, _ = run_cli("info", "--plain")
        self.assertIn("library\t" + pl.__version__, out.splitlines())

    def test_run_lists_its_tables_and_files_as_rows(self):
        status, out, _ = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "-p", "pre_s=5", "-p", "post_s=5", "--out", self.out, "--plain")
        self.assertEqual(status, 0)
        lines = out.splitlines()
        self.assertIn("auc\t2\t6", lines)
        self.assertTrue(any(l.endswith("_auc.csv") for l in lines))                           # the files, one a line

    def test_json_and_plain_are_exclusive(self):
        status, _, err = run_cli("info", "--json", "--plain")
        self.assertEqual(status, 2)
        self.assertIn("not allowed with argument", err)


class TestErrors(CliCase):
    def test_a_mistake_is_one_line_on_stderr_and_nothing_on_stdout(self):
        status, out, err = run_cli("run", "auc_window", os.path.join(self.tmp, "nope.txt"), "-p", "center_s=30", "--out", self.out)
        self.assertEqual((status, out), (1, ""))
        self.assertEqual(err.strip().splitlines()[0], f"error: {os.path.join(self.tmp, 'nope.txt')} does not exist")
        self.assertNotIn("Traceback", err)

    def test_with_json_the_error_is_a_json_document_on_stderr(self):
        status, out, err = run_cli("run", "auc_window", self.oxy, "-p", "centre=30", "--json")
        self.assertEqual((status, out), (1, ""))
        doc = json.loads(err)
        self.assertEqual((doc["ok"], doc["error"]["code"]), (False, "bad_parameter"))
        self.assertIn("centre", doc["error"]["message"])

    def test_a_bug_is_said_to_be_one_and_the_traceback_is_for_debug(self):
        with mock.patch.object(ops, "call", side_effect=ZeroDivisionError("division by zero")):
            status, out, err = run_cli("info")
            self.assertEqual((status, out), (1, ""))
            self.assertIn("error: ZeroDivisionError: division by zero", err)
            self.assertIn("this is a bug", err)
            self.assertNotIn("Traceback", err)
            with self.assertRaises(ZeroDivisionError):
                cli.main(["info", "--debug"])

    def test_a_malformed_jobs_file_is_a_message_not_a_traceback(self):
        jobs = os.path.join(self.tmp, "jobs.json")
        Path(jobs).write_text(json.dumps(["not a job"]), encoding="utf-8")
        status, _, err = run_cli("batch", jobs, "--out", self.out)
        self.assertEqual(status, 1)
        self.assertIn("a jobs file is", err)
        self.assertNotIn("Traceback", err)

    def test_a_bad_usage_is_exit_2_and_says_how(self):
        status, _, err = run_cli("run", "auc_window")
        self.assertEqual(status, 2)
        self.assertIn("usage:", err)

    def test_ctrl_c_stops_with_130(self):
        with mock.patch.object(ops, "call", side_effect=KeyboardInterrupt):
            status, out, err = run_cli("info")
        self.assertEqual((status, out), (130, ""))
        self.assertIn("interrupted", err)

    def test_debug_can_come_from_the_environment(self):
        with mock.patch.object(ops, "call", side_effect=ZeroDivisionError("x")), mock.patch.dict(os.environ, {"DEBUG": "1"}):
            with self.assertRaises(ZeroDivisionError):
                cli.main(["info"])
        with mock.patch.object(ops, "call", side_effect=ZeroDivisionError("x")), mock.patch.dict(os.environ, {"DEBUG": "0"}):
            self.assertEqual(run_cli("info")[0], 1)                                          # DEBUG=0 is not debugging


class TestGlobalFlagsAnywhere(CliCase):
    def test_before_the_command_after_it_and_after_a_subcommand(self):
        for argv in (["--json", "info"], ["info", "--json"], ["-q", "info"], ["info", "-q"]):
            with self.subTest(argv=argv):
                status, out, _ = run_cli(*argv)
                self.assertEqual(status, 0)
                if "--json" in argv:
                    json_of(out)
                else:
                    self.assertEqual(out, "")
        status, out, _ = run_cli("saved", "show", self.oxy, "--json")
        json_of(out)
        status, out, _ = run_cli("--json", "saved", "show", self.oxy)
        json_of(out)

    def test_run_accepts_debug_after_it_which_used_to_be_refused(self):
        with mock.patch.object(ops, "call", side_effect=ZeroDivisionError("x")):
            with self.assertRaises(ZeroDivisionError):
                cli.main(["run", "auc_window", self.oxy, "-d"])

    def test_the_short_flags(self):
        self.assertEqual(run_cli("info", "-v")[0], 0)
        self.assertEqual(run_cli("info", "-vv")[0], 0)
        self.assertEqual(run_cli("info", "--no-input")[0], 0)
        self.assertEqual(run_cli("info", "--no-progress")[0], 0)


class TestStandardInputAndOutput(CliCase):
    def test_parameters_from_standard_input(self):
        with mock.patch.object(sys, "stdin", io.StringIO(json.dumps({"parameters": {"center_s": 30, "pre_s": 5, "post_s": 5}}))):
            status, out, err = run_cli("run", "auc_window", self.oxy, "--params", "-", "--out", self.out, "--json")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(json_of(out)["result"]["parameters"]["center_s"], 30.0)

    def test_a_jobs_file_and_a_record_from_standard_input(self):
        jobs = json.dumps({"jobs": [{"kind": "auc_window", "recording": self.oxy, "parameters": {"center_s": 30, "pre_s": 5, "post_s": 5}}]})
        with mock.patch.object(sys, "stdin", io.StringIO(jobs)):
            status, out, err = run_cli("batch", "-", "--out", os.path.join(self.tmp, "b"), "--json")
        self.assertEqual((status, err), (0, ""))
        run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "-p", "pre_s=5", "-p", "post_s=5", "--out", self.out, "-q")
        record = Path(next(Path(self.out).glob("*.provenance.json"))).read_text(encoding="utf-8")
        with mock.patch.object(sys, "stdin", io.StringIO(record)):
            status, out, err = run_cli("reproduce", "-", "--json")
        self.assertEqual((status, err), (0, ""))
        self.assertTrue(json_of(out)["result"]["ok"])

class TestAsking(CliCase):
    def setUp(self):
        super().setUp()
        pl.saved_state.add_marker(self.oxy, 12.5)

    def test_clear_asks_when_a_person_is_there_and_does_what_they_say(self):
        with mock.patch.object(render.Output, "interactive", return_value=True), mock.patch.object(sys, "stdin", io.StringIO("y\n")):
            status, out, err = run_cli("saved", "clear", self.oxy)
        self.assertEqual(status, 0)
        self.assertIn("Delete them? [y/N]", err)
        self.assertEqual(pl.read_saved_markers(self.oxy), [])
        pl.saved_state.add_marker(self.oxy, 12.5)
        with mock.patch.object(render.Output, "interactive", return_value=True), mock.patch.object(sys, "stdin", io.StringIO("n\n")):
            status, _, err = run_cli("saved", "clear", self.oxy)
        self.assertEqual(status, 2)
        self.assertEqual(len(pl.read_saved_markers(self.oxy)), 1)

    def test_with_nobody_to_ask_or_no_input_it_fails_instead_of_deleting(self):
        status, _, err = run_cli("saved", "clear", self.oxy)                                 # the test's stdin is not a terminal
        self.assertEqual(status, 2)
        self.assertIn("say --yes to do it", err)
        with mock.patch.object(sys, "stdin", mock.Mock(isatty=lambda: True)):
            status, _, _ = run_cli("saved", "clear", self.oxy, "--no-input")
        self.assertEqual(status, 2)
        self.assertEqual(len(pl.read_saved_markers(self.oxy)), 1)

    def test_yes_goes_ahead_without_asking(self):
        status, out, _ = run_cli("saved", "clear", self.oxy, "-y")
        self.assertEqual(status, 0)
        self.assertEqual(pl.read_saved_markers(self.oxy), [])


class TestEventsAndProgress(CliCase):
    def test_events_are_json_lines_on_stderr_and_stdout_stays_the_result(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        status, out, err = run_cli("run", "process", folder, "--out", self.out, "--json", "--events")
        self.assertEqual(status, 0)
        json_of(out)
        events = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
        self.assertTrue(events)
        progress = [e for e in events if e["event"] == "progress"]
        self.assertTrue(progress and all(0.0 <= e["fraction"] <= 1.0 for e in progress))
        self.assertEqual(progress[-1]["fraction"], 1.0)
        self.assertIn("state", {e["event"] for e in events})                                 # the files written, as they were written

    def test_no_bar_without_a_terminal(self):
        status, out, err = run_cli("run", "process", make_tdt_folder(self.tmp, "Rat8-250101-101010"), "--out", self.out)
        self.assertEqual(status, 0)
        self.assertNotIn("%|", err)

    def test_progress_can_be_forced(self):
        status, out, err = run_cli("run", "process", make_tdt_folder(self.tmp, "Rat9-250101-101010"), "--out", self.out, "--progress")
        self.assertEqual(status, 0)
        self.assertIn("100%", err)


class TestOldSpellings(CliCase):
    def test_the_old_group_commands_still_work_and_say_what_they_are_now(self):
        status, out, err = run_cli("group-setup", os.path.join(self.tmp, "nowhere"), "--name", "G", "--out", self.out)
        self.assertIn("warning: `group-setup` is now `group setup`", err)
        settings = pl.write_group_settings(os.path.join(self.tmp, "s"), pl.GroupSpec(group_name="G", subjects=[{"subject": "a", "folder": "x"}], markers=["m"], metrics=["auc"]), [])
        with mock.patch("PyATCLI.ops.group.run_group_from_spec", side_effect=RuntimeError("stop here")):
            status, out, err = run_cli("group", settings)
        self.assertIn("warning: `group SETTINGS` is now `group run SETTINGS`", err)
        self.assertIn("stop here", err)                                                      # it got as far as running

    def test_group_alone_prints_its_help(self):
        status, out, _ = run_cli("group")
        self.assertEqual(status, 2)
        self.assertIn("setup", out)




class TestTheCommandLineNeverDraws(CliCase):
    """The command line computes and writes files. Drawing is the app's: no command, option or operation of it makes a figure or opens a window."""

    def test_no_command_is_about_drawing_and_no_option_asks_for_a_figure(self):
        from PyATCLI.cli import docs
        names = {node.name for node in docs.tree().walk() if node.name}
        self.assertTrue(names.isdisjoint({"view", "plot"}), names)
        flags = {flag for node in docs.tree().walk() for action, _ in node.own_options() for flag in action.option_strings}
        self.assertTrue(flags.isdisjoint({"--figures", "--dpi", "--transparent", "--text-as-outlines", "--no-data", "--xlim", "--ylim", "--save"}), flags)

    def test_the_engine_has_no_operation_that_draws(self):
        described = ops.describe()
        self.assertTrue({"view", "plot"}.isdisjoint(described))
        for name, op in described.items():
            self.assertTrue({"figures", "remember", "save", "dpi", "show"}.isdisjoint(p["name"] for p in op["parameters"]), name)

    def test_a_whole_run_imports_neither_matplotlib_nor_a_window_toolkit(self):
        import subprocess
        out = os.path.join(self.tmp, "no-draw")
        code = (
            "import sys\n"
            "from PyATCLI import cli\n"
            f"status = cli.main(['run', 'auc_window', r'{self.oxy}', '-p', 'center_s=30', '-p', 'pre_s=5', '-p', 'post_s=5', '--out', r'{out}', '-q'])\n"
            "from PyATCLI.engine.session import Session\n"
            "Session().handle('kinds', {}, lambda *a, **k: None)\n"
            f"Session().handle('run', {{'kind': 'auc_window', 'recording': r'{self.oxy}', 'parameters': {{'center_s': 30.0}}}}, lambda *a, **k: None)\n"
            "drawn = sorted(m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'tkinter', 'pyqtgraph'))\n"
            "print('STATUS', status, 'DRAWN', drawn)\n")
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.assertIn("STATUS 0 DRAWN []", done.stdout, done.stdout + done.stderr)

    def test_a_run_writes_tables_and_a_record_and_no_picture(self):
        status, _, err = run_cli("run", "auc_window", self.oxy, "-p", "center_s=30", "--out", self.out, "-q")
        self.assertEqual((status, err), (0, ""), err)
        names = os.listdir(self.out)
        self.assertTrue(any(n.endswith(".csv") for n in names) and any(n.endswith(".provenance.json") for n in names))
        self.assertFalse(any(n.lower().endswith((".png", ".pdf", ".svg", ".tif", ".tiff")) for n in names))


if __name__ == "__main__":
    unittest.main()
