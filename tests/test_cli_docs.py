"""The command line's reference documents (cli/docs.py): CLI_REFERENCE.md and the man page are generated from the parser, so a flag that is added,
renamed or removed changes them — and these tests fail until they are regenerated:

    python -m PyATCLI docs --format markdown --out CLI_REFERENCE.md
    python -m PyATCLI docs --format man --out man/pyatcli.1
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from PyATCLI import provenance
from PyATCLI.cli import docs
from PyATCLI.cli.parser import build_parser

from tests.test_cli import run_cli

ROOT = Path(__file__).resolve().parent.parent


def committed(path):
    """A committed document as the generator would write it: line endings as in the generator's (a Windows checkout may have CRLF)."""
    return (ROOT / path).read_text(encoding="utf-8").replace("\r\n", "\n")


def without_version(text):
    return text.replace(provenance.library_version(), "VERSION")


class TestTheDocumentsMatchTheProgram(unittest.TestCase):
    def test_the_markdown_reference_is_up_to_date(self):
        self.assertEqual(without_version(committed("CLI_REFERENCE.md")).rstrip(), without_version(docs.markdown()).rstrip(),
                         "CLI_REFERENCE.md is out of date: python -m PyATCLI docs --format markdown --out CLI_REFERENCE.md")

    def test_the_man_page_is_up_to_date(self):
        self.assertEqual(without_version(committed("man/pyatcli.1")).rstrip(), without_version(docs.man()).rstrip(),
                         "man/pyatcli.1 is out of date: python -m PyATCLI docs --format man --out man/pyatcli.1")

    def test_every_command_and_option_is_documented_in_both(self):
        top = docs.tree()
        md, page = docs.markdown(), docs.man()
        for node in top.walk():
            if not node.name:
                continue
            self.assertIn(f"## {node.name}", md, node.name)
            self.assertIn(node.name.upper().replace("-", "\\-"), page, node.name)
            for action, _ in node.own_options():
                for flag in action.option_strings:
                    self.assertIn(flag, md, f"{node.name} {flag}")
                    self.assertIn(flag.replace("-", "\\-"), page, f"{node.name} {flag}")

    def test_every_command_of_the_concise_help_is_in_the_parser(self):
        _, out, _ = run_cli()
        listed = re.findall(r"^  (\w+)  ", out.split("commands:")[1].split("examples:")[0], flags=re.M)
        parser_commands = {n.name for n in docs.tree().children}
        self.assertEqual(set(listed), parser_commands)

    def test_the_global_options_are_in_the_documents_once_not_per_command(self):
        md = docs.markdown()
        self.assertEqual(md.count("`--no-input`"), 1)
        self.assertEqual(md.count("`--events`"), 1)


class TestTheManPage(unittest.TestCase):
    def test_it_has_the_sections_a_man_page_has(self):
        page = docs.man()
        self.assertTrue(page.startswith(".TH PYATCLI 1"))
        for section in ("NAME", "SYNOPSIS", "DESCRIPTION", "COMMANDS", "GLOBAL OPTIONS", "EXIT STATUS", "ENVIRONMENT", "SEE ALSO"):
            self.assertIn(f".SH {section}\n", page)

    def test_no_text_line_can_be_taken_for_a_macro(self):
        known = {".TH", ".SH", ".TP", ".B", ".I", ".RI", ".br", ".PP", ".nf", ".fi"}
        for line in docs.man().splitlines():
            if line.startswith((".", "'")):
                self.assertIn(line.split()[0], known, line)

    def test_a_backslash_or_a_hyphen_cannot_break_the_page(self):
        page = docs.man()
        self.assertNotRegex(page, r"(?<!\\)\\(?![e\-&])")               # every backslash starts an escape roff knows
        self.assertNotIn("--json", page)                                 # (hyphens are written \-)

    @unittest.skipUnless(shutil.which("groff") or shutil.which("mandoc"), "no groff or mandoc here")
    def test_groff_renders_it_without_a_warning(self):
        tool = [shutil.which("mandoc"), "-Tlint"] if shutil.which("mandoc") else [shutil.which("groff"), "-man", "-ww", "-z"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pyatcli.1")
            with open(path, "w", encoding="utf-8") as f:
                f.write(docs.man())
            done = subprocess.run(tool + [path], capture_output=True, text=True)
        self.assertEqual(done.stderr.strip(), "")


class TestTheDocsCommand(unittest.TestCase):
    def test_it_prints_the_markdown_on_stdout_by_default(self):
        status, out, err = run_cli("docs")
        self.assertEqual((status, err), (0, ""))
        self.assertEqual(out.rstrip(), docs.markdown().rstrip())

    def test_the_man_page_goes_to_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pyatcli.1")
            status, out, _ = run_cli("docs", "--format", "man", "--out", path)
            self.assertEqual(status, 0)
            self.assertIn(f"wrote {path}", out)
            self.assertEqual(Path(path).read_text(encoding="utf-8").rstrip(), docs.man().rstrip())
            self.assertNotIn("\r", Path(path).read_bytes().decode("utf-8"))      # the same bytes on every platform

    def test_json_wraps_the_text(self):
        status, out, _ = run_cli("docs", "--format", "man", "--json")
        doc = json.loads(out)
        self.assertEqual((status, doc["command"], doc["result"]["format"]), (0, "docs", "man"))
        self.assertTrue(doc["result"]["text"].startswith(".TH"))

    def test_the_documents_are_in_the_commands_help_too(self):
        status, out, _ = run_cli("docs", "--help")
        self.assertEqual(status, 0)
        self.assertIn("examples:", out)
        self.assertIn("man ./pyatcli.1", out)


class TestServeIsDocumented(unittest.TestCase):
    def test_serve_has_help_and_an_example(self):
        status, out, _ = run_cli("serve", "--help")
        self.assertEqual(status, 0)
        self.assertIn("--data-dir", out)
        self.assertIn("examples:", out)

    def test_the_stub_in_the_parser_is_the_servers_own_options(self):
        from PyATCLI.engine import server
        import argparse
        parser_flags = {f for a in build_parser()._subparsers._group_actions[0].choices["serve"]._actions for f in a.option_strings}
        for flag in ("--data-dir", "--app-name", "--app-version"):
            self.assertIn(flag, parser_flags)
        self.assertTrue(callable(server.main))


if __name__ == "__main__":
    unittest.main()
