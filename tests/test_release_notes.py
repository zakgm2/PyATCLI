"""The checks the release workflow makes before anything is uploaded (.github/scripts/release_notes.py): the tag, pyproject.toml, CITATION.cff and CHANGELOG.md
agree, and the changelog's section becomes the GitHub Release's notes. The last test runs them on the repository as it is."""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, ".github", "scripts", "release_notes.py")

spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)

PYPROJECT = '[project]\nname = "PyATCLI"\nversion = "2026.11.5"\n'
CFF = 'cff-version: 1.2.0\ntitle: "PyATCLI"\nversion: "2026.11.5"\ndate-released: "2026-11-05"\n'
CHANGELOG = """# Changelog

---

## Version 2026.11.5 — PyATCLI:
  **A fix.** It reads back exactly.
  - one
  - two

---

## Version 2026.11.50 — PyATCLI:
  not this one

---

## Version 2026.9.28:
  An older heading, without the name.

---
"""


class TestTheSection(unittest.TestCase):
    def test_a_section_runs_to_the_next_rule_and_is_dedented(self):
        self.assertEqual(release_notes.changelog_section(CHANGELOG, "2026.11.5"), "**A fix.** It reads back exactly.\n- one\n- two")

    def test_a_longer_version_is_not_mistaken_for_it(self):
        self.assertEqual(release_notes.changelog_section(CHANGELOG, "2026.11.50"), "not this one")
        self.assertIsNone(release_notes.changelog_section(CHANGELOG, "2026.11.0"))

    def test_the_older_heading_without_the_name_works(self):
        self.assertEqual(release_notes.changelog_section(CHANGELOG, "2026.9.28"), "An older heading, without the name.")


class TestTheChecks(unittest.TestCase):
    def test_everything_agreeing_gives_the_notes(self):
        problems, notes = release_notes.check("v2026.11.5", PYPROJECT, CFF, CHANGELOG)
        self.assertEqual(problems, [])
        self.assertTrue(notes.startswith("**A fix.**"))
        self.assertIn("pip install PyATCLI==2026.11.5", notes)
        self.assertIn("https://pypi.org/project/PyATCLI/2026.11.5/", notes)

    def test_a_tag_that_is_not_a_version(self):
        for tag in ("2026.11.5", "v2026.11", "release-1", "v2026.11.5-rc1"):
            problems, notes = release_notes.check(tag, PYPROJECT, CFF, CHANGELOG)
            self.assertEqual(len(problems), 1, tag)
            self.assertIsNone(notes)

    def test_each_file_that_disagrees_is_named(self):
        problems, _ = release_notes.check("v2026.11.5", PYPROJECT.replace("2026.11.5", "2026.11.2"), CFF, CHANGELOG)
        self.assertEqual(len(problems), 1)
        self.assertIn("pyproject.toml says 2026.11.2", problems[0])
        problems, _ = release_notes.check("v2026.11.5", PYPROJECT, CFF.replace("2026.11.5", "2026.11.2"), CHANGELOG)
        self.assertIn("CITATION.cff says 2026.11.2", problems[0])
        problems, _ = release_notes.check("v2026.11.6", PYPROJECT.replace("2026.11.5", "2026.11.6"), CFF.replace("2026.11.5", "2026.11.6"), CHANGELOG)
        self.assertIn("no '## Version 2026.11.6' section", problems[0])

    def test_all_the_problems_are_reported_together(self):
        problems, notes = release_notes.check("v2026.11.7", PYPROJECT, CFF, CHANGELOG)
        self.assertEqual(len(problems), 3)
        self.assertIsNone(notes)

    def test_the_script_writes_the_notes_file_and_returns_nonzero_on_a_problem(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, text in (("pyproject.toml", PYPROJECT), ("CITATION.cff", CFF), ("CHANGELOG.md", CHANGELOG)):
                with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
                    f.write(text)
            out = os.path.join(folder, "notes.md")
            here = os.getcwd()
            os.chdir(folder)
            printed = io.StringIO()      # the script prints GitHub `::error::` lines, which the runner would show as annotations on a green run
            try:
                with contextlib.redirect_stdout(printed):
                    self.assertEqual(release_notes.main("v2026.11.5", out), 0)
                    with open(out, encoding="utf-8") as f:
                        self.assertTrue(f.read().startswith("**A fix.**"))
                    self.assertEqual(release_notes.main("v2026.11.9", os.path.join(folder, "never.md")), 1)
                self.assertFalse(os.path.exists(os.path.join(folder, "never.md")))
                self.assertEqual(printed.getvalue().count("::error::"), 3)
            finally:
                os.chdir(here)


class TestThisRepository(unittest.TestCase):
    def test_the_version_in_the_repository_could_be_released_as_it_stands(self):
        """A release that would fail the workflow's checks (CITATION.cff or the changelog left behind) fails here first."""
        def read(name):
            with open(os.path.join(ROOT, name), encoding="utf-8") as f:
                return f.read()

        version = release_notes.project_version(read("pyproject.toml"))
        problems, notes = release_notes.check(f"v{version}", read("pyproject.toml"), read("CITATION.cff"), read("CHANGELOG.md"))
        self.assertEqual(problems, [])
        self.assertGreater(len(notes), 100)


if __name__ == "__main__":
    unittest.main()
