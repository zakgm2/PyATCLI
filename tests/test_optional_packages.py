"""The optional packages (PyATCLI/_optional.py): `pip install PyATCLI` is the core, and each file type that needs a package of its own has an extra
(`PyATCLI[tdt]`, `[excel]`, `[fieldstudy]`, and `[all]`; `[oxysoft]`, `[csv]` and `[pt2]` need nothing). The library must import and run without them and ask
for a missing one by name.

This file must not import tdt (or the test modules that do): the CI job `core-install` runs it on an install that has none of them."""

import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import PyATCLI as pl
from PyATCLI import _optional, cli, ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def without(*modules):
    """The named packages behave as if they were not installed (`import x` raises ModuleNotFoundError), submodules already imported by an earlier test included."""
    blocked = {name: None for name in modules}
    blocked.update({name: None for name in sys.modules if name.split(".")[0] in modules})
    return mock.patch.dict(sys.modules, blocked)


def pyproject_section(name):
    with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    match = re.search(rf"^\[{re.escape(name)}\]\n(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""


def extras_in_pyproject():
    """{extra: set of package names} as pyproject.toml lists them."""
    found = {}
    for extra, body in re.findall(r"(?m)^(\w+)\s*=\s*\[(.*?)\]", pyproject_section("project.optional-dependencies")):
        found[extra] = set(re.findall(r'"([^"]+)"', body))
    return found


class TestRequire(unittest.TestCase):
    def test_a_missing_package_is_named_with_the_pip_line_that_installs_it(self):
        for extra, modules in _optional.EXTRAS.items():
            for module in modules:
                with without(_optional.top_level(module)):
                    with self.assertRaises(_optional.MissingOptionalPackage) as caught:
                        _optional.require(extra, module)
                message = str(caught.exception)
                self.assertIn(f"'{_optional.pip_name(module)}'", message, module)
                self.assertIn(f'pip install "PyATCLI[{extra}]"', message, module)
                self.assertIn("PyATCLI[all]", message, module)

    def test_scikit_learn_is_named_by_its_pip_name_not_its_import_name(self):
        with without("sklearn"), self.assertRaises(_optional.MissingOptionalPackage) as caught:
            _optional.require("tdt", "sklearn.linear_model")
        self.assertIn("'scikit-learn'", str(caught.exception))
        self.assertNotIn("'sklearn'", str(caught.exception))

    def test_it_is_an_import_error_so_code_that_catches_one_still_works(self):
        with without("tdt"), self.assertRaises(ImportError):
            _optional.require("tdt")

    def test_a_package_that_is_installed_but_broken_keeps_its_own_error(self):
        """An import that fails for another reason (a missing dependency of the package) is not "extra not installed"."""
        broken = ImportError("libfoo.so is missing", name="libfoo")
        with mock.patch.object(_optional.importlib, "import_module", side_effect=broken):
            with self.assertRaises(ImportError) as caught:
                _optional.require("tdt")
        self.assertIs(type(caught.exception), ImportError)

    def test_available_says_what_is_installed_without_importing_it(self):
        with without("tdt"):
            self.assertFalse(_optional.available()["tdt"])
        self.assertEqual(set(_optional.available()), set(_optional.EXTRAS))
        for extra in ("oxysoft", "csv", "pt2"):
            self.assertTrue(_optional.available()[extra], extra)         # they need nothing

    def test_info_reports_which_readers_can_be_imported(self):
        result = ops.call("info", {})
        self.assertEqual(set(result["optional"]), set(_optional.EXTRAS))
        self.assertTrue(all(isinstance(v, bool) for v in result["optional"].values()))


class TestTheLibraryWithoutThem(unittest.TestCase):
    def test_importing_the_library_imports_none_of_them(self):
        code = ("import sys\nimport PyATCLI\n"
                "loaded = [m for m in ('tdt', 'sklearn', 'openpyxl', 'sentence_transformers', 'torch') if m in sys.modules]\n"
                "print(loaded)\nsys.exit(1 if loaded else 0)\n")
        done = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT)
        self.assertEqual(done.returncode, 0, f"importing PyATCLI imported an optional package: {done.stdout} {done.stderr[-400:]}")

    def test_reading_a_tdt_tank_says_how_to_install_the_reader(self):
        with tempfile.TemporaryDirectory() as parent:
            folder = os.path.join(parent, "Rat7-250101-101010")
            os.makedirs(folder)
            with open(os.path.join(folder, "Rat7-250101-101010.Tbk"), "wb") as f:
                f.write(b"block")
            with without("tdt"):
                with self.assertRaises(_optional.MissingOptionalPackage) as caught:
                    pl.load_recording(folder)
        self.assertIn('pip install "PyATCLI[tdt]"', str(caught.exception))

    def test_a_robust_motion_correction_fit_asks_for_scikit_learn(self):
        from PyATCLI.sources import processing_TDT
        with without("sklearn"), self.assertRaises(_optional.MissingOptionalPackage) as caught:
            processing_TDT._linear_model()
        self.assertIn('pip install "PyATCLI[tdt]"', str(caught.exception))

    def test_the_command_line_reports_it_as_something_to_install_not_as_a_bug(self):
        with tempfile.TemporaryDirectory() as parent:
            folder = os.path.join(parent, "Rat7-250101-101010")
            os.makedirs(folder)
            with open(os.path.join(folder, "Rat7-250101-101010.Tbk"), "wb") as f:
                f.write(b"block")
            out, err = io.StringIO(), io.StringIO()
            with without("tdt"), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = cli.main(["run", "process", folder])
        self.assertEqual(status, 1)
        self.assertIn('pip install "PyATCLI[tdt]"', err.getvalue())
        self.assertNotIn("this is a bug", err.getvalue())

    def test_the_error_code_is_its_own_and_not_a_bug(self):
        info = ops.error_info(_optional.MissingOptionalPackage("needs a package"))
        self.assertEqual(info["code"], "missing_package")
        self.assertTrue(ops.is_expected(_optional.MissingOptionalPackage("x")))
        self.assertIn("missing_package", ops.ERROR_CODES)

    def test_an_excel_workbook_asks_for_the_excel_extra(self):
        with tempfile.TemporaryDirectory() as parent:
            path = os.path.join(parent, "book.xlsx")
            with open(path, "wb") as f:
                f.write(b"PK")
            with without("openpyxl"):
                with self.assertRaises(_optional.MissingOptionalPackage) as caught:
                    from PyATCLI.sources import tables
                    tables._parse_excel(path)
        self.assertIn('pip install "PyATCLI[excel]"', str(caught.exception))


class TestThePackaging(unittest.TestCase):
    def test_the_core_does_not_ask_for_the_optional_packages(self):
        core = pyproject_section("project")
        dependencies = re.search(r"^dependencies\s*=\s*\[(.*?)\]", core, re.MULTILINE | re.DOTALL).group(1)
        for package in ("tdt", "scikit-learn", "openpyxl", "sentence-transformers"):
            self.assertNotIn(f'"{package}"', dependencies, f"{package} belongs in an extra")
        for needed in ("numpy", "scipy", "pandas", "statsmodels", "tqdm"):
            self.assertIn(needed, dependencies)

    def test_there_is_an_extra_for_each_file_type_and_it_installs_what_the_library_asks_for_by_name(self):
        listed = extras_in_pyproject()
        self.assertEqual(set(listed), set(_optional.EXTRAS) | {"all"})
        for extra, modules in _optional.EXTRAS.items():
            self.assertEqual(listed[extra], {_optional.pip_name(m) for m in modules}, f"extra {extra}")

    def test_all_is_every_package_of_every_extra(self):
        everything = set()
        for modules in _optional.EXTRAS.values():
            everything |= {_optional.pip_name(m) for m in modules}
        self.assertEqual(extras_in_pyproject()["all"], everything)


if __name__ == "__main__":
    unittest.main()
