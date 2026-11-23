"""
_optional.py
------------
The packages that are not installed with the library itself, one pip "extra" per file type, and the error a missing one gives. The core install does the
analyses, the group analysis, the command line and the engine; each file type that needs a package of its own is asked for by name:

    pip install PyATCLI                   the core
    pip install "PyATCLI[tdt]"            + TDT tanks: the `tdt` package (GPL-3.0) and scikit-learn (the Huber and RANSAC motion-correction fits)
    pip install "PyATCLI[excel]"          + Excel workbooks: openpyxl
    pip install "PyATCLI[fieldstudy]"     + the text field study's language-model comparison: sentence-transformers (pulls in torch: about 650 MB installed on Windows, several GB on Linux, where torch brings NVIDIA CUDA packages)
    pip install "PyATCLI[all]"            all of them

`oxysoft`, `csv` and `pt2` need nothing beyond the core: those extras exist (and install nothing) so that a requirements file can name every format a project
reads. Several at once: `pip install "PyATCLI[tdt,excel]"`.

A missing package is found when it is first needed, never at import, so `import PyATCLI` works on the core install. A frozen program (PyAT.exe) cannot see these
imports by reading the code, so PhysicsAnalysis.spec in the PyAT repository names them; `info` reports which are importable, and PyAT's build checks that.
tests/test_optional_packages.py keeps the extras in pyproject.toml, this table and the messages in step.
"""

import importlib
import importlib.util

# extra -> {module it needs: what that is needed for}; an empty one needs nothing beyond the core
EXTRAS = {
    "tdt": {"tdt": "reading TDT recordings", "sklearn.linear_model": "motion correction by Huber or RANSAC regression"},
    "excel": {"openpyxl": "reading Excel (.xlsx) files"},
    "oxysoft": {},
    "csv": {},
    "pt2": {},
    "fieldstudy": {"sentence_transformers": "the text field study's language-model comparison"},
}
# the name pip knows a module's package by, where it differs from the import name (keyed by the module's top-level name)
PIP_NAMES = {"sklearn": "scikit-learn", "sentence_transformers": "sentence-transformers"}


class MissingOptionalPackage(ImportError):
    """An optional package is not installed; the message says which extra installs it."""


def top_level(module):
    return module.split(".")[0]


def pip_name(module):
    return PIP_NAMES.get(top_level(module), top_level(module))


def install_hint(extra):
    return f'pip install "PyATCLI[{extra}]"'


def require(extra, module=None):
    """A module an extra provides (its first by default), imported now. MissingOptionalPackage, naming the pip line, when its package is not installed (a
    package that is installed but fails to import for another reason keeps its own error: that is a broken install, not a missing extra)."""
    modules = EXTRAS[extra]
    module = module or next(iter(modules))
    what = modules[module]
    try:
        return importlib.import_module(module)
    except ImportError as error:
        if getattr(error, "name", None) not in (module, top_level(module)):
            raise
        raise MissingOptionalPackage(
            f"{what[0].upper()}{what[1:]} needs the optional package '{pip_name(module)}', which is not installed. Install it with:  {install_hint(extra)}"
            f'  (or "PyATCLI[all]" for every optional package)') from None


def available():
    """{extra: whether everything it needs is installed}, without importing any of it. An extra that needs nothing is always available."""
    return {extra: all(importlib.util.find_spec(top_level(module)) is not None for module in modules) for extra, modules in EXTRAS.items()}
