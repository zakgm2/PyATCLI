"""
_version.py
-----------
Which version of this library is running — the one place that decides, so the package's
`__version__`, the app's error-log header and every provenance record agree.

An installed copy reports its distribution's version. A copy running from a source checkout (a
pyproject.toml for this project sits beside the package) reports that checkout's own version
instead: an editable install's metadata is only refreshed by re-running pip, so it quietly goes
stale (a checkout at 2026.10.2 kept reporting 2026.8.20), and a result stamped with the wrong
library version is worse than one with none.
"""

import os
import re
from importlib.metadata import PackageNotFoundError, version as _dist_version

DISTRIBUTION = "PyATCLI"


def source_tree_version():
    """The version in the pyproject.toml beside the package, if this is a source checkout of this
    project; None for an installed copy (wheels carry no pyproject)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(here, "pyproject.toml"), encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    if not re.search(r'^name\s*=\s*"' + re.escape(DISTRIBUTION) + '"', text, re.M):
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def installed_version():
    """The installed distribution's version, or None if it isn't installed."""
    try:
        return _dist_version(DISTRIBUTION)
    except PackageNotFoundError:
        return None


def resolve_version():
    """The running version: the source checkout's if running from one, else the installed
    distribution's, else "unknown"."""
    return source_tree_version() or installed_version() or "unknown"
