"""
ops
---
The operations layer: everything the command line and the engine (`python -m PyATCLI serve`, which PyAT's window talks to) can
ask of the library, as functions of plain, JSON-ready parameters that return a plain, JSON-ready result.

    result = ops.call("run", {"kind": "event_peth", "recording": "D:/rats/Rat7", "parameters": {"event": "L1P¹"}, "out": "out"})

An operation is the one place that decides what a request means and what comes back. The command line is argparse in front of it and a
renderer (human text, `--plain` tables, `--json`) behind it; the engine is a JSON-lines transport in front of the same functions, with the
loaded recordings kept between calls. The two cannot disagree, because neither holds any logic of its own.

    @operation("run", "run one analysis on a recording")
    def run(ctx, *, kind, recording, parameters=None, ...): ...

An operation takes an `OpContext` and its parameters by keyword and returns a dict. While it works it tells the context what is going on —
`ctx.progress()` (the `progress=` argument of the library's slow functions), `ctx.warn(text)`, `ctx.note(text)` — and the front end shows
that however it likes (a bar, JSON lines, a toast). It reads recordings through `ctx.load(...)`, which is where the engine's recordings
live between calls; a one-shot run opens each fresh.

Errors are `OpError`s with a stable `code` (`error_info` turns any exception into one), so a caller can tell a mistake it can show (a bad
parameter) from a bug. Nothing here prints, reads sys.argv, or touches a window.

    ops.names()              every operation, in the order they are documented
    ops.describe()           {name: {"summary", "parameters": [...]}} — what `serve`'s `help` method returns
    ops.call(name, params, ctx=None)

The result of every operation is JSON-ready (numbers, strings, lists, dicts, None): `to_jsonable` makes it so, with NaN as null.
"""

import inspect
import math
import os

from ..progress import Cancelled
from ..analyses import AnalysisError, ParameterError
from ..reproduce import ReproduceError

# The engine's protocol: what its callers can count on (the operations and their parameters, the analyses, their tables and the columns that are
# always there — ENGINE_CONTRACT.json lists them, engine/contract.py explains what counts as breaking). Raise it when something a caller could rely
# on is removed or renamed; adding is not a reason. tests/test_engine_contract.py fails until the baseline of the new number is added.
PROTOCOL_VERSION = 1

_OPERATIONS = {}


class OpError(Exception):
    """A request that cannot be done, for a reason a person can act on. `code` is stable (see ERROR_CODES), `hint` says what to try."""

    def __init__(self, message, code="invalid_input", hint=None):
        super().__init__(message)
        self.code = code
        self.hint = hint


ERROR_CODES = {
    "bad_parameter": "a parameter is missing, unknown, or has a value the analysis cannot use",
    "invalid_input": "something in the request is not usable (a file that is not a recording, a value out of range)",
    "not_found": "a file, folder or recording the request names is not there",
    "io_error": "a file could not be read or written",
    "analysis_failed": "the analysis ran and could not give a result",
    "cannot_reproduce": "a provenance record cannot be replayed",
    "unknown_operation": "there is no such operation",
    "no_recording": "the request names a recording the engine has not opened",
    "confirmation_required": "the request would delete or replace something and was not told to go ahead",
    "cancelled": "the request was cancelled before it finished",
    "internal_error": "a bug: something unexpected went wrong",
}


def error_info(exc):
    """{"code", "message", "hint", "type"} for any exception — the one place an error becomes something a caller can show."""
    if isinstance(exc, Cancelled):
        return {"code": "cancelled", "message": "cancelled", "hint": None, "type": "Cancelled"}
    if isinstance(exc, OpError):
        return {"code": exc.code, "message": str(exc), "hint": exc.hint, "type": type(exc).__name__}
    message = str(exc) or type(exc).__name__
    if isinstance(exc, ParameterError):
        code = "bad_parameter"
    elif isinstance(exc, AnalysisError):
        code = "analysis_failed"
    elif isinstance(exc, ReproduceError):
        code = "cannot_reproduce"
    elif isinstance(exc, FileNotFoundError):
        code = "not_found"
    elif isinstance(exc, KeyError):
        code, message = "invalid_input", f"missing {message}"
    elif isinstance(exc, ValueError):
        code = "invalid_input"
    elif isinstance(exc, OSError):
        code = "io_error"
    else:
        return {"code": "internal_error", "message": f"{type(exc).__name__}: {message}", "type": type(exc).__name__,
                "hint": "this is a bug; it helps to say what was asked (re-run with --debug for the traceback)"}
    return {"code": code, "message": message, "hint": None, "type": type(exc).__name__}


def is_expected(exc):
    """True for the errors a person can act on (everything error_info does not call a bug)."""
    return error_info(exc)["code"] != "internal_error"


def to_jsonable(obj):
    """`obj` as plain JSON values: numpy numbers and arrays, tuples, sets and paths become Python ones, and NaN / infinity become None
    (strict JSON has no such numbers)."""
    import numpy as np

    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int,)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# The context an operation runs in
# ---------------------------------------------------------------------------

class OpContext:
    """What an operation is told about where it runs, and how it tells anyone what it is doing.

    emit        `emit(event, **fields)` — called for "progress" (fraction, message), "warning" (text), "note" (text) and "state" (path: a
                file was written). Default: nobody is listening.
    cancelled   `cancelled() -> bool`: True once the caller wants the work stopped; the next progress report raises Cancelled
    loader      `loader(path, **options) -> load_recording's dict`; the engine's keeps recordings between calls
    store       the engine's recording store (None for a one-shot run): operations that name a recording by id use it
    data_dir    where bulk results (arrays, big tables) are written for the caller to map; None: they are not
    app         {"name", "version"} of the program asking, written into the provenance records of what is made
    """

    def __init__(self, emit=None, cancelled=None, loader=None, store=None, data_dir=None, wants_progress=True, app=None):
        self.app = dict(app or {"name": "PyATCLI command line"})          # the program in a provenance record's software section
        self._emit = emit
        self._cancelled = cancelled
        self._loader = loader
        self.store = store
        self.data_dir = data_dir
        self.wants_progress = wants_progress and emit is not None
        self.warnings = []

    def emit(self, event, **fields):
        if self._emit is not None:
            self._emit(event, **fields)

    def warn(self, text):
        self.warnings.append(str(text))
        self.emit("warning", text=str(text))

    def note(self, text):
        self.emit("note", text=str(text))

    def written(self, path):
        """Tell the caller a file was written (a change of state the user should hear about)."""
        self.emit("state", path=os.fspath(path))

    def check_cancel(self):
        if self._cancelled is not None and self._cancelled():
            raise Cancelled()

    def progress(self):
        """The `progress=` argument for the library's slow functions: a callable(fraction, message) that reports through `emit` and
        stops the work (Cancelled) when the caller has cancelled — or None when nobody is listening and nothing can be cancelled."""
        if not self.wants_progress and self._cancelled is None:
            return None

        def report(fraction, message=""):
            if self.wants_progress:
                self.emit("progress", fraction=float(fraction), message=str(message))
            self.check_cancel()

        return report

    def load(self, path, *, progress=None, **options):
        """A recording as load_recording returns it ({"cache", "original_cache", "splices", "warnings"}); `options` are its own
        (regression_method, use_saved, table, x_column, y_columns). Its warnings are passed on (ctx.warn). `progress` is where the
        reading reports, if not the request's own progress (a stage of a longer request: see ops/analysis.py)."""
        progress = progress if progress is not None else self.progress()
        if self._loader is not None:
            loaded = self._loader(path, progress=progress, **options)
        else:
            from ..recording import load_recording
            loaded = load_recording(path, progress=progress, **options)
        for w in loaded.get("warnings", []):
            self.warn(w)
        return loaded


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

class Operation:
    def __init__(self, name, fn, summary, result=()):
        self.name, self.fn, self.summary = name, fn, summary
        self.result = tuple(result)               # the keys of the result that are always there (see ENGINE_CONTRACT.json); more may be
        sig = inspect.signature(fn)
        self.parameters = [p for n, p in sig.parameters.items() if n != "ctx"]

    def describe(self):
        return {"summary": self.summary,
                "parameters": [{"name": p.name, "required": p.default is inspect.Parameter.empty,
                                "default": None if p.default is inspect.Parameter.empty else to_jsonable(p.default)}
                               for p in self.parameters],
                "result": list(self.result)}


def operation(name, summary, result=()):
    """Registers `fn(ctx, **parameters) -> dict` as the operation `name`. `result` lists the keys of what it returns that are ALWAYS there: a
    caller may count on them (more can be present, and more can be added; removing one is a breaking change, see engine/contract.py)."""
    def register(fn):
        _OPERATIONS[name] = Operation(name, fn, summary, result)
        return fn
    return register


def names():
    _load_all()
    return list(_OPERATIONS)


def describe():
    _load_all()
    return {n: op.describe() for n, op in _OPERATIONS.items()}


def get(name):
    _load_all()
    try:
        return _OPERATIONS[name]
    except KeyError:
        raise OpError(f"there is no operation {name!r}; the operations are: {', '.join(_OPERATIONS)}", "unknown_operation") from None


def call(name, params=None, ctx=None):
    """Runs operation `name` with `params` (a dict of its keyword parameters) and returns its result, made JSON-ready. An unknown or
    missing parameter is a ParameterError naming it."""
    op = get(name)
    params = dict(params or {})
    valid = {p.name for p in op.parameters}
    unknown = sorted(set(params) - valid)
    if unknown:
        raise ParameterError(f"{name}: unknown parameter(s) {', '.join(unknown)}; valid: {', '.join(sorted(valid))}")
    missing = [p.name for p in op.parameters if p.default is inspect.Parameter.empty and p.name not in params]
    if missing:
        raise ParameterError(f"{name}: {', '.join(missing)} {'is' if len(missing) == 1 else 'are'} required")
    result = to_jsonable(op.fn(ctx or OpContext(), **params))
    if os.environ.get("PYATCLI_CHECK_RESULTS", "").strip().lower() not in ("", "0", "false", "no", "off"):
        missing = [k for k in op.result if k not in result]
        if missing:                                                    # the tests (and a developer) run like this: a promise broken is loud
            raise AssertionError(f"{name}: the result lacks {', '.join(missing)}, which the operation's contract promises (ops: result=...)")
    return result


_loaded = False


def _load_all():
    """Imports the modules that register operations (done on first use, so `import PyATCLI.ops` stays light)."""
    global _loaded
    if not _loaded:
        from . import analysis, compute, fieldstudy, files, group, records, recording, results, saved, system  # noqa: F401,E402 — importing each registers its operations
        _loaded = True
