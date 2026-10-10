"""
cli
---
The command line: `python -m PyATCLI <command>` (and the `pyatcli` command pip installs).

    run         run one analysis on a recording; writes CSV tables and a provenance record
    reproduce   run an analysis again from its .provenance.json and check the numbers
    batch       many analyses over many recordings (a folder of TDT blocks counts as many)
    group       group analysis: `group setup` (the wizard: find the recordings, list their markers, write analysis_settings.json) and `group run`
    saved       what the app's Save Changes saves, from a script: show, marker, unmark, splice, unsplice, rename, highlight, text, clear
    kinds       the analyses, and every parameter of each
    info        what is installed (versions: what a result would be stamped with)
    serve       the engine: the same operations as JSON lines on stdin/stdout, with recordings kept open (what PyAT's window talks to)

The command line is a thin front over ops/ — argparse in front of an operation, a renderer (human, --plain, --json) behind it — and the
engine is the same operations behind a different transport, so they cannot disagree; see ops/__init__.py.

Every command takes the same global flags, before or after its name: --json, --plain, -q, -v, -d, --no-input, --events, --progress.

Output: stdout is only ever the result (--json for a program, --plain for rows); warnings, errors, progress and anything a library prints
itself go to stderr or are dropped. Exit status: 0 done; 1 something failed (the message is on stderr); 2 bad usage; 3 `reproduce` ran but
an export did not match; 130 interrupted.
"""

import os
import sys

from .. import ops
from ..ops import error_info
from .commands import EXIT_ERROR, EXIT_MISMATCH, EXIT_OK, EXIT_USAGE, APP, collect_parameters, parse_value, recording_name   # noqa: F401 — the module's public names
from .parser import build_parser, concise_help
from .render import Discard, Output

INTERRUPTED = 130


def _command_index(argv):
    """Where the command name is: the first token that is not a flag (the global flags take no value)."""
    return next((n for n, token in enumerate(argv) if not token.startswith("-")), None)


def _legacy(argv):
    """The old spellings, rewritten (with a warning on stderr): `group-setup ...` is `group setup ...` and `group SETTINGS ...` is `group run
    SETTINGS ...`. Returns (argv, [warnings])."""
    argv = list(argv)
    i = _command_index(argv)
    if i is None:
        return argv, []
    if argv[i] == "group-setup":
        argv[i:i + 1] = ["group", "setup"]
        return argv, ["`group-setup` is now `group setup` (the old name still works)"]
    if argv[i] == "group" and i + 1 < len(argv) and not argv[i + 1].startswith("-") and argv[i + 1] not in ("setup", "run"):
        argv.insert(i + 1, "run")
        return argv, ["`group SETTINGS` is now `group run SETTINGS` (the old form still works)"]
    return argv, []


def _truthy(value):
    return bool(value) and value.strip().lower() not in ("0", "false", "no", "off")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for stream in (sys.stdout, sys.stderr):                  # a console that cannot show ¹ or ⁰ must not kill the run
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    argv, legacy = _legacy(argv)
    if not argv:
        print(concise_help(), end="")
        return EXIT_OK
    i = _command_index(argv)
    if i is not None and argv[i] == "serve" and not {"-h", "--help"} & set(argv):          # (its help is the parser's: with examples)
        from ..engine.server import main as serve_main
        return serve_main(argv[:i] + argv[i + 1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:                                 # only global flags were given
        print(concise_help(), end="")
        return EXIT_OK
    real_stdout, real_stderr = sys.stdout, sys.stderr
    out = Output(args, real_stdout, real_stderr)
    args.debug = args.debug or _truthy(os.environ.get("DEBUG", ""))
    out.debug = args.debug
    for text in legacy:
        out.warn(text)
    sys.stdout = real_stderr if out.verbose else Discard()   # what a library prints on its own must not mix into the result
    try:
        return args.fn(args, out)
    except (KeyboardInterrupt, ops.Cancelled):
        out.error("interrupted")
        return INTERRUPTED
    except Exception as e:                                   # noqa: BLE001 — every error becomes a message a person can read
        if args.debug:
            raise
        info = error_info(e)
        out.fail(info)
        return EXIT_USAGE if info["code"] == "confirmation_required" else EXIT_ERROR
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    sys.exit(main())
