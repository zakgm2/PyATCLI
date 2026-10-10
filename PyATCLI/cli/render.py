"""
cli/render.py
-------------
How a command's result reaches the person (or the program) that ran it. One `Output` per run:

    human    the default: short lines on stdout saying what was done, aligned tables, "wrote ..." for every file written
    --plain  tab-separated rows and nothing else on stdout: for `cut`, `awk`, a spreadsheet
    --json   one JSON document on stdout — {"schema": 1, "ok": true, "command": ..., "result": {...}, "warnings": [...]} — and nothing else

Whatever the mode, stdout is only ever the result: warnings, errors, progress and anything a library prints on its own go to stderr (or are
dropped), so `pyatcli run ... --json | jq` always sees valid JSON. Colour is never used, so NO_COLOR has nothing to turn off.
"""

import json
import sys

from ..ops import to_jsonable

SCHEMA = 1


class Output:
    def __init__(self, args, stdout, stderr):
        self.json = bool(getattr(args, "json", False))
        self.plain = bool(getattr(args, "plain", False))
        self.quiet = bool(getattr(args, "quiet", False))
        self.verbose = int(getattr(args, "verbose", 0) or 0)
        self.debug = bool(getattr(args, "debug", False))
        self.no_input = bool(getattr(args, "no_input", False))
        self.stdout, self.stderr = stdout, stderr
        self.warnings = []

    # ---- what the person reads ---------------------------------------------------------------------------------------------------------

    @property
    def human(self):
        return not (self.json or self.plain)

    def say(self, text=""):
        """A line of prose: stdout, in human mode, unless --quiet."""
        if self.human and not self.quiet:
            print(text, file=self.stdout)

    def detail(self, text):
        """A line only --verbose wants (stderr, so it never mixes with the result)."""
        if self.verbose:
            print(text, file=self.stderr)

    def table(self, header, rows, indent="  "):
        """Rows under a header. Human: aligned columns. --plain: tab-separated, no header. --json: nothing (the result carries the data)."""
        rows = [[("" if v is None else str(v)) for v in row] for row in rows]
        if self.plain:
            for row in rows:
                print("\t".join(row), file=self.stdout)
        elif self.human and not self.quiet:
            widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(header)]
            print(indent + "  ".join(str(h).ljust(w) for h, w in zip(header, widths)).rstrip(), file=self.stdout)
            for row in rows:
                print(indent + "  ".join(v.ljust(w) for v, w in zip(row, widths)).rstrip(), file=self.stdout)

    def files(self, paths, directory=None):
        """The files a command wrote — a change of state the person is always told about (even with --plain; --quiet says nothing)."""
        if not paths:
            return
        if self.plain:
            for p in paths:
                print(p, file=self.stdout)
        else:
            self.say(f"wrote {len(paths)} files to {directory}" if directory else "\n".join(f"wrote {p}" for p in paths))

    def warn(self, text):
        self.warnings.append(str(text))
        print(f"warning: {text}", file=self.stderr)

    def error(self, message, hint=None):
        print(f"error: {message}", file=self.stderr)
        if hint:
            print(f"  {hint}", file=self.stderr)

    def fail(self, info):
        """Prints an error_info dict: as `error: ...` (and the hint), or, with --json, as {"ok": false, "error": {...}} on stderr."""
        if self.json:
            print(json.dumps({"schema": SCHEMA, "ok": False, "error": to_jsonable(info)}, ensure_ascii=False), file=self.stderr)
        else:
            self.error(info["message"], info.get("hint"))

    # ---- what a program reads ----------------------------------------------------------------------------------------------------------

    def result(self, command, result):
        """With --json: the one document on stdout."""
        if self.json:
            doc = {"schema": SCHEMA, "ok": True, "command": command, "result": to_jsonable(result), "warnings": list(self.warnings)}
            print(json.dumps(doc, indent=2, ensure_ascii=False), file=self.stdout)

    def event(self, name, **fields):
        """--events: one JSON line on stderr per event (progress, warning, note, state)."""
        print(json.dumps({"event": name, **to_jsonable(fields)}, ensure_ascii=False), file=self.stderr, flush=True)

    # ---- asking -------------------------------------------------------------------------------------------------------------------------

    def interactive(self):
        """True if a person is there to ask: stdin is a terminal and --no-input was not given."""
        try:
            return not self.no_input and sys.stdin is not None and sys.stdin.isatty() and self.stderr.isatty()
        except (AttributeError, ValueError):
            return False

    def confirm(self, question):
        """Asks `question` on stderr and reads yes/no from stdin; False without asking when nobody is there to answer."""
        if not self.interactive():
            return False
        print(f"{question} [y/N] ", end="", file=self.stderr, flush=True)
        try:
            return sys.stdin.readline().strip().lower() in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            return False


class Discard:
    """What sys.stdout is while a command runs: anything a library prints on its own (the TDT SDK reports every block it opens) must not mix
    into a result."""

    def write(self, text):
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False
