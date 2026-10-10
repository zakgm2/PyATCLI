"""
cli/docs.py
-----------
The command line's reference documentation, generated from the parser itself so it cannot drift from what the program accepts:

    python -m PyATCLI docs --format markdown     the web reference (CLI_REFERENCE.md in the repository)
    python -m PyATCLI docs --format man          the manual page (pyatcli.1; `man ./pyatcli.1`)
    python -m PyATCLI docs --format contract     what a program using the engine can count on (ENGINE_CONTRACT.json; engine/contract.py)

Both walk the argparse tree — every command and subcommand, its arguments and options with their help text, its examples — and add the
parts that are the same for every command (the global options, the exit status, the environment variables, the engine's protocol).
`tests/test_cli_docs.py` regenerates both and compares them with the files committed beside the code, so a changed flag fails the tests
until the documents are regenerated (`python -m PyATCLI docs --format markdown --out CLI_REFERENCE.md`).
"""

import argparse
import re

from .. import provenance
from .parser import PROG, build_parser

EXIT_STATUS = [
    ("0", "done"),
    ("1", "something failed; the message is on standard error (with `--json`, a JSON error record: code, message, hint)"),
    ("2", "bad usage: an unknown flag, a missing argument, an action that needs a yes and was not given one"),
    ("3", "`reproduce` ran, but an export did not match the result"),
    ("130", "interrupted (Ctrl-C)"),
]

ENVIRONMENT = [
    ("DEBUG", "set to anything but 0/false/no/off: the same as `--debug` (a bug's full traceback)"),
    ("NO_COLOR", "accepted and unused: the command line never uses colour"),
]

ENGINE = (
    "`serve` runs the same operations as the other commands as a long-lived process, speaking one JSON object per line on standard input and "
    "output, and keeps recordings open between requests; it is what PyAT's window talks to. A request is "
    '`{"id": 7, "method": "recording.open", "params": {"path": "D:/rats/Rat7"}}`; it answers with `progress` events and one `result` or `error` '
    '(`{"id": 7, "event": "result", "ok": true, "result": {...}}`); `{"cancel": 7}` stops a request at its next progress report and '
    '`{"method": "shutdown"}` ends the process. Large arrays are never sent through the pipe: they are written as `.npy` files in the engine\'s '
    "data folder and named in the result. The protocol is described at the top of `PyATCLI/engine/server.py`."
)


# ---------------------------------------------------------------------------
# Walking the parser
# ---------------------------------------------------------------------------

class Node:
    """One command: its name ("group setup"), help, usage line, the arguments it takes, its options, its examples and its subcommands."""

    def __init__(self, parser, name, summary=""):
        self.name = name
        self.summary = summary
        self.parser = parser
        self.description = (parser.description or "").strip()
        self.epilog = (parser.epilog or "").strip()
        self.arguments, self.options, self.children = [], [], []
        formatter = parser._get_formatter()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child_name, child in action.choices.items():
                    helps = {a.dest: a.help for a in action._choices_actions}
                    self.children.append(Node(child, f"{name} {child_name}".strip(), helps.get(child_name) or ""))
                continue
            if isinstance(action, (argparse._HelpAction, argparse._VersionAction)) or action.help == argparse.SUPPRESS:
                continue
            row = (action, formatter._expand_help(action) if action.help else "")
            (self.options if action.option_strings else self.arguments).append(row)

    def usage(self):
        """`python -m PyATCLI run [options] kind recording`: the arguments in order, the options as one word."""
        if self.children:
            return f"{PROG} {self.name} <{'|'.join(c.name.split()[-1] for c in self.children)}> [options]"
        parts = [PROG, self.name, "[options]"] if self.name else [PROG, "<command>", "[options]"]
        for action, _ in self.arguments:
            label = _argument_label(action)
            if action.nargs == "?":
                parts.append(f"[{label}]")
            elif action.nargs == "*":
                parts.append(f"[{label} ...]")
            elif action.nargs == "+":
                parts.append(f"{label} ...")
            else:
                parts.append(label)
        return " ".join(parts)

    def is_global(self, action):
        return any(s in ("--json", "--plain", "--quiet", "--verbose", "--debug", "--no-input", "--events", "--progress", "--no-progress")
                   for s in action.option_strings)

    def own_options(self):
        return [(a, h) for a, h in self.options if not self.is_global(a)]

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


def tree():
    top = Node(build_parser(), "")
    return top


def _option_label(action):
    names = ", ".join(action.option_strings)
    if action.nargs == 0 or isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._CountAction)):
        return names
    metavar = action.metavar or action.dest.upper()
    if isinstance(metavar, tuple):
        metavar = " ".join(metavar)
    if action.nargs in ("+", "*"):
        metavar += " ..."
    return f"{names} {metavar}"


def _argument_label(action):
    return action.metavar or action.dest


def _examples(node):
    """The commands in a node's epilog, one per line, without the leading prose and the common indentation."""
    lines = []
    for line in node.epilog.splitlines():
        if line.strip().startswith(("examples:", "exit status", "Standard output", "Environment")):
            continue
        if line.strip():
            lines.append(line.rstrip())
    margin = min((len(l) - len(l.lstrip()) for l in lines), default=0)
    return [l[margin:] for l in lines]


ORDER = ["run", "reproduce", "batch", "group", "saved", "kinds", "info", "serve", "docs"]


def _commands_of(node):
    """Every command and subcommand, the top-level ones in the order --help lists them."""
    ranked = sorted(node.children, key=lambda c: ORDER.index(c.name) if c.name in ORDER else len(ORDER))
    out = []
    for child in ranked:
        out += [n for n in child.walk() if n.name]
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _md_escape(text):
    return text.replace("|", "\\|")


def markdown():
    top = tree()
    out = ["# PyATCLI command line", "", f"`{PROG}` (or `pyatcli`) — version {provenance.library_version()}.", "",
           "This page is generated from the program (`python -m PyATCLI docs --format markdown`); the same text is in `--help`.", ""]
    out += ["## Commands", ""]
    for n in _commands_of(top):
        if " " not in n.name:
            out.append(f"- [`{n.name}`](#{n.name}) — {n.summary}")
    out += ["", "## Options every command takes", "", "Before or after the command name:", ""]
    for action, help_text in top.options:
        if top.is_global(action):
            out.append(f"- `{_option_label(action)}` — {help_text}")
    out += ["", "`--version`, `-V` print the version; `--help`, `-h` print a command's options and examples.", ""]
    for n in _commands_of(top):
        level = "##" if " " not in n.name else "###"
        out += [f"{level} {n.name}", "", n.description or n.summary, "", "```", n.usage(), "```", ""]
        if n.arguments:
            out += ["| argument | |", "|---|---|"]
            out += [f"| `{_argument_label(a)}` | {_md_escape(h)} |" for a, h in n.arguments]
            out.append("")
        own = n.own_options()
        if own:
            out += ["| option | |", "|---|---|"]
            out += [f"| `{_option_label(a)}` | {_md_escape(h)} |" for a, h in own]
            out.append("")
        examples = _examples(n)
        if examples:
            out += ["Examples:", "", "```bash"] + examples + ["```", ""]
    out += ["## Exit status", ""] + [f"- `{code}` — {text}" for code, text in EXIT_STATUS] + [""]
    out += ["## Environment", ""] + [f"- `{name}` — {text}" for name, text in ENVIRONMENT] + [""]
    out += ["## The engine", "", ENGINE, ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# roff (man page)
# ---------------------------------------------------------------------------

def _roff(text):
    """Text as roff: backslashes and a leading dot or apostrophe made safe, markdown backticks dropped."""
    text = text.replace("`", "")
    text = text.replace("\\", "\\e").replace("-", "\\-")
    return re.sub(r"(?m)^([.'])", r"\\&\1", text)


def man():
    top = tree()
    version = provenance.library_version()
    out = [f'.TH PYATCLI 1 "" "PyATCLI {version}" "PyATCLI"',
           ".SH NAME", "pyatcli \\- analyse fibre-photometry, NIRS and tabular recordings from a script or the shell",
           ".SH SYNOPSIS", ".B pyatcli", ".I command", ".RI [ options ]", ".br", ".B python \\-m PyATCLI", ".I command", ".RI [ options ]",
           ".SH DESCRIPTION",
           _roff("Runs the analyses of PyAT (Python Analysis Tool) on a recording without the app, writes their tables and a provenance record, "
                 "and runs an analysis again from its record to check the numbers. Standard output is only ever the result (--json for a program, "
                 "--plain for tab-separated rows); warnings, errors and progress go to standard error."),
           ".SH COMMANDS"]
    for n in _commands_of(top):
        out += [".TP", f".B {_roff(n.name)}", _roff(n.summary or n.description)]
    out += [".SH GLOBAL OPTIONS", _roff("Accepted before or after the command name.")]
    for action, help_text in top.options:
        if top.is_global(action):
            out += [".TP", f".B {_roff(_option_label(action))}", _roff(help_text)]
    for n in _commands_of(top):
        out += [f".SH {_roff(n.name.upper())}", _roff(n.description or n.summary), ".PP", ".B " + _roff(n.usage())]
        for a, h in n.arguments:
            out += [".TP", f".I {_roff(_argument_label(a))}", _roff(h)]
        for a, h in n.own_options():
            out += [".TP", f".B {_roff(_option_label(a))}", _roff(h)]
        examples = _examples(n)
        if examples:
            out += [".PP", "Examples:", ".nf"] + [_roff(line) for line in examples] + [".fi"]
    out += [".SH EXIT STATUS"]
    for code, text in EXIT_STATUS:
        out += [".TP", f".B {code}", _roff(text)]
    out += [".SH ENVIRONMENT"]
    for name, text in ENVIRONMENT:
        out += [".TP", f".B {name}", _roff(text)]
    out += [".SH THE ENGINE", _roff(ENGINE),
            ".SH SEE ALSO", _roff("API_REFERENCE.md and CLI_REFERENCE.md in the repository; https://github.com/zakgm2/PyATCLI"),
            ""]
    return "\n".join(out)


def contract():
    """What a program that talks to the engine can count on (ENGINE_CONTRACT.json in the repository): see engine/contract.py."""
    import json

    from ..engine import contract as engine_contract

    return json.dumps(engine_contract.build(), indent=2, ensure_ascii=False) + "\n"


FORMATS = {"markdown": markdown, "man": man, "contract": contract}
