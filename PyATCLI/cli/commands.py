"""
cli/commands.py
---------------
One function per command: it turns the arguments into an operation's parameters (ops/), calls the operation, and has the result shown
(render.py). No analysis, file or figure logic lives here — that is in the operations, which the engine calls too.
"""

import json
import sys

from .. import ops
from ..analyses import ParameterError
from ..ops import OpContext, OpError
from ..ops.common import recording_name   # noqa: F401 — also this module's public name (cli/__init__.py)
from ..progress import coerce

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_MISMATCH = 0, 1, 2, 3
APP = {"name": "PyATCLI command line"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_value(text):
    """A command-line value: JSON if it is JSON (10, 2.5, true, [1, 2]), otherwise the text itself."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def read_json(path):
    """A JSON document from a file — or from standard input when `path` is "-"."""
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_parameters(param_args, params_file=None):
    """{name: value} from an optional JSON file ("-" is stdin) and then each K=V argument (which wins)."""
    out = {}
    if params_file:
        loaded = read_json(params_file)
        out.update(loaded.get("parameters", loaded) if isinstance(loaded, dict) else {})
    for item in param_args or []:
        if "=" not in item:
            raise ParameterError(f"--param expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = parse_value(v)
    return out


def pair(text, what):
    """(left, right) from "LEFT=RIGHT"."""
    left, sep, right = text.partition("=")
    if not sep or not left:
        raise ValueError(what)
    return left, right


def name_pair(text):
    return pair(text, f"a rename is KEY=NAME (the signal's key, and what to call it; NAME empty to take the rename away), not {text!r}")


def load_from_flags(args):
    """The `load` parameter of an operation from the load flags (--regression, --no-saved, --table, --x, --y)."""
    load = {}
    if getattr(args, "regression", None):
        load["regression_method"] = args.regression
    if getattr(args, "no_saved", False):
        load["use_saved"] = False
    if getattr(args, "table", None):
        load["table"] = args.table
    if getattr(args, "x", None):
        load["x_column"] = args.x
    if getattr(args, "y", None):
        load["y_columns"] = args.y
    return load


def make_context(args, out):
    """The OpContext for a run: what the operation reports goes to the person (a progress bar when stderr is a terminal, --progress to
    force it, --events for JSON lines), never into the result."""
    events = bool(getattr(args, "events", False))
    mode = getattr(args, "progress", None)
    bar = not events and not out.quiet and (mode is True or (mode is None and _is_tty(out.stderr)))
    state = {"sink": None}

    def emit(event, **fields):
        if events:
            out.event(event, **fields)
        if event == "progress" and bar:
            if state["sink"] is None:
                state["sink"] = coerce(True)
            state["sink"](fields["fraction"], fields.get("message", ""))
            if fields["fraction"] >= 1.0:
                state["sink"] = None
        elif event == "warning":
            if events:
                out.warnings.append(fields["text"])
            else:
                out.warn(fields["text"])
        elif event == "note" and not events:
            out.say(fields["text"])

    return OpContext(emit=emit, wants_progress=events or bar, app=APP)


def _is_tty(stream):
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def call(args, out, name, params):
    """Runs operation `name` for a command and returns its result (errors propagate to main, which shows them)."""
    return ops.call(name, params, make_context(args, out))



def _tables_lines(out, tables):
    """What tables a result has: a line each in human mode, rows in --plain."""
    if out.plain:
        out.table([], [[name, t["rows"], len(t["columns"])] for name, t in tables.items()])
    for name, t in tables.items():
        out.say(f"  {name}: {t['rows']} rows x {len(t['columns'])} columns")


def _out_dir(args, default):
    """The folder `--out` names (or `default`) — or None when `--output-folder` is given instead (each recording's results then go in its
    own folder under it)."""
    return None if args.output_folder else (args.out or default)


# ---------------------------------------------------------------------------
# info, kinds
# ---------------------------------------------------------------------------

def cmd_info(args, out):
    result = call(args, out, "info", {})
    if out.plain:
        out.table([], [["library", result["library"]], ["python", result["python"]], ["platform", result["platform"]]]
                  + [[name, v] for name, v in result["packages"].items()])
    else:
        out.say(f"PyATCLI {result['library']}")
        out.say(f"Python {result['python']} ({result['implementation']}, {result['bits']}-bit) on {result['platform']}")
        for name, v in result["packages"].items():
            out.say(f"  {name} {v}")
    out.result("info", result)
    return EXIT_OK


def cmd_serve(args, out):
    """`serve` is routed before argument parsing (cli.main); this is only here so that parsing it by itself does the same."""
    from ..engine.server import main as serve_main

    argv = ["--app-name", args.app_name]
    argv += ["--data-dir", args.data_dir] if args.data_dir else []
    argv += ["--app-version", args.app_version] if args.app_version else []
    return serve_main(argv + (["--verbose"] if out.verbose else []))


def cmd_docs(args, out):
    from . import docs

    text = docs.FORMATS[args.format]()
    if args.out and args.out != "-":
        with open(args.out, "w", encoding="utf-8", newline=chr(10)) as f:
            f.write(text if text.endswith(chr(10)) else text + chr(10))
        out.files([args.out])
        out.result("docs", {"format": args.format, "file": args.out})
    elif out.json:
        out.result("docs", {"format": args.format, "text": text})
    else:
        print(text, file=out.stdout)
    return EXIT_OK


def cmd_kinds(args, out):
    result = call(args, out, "kinds", {"kind": args.kind})
    if out.plain:
        out.table([], [[k, p["name"], p["type"], "required" if p["required"] else p["default"], p["doc"]]
                       for k, a in result["analyses"].items() for p in a["parameters"]])
    else:
        for k, a in result["analyses"].items():
            out.say(f"{k}\n  {a['summary']}")
            for p in a["parameters"]:
                default = "required" if p["required"] else f"default {p['default']!r}"
                choices = f" one of {', '.join(p['choices'])};" if p["choices"] else ""
                out.say(f"    {p['name']:<16} {p['type']:<8}{choices} {default}. {p['doc']}")
        if "group_analysis" in result:
            out.say(f"\n{result['group_analysis']['kind']}\n  {result['group_analysis']['summary']}")
    out.result("kinds", result)
    return EXIT_OK


# ---------------------------------------------------------------------------
# run, reproduce, batch
# ---------------------------------------------------------------------------

def cmd_run(args, out):
    parameters = collect_parameters(args.param, args.params)
    result = call(args, out, "run", {"kind": args.kind, "recording": args.recording, "parameters": parameters, "load": load_from_flags(args),
                                     "out": _out_dir(args, "pyatcli-output"), "output_folder": args.output_folder})
    out.say(f"{args.kind} on {recording_name(args.recording)}")
    _tables_lines(out, result["tables"])
    for n in result["notes"]:
        out.say(f"  note: {n}")
    out.files(result["files"], result["directory"])
    out.result("run", result)
    return EXIT_OK


def cmd_reproduce(args, out):
    rounded = True if args.rounded else (False if args.exact else None)
    record = read_json("-") if args.record == "-" else args.record
    result = call(args, out, "reproduce", {"record": record, "recording": args.recording, "out": args.out, "compare": args.compare or [],
                                           "tolerance": args.tolerance, "strict": args.strict, "rounded": rounded})
    _tables_lines(out, result["tables"])
    rows = []
    for c in result["comparisons"]:
        verdict = "MATCH" if c["ok"] else "MISMATCH"
        detail = c["message"] + (f" (largest difference {c['max_abs_diff']:.3g} over {c['cells']} cells)" if c["ok"] else "")
        rows.append([verdict, c["file"], c.get("table", "?"), detail])
        out.say(f"{'MATCH   ' if c['ok'] else 'MISMATCH'} {c['file']} vs {c.get('table', '?')}: {detail}")
    if out.plain:
        out.table([], rows)
    out.files(result["files"], args.out)
    out.result("reproduce", result)
    return EXIT_OK if result["ok"] else EXIT_MISMATCH


def cmd_batch(args, out):
    jobs = read_json("-") if args.jobs == "-" else args.jobs
    result = call(args, out, "batch", {"jobs": jobs, "out": None if args.output_folder else (args.out or "pyatcli-batch"),
                                       "output_folder": args.output_folder, "load": load_from_flags(args)})
    for f in result["failed"]:
        out.error(f"job {f['job']} ({f['kind']}) on {f['recording']}: {f['error']}")
    out.result("batch", result)
    return EXIT_OK if result["ok"] else EXIT_ERROR


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------

def cmd_group_run(args, out):
    result = call(args, out, "group.run", {"settings": args.settings, "out": args.out, "output_folder": args.output_folder})
    out.say(f"group analysis {result['name']!r}: {result['tables']} tables")
    for n in result["notes"]:
        out.say(f"  note: {n}")
    if result["directory"]:
        out.say(f"wrote {len(result['files'])} files to {result['directory']}")
    out.result("group.run", result)
    return EXIT_OK


def cmd_group_setup(args, out):
    result = call(args, out, "group.setup", {
        "folders": args.folders, "name": args.name, "markers": args.marker, "exclude": args.exclude,
        "store_labels": dict(name_pair(t) for t in args.store_label), "pre": args.pre, "post": args.post, "baseline": list(args.baseline),
        "response": list(args.response), "signal": args.signal, "smooth": args.smooth, "measures": args.measure,
        "peak_direction": args.peak_direction, "decay_fraction": args.decay_fraction, "correction": args.correction, "alpha": args.alpha,
        "regression": args.regression, "out": args.out, "output_folder": args.output_folder, "run": args.run})
    if result["stage"] == "markers":
        out.say(f"{len(result['recordings'])} recordings under {result['base']}; the markers they have:")
        n = len(result["recordings"])
        out.table(["marker (* = in every one)", "recordings", "events"],
                  [[("* " if m["shared"] else "  ") + m["name"], f"{m['recordings']} of {n}", m["events"]] for m in result["markers"]])
        out.result("group.setup", result)
        out.error("say which markers to compare with --marker NAME (repeat it for each)")
        return EXIT_USAGE
    for w in result["design_warnings"]:
        out.say(f"{w['level']}: {w['text']}")
    out.say(f"settings for {result['subjects']} recordings and {len(result['marker_names'])} markers saved: {result['settings']}")
    if result["ran"]:
        for n in result["run"]["notes"]:
            out.say(f"  note: {n}")
        out.say(f"wrote {len(result['run']['files'])} files to {result['directory']}")
    else:
        out.say(f"run it with: python -m PyATCLI group run \"{result['directory']}\"")
    out.result("group.setup", result)
    return EXIT_OK


# ---------------------------------------------------------------------------
# saved
# ---------------------------------------------------------------------------

def cmd_saved_show(args, out):
    result = call(args, out, "saved.show", {"recording": args.recording})
    kinds_saved = ("markers", "splices", "signal_labels", "highlights", "text_annotations")
    out.say(f"saved state of {result['name']} ({result['folder']})")
    if not any(result[k] for k in kinds_saved):
        out.say("  nothing saved")
    rows = []
    for m in result["markers"]:
        rows.append(["marker", f"{m['time']:.3f}", m.get("label", ""), m.get("color", "")])
        out.say(f"  marker      {m['time']:.3f} s  {m.get('label', '')}  ({m.get('color', '')})")
    for i, s in enumerate(result["splices"], 1):
        rows.append(["splice", i, s["mode"], f"{s['start']:.3f}..{s['end']:.3f}"])
        out.say(f"  splice {i}    {s['mode']} {s['start']:.3f}..{s['end']:.3f} s")
    for key, name in result["signal_labels"].items():
        rows.append(["signal_name", key, name, ""])
        out.say(f"  signal name {key} -> {name}")
    for h in result["highlights"]:
        rows.append(["highlight", f"{h['start']:.3f}..{h['end']:.3f}", h["color"], ""])
        out.say(f"  highlight   {h['start']:.3f}..{h['end']:.3f} s  {h['color']}")
    for t in result["text_annotations"]:
        rows.append(["text", f"{t['x']:.3f}", f"{t['y']:g}", t["text"]])
        out.say(f"  text        ({t['x']:.3f} s, {t['y']:g})  {t['text']}")
    if out.plain:
        out.table([], rows)
    out.result("saved.show", result)
    return EXIT_OK


def cmd_saved_marker(args, out):
    result = call(args, out, "saved.marker", {"recording": args.recording, "time": args.time, "label": args.label, "color": args.color,
                                              "fontsize": args.fontsize, "load": load_from_flags(args)})
    m = result["marker"]
    out.say(f"marker {m['label']!r} saved at {m['time']:.3f} s")
    out.result("saved.marker", result)
    return EXIT_OK


def cmd_saved_unmark(args, out):
    result = call(args, out, "saved.unmark", {"recording": args.recording, "time": args.time, "tolerance": args.tolerance,
                                              "label": args.label, "load": load_from_flags(args)})
    if not result["removed"]:
        out.error(f"no saved marker within {args.tolerance:g} s of {args.time:g} s" + (f" named {args.label!r}" if args.label else ""))
        return EXIT_ERROR
    n = len(result["removed"])
    out.say(f"removed {n} marker{'s' if n != 1 else ''}")
    out.result("saved.unmark", result)
    return EXIT_OK


def cmd_saved_splice(args, out):
    result = call(args, out, "saved.splice", {"recording": args.recording, "mode": args.mode, "start": args.start, "end": args.end,
                                              "load": load_from_flags(args)})
    out.say(f"splice saved ({len(result['splices'])} stacked): {args.mode} {args.start:g}..{args.end:g} s")
    out.result("saved.splice", result)
    return EXIT_OK


def cmd_saved_unsplice(args, out):
    result = call(args, out, "saved.unsplice", {"recording": args.recording, "number": args.number})
    out.say(f"splice {args.number} removed ({len(result['splices'])} left)")
    out.result("saved.unsplice", result)
    return EXIT_OK


def cmd_saved_rename(args, out):
    result = call(args, out, "saved.rename", {"recording": args.recording, "names": dict(name_pair(t) for t in args.names),
                                              "load": load_from_flags(args)})
    n = len(result["names"])
    out.say(f"{n} signal name{'s' if n != 1 else ''} saved")
    out.result("saved.rename", result)
    return EXIT_OK


def cmd_saved_highlight(args, out):
    result = call(args, out, "saved.highlight", {"recording": args.recording, "start": args.start, "end": args.end, "color": args.color})
    out.say(f"highlight saved: {result['highlight']['start']:g}..{result['highlight']['end']:g} s")
    out.result("saved.highlight", result)
    return EXIT_OK


def cmd_saved_text(args, out):
    result = call(args, out, "saved.text", {"recording": args.recording, "x": args.x, "y": args.y, "text": args.text, "fontsize": args.fontsize})
    t = result["text"]
    out.say(f"text box saved at ({t['x']:g} s, {t['y']:g})")
    out.result("saved.text", result)
    return EXIT_OK


def cmd_saved_clear(args, out):
    params = {"recording": args.recording, "yes": args.yes}
    try:
        result = call(args, out, "saved.clear", params)
    except OpError as e:
        if e.code != "confirmation_required":
            raise
        if not out.confirm(f"{e}. Delete them?"):
            out.error(str(e), "say --yes to do it")
            return EXIT_USAGE
        result = call(args, out, "saved.clear", {**params, "yes": True})
    n = len(result["removed"])
    out.say(f"removed {n} file{'s' if n != 1 else ''}" + (": " + ", ".join(result["removed"]) if result["removed"] else ""))
    out.result("saved.clear", result)
    return EXIT_OK
