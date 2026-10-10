"""
cli/parser.py
-------------
The argument parser: every command, its flags, its help and its examples.

Every command takes the same global flags — `--json`, `--plain`, `-q/--quiet`, `-v/--verbose`, `-d/--debug`, `--no-input`, `--events`,
`--progress/--no-progress` — before or after the command name, so they can be given wherever they come to mind.
"""

import argparse

from .. import provenance, saved_state
from ..analyses import kinds
from ..analysis import group as group_design
from ..sources.processing_TDT import REGRESSION_METHODS
from . import commands as C

PROG = "python -m PyATCLI"

CONCISE = """\
PyATCLI {version} — analyse fibre-photometry, NIRS and tabular recordings from a script or the shell.

usage: {prog} <command> [options]

commands:
  run         run one analysis on a recording
  reproduce   run an analysis again from its provenance record and check the numbers
  batch       many analyses over many recordings
  group       group analysis: `group setup` (the wizard), `group run`
  saved       what the app's Save Changes saves: markers, splices, signal names, highlights, text
  kinds       the analyses and every parameter of each
  info        versions of this library, Python and the packages
  serve       the engine: these operations as JSON lines on stdin/stdout, with recordings kept open (what PyAT's window talks to)
  docs        this command line's reference: markdown, or a man page

examples:
  {prog} run event_peth "D:/rats/Rat7" -p event="L1P¹" -p pre_s=10 -p post_s=20 --out out
  {prog} batch jobs.json --out batch-out
  {prog} run zscore_window "D:/rats/Rat7" -p center_s=312.5 --json | jq .result.summary

`{prog} <command> --help` lists a command's options; `--help` shows everything. Every command takes --json and --plain.
"""

EPILOGS = {
    "info": f"""examples:
  {PROG} info
  {PROG} info --json
""",
    "serve": f"""examples:
  {PROG} serve                       # then one JSON request per line on stdin, one JSON answer per line on stdout
  echo '{{"id": 1, "method": "kinds"}}' | {PROG} serve
  {PROG} serve --data-dir ./engine-data -v
""",
    "docs": f"""examples:
  {PROG} docs --format man --out pyatcli.1 && man ./pyatcli.1
  {PROG} docs --format markdown --out CLI_REFERENCE.md
""",
    "kinds": f"""examples:
  {PROG} kinds                      every analysis and its parameters
  {PROG} kinds event_peth           just one
  {PROG} kinds --plain | cut -f1,2   tab-separated: kind, parameter
""",
    "run": f"""examples:
  {PROG} run event_peth "D:/rats/Rat7-250101" --param event="L1P¹" --param pre_s=10 --param post_s=20 --out out
  {PROG} run zscore_window "D:/rats/Rat7-250101" --param center_s=312.5 --param signal=main_driver --json
  {PROG} run auc_window recording.txt -p center_s=30 --output-folder Output   # Output/Oxysoft/recording (Analysis)/
""",
    "reproduce": f"""examples:
  {PROG} reproduce "out/Rat7_event_peth.provenance.json" --compare "out/Event PETH.csv"
  {PROG} reproduce record.provenance.json --recording "E:/moved/Rat7" --strict
""",
    "batch": f"""examples:
  {PROG} batch jobs.json --out batch-out
  {PROG} batch jobs.json --output-folder Output
  cat jobs.json | {PROG} batch - --out batch-out
""",
    "group": f"""examples:
  {PROG} group setup "D:/cohort" --name Sucrose                       # lists the markers the recordings share
  {PROG} group setup "D:/cohort" --name Sucrose --marker "L1P¹" --marker "L2P¹" --output-folder Output
  {PROG} group run "Output/Group Analysis/Sucrose"
""",
    "group setup": f"""examples:
  {PROG} group setup "D:/cohort" --name Sucrose
  {PROG} group setup "D:/cohort" --name Sucrose --marker "L1P¹" --marker "L2P¹" --output-folder Output --run
""",
    "group run": f"""examples:
  {PROG} group run "Output/Group Analysis/Sucrose"
  {PROG} group run analysis_settings.json --out results
""",
    "saved": f"""examples:
  {PROG} saved show "D:/rats/Rat7"
  {PROG} saved marker "D:/rats/Rat7" --time 312.5 --label Lever
  {PROG} saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
  {PROG} saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
""",
}


def concise_help():
    return CONCISE.format(version=provenance.library_version(), prog=PROG)


def add_global(parser, top):
    """The flags every command takes. On a command's own parser their defaults are suppressed, so a flag given before the command is not
    overwritten by the command's absence of it."""
    d = (lambda value: value) if top else (lambda value: argparse.SUPPRESS)
    g = parser.add_argument_group("global options (accepted before or after the command)")
    shape = g.add_mutually_exclusive_group()
    shape.add_argument("--json", action="store_true", default=d(False), help="print the result as one JSON document on stdout")
    shape.add_argument("--plain", action="store_true", default=d(False), help="print tab-separated rows on stdout, nothing else (for cut, awk, spreadsheets)")
    g.add_argument("--quiet", "-q", action="store_true", default=d(False), help="print nothing but errors")
    g.add_argument("--verbose", "-v", action="count", default=d(0), help="say more on stderr (and show what libraries print)")
    g.add_argument("--debug", "-d", action="store_true", default=d(False), help="show the full traceback of an error (also set by DEBUG=1)")
    g.add_argument("--no-input", action="store_true", default=d(False), help="never ask a question (an action that needs a yes fails instead)")
    g.add_argument("--events", action="store_true", default=d(False),
                   help="report progress, warnings and notes as JSON lines on stderr (for a program driving this one)")
    g.add_argument("--progress", dest="progress", action="store_true", default=d(None), help="always show a progress bar on stderr for slow steps")
    g.add_argument("--no-progress", dest="progress", action="store_false", default=d(None),
                   help="never show a progress bar (the default shows one when stderr is a terminal)")


def _load_flags(sp):
    sp.add_argument("--regression", default="ols", choices=REGRESSION_METHODS, help="motion-correction method for TDT (default ols)")
    sp.add_argument("--no-saved", action="store_true", help="ignore the recording's saved splices and markers (JSON saves/)")
    sp.add_argument("--table", help="generic file: which table to open")
    sp.add_argument("--x", help="generic file: the x column (default the first)")
    sp.add_argument("--y", nargs="+", help="generic file: the trace columns (default all the others)")


def _where_to(sp, default, what, group=False):
    """--out DIR (the results exactly there), or --output-folder DIR (PyAT's layout under an Output folder)."""
    choice = sp.add_mutually_exclusive_group()
    choice.add_argument("--out", "-o", metavar="DIR", help=f"folder for {what}" + (f" (default {default})" if default else ""))
    choice.add_argument("--output-folder", metavar="DIR",
                        help="an Output folder, laid out as PyAT's exports are: " + (
                            "<DIR>/Group Analysis/<group name>, a new folder if the name was used before" if group else
                            "<DIR>/<data type>/<recording> (Analysis)/, and a repeat run never overwrites an earlier one"))


def build_parser():
    top = argparse.ArgumentParser(
        prog=PROG, formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run PyATCLI analyses from the command line, and run them again from a provenance record.",
        epilog="""exit status: 0 done; 1 something failed (the message is on stderr); 2 bad usage; 3 `reproduce` ran but an export did not match; 130 interrupted.
Standard output is only ever the result (--json for a program, --plain for rows); everything else goes to standard error.
Environment: DEBUG=1 (same as --debug).
""" + "\n" + EPILOGS["run"])
    top.add_argument("--version", "-V", action="version", version=f"PyATCLI {provenance.library_version()}")
    add_global(top, True)
    sub = top.add_subparsers(dest="command", metavar="<command>")

    def command(name, help_text, fn, epilog_key=None, description=None, parent=sub):
        sp = parent.add_parser(name, help=help_text, description=description or help_text, formatter_class=argparse.RawDescriptionHelpFormatter,
                               epilog=EPILOGS.get(epilog_key or name), add_help=True)
        sp.set_defaults(fn=fn)
        add_global(sp, False)
        return sp

    command("info", "versions of this library, Python and the numerical packages", C.cmd_info)

    sp = command("serve", "the engine: these operations as JSON lines on stdin/stdout, with recordings kept open", C.cmd_serve, description=(
        "Runs the engine: the operations of the other commands as a long-lived process that speaks one JSON object per line on standard input "
        "and output, and keeps recordings open between requests. It is what PyAT's window talks to. Requests: "
        '{"id": 7, "method": "recording.open", "params": {...}}; {"cancel": 7} stops one; {"method": "shutdown"} ends the process. '
        "Arrays travel as .npy files in the data folder, never through the pipe."))
    sp.add_argument("--data-dir", metavar="DIR", help="where bulk results (arrays, big tables) are written (default: a temporary folder, removed on exit)")
    sp.add_argument("--app-name", default="PyATCLI engine", metavar="NAME", help="the program asking, for the provenance records the engine writes")
    sp.add_argument("--app-version", metavar="VERSION", help="its version")

    sp = command("docs", "print this command line's reference: markdown, or a man page (or the engine's contract)", C.cmd_docs)
    sp.add_argument("--format", "-f", choices=["markdown", "man", "contract"], default="markdown",
                    help="markdown (CLI_REFERENCE.md), man (pyatcli.1), or contract (ENGINE_CONTRACT.json: what the engine's callers can count on)")
    sp.add_argument("--out", "-o", metavar="FILE", help="write it to this file instead of standard output (- is standard output)")

    sp = command("kinds", "list the analyses and their parameters", C.cmd_kinds)
    sp.add_argument("kind", nargs="?", help="describe just this one")

    sp = command("run", "run one analysis on a recording", C.cmd_run)
    sp.add_argument("kind", choices=kinds(), help="the analysis (see `kinds`)")
    sp.add_argument("recording", help="a TDT folder, an Oxysoft .txt export, or a generic table file")
    sp.add_argument("--param", "-p", action="append", metavar="NAME=VALUE", help="an analysis parameter; repeat for several")
    sp.add_argument("--params", metavar="FILE.json", help="parameters from a JSON file, or - for standard input (--param overrides it)")
    _where_to(sp, "pyatcli-output", "the CSV tables and the provenance record")
    _load_flags(sp)

    sp = command("reproduce", "run an analysis again from its provenance record and check the numbers", C.cmd_reproduce)
    sp.add_argument("record", help="a .provenance.json written beside an export, or - for standard input")
    sp.add_argument("--recording", help="where the recording is now, if it is not where the record says")
    sp.add_argument("--compare", "-c", nargs="+", metavar="EXPORT.csv", help="exports to check the result against")
    sp.add_argument("--tolerance", type=float, default=1e-4, help="|a-b| <= tolerance*(1+|b|) counts as equal (default 1e-4)")
    sp.add_argument("--strict", action="store_true", help="fail if the recording or the software differs from the record's")
    rounding = sp.add_mutually_exclusive_group()
    rounding.add_argument("--rounded", action="store_true", help="the exports hold rounded numbers: accept the digits they show (the default for PyAT's exports)")
    rounding.add_argument("--exact", action="store_true", help="compare to the tolerance only, whoever wrote the exports")
    sp.add_argument("--out", "-o", metavar="DIR", help="write the result tables and a new record here")

    sp = command("batch", "run many analyses over many recordings", C.cmd_batch)
    sp.add_argument("jobs", help='JSON: {"defaults": {...}, "jobs": [{"kind", "recording", "parameters"}, ...]}, or - for standard input')
    _where_to(sp, "pyatcli-batch", "the results, a folder per job and recording")
    _load_flags(sp)

    # -- group: setup, run
    group = command("group", "group analysis across recordings: `setup` (the wizard) and `run`", None, description=(
        "Group analysis (Hypothesis Testing): `group setup` finds the recordings, lists the markers they share and writes analysis_settings.json; "
        "`group run` runs it."))
    group.set_defaults(fn=lambda args, out: group.print_help(out.stdout) or C.EXIT_USAGE)
    gsub = group.add_subparsers(dest="group_command", metavar="<setup|run>")

    sp = command("setup", "set up a group analysis as the wizard does, and write its analysis_settings.json", C.cmd_group_setup, "group setup",
                 description="Finds the recordings (a cohort folder is searched, at any depth), lists the markers they share, takes the choices the "
                             "wizard asks for, checks the design and writes analysis_settings.json. Without --marker it only lists the markers. "
                             "The defaults are the wizard's.", parent=gsub)
    sp.add_argument("folders", nargs="+", help="a cohort folder to search, or recordings (TDT blocks)")
    sp.add_argument("--name", required=True, help="the group's name (its folder and its figures' titles)")
    sp.add_argument("--marker", "-m", action="append", default=[], metavar="NAME", help="a marker to compare, as `group setup` lists it; repeat for each")
    sp.add_argument("--exclude", action="append", default=[], metavar="RECORDING", help="leave a recording out, by its subject name; repeat")
    sp.add_argument("--store-label", action="append", default=[], metavar="ID=NAME", help="show an event store under another name (what Edit Attributes does); repeat")
    sp.add_argument("--pre", type=float, default=group_design.DEFAULT_PRE, help="seconds before each event (default %(default)g)")
    sp.add_argument("--post", type=float, default=group_design.DEFAULT_POST, help="seconds after each event (default %(default)g)")
    sp.add_argument("--baseline", type=float, nargs=2, default=group_design.DEFAULT_BASELINE, metavar=("START", "END"),
                    help="the baseline window, seconds from the event (default %(default)s)")
    sp.add_argument("--response", type=float, nargs=2, default=group_design.DEFAULT_RESPONSE, metavar=("START", "END"),
                    help="the response window, seconds from the event (default %(default)s)")
    sp.add_argument("--signal", choices=group_design.SIGNALS, default="dff", help="dff, or zscore (also divide by the baseline's SD)")
    sp.add_argument("--smooth", type=float, default=0.5, help="moving-average smoothing before slicing, seconds; 0 for none (default 0.5)")
    sp.add_argument("--measure", action="append", choices=group_design.METRICS, help="a measure to compute and test; repeat (default: all five)")
    sp.add_argument("--peak-direction", choices=group_design.PEAK_DIRECTIONS, default="absolute", help="default absolute")
    sp.add_argument("--decay-fraction", type=float, default=0.5, help="how far the trace must fall for the decay time, as a fraction of the peak (default 0.5)")
    sp.add_argument("--correction", choices=group_design.CORRECTIONS, default="holm", help="multiple-comparison correction (default holm)")
    sp.add_argument("--alpha", type=float, default=0.05, help="significance level (default 0.05)")
    sp.add_argument("--regression", default="ols", choices=REGRESSION_METHODS, help="motion-correction method for every recording (default ols)")
    _where_to(sp, None, "the settings (and, with --run, the results)", group=True)
    sp.add_argument("--run", action="store_true", help="run the analysis right after, as the wizard's Run Analysis does")

    sp = command("run", "run a group analysis from its analysis_settings.json", C.cmd_group_run, "group run", parent=gsub)
    sp.add_argument("settings", help="analysis_settings.json, or the folder holding it")
    _where_to(sp, None, "the tables and the report", group=True)

    # -- saved
    saved = command("saved", "edit what the app's Save Changes saves (markers, splices, signal names, highlights, text)", None, description=(
        "Writes the files PyAT writes in a recording's JSON saves folder, so what is done here is what was done in the app: either reads it "
        "back when it opens the recording."))
    saved.set_defaults(fn=lambda args, out: saved.print_help(out.stdout) or C.EXIT_USAGE)
    actions = saved.add_subparsers(dest="action", metavar="<action>")

    def action(name, help_text, fn, load=False):
        sp = command(name, help_text, fn, "saved", parent=actions)
        sp.add_argument("recording", help="a TDT folder, an Oxysoft .txt export, or a generic table file")
        if load:
            sp.add_argument("--regression", default="ols", choices=REGRESSION_METHODS, help="motion-correction method for TDT (default ols)")
            sp.add_argument("--table", help="generic file: which table to open")
            sp.add_argument("--x", help="generic file: the x column (default the first)")
            sp.add_argument("--y", nargs="+", help="generic file: the trace columns (default all the others)")
        return sp

    action("show", "list what is saved for a recording", C.cmd_saved_show)
    sp = action("marker", "Add Marker + Save Changes: put a marker at a time (seconds, on the time axis as the recording now stands)", C.cmd_saved_marker, load=True)
    sp.add_argument("--time", "-t", type=float, required=True, help="seconds")
    sp.add_argument("--label", help="what the marker says (default Marker)")
    sp.add_argument("--color", help="its colour (default green)")
    sp.add_argument("--fontsize", type=int, help="its text size (default 8)")
    sp = action("unmark", "delete the saved markers near a time", C.cmd_saved_unmark, load=True)
    sp.add_argument("--time", "-t", type=float, required=True, help="seconds")
    sp.add_argument("--tolerance", type=float, default=0.05, help="how near counts, in seconds (default 0.05)")
    sp.add_argument("--label", help="only markers with this name")
    sp = action("splice", "Splice + Save Changes: stack a splice on the saved ones (it is replayed whenever the recording opens)", C.cmd_saved_splice, load=True)
    sp.add_argument("--mode", required=True, choices=saved_state.SPLICE_MODES, help="cut_out removes start..end and closes the gap; keep_inside keeps only start..end")
    sp.add_argument("--start", type=float, required=True, help="seconds, on the time axis as the recording now stands (after the saved splices)")
    sp.add_argument("--end", type=float, required=True, help="seconds")
    sp = action("unsplice", "take one saved splice out of the stack", C.cmd_saved_unsplice)
    sp.add_argument("number", type=int, help="which one: 1 is the first applied (see `saved show`)")
    sp = action("rename", "give signals names of your own, kept with the recording", C.cmd_saved_rename, load=True)
    sp.add_argument("names", nargs="+", metavar="KEY=NAME", help="the signal's key (main_driver, isosbestic, normalized, ...) and its name; empty NAME takes the rename away")
    sp = action("highlight", "Highlighter + Save Changes: shade a stretch of the recording", C.cmd_saved_highlight)
    sp.add_argument("--start", type=float, required=True, help="seconds")
    sp.add_argument("--end", type=float, required=True, help="seconds")
    sp.add_argument("--color", default="Yellow", help="Yellow, Green, Pink, Orange or Blue (the app's five), or a hex colour (default Yellow)")
    sp = action("text", "Text tool + Save Changes: put a text box on the plot", C.cmd_saved_text)
    sp.add_argument("--x", type=float, required=True, help="seconds")
    sp.add_argument("--y", type=float, required=True, help="in the signal's units")
    sp.add_argument("--text", required=True, help="what it says")
    sp.add_argument("--fontsize", type=int, default=saved_state.DEFAULT_TEXT_FONT_SIZE, help="6 to 72 (default 10)")
    sp = action("clear", "Undo All Changes, saved: delete every file in the recording's JSON saves folder (asks first)", C.cmd_saved_clear)
    sp.add_argument("--yes", "-y", action="store_true", help="go ahead without asking")

    return top
