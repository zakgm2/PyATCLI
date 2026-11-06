# PyATCLI command line

`python -m PyATCLI` (or `pyatcli`) — version 2026.11.5.

This page is generated from the program (`python -m PyATCLI docs --format markdown`); the same text is in `--help`.

## Commands

- [`run`](#run) — run one analysis on a recording
- [`reproduce`](#reproduce) — run an analysis again from its provenance record and check the numbers
- [`batch`](#batch) — run many analyses over many recordings
- [`group`](#group) — group analysis across recordings: `setup` (the wizard) and `run`
- [`saved`](#saved) — edit what the app's Save Changes saves (markers, splices, signal names, highlights, text)
- [`kinds`](#kinds) — list the analyses and their parameters
- [`info`](#info) — versions of this library, Python and the numerical packages
- [`serve`](#serve) — the engine: these operations as JSON lines on stdin/stdout, with recordings kept open
- [`docs`](#docs) — print this command line's reference: markdown, or a man page (or the engine's contract)

## Options every command takes

Before or after the command name:

- `--json` — print the result as one JSON document on stdout
- `--plain` — print tab-separated rows on stdout, nothing else (for cut, awk, spreadsheets)
- `--quiet, -q` — print nothing but errors
- `--verbose, -v` — say more on stderr (and show what libraries print)
- `--debug, -d` — show the full traceback of an error (also set by DEBUG=1)
- `--no-input` — never ask a question (an action that needs a yes fails instead)
- `--events` — report progress, warnings and notes as JSON lines on stderr (for a program driving this one)
- `--progress` — always show a progress bar on stderr for slow steps
- `--no-progress` — never show a progress bar (the default shows one when stderr is a terminal)

`--version`, `-V` print the version; `--help`, `-h` print a command's options and examples.

## run

run one analysis on a recording

```
python -m PyATCLI run [options] kind recording
```

| argument | |
|---|---|
| `kind` | the analysis (see `kinds`) |
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |

| option | |
|---|---|
| `--param, -p NAME=VALUE` | an analysis parameter; repeat for several |
| `--params FILE.json` | parameters from a JSON file, or - for standard input (--param overrides it) |
| `--out, -o DIR` | folder for the CSV tables and the provenance record (default pyatcli-output) |
| `--output-folder DIR` | an Output folder, laid out as PyAT's exports are: <DIR>/<data type>/<recording> (Analysis)/, and a repeat run never overwrites an earlier one |
| `--regression REGRESSION` | motion-correction method for TDT (default ols) |
| `--no-saved` | ignore the recording's saved splices and markers (JSON saves/) |
| `--table TABLE` | generic file: which table to open |
| `--x X` | generic file: the x column (default the first) |
| `--y Y ...` | generic file: the trace columns (default all the others) |

Examples:

```bash
python -m PyATCLI run event_peth "D:/rats/Rat7-250101" --param event="L1P¹" --param pre_s=10 --param post_s=20 --out out
python -m PyATCLI run zscore_window "D:/rats/Rat7-250101" --param center_s=312.5 --param signal=main_driver --json
python -m PyATCLI run auc_window recording.txt -p center_s=30 --output-folder Output   # Output/Oxysoft/recording (Analysis)/
```

## reproduce

run an analysis again from its provenance record and check the numbers

```
python -m PyATCLI reproduce [options] record
```

| argument | |
|---|---|
| `record` | a .provenance.json written beside an export, or - for standard input |

| option | |
|---|---|
| `--recording RECORDING` | where the recording is now, if it is not where the record says |
| `--compare, -c EXPORT.csv ...` | exports to check the result against |
| `--tolerance TOLERANCE` | \|a-b\| <= tolerance*(1+\|b\|) counts as equal (default 1e-4) |
| `--strict` | fail if the recording or the software differs from the record's |
| `--rounded` | the exports hold rounded numbers: accept the digits they show (the default for PyAT's exports) |
| `--exact` | compare to the tolerance only, whoever wrote the exports |
| `--out, -o DIR` | write the result tables and a new record here |

Examples:

```bash
python -m PyATCLI reproduce "out/Rat7_event_peth.provenance.json" --compare "out/Event PETH.csv"
python -m PyATCLI reproduce record.provenance.json --recording "E:/moved/Rat7" --strict
```

## batch

run many analyses over many recordings

```
python -m PyATCLI batch [options] jobs
```

| argument | |
|---|---|
| `jobs` | JSON: {"defaults": {...}, "jobs": [{"kind", "recording", "parameters"}, ...]}, or - for standard input |

| option | |
|---|---|
| `--out, -o DIR` | folder for the results, a folder per job and recording (default pyatcli-batch) |
| `--output-folder DIR` | an Output folder, laid out as PyAT's exports are: <DIR>/<data type>/<recording> (Analysis)/, and a repeat run never overwrites an earlier one |
| `--regression REGRESSION` | motion-correction method for TDT (default ols) |
| `--no-saved` | ignore the recording's saved splices and markers (JSON saves/) |
| `--table TABLE` | generic file: which table to open |
| `--x X` | generic file: the x column (default the first) |
| `--y Y ...` | generic file: the trace columns (default all the others) |

Examples:

```bash
python -m PyATCLI batch jobs.json --out batch-out
python -m PyATCLI batch jobs.json --output-folder Output
cat jobs.json | python -m PyATCLI batch - --out batch-out
```

## group

Group analysis (Hypothesis Testing): `group setup` finds the recordings, lists the markers they share and writes analysis_settings.json; `group run` runs it.

```
python -m PyATCLI group <setup|run> [options]
```

Examples:

```bash
python -m PyATCLI group setup "D:/cohort" --name Sucrose                       # lists the markers the recordings share
python -m PyATCLI group setup "D:/cohort" --name Sucrose --marker "L1P¹" --marker "L2P¹" --output-folder Output
python -m PyATCLI group run "Output/Group Analysis/Sucrose"
```

### group setup

Finds the recordings (a cohort folder is searched, at any depth), lists the markers they share, takes the choices the wizard asks for, checks the design and writes analysis_settings.json. Without --marker it only lists the markers. The defaults are the wizard's.

```
python -m PyATCLI group setup [options] folders ...
```

| argument | |
|---|---|
| `folders` | a cohort folder to search, or recordings (TDT blocks) |

| option | |
|---|---|
| `--name NAME` | the group's name (its folder and its figures' titles) |
| `--marker, -m NAME` | a marker to compare, as `group setup` lists it; repeat for each |
| `--exclude RECORDING` | leave a recording out, by its subject name; repeat |
| `--store-label ID=NAME` | show an event store under another name (what Edit Attributes does); repeat |
| `--pre PRE` | seconds before each event (default 10) |
| `--post POST` | seconds after each event (default 10) |
| `--baseline START END` | the baseline window, seconds from the event (default (-10.0, -6.0)) |
| `--response START END` | the response window, seconds from the event (default (0.0, 10.0)) |
| `--signal SIGNAL` | dff, or zscore (also divide by the baseline's SD) |
| `--smooth SMOOTH` | moving-average smoothing before slicing, seconds; 0 for none (default 0.5) |
| `--measure MEASURE` | a measure to compute and test; repeat (default: all five) |
| `--peak-direction PEAK_DIRECTION` | default absolute |
| `--decay-fraction DECAY_FRACTION` | how far the trace must fall for the decay time, as a fraction of the peak (default 0.5) |
| `--correction CORRECTION` | multiple-comparison correction (default holm) |
| `--alpha ALPHA` | significance level (default 0.05) |
| `--regression REGRESSION` | motion-correction method for every recording (default ols) |
| `--out, -o DIR` | folder for the settings (and, with --run, the results) |
| `--output-folder DIR` | an Output folder, laid out as PyAT's exports are: <DIR>/Group Analysis/<group name>, a new folder if the name was used before |
| `--run` | run the analysis right after, as the wizard's Run Analysis does |

Examples:

```bash
python -m PyATCLI group setup "D:/cohort" --name Sucrose
python -m PyATCLI group setup "D:/cohort" --name Sucrose --marker "L1P¹" --marker "L2P¹" --output-folder Output --run
```

### group run

run a group analysis from its analysis_settings.json

```
python -m PyATCLI group run [options] settings
```

| argument | |
|---|---|
| `settings` | analysis_settings.json, or the folder holding it |

| option | |
|---|---|
| `--out, -o DIR` | folder for the tables and the report |
| `--output-folder DIR` | an Output folder, laid out as PyAT's exports are: <DIR>/Group Analysis/<group name>, a new folder if the name was used before |

Examples:

```bash
python -m PyATCLI group run "Output/Group Analysis/Sucrose"
python -m PyATCLI group run analysis_settings.json --out results
```

## saved

Writes the files PyAT writes in a recording's JSON saves folder, so what is done here is what was done in the app: either reads it back when it opens the recording.

```
python -m PyATCLI saved <show|marker|unmark|splice|unsplice|rename|highlight|text|clear> [options]
```

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved show

list what is saved for a recording

```
python -m PyATCLI saved show [options] recording
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved marker

Add Marker + Save Changes: put a marker at a time (seconds, on the time axis as the recording now stands)

```
python -m PyATCLI saved marker [options] recording
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |

| option | |
|---|---|
| `--regression REGRESSION` | motion-correction method for TDT (default ols) |
| `--table TABLE` | generic file: which table to open |
| `--x X` | generic file: the x column (default the first) |
| `--y Y ...` | generic file: the trace columns (default all the others) |
| `--time, -t TIME` | seconds |
| `--label LABEL` | what the marker says (default Marker) |
| `--color COLOR` | its colour (default green) |
| `--fontsize FONTSIZE` | its text size (default 8) |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved unmark

delete the saved markers near a time

```
python -m PyATCLI saved unmark [options] recording
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |

| option | |
|---|---|
| `--regression REGRESSION` | motion-correction method for TDT (default ols) |
| `--table TABLE` | generic file: which table to open |
| `--x X` | generic file: the x column (default the first) |
| `--y Y ...` | generic file: the trace columns (default all the others) |
| `--time, -t TIME` | seconds |
| `--tolerance TOLERANCE` | how near counts, in seconds (default 0.05) |
| `--label LABEL` | only markers with this name |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved splice

Splice + Save Changes: stack a splice on the saved ones (it is replayed whenever the recording opens)

```
python -m PyATCLI saved splice [options] recording
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |

| option | |
|---|---|
| `--regression REGRESSION` | motion-correction method for TDT (default ols) |
| `--table TABLE` | generic file: which table to open |
| `--x X` | generic file: the x column (default the first) |
| `--y Y ...` | generic file: the trace columns (default all the others) |
| `--mode MODE` | cut_out removes start..end and closes the gap; keep_inside keeps only start..end |
| `--start START` | seconds, on the time axis as the recording now stands (after the saved splices) |
| `--end END` | seconds |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved unsplice

take one saved splice out of the stack

```
python -m PyATCLI saved unsplice [options] recording number
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |
| `number` | which one: 1 is the first applied (see `saved show`) |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved rename

give signals names of your own, kept with the recording

```
python -m PyATCLI saved rename [options] recording KEY=NAME ...
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |
| `KEY=NAME` | the signal's key (main_driver, isosbestic, normalized, ...) and its name; empty NAME takes the rename away |

| option | |
|---|---|
| `--regression REGRESSION` | motion-correction method for TDT (default ols) |
| `--table TABLE` | generic file: which table to open |
| `--x X` | generic file: the x column (default the first) |
| `--y Y ...` | generic file: the trace columns (default all the others) |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved highlight

Highlighter + Save Changes: shade a stretch of the recording

```
python -m PyATCLI saved highlight [options] recording
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |

| option | |
|---|---|
| `--start START` | seconds |
| `--end END` | seconds |
| `--color COLOR` | Yellow, Green, Pink, Orange or Blue (the app's five), or a hex colour (default Yellow) |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved text

Text tool + Save Changes: put a text box on the plot

```
python -m PyATCLI saved text [options] recording
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |

| option | |
|---|---|
| `--x X` | seconds |
| `--y Y` | in the signal's units |
| `--text TEXT` | what it says |
| `--fontsize FONTSIZE` | 6 to 72 (default 10) |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

### saved clear

Undo All Changes, saved: delete every file in the recording's JSON saves folder (asks first)

```
python -m PyATCLI saved clear [options] recording
```

| argument | |
|---|---|
| `recording` | a TDT folder, an Oxysoft .txt export, or a generic table file |

| option | |
|---|---|
| `--yes, -y` | go ahead without asking |

Examples:

```bash
python -m PyATCLI saved show "D:/rats/Rat7"
python -m PyATCLI saved marker "D:/rats/Rat7" --time 312.5 --label Lever
python -m PyATCLI saved splice "D:/rats/Rat7" --mode cut_out --start 40 --end 55
python -m PyATCLI saved clear "D:/rats/Rat7"          # asks first; --yes to go ahead without asking
```

## kinds

list the analyses and their parameters

```
python -m PyATCLI kinds [options] [kind]
```

| argument | |
|---|---|
| `kind` | describe just this one |

Examples:

```bash
python -m PyATCLI kinds                      every analysis and its parameters
python -m PyATCLI kinds event_peth           just one
python -m PyATCLI kinds --plain | cut -f1,2   tab-separated: kind, parameter
```

## info

versions of this library, Python and the numerical packages

```
python -m PyATCLI info [options]
```

Examples:

```bash
python -m PyATCLI info
python -m PyATCLI info --json
```

## serve

Runs the engine: the operations of the other commands as a long-lived process that speaks one JSON object per line on standard input and output, and keeps recordings open between requests. It is what PyAT's window talks to. Requests: {"id": 7, "method": "recording.open", "params": {...}}; {"cancel": 7} stops one; {"method": "shutdown"} ends the process. Arrays travel as .npy files in the data folder, never through the pipe.

```
python -m PyATCLI serve [options]
```

| option | |
|---|---|
| `--data-dir DIR` | where bulk results (arrays, big tables) are written (default: a temporary folder, removed on exit) |
| `--app-name NAME` | the program asking, for the provenance records the engine writes |
| `--app-version VERSION` | its version |

Examples:

```bash
python -m PyATCLI serve                       # then one JSON request per line on stdin, one JSON answer per line on stdout
echo '{"id": 1, "method": "kinds"}' | python -m PyATCLI serve
python -m PyATCLI serve --data-dir ./engine-data -v
```

## docs

print this command line's reference: markdown, or a man page (or the engine's contract)

```
python -m PyATCLI docs [options]
```

| option | |
|---|---|
| `--format, -f FORMAT` | markdown (CLI_REFERENCE.md), man (pyatcli.1), or contract (ENGINE_CONTRACT.json: what the engine's callers can count on) |
| `--out, -o FILE` | write it to this file instead of standard output (- is standard output) |

Examples:

```bash
python -m PyATCLI docs --format man --out pyatcli.1 && man ./pyatcli.1
python -m PyATCLI docs --format markdown --out CLI_REFERENCE.md
```

## Exit status

- `0` — done
- `1` — something failed; the message is on standard error (with `--json`, a JSON error record: code, message, hint)
- `2` — bad usage: an unknown flag, a missing argument, an action that needs a yes and was not given one
- `3` — `reproduce` ran, but an export did not match the result
- `130` — interrupted (Ctrl-C)

## Environment

- `DEBUG` — set to anything but 0/false/no/off: the same as `--debug` (a bug's full traceback)
- `NO_COLOR` — accepted and unused: the command line never uses colour

## The engine

`serve` runs the same operations as the other commands as a long-lived process, speaking one JSON object per line on standard input and output, and keeps recordings open between requests; it is what PyAT's window talks to. A request is `{"id": 7, "method": "recording.open", "params": {"path": "D:/rats/Rat7"}}`; it answers with `progress` events and one `result` or `error` (`{"id": 7, "event": "result", "ok": true, "result": {...}}`); `{"cancel": 7}` stops a request at its next progress report and `{"method": "shutdown"}` ends the process. Large arrays are never sent through the pipe: they are written as `.npy` files in the engine's data folder and named in the result. The protocol is described at the top of `PyATCLI/engine/server.py`.
