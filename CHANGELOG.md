# Changelog — PyATCLI

---

## Version 2026.11.2 — PyATCLI:
  **The Peak Finder gets a start buffer, and the search for a response near events gets a chance level.** Both come from the benchmarks written for the paper
  (synthetic recordings with known truth; the scripts, the criteria written before each run and every result, failures included, are in the `PyAT-paper` repository, which is not public yet):
  on recordings with no transients at all the whole-recording detector found about 22 false five-sigma peaks per hour, nearly all in the first seconds, and the
  search near events reports a "response" by chance in a share of random windows (about one in ten at z = 5 on a real recording).
  Changed defaults (two results change for an unchanged call — set the parameter to its old value for the old numbers):
  - **`find_significant_peaks(..., start_buffer_s=10.0)`, and the `peak_finder` analysis with scope `whole`: no peak is reported in the first 10 seconds.** Only the
    search skips the start: the mean and standard deviation still come from the whole recording, so no peak's z-score changes with the buffer (cutting the start
    out instead was tried and made false peaks *more* frequent with motion in the data, because the standard deviation shrinks). On the benchmark's recordings
    with no transients the false peaks fell from 22 to 0 per hour (new seeds: criteria and numbers in `PyAT-paper/results/b2d_report.md`) at a cost of under
    0.01 in recall. The 10 s was chosen on synthetic data, and the buffer does not remove false peaks that motion artefacts cause later; `0` searches
    everything, as before.
  - **The `peak_finder` analysis near events (scopes `one_event`, `all_same_name`, `scan_all`) runs the shuffle-null by default (`n_shuffles=1000`, `seed=0`):**
    the `alignment` table gets a `p (shuffle)` column, the `scan` table gets `Chance Hit Rate`, `p (shuffle)` and `p (Holm)` (Holm across the event types
    scanned), and the summary gets `chance_hit_rate`, `expected_found_by_chance` and `p_shuffle`. `n_shuffles=0` gives the tables as before. About two seconds
    per 20 events. The engine operation `compute.peaks` keeps `n_shuffles=0` unless asked, so an older client is not slowed.
  New:
  - **`shuffle_null_near_events(time_array, signal, event_times, pre, post, z_threshold, include_troughs, n_shuffles, seed, progress)`** (`PyATCLI.analysis.peak_finder`,
    exported from `PyATCLI`): how often `find_peak_near_events` finds a "response" by chance on the recording itself, and whether the given events do better. The
    whole train of events is moved together by a random circular shift (their spacing is kept) and every shifted event is searched exactly as a real one is. It
    returns the chance hit rate, a permutation p-value for the number of events with a response (`p_count`) and one per event (`p_event`); the p-values are
    conservative when the recording really has responses, never optimistic. Calibration on recordings with no transients (200 per condition, new seeds):
    `p_count` ≤ 0.05 in 1.0–6.0% of trains, `p_event` ≤ 0.05 for 4.9–5.2% of random events; trains locked to real transients came out significant in 99–100% of
    recordings against 2–7% for random times (`PyAT-paper/results/b5_report.md`).
  - `Param.legacy` and `analyses.fill_legacy(kind, parameters)`: what a record made before a parameter existed means by leaving it out. `reproduce_record` fills
    it, so **a Peak Finder record written before this version replays with `start_buffer_s=0` and `n_shuffles=0`** and gives its own numbers, not today's defaults.
    (Running an analysis afresh uses today's defaults.)
  - `compute.peaks` takes `start_buffer_s` (default 10, as above), `n_shuffles` (default 0) and `seed`; with `n_shuffles` the answer carries `null`, one
    shuffle-null result per event type (plus `p_count_holm`, the train-level p corrected across the types asked about together), and the progress
    follows the shuffles. The generated `ENGINE_CONTRACT.json`, `CLI_REFERENCE.md` and the man page list the
    new parameters (additions only: the released protocol 1 still holds).
  Also `analysis.peak_finder.holm_adjust`, the Holm step-down the scan's `p (Holm)` and that `p_count_holm` both use.
  Tests: 894 (31 new): the buffer hides only the start and leaves every z alone; the null's chance rate equals the search's rate on noise, a locked train beats it and a
    noise train does not, the same seed gives the same answer, edge and short-recording cases say why; the analysis tables with and without it; the engine operation;
    an old record replays with the old values; float32 input; progress.

## Version 2026.10.10 — PyATCLI:
  **PhysicsLibrary is now PyATCLI.** It is the analysis library, engine and command line behind PyAT, and PyAT 4.0.0 needs it. This is a new PyPI project:
  `ZaksPhysicsLibrary` stays on PyPI at 2026.9.28 and gets no further releases; version numbers stay release dates (this one is 2026.10.10). The two packages have different import names, so they can sit side by side while you move over; there is no alias for the old name.
  Renamed:
  - `pip install ZaksPhysicsLibrary` → `pip install PyATCLI`; `import PhysicsLibrary as pl` → `import PyATCLI as pl`; `python -m PhysicsLibrary` →
    `python -m PyATCLI`; the `physicslib` command → `pyatcli`; the environment variable `PHYSICSLIB_CHECK_RESULTS` → `PYATCLI_CHECK_RESULTS`; the default
    output folders `physicslib-output` / `physicslib-batch` → `pyatcli-output` / `pyatcli-batch`; the man page `physicslib.1` → `pyatcli.1`; the repository
    `zakgm2/PhysicsLibrary` → `zakgm2/PyATCLI` (the old address redirects). Everything importable from the top-level package keeps its name.
  - Modules that moved into folders (only matters if you imported a module by its path rather than the name from the package):
    `processing_TDT`, `processing_oxysoft`, `dataset` → `sources.processing_TDT`, `sources.processing_oxysoft`, `sources.dataset`;
    `file_parser`, `loaders.tdt_loader`, `loaders.oxysoft_loader` → `sources.load`; `file_parser_generic` → `sources.tables`; `loaders.pt2_loader` →
    `sources.pt2`; `text_field_study` → `fieldstudy.study`; `field_study_validation` → `fieldstudy.validation`; `analysis.group` stays a module path but is
    now a package, and `analysis.group_trials` / `group_stats` / `group_report` / `group_scan` / `group_run` → `analysis.group.trials` / `stats` /
    `report` / `scan` / `run`.
  - Records and settings written by PhysicsLibrary open as before (`reproduce` names a library version that differs from the running one as a warning,
    as it always has); nothing about the files on disk changed.
  New:
  - **Reading results back, with no drawing: `results.open`.** `open_results(path)` (`PyATCLI.results`; the engine operation `results.open` is the
    same) reads a finished analysis from the files it left: a group analysis's folder (or its `analysis_settings.json`), a `.provenance.json` record, a
    folder of records or tables, or one `.csv`. It returns an `OpenedResults`: what made it (`kind`), what it was asked (`parameters`), its `tables`,
    what the analysis said (`notes`) and ran on (`software`), and the `files` it read; `NoResults` (a `ValueError`) if there is nothing to read. It is what
    PyAT's Open Results… asks for. A group analysis's folder holds `analysis_notes.json` (the notes
    and software, which no table does), so a folder read back reports as the run did; and a figure exported into a group's folder no longer changes what
    the folder opens as (the group's settings decide first).
  - **The contract declares what is written besides an analysis's tables** (`outputs` in `ENGINE_CONTRACT.json`, `analyses.describe_outputs()`): a group
    analysis's folder — each table with the columns that are always there and the file it is written to (`analysis.group.report.TABLE_FILES`,
    `GROUP_TABLES`), and the settings, report and notes files — and an Export View's `data` and `markers` tables. Removing or renaming one is a breaking
    change like any other (`engine/contract.py`); tests check that a simulated group analysis writes every declared table, column and file.
  - `run_cli.py`, an entry point for the command line at the top of the repository: run with nothing (or double-clicked, or from a shortcut to it) it opens Git Bash in that folder, shows the commands and leaves the prompt open; run with arguments it is the command line in the terminal you are in. `--terminal` and `--inline` choose which; without Git for Windows it opens a Command Prompt, and anywhere but Windows it is always the command line itself.
  - **Operations and the engine: the app no longer runs the library in its own process.** `ops/` is a layer of JSON-in /
    JSON-out operations (`@operation(name, summary)`, `ops.call(name, params, ctx)`, `OpContext`, `OpError(message, code, hint)`,
    `error_info`, `to_jsonable`; `ops.describe()` lists them) under both the command line and `python -m PyATCLI serve`,
    so the two cannot disagree. `serve` is an engine: one JSON object per line on standard input and output
    (`{"id", "method", "params"}` in; `ready`, `progress`, `warning`, `note`, `result`, `error` out; `{"cancel": id}`;
    `{"method": "shutdown"}`), recordings kept open between requests (`recording.open`, `.view`, `.splice`, `.labels`, `.close`,
    `.list`), one variant per list of splices so Ctrl+Z / removing a splice / Restore Full Recording cost nothing, and arrays handed
    over as `.npy` files to memory-map instead of text through the pipe (`engine/wire.py`; a 17-minute TDT recording is 34 MB,
    one trace as JSON text is 44 MB and a second to parse). PyAT's window talks to it, so a slow load or fit cannot freeze it and a
    crash in a library cannot take it down. New operations for a window: `compute.event_peth`, `compute.peaks`, `group.scan`,
    `group.run` (`tables="files"` hands the tables back as CSVs; the full note records
    and software come with it), `fieldstudy.fields` / `.run` / `.validate`, `file.tables` (with the first rows), `saved.write`.
    `recording.open` returns the recording as recorded when a saved splice was replayed, the names saved for its signals and what
    the saved state restored; `fresh` reads the disk again. `analysis.group.rebuild.results_from_tables` takes the analysis's `notes` and
    `software`, so results rebuilt from files report as the original does. The TDT cache carries `motion_correction_inlier_fraction`
    (refreshed by a splice's re-fit). `Cancelled` (a `BaseException`) is exported from `progress`: a cooperative cancel raised by a
    progress report, which `Plan` does not swallow. See API_REFERENCE.md, "Operations and the engine".
  - For the rest of what a window does with the disk: `files.inspect` (what a path is: a TDT block, a folder with blocks under it — walked
    at any depth —, an Oxysoft export or a table file), `pt2.load` (the EFNMR image as an `.npy` file to map), and `provenance.write`
    (the record beside an export — the recording's fingerprint, remembered while the folder is unchanged, its state, the pipeline's settings,
    the files made). `group.run` can save the wizard's settings first
    (`write_settings`, `scans`), so a run that fails still leaves them, and its answer carries `settings`. The `curve_fit` summary says the
    `view` (the stretch of signal the figure shows around the fit). `compute.event_peth` / `compute.peaks` / the `run` analyses are what
    PyAT's windows now ask the engine for; their answers are the tables the windows export.
  - **The command line follows the usual conventions** (`cli/` is now a package: `parser.py`, `commands.py`, `render.py`,
    `docs.py`). Every command takes `--json` (one document `{"schema", "ok", "command", "result", "warnings"}` on standard output;
    failures as `{"ok": false, "error": {code, message, hint, type}}` on standard error), `--plain` (tab-separated rows), `-q`,
    `-v`, `-d/--debug` (also `DEBUG=1`), `--no-input`, `--events` (JSON event lines on standard error), `--progress` /
    `--no-progress`, before or after the command name. Standard output is only ever the result: the TDT SDK's own chatter no
    longer breaks `--json` (it used to print into it). No arguments prints a short help; each command's `--help` has examples;
    `-` means standard input or output; `saved clear` asks when it can and says `--yes` when it cannot; Ctrl-C exits 130; an
    unexpected exception is reported as a bug rather than a traceback. `group-setup` is `group setup` and `group SETTINGS` is
    `group run SETTINGS` (the old spellings still work, with a warning).
  - **`python -m PyATCLI docs`** writes the command line's reference — `--format markdown` (`CLI_REFERENCE.md`) or `--format
    man` (`man/pyatcli.1`) — generated from the parser; a test regenerates both and fails if the committed files differ.
  - Everything between "open a recording" and "run an analysis" that PyAT used to do itself now lives
    in the library, so a script gets exactly the numbers the app shows. `recording.py`: the saved
    state beside a recording (`saved_state_dir`, `marker_sidecar_path`, `splice_sidecar_path`,
    `read_saved_splices`, `read_saved_markers`, ...), building a cache from a load (`make_tdt_cache`,
    `make_oxysoft_cache`, `make_generic_cache`, `build_signal_options`, `normalized_signal`),
    splicing (`apply_splice`, `apply_splices`, `spliced_store_name` — a TDT splice re-fits the dF/F
    on what is left, as the app does), the one-call headless load `load_tdt_recording` (processed,
    saved splices replayed, saved markers on top) and `find_tdt_recordings`. `markers.py`:
    `marker_display_label`, `store_display_name`, `group_markers_by_name`.
    `analysis/group/scan.py` and `analysis/group/run.py`: `scan_recording(s)`,
    `default_subject_names`, `load_group_recording`, `run_group_analysis` (headless; figures are
    the caller's, via `extra_writers`), and `write_group_settings` / `read_group_settings` for the
    `analysis_settings.json` that makes a group analysis repeatable.
  - `CURVE_FIT_MODELS` in `models.py`: the curve-fit registry (model, initial-guess function,
    parameter names) the app's Curve Fit tool lists, importable by scripts.
  - `numeric.py`: float64 is now a rule of the library, enforced rather than assumed. TDT hands its
    streams over as float32, and numpy is unsafe on single precision at a realistic length
    (`np.polyfit` on a float32 regressor drops the slope term past ~3 minutes at 1 kHz). Every file
    load returns float64 — `get_tdt_struct` widens each stream as it is read, so
    `process_tdt_folder`'s raw per-wavelength `channels` are float64 too (they were the one place
    float32 survived into the app's analyses), and so are the Oxysoft readers, `load_any_file`,
    `load_pt2` and `Dataset.signals` — and every public function that takes a signal, a time axis
    or a trial matrix widens its input, so arrays from a script's own pipeline are covered as well.
    Widening is exact: results for float64 input are unchanged, and float32 input now gives
    bit-identical results to the same values as float64.
  - Scripted use: every analysis PyAT offers can be run by name, from Python or the command line, and any
    result can be reproduced from the record written beside it.
    `analyses.py`: `run_analysis(kind, cache, parameters)` for `zscore_window`, `auc_window`, `fft_window`,
    `curve_fit`, `event_peth`, `peak_finder`, `event_intervals` and `process`, returning an `AnalysisResult`
    (`tables`, `parameters` as run with the defaults filled in and anything looked up in the recording, such as an
    event's times, written out, `summary`, `notes`). Each table has the columns, in order, of the CSV PyAT exports
    for that analysis, so a script's output compares column for column. `kinds()`, `describe_kinds()`,
    `normalize_parameters()`, `ParameterError`, `AnalysisError`.
  - Provenance: `provenance.py` describes how a result was produced as data — the software (library, Python,
    platform, numerical packages, the app), a fingerprint of the recording's files, its state (splices in order,
    motion-correction method, marker counts and digests, store renames, signal names), what the pipeline does that
    no caller chose (filter, bleaching model, motion-correction settings, z-score baseline rule, read from the
    constants the code uses), the analysis parameters and the output files with hashes. It holds no recording
    data. `build_record`, `write_record`, `read_record`, `recording_state`, `file_fingerprint`,
    `pipeline_parameters`, `software_versions`, `check_environment` and friends; PyAT writes
    `<export>.provenance.json` beside every export.
  - `reproduce.py`: `reproduce_record(record, recording=None, out_dir=None, compare=(), ...)` rebuilds the
    recording in exactly the recorded state, runs the analysis with the recorded parameters and compares any
    exports column for column (`compare_table`; `rounded=True` also accepts the digits an export shows, which is
    what PyAT's fixed-decimal exports need). It reports a changed recording or software as warnings, or raises
    `ReproduceError` with `strict=True`. (`PyATCLI.reproduce` stays the module; the function's top-level
    name is `reproduce_record`.)
  - **Drawing is PyAT's.** The library computes and writes numbers and draws nothing: it has no plotting code and no `matplotlib` dependency. The
    figures — the analysis windows', a group analysis's, Export View, Edit Graph Attributes, and writing a figure to PNG / TIFF / PDF / SVG with the CSV
    of its data — are `physicsanalysis_qt/figures` in PyAT, which draws from the tables the library writes (the columns each one always has are in
    `ENGINE_CONTRACT.json`). A script that wants a picture reads the tables (`run_analysis`, `open_results`) and plots them itself.
  - Command line: `python -m PyATCLI` (and a `pyatcli` command): `info`, `kinds`, `run`, `reproduce`,
    `batch` (many analyses over many recordings, with combined tables) and `group` (from an
    `analysis_settings.json`). Exit status 0 done, 1 failed, 2 bad usage, 3 an export did not match.
  - Export View's numbers: `viewplan.py` says what the main plot of a recording shows — `plan_view(cache, signals=, xlim=, ylim=, store_labels=, attributes=)`,
    `view_traces`, `padded_xlim`; a `ViewPlan` has the lines, the range, the markers in view and `data_table()` / `markers_table()` / `write_data()` /
    `parameters()` — and `run_view(cache, parameters)` gives them as an `AnalysisResult`, which `reproduce_record` replays for a `view` record and compares
    with its data CSV, so an Export View can be reproduced like any analysis. A planned line says what it is (`role`, `signal`, `key`, `csv_name`) and
    never how it looks: PyAT styles and draws it.
  - The rules for naming signals are the recording's: `name_problem`, `plan_signal_renames` (`recording.py`); a name must say something and two signals
    cannot share one, and `saved_state.rename_signals` refuses a name two signals would share, as the app does.
  - `analysis/group/rebuild.py`: `results_from_tables(tables, parameters, notes, software)` puts a group analysis's `GroupResults` back together from its
    tables, for a viewer that shows a result that is no longer in memory.
  - Where results go: `outputs.py` is the layout PyAT's exports use under an Output folder —
    `<Output>/<TDT|Oxysoft|Generic|PT2|Text Field Study>/<recording> (Analysis)/` and `<Output>/Group Analysis/<group>/`
    — with its rules: the " (Analysis)" suffix, a marker file that tells two recordings of one name apart, and " (2)"
    instead of overwriting (`recording_dir`, `group_dir`, `unique_path`, `unique_stem`, `safe_folder_name`). PyAT now
    calls these; the command line writes into the same folders with `--output-folder DIR` on `run`, `batch`, `group`
    and `group-setup` (`--out DIR` still writes exactly where it is told).
  - Saved state, written: `saved_state.py` writes what PyAT's Save Changes writes into a recording's `JSON saves/`
    folder — markers, splices, signal names, highlights, text boxes — and PyAT's Save Changes now writes through it, so
    the files are the same bytes either way. The writers (`write_saved_markers`, `write_saved_splices`,
    `write_saved_signal_labels`, `write_saved_annotations`, `clear_saved_state`) refuse what the app could not read
    back; the edits do what one action in the app does to what is already saved (`add_marker`, `remove_markers`,
    `add_splice`, `remove_splice`, `rename_signals`, `add_highlight`, `add_text`, `saved_state_summary`; a splice stacks
    on the saved ones, on the time axis as it now stands). `load_recording` now restores saved highlights and text boxes
    too (`cache["highlights"]`, `cache["text_annotations"]`, as the app does); `read_saved_annotations`.
    `python -m PyATCLI saved show|marker|unmark|splice|unsplice|rename|highlight|text|clear RECORDING ...`
    (`clear` needs `--yes`).
  - `group-setup` on the command line is the Hypothesis Testing wizard: it finds the recordings under a folder, lists the
    markers they share (without `--marker`), takes the wizard's choices as options (windows, signal, smoothing,
    measures, peak direction, correction, store renames, `--exclude`), checks the design as the wizard does and writes
    `analysis_settings.json`; `--run` goes on to run it. `group SETTINGS --out DIR` now leaves the settings beside the
    tables, so the folder opens under the group's own name and can be reproduced on its own.
  - `load_recording(path, ...)`: opens a TDT folder, an Oxysoft export or a generic table file the way PyAT
    does, with saved splices, markers and signal names applied (`use_saved=False` for the recording as recorded).
  - Signal names: the names a person gives a recording's signals in PyAT are saved as `signal_labels.json` in
    `JSON saves/` and applied on load; `signal_labels_path`, `read_saved_signal_labels`, `apply_signal_labels`,
    `current_signal_labels`.
  - `_version.py`: `__version__` (and every provenance record) reports a source checkout's own `pyproject.toml`
    version instead of the stale one an editable install's metadata keeps until pip is re-run.
  - **The engine's contract is written down and guarded.** Each analysis declares the tables it returns and the columns of each that are always there
    (`analyses.TABLES`; `kinds` / `describe_kinds()` show them), and `engine/contract.py` builds the contract — the protocol number, every operation with its
    parameters, every analysis with parameters and tables — as `ENGINE_CONTRACT.json` (`python -m PyATCLI docs --format contract`).
    `breaking_changes(old, new)` says what a caller written against the old one would lose: an operation, parameter, analysis, choice, table or
    always-there column that is gone, a parameter that must now be given, a type that changed, a protocol that went down. `ops.PROTOCOL_VERSION` is raised for
    those and only those, and `tests/test_engine_contract.py` fails when the engine no longer honours `tests/contracts/protocol_N.json` (the contract as
    protocol N began), when the file is stale, or when an analysis makes a table or lacks a column it declared. The same goes for what an operation returns
    (`@operation(name, summary, result=[...])`: the keys of its result that are always there; `PYATCLI_CHECK_RESULTS=1` makes `ops.call` fail loudly when one is
    missing, and the tests run with it on) and for the loaded recording (`recording.CACHE_KEYS`: the keys a TDT, Oxysoft or generic recording always has). CI also
    runs PyAT's own tests against each push of the library (the `gui-compat` job), so a change the app cannot live with is seen before it is released. See
    API_REFERENCE.md, "The engine".
  Changed:
  - **The command line and the engine draw nothing.** `python -m PyATCLI` computes and writes files (CSV tables, JSON records, settings); it never makes a figure
    or opens a window, and a run writes nothing to a cache folder in the home directory. `tests/test_cli_conventions.py` checks that no command, option or
    operation draws and that a whole run imports neither matplotlib nor a window toolkit.
  - **The modules are in folders by what they do** (nothing changed in what they do). `sources/` is everything that reads a recording from
    disk: `load.py` (`load_dataset`, `load_dataset_file`, `load_tdt`, `load_oxysoft`, `load_oxysoft_file` — the old `file_parser.py` and the
    `loaders/tdt_loader.py` and `loaders/oxysoft_loader.py` that only wrapped a reader's result, now one module, and `loaders/` is gone),
    `dataset.py`, `processing_TDT.py`, `processing_oxysoft.py`, `tables.py` (was `file_parser_generic.py`) and `pt2.py` (was
    `loaders/pt2_loader.py`). `fieldstudy/` holds the text-field study (`study.py`, was `text_field_study.py`) and its validation
    (`validation.py`, was `field_study_validation.py`). `analysis/group/` is the group analysis as a package: `analysis/group.py` is its
    `__init__.py` and `group_trials.py`, `group_stats.py`, `group_report.py`, `group_scan.py`, `group_run.py` are `trials.py`, `stats.py`,
    `report.py`, `scan.py`, `run.py` in it. Everything exported from `PyATCLI` itself is as it was; code that imported one of these modules by its path needs the new path (the table under
    Renamed, above).
  - `curve_fit` takes `window_s` (the signal shown either side of the fitted stretch, as the app's Curve Fit window does)
    and its result carries a `context` table (that signal) and a `Model` column on `curve`, so the figure can be rebuilt
    from the CSVs; a segment whose fit failed is kept with its fit as NaN. A group analysis result carries a
    `mean_traces` table (what `mean_traces.csv` holds).
  - `_parse_oxysoft_txt` and the folder / single-file readers moved out of
    `loaders/oxysoft_loader.py` into the new `processing_oxysoft.py` (public:
    `process_oxysoft_folder`, `process_oxysoft_file`), the counterpart of `processing_TDT.py`.
    `loaders/oxysoft_loader.py` now only wraps the result into a `Dataset`, as
    `loaders/tdt_loader.py` does. `load_oxysoft` / `load_oxysoft_file` and their results are
    unchanged.
  - `load_pt2` returns float64 (it was float32; the file stores float32).
  - `get_plot_data` returns the sampling rate as a Python `float`.
  - CI only: `actions/checkout` (v4→v7), `actions/setup-python` (v5→v7), `actions/upload-artifact` (v4→v7) and `actions/download-artifact` (v4→v8) in both
    workflows, all on the Node 24-native release of each (GitHub is deprecating the Node 20 runtime they were built on). `CITATION.cff` brought up to date.
  Fixed:
  - A `run` request now reports progress as stages (opening the recording, then the analysis) and ends at 100%, so an analysis that is one
    call with nothing to tick (a curve fit, a z-score window) no longer leaves a progress display at 0% until it vanishes.
  - `API_REFERENCE.md` and `README.md` described a structure that no longer existed: every code
    example in the API reference had its underscores backslash-escaped (a Word-export artefact, so
    `pl.process\_tdt\_folder(...)` could not be pasted into Python), `annotate_fft_peaks` was
    documented though the library has never drawn anything, the README listed an `analysis.py`
    that is a package, and the optional-dependencies section said `statsmodels` was used by one
    function. All brought in line with the code, and every documented name checked to import.
  Known, not changed:
  - For a folder of Oxysoft exports each event's `sample` is its index within its own file; it is
    not shifted by the length of the files before it (documented by an expected-failure test).
  - `validate_tdt_folder` raises `FileNotFoundError` for a path that does not exist, rather than
    returning `(False, message)`.
  - The registry's sinusoid starts every fit at 1 Hz, so it finds a signal only if the phase it
    drifts from that guess across the segment stays small.
  - A splice whose `mode` is neither `"cut_out"` nor `"keep_inside"` is read as `"keep_inside"` (as PyAT has
    always read it) rather than rejected.

---

## Version 2026.9.28:
  Fixed:
  - Group analysis with two or more markers raised `AttributeError: 'PandasData' object has no attribute
    'design_info'` on a fresh install. `2026.9.27` (yanked from active use by this release; the CI test job
    that would have caught it ran *after* the PyPI publish job and failed) read a patsy-specific attribute
    off the fitted mixed model to reconstruct each marker's design row; `statsmodels` 0.15.0, released the
    same day, switched its formula backend from `patsy` to `formulaic` by default and dropped that
    attribute. `group_stats.py`'s `_design_rows` no longer touches the formula engine's internals at all —
    it reads the `C(marker)[T.<level>]` dummy-column names straight off the fitted result, a naming
    convention unchanged across both engines — so this works on `statsmodels` 0.14.x and 0.15.x alike.
    Verified in a clean install of each. No API change; single-marker analyses were never affected.

---

## Version 2026.9.27:
  New:
  - Group analysis (Hypothesis Testing across recordings): `analysis/group.py` (`GroupSpec`,
    `marker_index`, `common_markers`, `design_summary`), `analysis/group_trials.py`
    (`extract_group_trials`, `measures_for_trial`, `decay_time`), `analysis/group_stats.py`
    (`fit_group_models` -> `GroupResults`) and `analysis/group_report.py`
    (`write_group_results`, `group_report_text`). Fits a linear mixed-effects model
    (statsmodels `MixedLM`, REML, Powell/Nelder-Mead optimizer) per measure (AUC, peak
    amplitude, mean amplitude, latency to peak, decay time) with random intercepts for
    subject and subject x marker, then Holm-corrected pairwise post hoc comparisons using
    subject-based degrees of freedom rather than statsmodels' own Wald z. A single-marker
    design tests AUC/mean amplitude against zero (peak/latency/decay have no null value and
    are reported as estimates with 95% CIs only). Every fit is cross-checked against subject
    x marker means as an independence-free sanity check. New dependencies: `statsmodels`,
    `scikit-learn`. 92 new tests across `tests/test_group_setup.py`,
    `tests/test_group_trials.py` and `tests/test_group_stats.py` (262 total, including the
    34 progress-toast tests below).
  - `progress.py` (`Plan`, `track`): a small `tqdm`-based progress-reporting helper. Thirteen
    slow functions (loaders, splice, motion correction, Event PETH, Peak Finder, text-field
    study/validation) accept `progress=None|True|callable` to report percent-complete without
    printing to stdout when driven by a GUI. New dependency: `tqdm`. 34 new tests in
    `tests/test_progress.py`.
  - `get_tdt_struct` accepts `read_streams=` to skip loading stream data when only marker/epoc
    structure is needed (used by group analysis's marker-scan step to stay fast on long
    recordings), and `replay_splices` reapplies a recording's saved splice history to a
    freshly loaded struct.

---

## Version 2026.9.24:
  Fixed:
  - Motion correction is now fitted in float64. TDT delivers float32, and `np.polyfit` on float32
    data silently drops the slope term once a recording is long (about 200,000 samples for a 415 nm
    channel of 25 ± 1, i.e. ~3 minutes at 1 kHz): it returned a line at roughly half the true slope
    with a large intercept, with only a `RankWarning`. That broke the OLS method (the default since
    2026.9.22) and RANSAC's noise estimate on any longer recording; on a 17-minute test recording the
    default's peak PETH z-scores were 2-3x too large. RANSAC and Huber results barely change.
  - `compute_dff` (two-channel): ΔF/F is now the motion-corrected residual divided by the 465
    channel's bleaching baseline. It used to divide the residual by the bleaching trend of that same
    near-zero residual, so the values were not ΔF/F and their scale changed by orders of magnitude
    with the regression method (a simulated 10% transient read +1,650% with Huber and +399,000% with
    OLS). `f0` now returns the baseline fluorescence; `raw` is unchanged. Everything is computed in
    float64.
  - `get_zscore_slice` no longer clips the signal to ±5 in raw units. That made z-scores depend on
    the units of the signal, returned all zeros for signals sitting above 5, truncated large
    responses and biased latencies early. The baseline mean/SD are now taken after winsorising the
    baseline at 5 robust SDs (so a brief artifact still can't inflate the SD), and the flat-baseline
    test is relative instead of an absolute 1e-6.
  - `correct_bleaching` and the Huber regression no longer depend on the units of the recording
    (the same data ×1000 gave a baseline up to 4.5% different, and a Huber inlier fraction of 0.2%).
  Changed:
  - `find_significant_peaks` and `find_peak_near_events`: the default `z_threshold` is now 5 (was
    2.5). The response search takes the largest z in the window, so 2.5 reported a "response" for
    about 58% of random pseudo-events on a real recording (10% at 5).
  - License: declared as AGPL-3.0-only in `pyproject.toml`, `README.md` and `CITATION.cff`, matching
    the `LICENSE` file. Earlier releases' PyPI metadata said MIT while the shipped LICENSE was
    AGPL-3.0. `pyproject.toml` now asks for `setuptools>=77`, the version that understands that field.
  - `API_REFERENCE.md` brought in line with the code: the new defaults, the meaning of `raw` / `f0`,
    the z-score baseline, and the `regression_method` default shown as `"ols"` (it still said
    `"ransac"`).
  - Docstrings and the README say PyAT instead of "Physics Analysis GUI".
  New:
  - `tests/`: 136 automated tests (`python -m unittest discover -s tests`). Each checks an analytic
    result, an independent reference (statsmodels, scipy) or a simulated recording with a known true
    ΔF/F. `.github/workflows/tests.yml` runs them on pushes to `main` and on pull requests.
  - `CITATION.cff` (GitHub's "Cite this repository").

---

## Version 2026.9.22:
  New:
  - `compute_dff(y_465, y_415, fs, regression_method)` factored out of `process_tdt_folder` —
    the motion-correction + bleaching-correction + ΔF/F + denoise steps, callable on their own
    against any raw 465/415 signal pair, not just a fresh TDT block load. Built for
    PhysicsAnalysis's Splice "Cut out this range": re-running this on a stitched, artifact-removed
    signal gives a genuinely improved correction (the artifact no longer drags the fit) rather
    than just a shorter view of the original one. `process_tdt_folder` now calls this internally
    instead of duplicating the same steps inline.

  Changed:
  - `analysis.py` (824 lines, every analysis tool's numeric implementation in one file) split
    into an `analysis/` package, one file per tool — `shared.py`, `zscore_peth.py`,
    `event_peth.py`, `auc.py`, `fft.py`, `peak_finder.py`, `curve_fit.py`, `intervals.py` — mirroring
    how physicsanalysis_qt/analysis/ already splits the GUI side of each tool. `analysis/__init__.py`
    re-exports every name under the same `PhysicsLibrary.analysis` namespace, so this is purely
    internal reorganization — no public name moved, renamed, or changed behavior, except
    `annotate_fft_peaks` (see Removed below).

  Removed:
  - `annotate_fft_peaks` — it drew matplotlib annotations (fontsize, arrow styling, a vertical
    line) directly onto a caller-supplied Axes, which is presentation logic that doesn't belong in
    a library with no other GUI/rendering code anywhere in it. The peak-finding it depended on
    (`find_fft_peaks`) stays here; the drawing moved to physicsanalysis_qt's own analysis/fft.py
    as a local `_annotate_fft_peaks`, the only place that ever called it.

  Fixed:
  - `splice_cut_out` refused to cut a range touching either edge of the recording (it required
    "usable signal on both sides" of the cut), so removing e.g. a bad first few seconds wasn't
    possible — only requires at least 2 samples to remain overall now. Also fixes a latent
    IndexError this same border case would have hit (`x[i1]` when the cut ran to the very end,
    i1 == len(x)).

  Changed:
  - Default `regression_method` for TDT motion correction is now `"ols"` instead of `"ransac"`
    (`_robust_linear_fit`, `process_tdt_folder`, `load_tdt`, `load_dataset`). RANSAC excludes
    whatever it judges an outlier from the fit but still applies the fit there — a real, large
    biological transient that merely looks unusual gets a distorted correction, not just an
    imperfect one. OLS never actively excludes or downweights a sample, so it can't misjudge
    signal as noise that way; its own weakness (the fit line can get dragged toward a severe
    artifact) is a known, predictable trade-off instead. RANSAC/Huber remain available and are
    worth trying per-recording when artifacts are severe and occasional enough to favor excluding
    them outright.

---

## Version 2026.8.21:
  Fixed:
  - .github/workflows/publish.yml's environment.url pointed at https://pypi.org/p/PhysicsLibrary —
    the actual published project is named ZaksPhysicsLibrary (pyproject.toml's [project] name has
    always said so), so that URL named a project that doesn't exist. Cosmetic only (it's just the
    link shown in GitHub's environment UI, not something that affects what actually gets built or
    uploaded), found while diagnosing why the 2026.8.20 release ended up published manually instead
    of through this workflow.

---

## Version 2026.8.20:
  (Switched to date-based versioning (YYYY.M.D) from this release on — PhysicsAnalysis stays on
  its own numbered scheme, this library just no longer tracks it 1:1, so a date reads more
  honestly than a semver bump would.)

  New:
  - Motion correction's isosbestic-vs-signal regression is now a choice, not just RANSAC:
    _robust_linear_fit()/process_tdt_folder()/loaders/tdt_loader.py's load_tdt() all take a new
    regression_method parameter ("ransac" | "huber" | "ols", the valid set exported as
    REGRESSION_METHODS). Huber down-weights outliers instead of excluding them outright — gentler
    than RANSAC's hard in/out cut, better suited to noise that's elevated throughout a recording
    rather than concentrated in a few bad stretches. OLS is a plain least-squares baseline with no
    robustness at all, included for comparison. Defaults to "ransac" everywhere, unchanged from
    before this existed.
  - New analysis.py helpers backing the GUI's AUC and Event PETH tools: compute_auc_from_trace()/
    compute_auc_window()/compute_auc_matrix() (area under a z-score trace, split into
    total/positive/negative) and compute_peri_event_from_trace()/compute_peri_event_matrix() (peak
    amplitude, latency, and mean time-bins around an event, single-trace or across a whole trial
    matrix). find_fft_peaks() factored out of annotate_fft_peaks() (which now calls it and only
    handles the drawing) so a caller building a stats/export panel can get the same peak list
    without needing a matplotlib axes to draw on.
  - estimate_sample_rate(), mean_channels(), compute_group_stats(), compute_marker_intervals():
    small formulas (median-diff sample rate, 2D-detector-array channel averaging, trial mean/SEM,
    marker-to-marker interval diffing) that were previously duplicated across several call sites in
    the GUI, now defined once here.

  Changed:
  - splice_keep_inside()/splice_cut_out()'s extra_channels slicing is now axis-agnostic
    (y[..., i0:i1] / np.concatenate(..., axis=-1) instead of y[i0:i1] / np.concatenate([...])) — a
    2D multi-detector array (e.g. Oxysoft's o2hb/hhb, shape (n_detectors, n_samples)) now splices
    correctly alongside the 1D case this already handled, with no special-casing needed by callers.
  - Added openpyxl to dependencies — loaders/file_parser_generic.py's Excel support has needed it
    all along; it just wasn't declared, so a fresh install without pandas' own optional Excel
    extras would fail on the first .xlsx file.

---

## Version 1.10.0:
  New:
  - process_tdt_folder() now returns "motion_correction_inlier_fraction": the fraction (0-1)
    of samples RANSAC kept as inliers when fitting the 415-vs-465 motion correction line, or
    None if there was no isosbestic reference stream to correct against. Lets a caller (e.g. the
    GUI) show how much of the recording was judged to be motion artifact and excluded from the fit.

  Changed:
  - get_event_markers() now skips the 'Tick' epoc store entirely. It's TDT's own 1-second
    heartbeat/reference signal, not a behavioral event — every recording has it, nothing ever
    wants to plot/analyze/splice against it, and its regularity would otherwise flood every
    marker and event-name picker a caller builds from this function's output.

  Fixed:
  - _robust_linear_fit()'s noise estimate (used to set RANSAC's residual_threshold) was based on
    consecutive-sample differences, which measures noise at the timescale of one sample —
    fine in principle, but at TDT's actual sampling rates (~1 kHz, adjacent samples ~1ms apart)
    that's dominated by ADC/thermal noise far smaller than the residual scale a real 415-vs-465
    regression produces once genuine biological signal is accounted for. Confirmed on a real
    recording (PFC-GCaMP-Sucrose): diff-based noise estimation set a threshold ~350x tighter
    than the fit's actual residual scale (OLS residual std ≈ 5.19 vs. a 0.0146 threshold),
    rejecting 98% of samples as "outliers" — including essentially all real signal. Replaced
    with the MAD of an initial OLS fit's own residuals, which is on the right scale by
    construction and stays robust to the artifacts it's meant to exclude; same recording now
    lands at ~95% inliers. (1.9.1's sqrt(2)/duplicate-scaling fix and this session's earlier
    zero-diff guard were both correct fixes to the diff-based approach, but the approach itself
    was the wrong tool at this sampling rate — this replaces it rather than patching further.)

---

## Version 1.9.1:
  Fixed:
  - processing_TDT.py: _robust_linear_fit()'s noise estimate (used to set RANSAC's
    residual_threshold — see 1.9.0 below) applied the MAD-to-standard-deviation scale factor
    twice (`/ 0.6745 * 1.4826`, where 1.4826 ≈ 1/0.6745), inflating the estimate by roughly 2x,
    and never corrected for consecutive-sample differencing doubling the variance
    (Var(y[i+1]-y[i]) = 2·Var(y) for independent samples) — so the residual threshold RANSAC
    used was calibrated well above the recording's actual noise level, meaning it excluded fewer
    genuine outliers than intended. Corrected to `MAD / (0.6745 * sqrt(2))` — each factor applied
    exactly once. Verified against synthetic data with known noise: the old formula estimated
    noise at roughly 2x the true level; the corrected one lands within ~2%.

---

## Version 1.9.0:
  Changed:
  - processing_TDT.py: process_tdt_folder() now motion-corrects (regresses the isosbestic/415
    stream onto the signal/465 stream) with RANSAC robust regression (sklearn's
    RANSACRegressor) instead of ordinary least squares (np.polyfit). OLS lets exactly the kind
    of thing this regression exists to remove — a burst of motion artifact, a fiber-cord twist —
    drag the fitted line toward itself, corrupting the "motion-free" prediction across the whole
    recording rather than just where the artifact happened; RANSAC instead fits to random small
    subsets, keeps whichever gets the most points within a residual threshold ("inliers"), and
    does one final fit on just those — points that never look like they belong to the same line
    are excluded from the fit entirely, not merely downweighted. The residual threshold is set
    explicitly (3x a robust noise estimate from consecutive-sample differences) since sklearn's
    own default is calibrated for roughly-flat data and comes out far too loose here (y's spread
    is dominated by the real 415-vs-465 trend, not noise) — left at the default, RANSAC would
    accept every point as an inlier and silently degrade to a plain OLS fit. Verified against
    synthetic data with an injected outlier burst: recovers the true slope/intercept almost
    exactly where OLS was measurably pulled off, and against a real recording with a
    poorly-conditioned 415/465 relationship (OLS previously threw `RankWarning: Polyfit may be
    poorly conditioned`) where the bleaching-trend baseline that depends on this fit went from
    swinging across 6 orders of magnitude (0.5 to 117,177 — a failed fit) to a tight, sane 0.49–0.88
    range. New dependency: scikit-learn (was already present transitively via
    sentence-transformers, now declared explicitly). This changes dF/F output values for every
    TDT recording that has an isosbestic reference stream — not a bug fix to a wrong number, but
    a real change in which regression method computes it.
    
  New:
  - processing_TDT.py: process_tdt_folder() now also returns "channels" — a list of
    {"key", "label", "y"} for each raw per-wavelength stream found (always "main_driver",
    plus "isosbestic" too if a 415 reference stream exists), the un-motion-corrected,
    un-normalized traces. Lets a caller plot/analyze the isosbestic control or the main
    driver/probe channel directly instead of only the final "raw"/"corr" result. Nothing about
    stream naming or count is hardcoded beyond the existing 465/415 detection, so a recording
    with no isosbestic stream simply has no "isosbestic" entry — still works.
  - splice.py: splice_keep_inside()/splice_cut_out() take an optional extra_channels dict of
    {name: array} — any other arrays sample-aligned with x (e.g. the new raw per-wavelength
    channels above) that need trimming/cutting identically to x/raw/corr, returned back under
    "extra_channels". Optional and backward compatible: omitting it behaves exactly as before.

---

## Version 1.8.0:
  New:
  - processing_TDT.py: debounce_events(times, min_isi) — collapses switch-bounce/double-tap
    duplicate event timestamps (e.g. a lever registering more than one contact for what was
    physically a single press) by sorting the timestamps and dropping anything within min_isi
    seconds of the previous *kept* event. Independent of any fixed-ratio schedule (FR1, FR3, ...)
    — it works on raw inter-event spacing, not an expected press count, and a min_isi comfortably
    below the animal's real max press rate leaves genuinely fast consecutive presses (e.g. an FR3
    burst) intact while still removing sub-threshold hardware bounce.

---

## Version 1.7.0:
  New:
  - splice.py: splice_keep_inside(x, raw, corr, markers, detected_markers, start, end) and
    splice_cut_out(x, raw, corr, markers, detected_markers, start, end) — non-destructive
    time-range edits of a signal (trim to a range, or remove one and stitch the remainder back
    together). cut_out shifts everything after the removed range backward by its duration so the
    timeline stays contiguous — a gap in x would break any downstream analysis assuming uniform
    sampling right at the cut boundary — and drops/shifts marker dicts to match. Pure array/dict
    operations, no knowledge of any GUI's state shape.

  Fixed:
  - loaders/oxysoft_loader.py: load_oxysoft_file() failed with a cryptic
    "not enough values to unpack (expected 2, got 1)" when a file's Legend block didn't match the
    expected O2Hb/HHb column format (o2hb ended up shape (0,), a 1-tuple). Now raises a clear
    ValueError naming the actual problem.

---

## Version 1.6.0:
  New:
  - analysis.py: compute_event_zscore_peth(time_array, signal, event_times, pre, post, num_bins) —
    Z-scores every occurrence of one event type against its own pre-event baseline and aligns
    them into a trial x time matrix (plus mean/SEM traces), for a GuPPy-style stacked-heatmap
    PETH rather than a single click-triggered one.
  - analysis.py: find_significant_peaks(time_array, signal, z_threshold, min_distance_sec,
    include_troughs) — z-scores the whole recording against its own global mean/std and returns
    local peaks (scipy.signal.find_peaks) at or above threshold, for finding candidate events
    directly from the signal rather than trusting externally-supplied event markers.
  - analysis.py: find_peak_near_events(time_array, signal, event_times, pre, post, z_threshold,
    include_troughs) — checks whether a statistically significant peak actually shows up near
    each given event time (single event or many occurrences of one type), reporting found/
    latency/z-score per event, so alignment between a marker and the real signal can be verified
    rather than assumed.

  Fixed:
  - processing_TDT.py: get_event_markers() — a level/buffered-logic epoc store already "high"
    the instant recording started got a spurious onset marker at exactly t=0 (TDT's synthetic
    starting-state entry, since it has no signal history before t=0 — not a real event). Now
    filtered out, mirroring the existing offset == inf guard for the opposite edge case (a
    strobe still active at the recording's end).

---

## Version 1.5.0:
  New:
  - text_field_study.py: a domain-agnostic pipeline for a study where each subject produces
    one JSON file with several free-text fields, letting a caller pick any pair of fields to
    compare directly (e.g. does the answer to one question track another for the same
    subject) — nothing about which fields exist or which pairs to compare is hardcoded.
    run_field_study_pipeline() is the one-call entry point: loads every file matching a glob
    pattern in a folder into a DataFrame, flags (never drops) near-empty responses as a data
    quality signal, embeds only the fields actually referenced by a comparison or delta pair
    with a sentence-transformers model, optionally computes a delta-vector magnitude between
    two fields, and optionally computes per-pair cosine similarity for each subject plus a
    permutation test (p-value, effect size) of whether same-subject similarity beats a
    shuffled-pairing null, and a word-count-vs-similarity confound check. peek_fields() reads
    just the field names out of a folder's first file so a caller can show a user their
    actual fields before picking which to compare.
  - field_study_validation.py: statistical validation for the paired-similarity metric above.
    run_validation_pipeline() (or build_validation_summary() if you already have similarity
    results) produces one row per field pair: a permutation-test p-value (with
    Benjamini-Hochberg FDR correction across every pair tested together), Cohen's d effect
    size against the pooled null distribution, an OLS regression of similarity on both
    fields' word counts (statsmodels — checks whether the effect is just a "longer answers
    look more similar" artifact), a bootstrap 95% confidence interval on the mean similarity,
    and a leave-one-out sensitivity check flagging any subject whose removal shifts the mean
    by more than one standard deviation of the leave-one-out distribution. Every function's
    docstring explains what its statistic means, not just how to call it. Every one of these
    functions returns NaN rather than crashing when there isn't enough data for a statistic
    to be defined (e.g. a correlation needs at least 2 subjects) — a genuinely undefined
    result, not an error.
  - New dependencies: pandas, sentence-transformers, statsmodels.

---

## Version 1.4.0:
  New:
  - get_event_markers() now emits both edges of every epoc's state — onset ("high") and
    offset ("low") — tagged with a new 'phase' key, instead of only onset. A pump or light's
    on-duration needs both edges; something like a lever press usually only needs the
    onset, so a caller now gets to choose per store instead of the library deciding for it.
    The 'Note' store is unaffected — its markers are instantaneous free-text annotations,
    not a state with a duration, so they never get a 'phase' key. Offsets still equal to
    infinity (a strobe/epoch still active when the recording was stopped) are skipped, since
    there's no real timestamp there.

---

## Version 1.3.0:
  New:
  - get_event_markers() now extracts markers from every populated epoc store in a TDT
    recording, not just 'Note' — I/O strobes (PP1/PP2/EE1/EE2/Pmp/Tne/ShK), Epoch Event
    Storage stores (L1P/L2P/L1E/L2E/pump/Tne), and the Tick reference all become their own
    marker group. Each marker dict now includes a 'store' key so a caller (e.g. a GUI) can
    group/toggle markers by source store instead of showing every store at once. Nothing
    about which stores exist is hardcoded — this walks whatever data.epocs actually
    contains, so it works for any TDT block regardless of Synapse configuration.
  - get_zscore_slice() and compute_fft_slice() now accept optional pre/post parameters for
    an asymmetric window around an event (e.g. 10s before, 20s after), instead of only a
    symmetric total `window` split evenly. `window` is still supported and used as the
    default if pre/post aren't given. get_zscore_slice()'s baseline is now computed from
    the actual pre-event portion of the segment rather than "the first half," since those
    differ once pre != post.

  Bug fixes:
  - Fixed get_tdt_struct() crashing (IndexError from inside the tdt SDK) on recordings
    where any epoc store has a mismatched onset/offset count — e.g. a strobe/epoch that
    was still active when the recording was stopped, a completely normal way for a session
    to end. Root cause: the tdt SDK breaks when reading multiple epoc stores in one call if
    any of them has this mismatch. Fixed by reading each epoc store with its own
    tdt.read_block() call (which correctly reconstructs the missing offset) instead of
    batching them; any store that still fails on its own (a genuine data issue, not this
    bug) is skipped with a warning instead of crashing the whole load.

  Cleanup:
  - Removed dataset.choose_file() (and its tkinter dependency) — a GUI file-dialog concern
    that had no business in a data-processing library, and was unused by any caller (the
    Qt GUI opens its own QFileDialog directly). No replacement needed.

---

## Version 1.2.2:
  Packaging:
  - Fixed the PyPI package name in pyproject.toml (was "PhysicsLibrary", registered trusted
    publisher is "ZaksPhysicsLibrary") — this mismatch caused the GitHub Actions trusted
    publishing OIDC exchange to fail on upload. No code changes.

---

## Version 1.2.1:
  Refactor:
  - Moved load_pt2() from file_parser_generic.py into file_parser.py where it belongs
    alongside TDT and Oxysoft parsers. Public API unchanged.

---

## Version 1.2.0:
  New features:
  - load_pt2(): Terranova Prospa .pt2 EFNMR/MRI image parser added to file_parser_generic.py.
    Detects magnitude image data after the LAER marker, supports all square power-of-two sizes
    (16x16 through 256x256) with a perfect-square fallback.
  - Exported load_pt2 from the top-level PhysicsLibrary package.
---

## Version 1.1.0:
  New features:
  - Generic file parser (file_parser_generic.py): supports Excel (.xlsx), CSV, TSV, and plain
    text with automatic sub-table detection for side-by-side data layouts.
  - Open button consolidated into a single dropdown menu (Open TDT / Open TXT Oxysoft / Open Excel).
  - Excel/generic data now loads directly into the main GUI plot with full snap and hover support.
  - TSI Fit Factor extracted from Oxysoft .txt files and displayed as [FF: x.x%] in the legend.
  - Curve fit parameters can now be copied to clipboard or exported as a CSV file.
  - Grid toggle checkbox added to the toolbar.
  - Marker enhancements: font size and colour are now editable both when adding and via right-click
    edit. Colour options include green, red, blue, orange, purple, and black (matching TDT default).

  Bug fixes and performance:
  - Blit-based hover animation: tracker dots and connector line are drawn via restore_region /
    draw_artist / blit instead of canvas.draw_idle(), making hover ~20-50x faster.
  - Scroll zoom debounced (150 ms) and scale reduced to 1.1x per tick for smoother zooming.
  - Tracker dots no longer disappear during scroll zoom; dots no longer cause autoscale zoom.
  - Rect-select zoom no longer accidentally triggers a curve fit click on release.
  - Oxysoft hover snap now correctly targets mean lines (O2Hb, HHb, tHb) by linewidth filter.
  - Edit Attributes changes now persist correctly across all view interactions (zoom, pan, hover).
---

## Version 1.0.0: 
  - Fully loadable data sets, graphing functionality, so far only TDT and Oxysoft.

---

## Version 0.1.0: 
  - Made a GUI capable of handling different file types. Parser implemented.
