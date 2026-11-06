# PyATCLI API Reference

Data processing and analysis library for TDT fibre-photometry, Oxysoft
NIRS, Terranova EFNMR, generic tabular data, and text-field studies.
Everything below is importable directly off the top-level package:

```python
import PyATCLI as pl
pl.process_tdt_folder(folder_path)
```

Install: `pip install PyATCLI` (called PhysicsLibrary — PyPI `ZaksPhysicsLibrary` — before version 2026.10.10: change `import PhysicsLibrary` to
`import PyATCLI`).

---

## Contents

* [Loading Data](#loading-data)
* [Numeric precision (float64)](#numeric-precision-float64)
* [TDT Photometry Processing](#tdt-photometry-processing)
* [Signal Analysis](#signal-analysis)

  * [General helpers](#general-helpers)
  * [Z-score / event-triggered analysis](#z-score--event-triggered-analysis)
  * [FFT](#fft)
  * [Curve fitting](#curve-fitting)
  * [Peak detection](#peak-detection)
  * [Area under the curve](#area-under-the-curve)
  * [Peri-event statistics](#peri-event-statistics)
* [Non-destructive splicing](#non-destructive-splicing)
* [Recordings and saved state](#recordings-and-saved-state)
* [Writing saved state](#writing-saved-state)
* [Output folders](#output-folders)
* [Markers](#markers)
* [Group analysis](#group-analysis)
* [Curve-fit models](#curve-fit-models)
* [Running analyses by name](#running-analyses-by-name)
* [Provenance records](#provenance-records)
* [Reproducing a result](#reproducing-a-result)
* [Command line](#command-line)
* [Reading results back](#reading-results-back)
* [Drawing results](#drawing-results)
* [Export View's numbers](#export-views-numbers)
* [Text Field Study](#text-field-study)
* [Text Field Study — Statistical Validation](#text-field-study--statistical-validation)
* [Progress reporting](#progress-reporting)
* [Optional dependencies](#optional-dependencies)

---

## Loading Data

Everything here returns a `Dataset` — a universal container regardless of
source format — except `load_any_file` (tabular files, which can hold
several independent tables) and `load_pt2` (a raw image array).

### `Dataset`

The struct every loader returns:

```python
@dataclass
class Dataset:
    source_format: str          # 'TDT' | 'Oxysoft'
    folder_path:   str
    folder_name:   str

    sample_rate: float = 0.0    # Hz
    num_samples: int   = 0
    duration_s:  float = 0.0    # seconds

    signals: Optional[np.ndarray] = None   # shape (num_channels, num_samples), always float64

    channel_names: list[str] = field(default_factory=list)
    num_channels:  int       = 0

    events: list[dict] = field(default_factory=list)   # {'label': str, 'sample': int}
    metadata: dict = field(default_factory=dict)        # format-specific raw header blob
```

### `DataFormat`

`Enum` with members `TDT`, `OXYSOFT`, `UNKNOWN`.

### `detect_format(folder_path: str) -> DataFormat`

Inspects a folder's contents and returns the matching format. TDT is
checked first (proprietary extensions), then Oxysoft (`.txt` marker).

### `detect_format_file(file_path: str) -> DataFormat`

Same, for a single file rather than a folder. Currently only recognizes
Oxysoft `.txt` exports.

### `load_dataset(folder_path, fmt=None, regression_method="ols", progress=None) -> Dataset`

Top-level dispatcher — detects the format (unless `fmt` is given
explicitly) and routes to the matching loader.

* `regression_method` (`"ransac" | "huber" | "ols"`): TDT-only, forwarded
to the motion-correction step (see [`process_tdt_folder`](#process_tdt_folderfolder_path-regression_methodols-progressnone)). Ignored for Oxysoft, which has no
such correction step.

### `load_dataset_file(file_path: str, progress=None) -> Dataset`

Loads a single file (rather than a folder) into a `Dataset`. Currently
supports Oxysoft `.txt` exports.

### `load_any_file(path: str, progress=None) -> list[GenericTable]`

Best-effort parser for arbitrary tabular data — `.xlsx`/`.xls` (needs
`openpyxl`), `.csv`, `.tsv`, or `.txt`/`.dat` with a sniffed delimiter.
Detects **multiple independent sub-tables** within one sheet/file (laid
out side-by-side or stacked, separated by empty rows/columns) and
returns each as its own table so a caller can pick which one to use.
Raises `ValueError` if nothing usable is found.

Each result is a `GenericTable`:

```python
@dataclass
class GenericTable:
    name:     str
    headers:  list[str]
    data:     np.ndarray   # (n_rows, n_cols) float64, NaN for missing
    metadata: dict = field(default_factory=dict)
```

### `load_pt2(path: str) -> np.ndarray`

Parses a Terranova/Prospa `.pt2` 2D NMR/MRI image file (magnitude image
stored as little-endian float32 after a `LAER` marker). Returns a 2D
`(n, n)` float64 array (widened on the way out, like every load — see
[Numeric precision](#numeric-precision-float64)).

> `load_oxysoft`/`load_oxysoft_file`/`load_tdt` (in `sources/load.py`) are the thin
> wrappers behind `load_dataset` that package a reader's result into a
> `Dataset`; they aren't re-exported at the top level — reach them via
> `load_dataset`/`load_dataset_file` unless you specifically need to bypass
> format auto-detection. The readers they wrap are public: `process_tdt_folder`
> (see [TDT Photometry Processing](#tdt-photometry-processing)) and the two
> Oxysoft ones below.

### `process_oxysoft_folder(folder_path, progress=None) -> dict`
### `process_oxysoft_file(file_path, progress=None) -> dict`

`sources/processing_oxysoft.py` is Oxysoft's counterpart of `sources/processing_TDT.py`: it
reads the export (the Legend block maps columns to channels; then the numeric
rows, the event column and the fit factor), and `load_dataset` /
`load_dataset_file` only wrap the result into a `Dataset`. Use these directly
when you want the arrays without one. The folder version concatenates every
`.txt` in it in alphabetical order (`FileNotFoundError` if there is none);
the file version raises `ValueError` naming the file if no O2Hb channel is
recognised in its Legend.

**Returns** `dict`:

|key|meaning|
|-|-|
|`o2hb`, `hhb`|`(n_channels, n_samples)` float64, channels in label order (`Tx1`, `Tx2`, ...)|
|`events`|`[{"label", "sample"}, ...]`: the event column's code as text (`"1"`, `"2"`, ...) and the 0-based sample it is on|
|`metadata`|`n_channels` (physical channels, not O2Hb + HHb) and, if the export has a Fit Factor column, `fit_factor_mean`; for a folder, from its first file|
|`channel_labels`|e.g. `["Tx1", "Tx2"]`|
|`sample_rate`|Hz, from the first file|

Row 0 of an export is the absolute baseline, not a delta, and is skipped.
Known limitation: for a folder, each event's `sample` is its index within its
own file; it is not shifted by the length of the files before it.

---

## Numeric precision (float64)

Signals, time axes, trial matrices and images are **float64 everywhere** in
this library. TDT's SDK delivers float32, and numpy is not safe on
single-precision data of a realistic length: `np.polyfit` on a float32
regressor silently drops the slope term once `n * float32-eps` exceeds the
ratio of the regressor's spread to its mean (about 200,000 samples, three
minutes at 1 kHz, for a 415 channel of 25 ± 1), returning a line at roughly
half the true slope with only a `RankWarning`; sums, means and FFTs lose
digits the same way, less visibly. So the rule is enforced, not left to
callers:

* **Every file load returns float64**, whatever the file stores.
  `get_tdt_struct` widens each stream as it reads the block;
  `process_tdt_folder`'s `x`, `raw`, `corr`, `f0` and every raw `channels`
  trace are float64; so are the Oxysoft readers, `load_any_file`
  (`GenericTable.data`), `load_pt2` (the file stores float32), and
  `Dataset.signals` — including a `Dataset` or `GenericTable` you build by
  hand.
* **Every public function that takes a signal, a time axis or a trial matrix
  widens it on the way in** — smoothing, z-scoring, PETH, FFT, AUC, curve
  fits, peak finding, splicing, motion correction, bleaching, denoising,
  `extract_group_trials` — so arrays that did not come through a loader
  (straight from the `tdt` SDK, or from your own pipeline) are covered too.
  Hand any of them float32 and you get float64 results, bit-identical to what
  the same values as float64 give.
* Widening float32 to float64 is exact, so no value changes; an array that is
  already float64 is passed through without a copy.

### `numeric.as_float64(a) -> np.ndarray`

The one helper that does it (`from PyATCLI.numeric import as_float64`;
`numeric.FLOAT` is `np.float64`). Returns `a` itself if it already is a float64
array, otherwise a widened copy (float32, float16, integer arrays, lists);
NaN and inf are kept as they are.

---

## TDT Photometry Processing

### `process_tdt_folder(folder_path, regression_method="ols", progress=None)`

The full photometry pipeline, and the main TDT entry point: load block →
extract 465nm signal (+ optional 415nm isosbestic reference) → motion
correction → bleaching baseline → ΔF/F → denoise → event markers.
ΔF/F is the motion-corrected residual divided by the 465 channel's
bleaching baseline (`f0`); everything is computed in float64.

* `regression_method` (`"ransac" | "huber" | "ols"`, see
[`REGRESSION_METHODS`](#regression_methods) below) — which regression the
isosbestic-vs-signal motion correction uses.

**Returns** `dict`:

|key|meaning|
|-|-|
|`x`|time vector|
|`raw`|motion-corrected fluorescence, in the recording's own units: the 465 minus its isosbestic-predicted motion component (the 465 itself if there is no 415 stream)|
|`corr` / `dff`|final ΔF/F (denoised) — `dff` is an alias for `corr`|
|`f0`|baseline fluorescence: the 465 channel's photobleaching trend, the denominator of ΔF/F|
|`fs`|sampling frequency|
|`store`|signal label|
|`markers`|behavioral event markers (see [`get_event_markers`](#get_event_markersdata---listdict))|
|`channels`|`[{"key", "label", "y"}, ...]` — raw, un-corrected per-wavelength streams (`main_driver`, plus `isosbestic` if a 415 reference exists)|
|`motion_correction_inlier_fraction`|fraction (0–1) of samples kept as inliers during motion correction, or `None` if there was no 415 reference stream|

### `REGRESSION_METHODS`

```python
REGRESSION_METHODS = ("ransac", "huber", "ols")
```

The three motion-correction regressions `process_tdt_folder`/`load_dataset`
accept:

* **`ols`** *(default)* — plain least-squares, no robustness at all. The
fit line can get dragged toward a severe artifact, but that's a
predictable weakness; unlike RANSAC/Huber below it never actively
discards or downweights a sample, so it can't mistake a real, large
biological transient for an artifact and distort the correction there.
* **`ransac`** — fits to random small subsets, keeps whichever
gets the most points within a residual threshold, refits on just those
inliers. Excludes artifact points from the fit entirely rather than
downweighting them — worth trying when artifacts are occasional and
severe enough (a fiber-cord twist, a brief motion burst) that this
trade-off clearly favors excluding them.
* **`huber`** — every point stays in the fit, but points beyond a
residual threshold get down-weighted instead of excluded. Gentler than
RANSAC's hard cut; better suited to noise elevated throughout the
recording rather than concentrated in a few bad stretches.

### `validate_tdt_folder(path) -> (bool, str)`

Checks whether a directory contains a valid TDT recording block. Returns
`(True, folder_name)` if valid, `(False, error_message)` if not.

### `get_tdt_struct(path, progress=None, read_streams=True)`

Loads the raw TDT block (streams, epocs, scalars) via the `tdt` SDK.
Handles a real SDK bug around epoc stores with mismatched onset/offset
counts by reading each epoc store with its own fresh call rather than
all at once. `read_streams=False` skips the streams (the slow part) and
still reads the block's info, scalars and epoc stores.

### `scan_tdt_markers(folder_path, splices=None, progress=None) -> dict`

Lists a TDT block's event markers **without processing the recording** —
no ΔF/F, and the streams are only read when `splices` need the time axis.
About a third of a second per block, against about two seconds to process
it. `splices` (optional): the recording's saved splices, a list of
`{"mode", "start", "end"}`, applied to the marker times and the recording's
span exactly as PyAT does when it restores `splice.json` (see
`replay_splices`).

**Returns** `{"markers", "t_range", "n_splices", "block_name", "start_time"}`:
`markers` is what `get_event_markers` returns (after the splices);
`t_range` is `(first, last)` sample time — exact when splices were given,
otherwise `(0, the block's recorded duration)`, which can run a few
hundredths of a second past the last sample — or `None` if the block does
not say how long it is; `n_splices` is how many splices took effect;
`block_name` is Synapse's `Subject-YYMMDD-HHMMSS`.

### `get_plot_data(data, store_name, channel=0, max_points=None) -> (time, signal, fs)`

Extracts one stream's time series from a loaded TDT struct, with
optional downsampling.

### `correct_bleaching(y, fs) -> (corrected, trend)`

Estimates and removes the photobleaching trend via masked curve fitting.

### `denoise_signal(signal, fs, cutoff=5, order=2)`

Low-pass Butterworth filter for ΔF/F signals.

### `get_event_markers(data) -> list[dict]`

Extracts behavioral event markers from every populated TDT epoc store
(auto-discovered — nothing about store names is hardcoded). Each onset
and offset becomes its own marker (`phase`: `"high"`/`"low"`), except the
`Note` store (instantaneous, text-labeled, no offset) and `Tick` (TDT's
own 1-second heartbeat, skipped entirely — not a real event).

**Returns** list of `{"time", "label", "color", "store", "phase"}`
(`phase` omitted for `Note` markers).

### `debounce_events(times, min_isi) -> list[float]`

Collapses switch-bounce/double-tap duplicate timestamps — drops any
event within `min_isi` seconds of the last *kept* event. Independent of
any fixed schedule; a `min_isi` below the subject's real max event rate
leaves genuinely fast consecutive events intact.

---

## Signal Analysis

### General helpers

#### `estimate_sample_rate(time_array) -> float`

Sample rate (Hz) from a timestamp array, via the **median** inter-sample
interval — robust to one irregular gap skewing a naive
`1/(t[1]-t[0])` estimate.

#### `mean_channels(arr) -> np.ndarray`

Collapses a multi-channel `(n_channels, n_samples)` array to its
across-channel mean (e.g. Oxysoft's per-detector o2hb/hhb/thb). A no-op
if `arr` is already 1D.

#### `compute_group_stats(trial_matrix) -> (mean_trace, sem_trace)`

Mean + SEM across trials (rows) of a trial × time matrix. `sem_trace` is
all zeros for fewer than 2 trials (SEM with ddof=1 is undefined below
that); `mean_trace` is all zeros for 0 trials.

#### `compute_marker_intervals(markers) -> list[dict]`

Time-since-previous-marker, two ways, for a list of markers from any
source (TDT epocs, Oxysoft events, manual). Each marker needs at least
`time`/`label` keys (`store` falls back to `label`, `phase` optional).

**Returns** one dict per marker: `{"time", "store", "label", "phase", "dt_store", "dt_global"}` — `dt_store` is the interval since that
store's last event (for a high/low pair, a low marker's `dt_store` *is*
its on-duration); `dt_global` is the interval since whatever marker came
immediately before it, regardless of store. Both are `None` for the
first event in their sequence.

#### `smooth_signal(data, fs, window_sec=0.5)`

Moving-average smoothing filter.

#### `bin_for_heatmap(z_seg, num_bins=300)`

Bins a signal into `num_bins` equal segments, for heatmap plotting.

### Z-score / event-triggered analysis

#### `get_zscore_slice(time_array, signal, center_t, window=None, pre=None, post=None) -> (time_segment, z_scored_signal)`

Extracts and z-scores a window around one event time. Accepts either a
symmetric `window` (split evenly) or an asymmetric `pre`/`post` pair
(`pre`/`post` take precedence if both are given).

The baseline is the pre-event part of the window. Its mean/SD are taken
after pulling any sample further than 5 robust SDs (1.4826 × MAD) from the
baseline median in to that limit, so a brief artifact can't inflate the SD
and flatten the whole trial. The samples being scored are never altered
(no clipping), and the result does not depend on the units of the signal.
A flat baseline returns all zeros.

#### `compute_event_zscore_peth(time_array, signal, event_times, pre, post, num_bins=300, progress=None) -> dict`

Z-scores and aligns **every occurrence** of one event type into a
trial × time matrix (GuPPy-style), for a stacked-heatmap + trial-average
view — as opposed to a single click-triggered PETH. Each trial is
z-scored independently against its own pre-event baseline, then
resampled onto a shared `num_bins`-point relative-time axis so trials
with slightly different raw sample counts can still stack into one
matrix.

**Returns** `dict`:

|key|meaning|
|-|-|
|`time_axis`|`(num_bins,)`, relative time from `-pre` to `+post`|
|`trial_matrix`|`(n_valid_trials, num_bins)`|
|`trial_event_times`|the `event_times` that produced a usable trial (too-close-to-recording-edge events are skipped)|
|`mean_trace` / `sem_trace`|`(num_bins,)` each|

### FFT

#### `compute_fft_slice(time_array, signal, center_t, fs, window=None, pre=None, post=None) -> (freqs, power, seg_x, seg_y)`

Extracts a window around `center_t` and computes its FFT, after mean
removal + linear detrending (eliminates the DC spike/slow drift so
physiological frequencies — breathing \~0.3Hz, heart rate \~1Hz — are
visible). Same symmetric-`window`-vs-asymmetric-`pre`/`post` convention
as `get_zscore_slice`.

#### `find_fft_peaks(freqs, power, n_peaks=3) -> list[dict]`

Top-N peaks in a power spectrum, as data. Returns
`[{"freq_hz", "power", "bpm"}, ...]` sorted by power descending — empty
if fewer than 3 usable frequency bins (≥0.05Hz) exist or nothing clears
the prominence bar.

### Curve fitting

#### `compute_slope_segment(x_data, y_data, p1_idx, p2_idx, padding_pct=0.05) -> dict`

Least-squares linear regression slope between two index boundaries.
Returns `{"slope", "intercept", "crop_x", "crop_y", "x1", "y1", "x2", "y2"}`.

#### `fit_model_to_segment(x_seg, y_seg, model_fn, p0_fn) -> dict`

Fits `model_fn` (signature `f(x, *params) -> y`, see
[Curve-fit models](#curve-fit-models)) to a segment via `scipy.optimize.curve_fit`,
with `p0_fn(x_seg, y_seg) -> initial_guesses` supplying the starting
parameters. Returns `{"popt", "y_fit", "r2", "success", "error"}`.

### Peak detection

#### `find_significant_peaks(time_array, signal, z_threshold=5.0, min_distance_sec=1.0, include_troughs=False, start_buffer_s=10.0) -> list[dict]`

Auto-detects statistically significant transients directly from the
signal — instead of trusting that externally-supplied event markers
actually line up with real activity. Z-scores the whole recording
against its own global mean/std (not per-event), then `scipy.signal.find_peaks`
picks local maxima at or above `z_threshold` (default 5, the usual
five-sigma convention), at least `min_distance_sec` apart.
`include_troughs` also detects significant negative deflections
(off by default).

`start_buffer_s` (default 10) is a number of seconds at the start of the
recording in which no peak is reported. A trace is least settled right after
the light comes on, and the tail of the bleaching there reads as unusual: on
synthetic recordings with no transients at all, the false five-sigma peaks of
the detector sat in the first seconds, and a 10 s buffer removed them. Only
the search skips the start — the mean and standard deviation still come from
the whole recording, so no peak's z-score depends on the buffer (a peak within
`min_distance_sec` of the buffer's end that a taller peak inside it used to
hide can appear). The buffer does not remove false peaks that motion artefacts
cause later in the recording. 10 s was chosen on synthetic data; look at the
first minute of your own recording and set what it needs (`0` searches
everything). Records written before 2026.11.2 replay with `0`.

**Returns** `[{"time", "z_score", "kind": "peak"|"trough"}, ...]`, sorted
by time.

#### `find_peak_near_events(time_array, signal, event_times, pre, post, z_threshold=5.0, include_troughs=False, progress=None) -> list[dict]`

Checks whether a significant peak actually shows up near each given
event time (baselined the same way as `get_zscore_slice`, i.e. relative
to that event's own local baseline — not the whole recording). Works for
one event or many occurrences of the same type.

The default `z_threshold` of 5 is deliberately strict: the search takes
the *largest* z over every sample in the `pre`/`post` window, so a low
threshold reports a "response" by chance in most windows. On a real
fibre-photometry recording searched the way PyAT does it (0.5 s-smoothed
ΔF/F, 5 s baseline, 10 s response window), random pseudo-events were
reported as having a response 58% of the time at 2.5, 40% at 3, 21% at 4
and 10% at 5; the exact rates depend on the window lengths. Even at 5,
"found" is not proof: `shuffle_null_near_events` measures the chance rate on
the recording itself.

**Returns** one dict per `event_time` (same order):
`{"event_time", "found", "peak_time", "latency", "z_score", "kind"}` —
`found` is `False` (with the rest `None`) if the window was unusable or
nothing cleared `z_threshold`.

#### `shuffle_null_near_events(time_array, signal, event_times, pre, post, z_threshold=5.0, include_troughs=False, n_shuffles=1000, seed=0, progress=None) -> dict`

How often `find_peak_near_events` finds a "response" **by chance** on this
recording, and whether the given events do better than that.

The whole train of events is moved as one by a random circular shift of the
part of the recording where a complete window fits (so the spacing between the
events, and with it how much their windows overlap, is kept), `n_shuffles`
times, and each shifted event is searched exactly as a real one is
(the same `pre`/`post`, baseline, `z_threshold` and `include_troughs`). From that:

- `chance_hit_rate`: the share of shifted windows that reached `z_threshold`.
- `p_count`: how often a shifted train had at least as many events with a
  response as the real one: `(1 + the shuffles that did) / (1 + n_shuffles)`, over the
  events whose window is complete. It is a permutation p-value for "these event
  times are unrelated to the signal".
- `p_event`: per event, the share of all shifted windows whose largest |z| was at
  least that event's, in the same `(1 + k) / (1 + n)` form. Not corrected for the
  number of events or event types looked at (the `peak_finder` analysis adds a Holm
  column across event types when it scans them all).

The smallest p-value possible is `1 / (1 + n_shuffles)`. `seed` fixes the
shifts: the same inputs and seed give the same answer. One shuffle of 20 events
costs about two milliseconds, so the default 1,000 shuffles take a couple of
seconds. `progress` ticks once per shuffle.

Events whose window is cut by the recording's edge cannot be moved: they stay put
in every shuffle and take no part in `p_count`; their `p_event` is read against the
complete shifted windows. If the recording really has responses, some shifted
windows land on them and count as chance, so `chance_hit_rate` reads higher than it
would without them and the p-values are conservative — a response is never made to
look more significant than it is, the more so the denser the events.

**Returns** a dict: `available` (`False` with a `reason` when nothing could be
shifted: a recording shorter than the window, or no event with a complete window —
the numbers are then `None`), `n_events`, `n_tested` (events with a complete window),
`n_shuffles`, `seed`, `found` (events with a response among those tested),
`expected_by_chance` (the mean of that over the shuffles), `chance_hit_rate`,
`p_count`, and `p_event` (one per event, `None` where the window holds no samples).

### Area under the curve

#### `compute_auc_from_trace(x, y) -> dict`

Trapezoidal AUC, split by sign — the shared core `compute_auc_window`
and `compute_auc_matrix` both delegate to. `positive_auc` integrates
only where `y >= 0`; `negative_auc` integrates only where `y <= 0` and
is itself signed (≤ 0, not an absolute value); `total_auc` is their sum
(equivalent to the plain net signed integral).

**Returns** `{"total_auc", "positive_auc", "negative_auc"}` — all zero
if fewer than 2 points.

#### `compute_auc_window(time_array, signal, center_t, pre, post) -> dict`

Slices a pre/post window around an event and integrates it (same
windowing convention as `get_zscore_slice`). Deliberately does **not**
z-score first — AUC quantifies the signal in its own (already
dF/F-normalized) units. Returns `{"seg_x", "seg_y", "total_auc", "positive_auc", "negative_auc"}`.

#### `compute_auc_matrix(time_axis, trial_matrix) -> dict`

Per-trial + group AUC for a trial × time matrix (see
`compute_event_zscore_peth`). Returns `{"per_trial": [...], "group": {...}}`
— `per_trial` one entry per row (same order), `group` computed on the
matrix's mean trace.

### Peri-event statistics

#### `compute_peri_event_from_trace(rel_x, y, post, bin_width=1.0) -> dict`

Peak amplitude, latency (time-to-peak), and mean-time-bins from an
already event-relative trace (`rel_x`: 0 = event time). Peak/latency are
post-event only (`rel_x >= 0` — a pre-event sample can't be a response),
using signed `argmax(|y|)` so a strong inhibitory (negative-going)
response is captured correctly, not just the largest positive value.

**Returns** `{"peak_amplitude", "latency", "mean_bins"}` — `mean_bins` is
a list of `{"bin_start", "bin_end", "mean"}` covering `[0, post]` in
`bin_width`-second chunks (last bin truncated, not dropped, if `post`
isn't evenly divisible).

#### `compute_peri_event_matrix(time_axis, trial_matrix, post, bin_width=1.0) -> dict`

Per-trial + group version of the above, mirroring `compute_auc_matrix`.
Returns `{"per_trial": [...], "group": {...}}`, each entry
`{"peak_amplitude", "latency", "mean_bins"}`.

---

## Non-destructive splicing

Both functions work on 1D arrays *and* 2D `(n_channels, n_samples)`
arrays via `extra_channels` (sliced along the last axis) — so Oxysoft's
multi-detector data splices exactly like TDT's single-channel data, no
special-casing needed by the caller.

### `splice_keep_inside(x, raw, corr, markers, detected_markers, start, end, extra_channels=None) -> dict | None`

Trims `x`/`raw`/`corr` to `[start, end]`, filters both marker lists to
the same range. `extra_channels` (optional): any other same-length
arrays that need trimming identically — e.g. TDT's raw per-wavelength
channels, or Oxysoft's `(n_detectors, n_samples)` arrays.

**Returns** `{"x", "raw", "corr", "markers", "detected_markers", "n_samples", "extra_channels"}` (copies, not views), or `None` if the
range doesn't contain at least 2 samples. Marker timestamps stay
absolute (not re-zeroed).

### `splice_cut_out(x, raw, corr, markers, detected_markers, start, end, extra_channels=None) -> dict | None`

Removes `[start, end]` and stitches the remainder together, shifting
everything after the cut backward by the cut's duration so the timeline
stays contiguous. Markers inside the cut are dropped; markers after it
are shifted by the same amount. Same `extra_channels` support as above.

**Returns** same shape as `splice_keep_inside`, or `None` if there isn't
usable signal on both sides of the cut to stitch together.

### `replay_splices(x, splices, markers=None, detected_markers=None) -> dict`

Replays a saved list of splices (PyAT's `splice.json`: `{"mode", "start",
"end"}` each, in the order they were applied; `"cut_out"` cuts, any other
mode keeps the range) on a time axis and marker lists, **without touching
any signal**. It goes through `splice_cut_out` / `splice_keep_inside`
themselves, so `x` and the marker times come out exactly as a full replay
gives them, down to the last digit. A splice that leaves fewer than 2
samples is skipped, as PyAT skips it.

**Returns** `{"x", "markers", "detected_markers", "applied"}` (`applied` =
how many splices took effect).

---

## Recordings and saved state

`recording.py` is a recording as PyAT holds it, without the GUI: everything
between "open a recording" and "run an analysis" that isn't drawing. The app
keeps this state in `ctx.cache`; these functions build, edit and query exactly
that dict, so a script gets the same numbers the app shows.

### The cache dict

|key|meaning|
|-|-|
|`source`|`"TDT"`, `"Oxysoft"` or `"Generic"`|
|`source_path`|the folder / file it was read from|
|`store`|a display name: the folder / file / table name, plus a tag per splice|
|`x`, `fs`|time axis (s) and sampling rate (Hz)|
|`markers`|user-placed markers (see [Markers](#markers))|
|TDT: `raw`, `corr`, `detected_markers`, `signals`|motion-corrected fluorescence, dF/F, TDT's own events, and `{key: {"label", "y"}}` for `normalized`, `main_driver`, `isosbestic`|
|Oxysoft: `o2hb`, `hhb`, optional `thb`, `detected_markers`, `fit_factor_mean`|`(n_channels, n_samples)` arrays|
|Generic: `y_columns`, `x_label`|`{header: y}` and the x column's header|

### Saved state ("JSON saves")

The app saves markers and splices next to a recording, in a `JSON saves/`
folder: inside a folder source (TDT), beside a file source (Oxysoft, Generic).

* `saved_state_dir(source_path)` — that folder.
* `marker_sidecar_path(source_path)` — `markers.json` inside a folder source,
  `<file name>.markers.json` beside a file source.
* `legacy_marker_sidecar_path(source_path)` — `<source>.markers.json`, where
  markers were saved before the `JSON saves/` folder existed (only read).
* `annotation_sidecar_path(source_path, filename)` — the same place for
  `highlights.json` / `text_annotations.json`.
* `splice_sidecar_path(source_path)` — `splice.json`, next to `markers.json`.
* `read_saved_splices(source_path) -> list[dict]` — every saved splice in the
  order it was applied (`[]` if none; the old single-splice dict format is
  accepted).
* `read_saved_markers(source_path) -> list[dict]` — the saved markers, from
  `JSON saves/` else the legacy location (`[]` if none; `ValueError` if the file
  does not hold a list).
* `read_saved_annotations(source_path, filename) -> list[dict]` — the saved highlights
  (`HIGHLIGHTS_FILE`) or text boxes (`TEXT_ANNOTATIONS_FILE`) (`[]` if none).
  `load_recording` puts them in the cache as `cache["highlights"]` and
  `cache["text_annotations"]` when the recording has them, as the app does.

### Building a cache

* `build_signal_options(result) -> dict` — the plot/analysis signals a processed
  TDT recording offers, `{key: {"label", "y"}}` (the app adds a colour to each).
* `make_tdt_cache(folder_path, result, signals=None) -> dict` — the cache for a
  `process_tdt_folder` result, before any splice or saved marker.
* `make_oxysoft_cache(file_path, dataset) -> dict` — from `load_dataset_file`'s
  `Dataset`; events become `detected_markers` in seconds.
* `make_generic_cache(path, table, x_index=0, y_indices=None) -> dict` — one
  `GenericTable`: column `x_index` is x (rows with no x are dropped), each of
  `y_indices` (default every other column) a trace.
* `normalized_signal(cache) -> (key, label, y)` — the trace the analyses that
  work on the corrected signal (Event PETH, peak finder, group analysis) use:
  `"normalized"` if offered, else the first signal; for a source with no signals
  map, its `corr` else `raw`.

### Splicing a recording

* `apply_splice(cache, mode, start, end, regression_method="ols", progress=None) -> dict | None`
  — one splice (`MODE_CUT_OUT` / `MODE_KEEP_INSIDE`) applied to a cache of any
  source; a new cache, `cache` itself is untouched; `None` if the range isn't
  usable. For TDT both modes **re-fit the dF/F** (`compute_dff`, with
  `regression_method`) on the raw channels that survive: an artifact dragging
  the original fit may now be excluded, so a splice is a fresh fit, not a
  shorter view of the old one.
* `apply_splices(cache, splices, regression_method="ols", progress=None) -> (cache, applied)`
  — a saved list, each stacked on the one before; an unusable one is skipped.
  `applied` is the list that took effect, and `store` is tagged with them.
* `spliced_store_name(base_name, splices) -> str` — `"Rec1 [cut 5.0s-8.0s, kept 2.0s-12.0s]"`.

### `load_tdt_recording(folder_path, regression_method="ols", use_saved=True, progress=None) -> dict`

Loads a TDT block the way the app does: processed to dF/F, its saved splices
replayed, its saved markers put on top. Nothing is written. `use_saved=False`
skips the `JSON saves/` folder (the recording exactly as recorded).

**Returns** `{"cache", "original_cache", "splices", "warnings"}`: the cache after
splices and markers; the unspliced one (or `None` if no splice applied), which
is what the app keeps to restore the full recording; the splices that took
effect; and a sentence for each saved file that could not be read (the recording
is then used without it). `ValueError` if `folder_path` is an existing folder
that is not a TDT block.

### `load_recording(path, regression_method="ols", use_saved=True, table=None, x_column=None, y_columns=None, progress=None) -> dict`

Opens any recording the app can: a TDT folder, an Oxysoft `.txt` export, or a
generic table file (Excel, CSV, TSV, text), with its saved splices replayed and
its saved markers and signal names on top (`use_saved=False` skips all three).
Returns what `load_tdt_recording` returns: `{"cache", "original_cache", "splices", "warnings"}`.

For a generic file, `table` names which of its tables to open (needed only if it
has more than one), `x_column` is its x column (default the first) and
`y_columns` the traces (default every other column). `regression_method` matters
for TDT only. `ValueError`, saying what is wrong, if `path` is not a recording it
can open or a named table/column does not exist.

### Signal names

A TDT recording's signals get names built from its stores (`"Main Driver (x465A)"`,
`"Isosbestic (x415A)"`). A person can give them names of their own in the app
(Plot menu, Rename signals), and the app keeps them in `signal_labels.json` in the
`JSON saves/` folder. The name appears in figures, legends and the `Channel` column
of exported tables, so a script that wants the same output applies the same names —
`load_recording` / `load_tdt_recording` already do.

* `signal_labels_path(source_path)` — where `signal_labels.json` lives.
* `read_saved_signal_labels(source_path) -> dict` — `{signal key: name}` (`{}` if
  none; `ValueError` if the file does not hold an object).
* `apply_signal_labels(cache, labels) -> dict` — gives the cache's signals the names in
  `labels` (a key the recording lacks, or an empty name, is ignored). The cache dict is
  updated in place and returned; the arrays are not copied. A splice made afterwards keeps
  the names.
* `current_signal_labels(cache) -> dict` — `{signal key: name in effect}`; this is what a
  provenance record stores so a replay names the signals the same.

### `find_tdt_recordings(path) -> list[str]`

Every valid TDT block anywhere under `path`, at any depth, sorted. Doesn't
descend into a block, doesn't report `path` itself, and is safe against a
folder that links back to one of its own ancestors (including Windows NTFS
junctions, which `os.walk`'s own link check misses).

---

## Writing saved state

`PyATCLI.saved_state` writes what PyAT's Save Changes writes, into the same files, so an edit made
in a script is an edit made in the app — and PyAT's Save Changes writes through it, so the files are the same
bytes either way.

```
JSON saves/
  markers.json           [{"time", "label", "color", "fontsize"}]
  splice.json            [{"mode": "cut_out" | "keep_inside", "start", "end"}, ...]   every splice, in order
  signal_labels.json     {signal key: name}
  highlights.json        [{"start", "end", "color"}]
  text_annotations.json  [{"x", "y", "text", "fontsize"}]
```

**Writers** take what to write, replace what was saved, return the file's path, and refuse what the app could not
read back (`ValueError`):

* `write_saved_markers(source_path, markers)`, `write_saved_annotations(source_path, filename, items)`
* `write_saved_splices(source_path, splices)`, `write_saved_signal_labels(source_path, labels)` — with nothing to
  write the file is removed (returns `None`), as Save Changes never leaves behind an edit that was since undone.
* `clear_saved_state(source_path) -> [names]` — Undo All Changes, saved: deletes every file in `JSON saves/` (not
  the folder, nothing outside it).
* `saved_state_summary(source_path) -> dict` — `markers`, `splices`, `signal_labels`, `highlights`,
  `text_annotations` and the `folder`; an unreadable file is a `ValueError` naming it.

**Edits** do what one action in the app does, to what is already saved. Each opens the recording as the app does
(saved splices replayed, so times are on the time axis as it now stands) and writes only its own file; `**load`
is `load_recording`'s options (`regression_method`, `table`, `x_column`, `y_columns`):

* `add_marker(source_path, time, label=None, color=None, fontsize=None, **load) -> dict` — a time outside the
  recording is an error. `remove_markers(source_path, time, tolerance=0.05, label=None, **load) -> list` removes
  those near `time` and returns them.
* `add_splice(source_path, mode, start, end, **load) -> list` — stacks a splice on the saved ones; tried first,
  so a range that leaves fewer than 2 samples is an error and nothing is written. `remove_splice(source_path,
  number) -> list` takes one out of the stack (1 = the first applied).
* `rename_signals(source_path, {key: name}, **load) -> dict` — an empty name takes a rename away; a key the
  recording does not have is an error that lists the ones it does.
* `add_highlight(source_path, start, end, color="Yellow") -> dict` — `color` is one of `HIGHLIGHT_COLORS` (the
  Highlighter's five) by name, or a hex colour. `add_text(source_path, x, y, text, fontsize=10) -> dict`.
* `new_marker`, `new_highlight`, `new_text_annotation` — the dicts those make, validated; `DEFAULT_MARKER`,
  `HIGHLIGHT_COLORS`, `HIGHLIGHT_ALPHA`, `DEFAULT_TEXT_FONT_SIZE`, `SPLICE_MODES`.

```python
pl.saved_state.add_marker("D:/rats/Rat7-250101", 312.5, label="Lever")
pl.saved_state.add_splice("D:/rats/Rat7-250101", "cut_out", 40.0, 55.0)
pl.load_recording("D:/rats/Rat7-250101")["splices"]        # [{"mode": "cut_out", "start": 40.0, "end": 55.0}]
```

`python -m PyATCLI saved ...` runs these (see [Command line](#command-line)).

---

## Output folders

`PyATCLI.outputs` is where results go once there is an Output folder: a folder per kind of data and,
inside it, one per recording, so a recording's tables, figures and reports accumulate together instead of a flat
pile of similar-looking filenames. PyAT's exports use it (Options → Output folder) and so does the command line
(`--output-folder`), so a recording analysed from a script and in the app ends up in one place.

```
<Output>/
  TDT/<recording name> (Analysis)/       Oxysoft/...   Generic/...   PT2/...   Text Field Study/...
  Group Analysis/<group name>/
```

* `recording_dir(base, data_type, source_path) -> str` — the recording's folder (made if it is not there; the
  same one every time for the same recording, however its path is spelt). The name is the recording's own
  basename plus ` (Analysis)` — never the display name, which grows a `[cut ...]` tag per splice — so it can't be
  mistaken for the recording. Two different recordings of one name (two subjects' `Day1`) get ` (2)`: a marker
  file in each folder (`SOURCE_MARKER_FILENAME`) says which source it belongs to.
* `group_dir(base, group_name) -> str | None` — `Group Analysis/<name>`, not made; a name already used for a finished
  run (its folder holds an `analysis_settings.json`) gets ` (2)`, ` (3)`. `None` if the name leaves nothing usable.
* `unique_path(path)`, `unique_stem(folder, stem, endings)` — a repeat run is ` (2)`, never an overwrite.
* `safe_folder_name(name)`, `recording_folder_name(source_path)`.

---

## Markers

`markers.py`: how a marker is named and grouped, shared by every analysis that
works per event so they all agree. A marker is a dict with `time` (s) and
`label`; TDT's own carry `store` (the epoc store's id) and, for onset/offset
pairs, `phase` (`"high"` / `"low"`).

* `store_display_name(store_id, store_labels=None) -> str` — the renamed name
  (`store_labels` = `{store id: shown name}`) or the id itself.
* `marker_display_label(marker, store_labels=None) -> str` — the text to show:
  an onset/offset marker gets its store's (renamed) name plus `¹` (`PHASE_SUFFIX["high"]`)
  or `⁰`; anything else (a `Note`, a hand-placed marker) keeps its own label, so
  renaming a store never clobbers a note that shares its store id.
* `group_markers_by_name(cache, store_labels=None) -> dict` — `{display name:
  [sorted event times]}` over `cache["detected_markers"]` and `cache["markers"]`,
  deduped by `(name, time)`. High and low phases are separate names.

---

## Group analysis

The same event-locked responses measured in every recording of a group,
reduced to a few numbers per trial, and compared with a linear mixed-effects
model. Every recording is one subject; the variables are event markers; a
trial is one occurrence of a marker.

### `GroupSpec`

A dataclass holding everything decided for one group analysis. It
round-trips through JSON with `to_dict()` / `GroupSpec.from_dict(d)`.

| field | default | meaning |
|---|---|---|
| `group_name` | `""` | the group's name (its results folder is named after it) |
| `subjects` | `[]` | `[{"subject": name, "folder": path}, ...]`, names unique |
| `markers` | `[]` | marker names, as PyAT lists them (a store's name plus `¹` for onset or `⁰` for offset, or a note's own text) |
| `store_labels` | `{}` | store renames in effect when the markers were listed |
| `pre`, `post` | `10.0`, `10.0` | seconds before / after each event |
| `baseline` | `(-10.0, -6.0)` | baseline window, seconds relative to the event |
| `response` | `(0.0, 10.0)` | response window, seconds relative to the event |
| `signal` | `"dff"` | `"dff"`: trial minus its baseline mean; `"zscore"`: also divided by the baseline SD |
| `smooth_seconds` | `0.5` | moving-average smoothing before slicing (0 = none); 0.5 matches the single-recording Event PETH |
| `metrics` | all five | `"auc"`, `"peak"`, `"mean"`, `"latency"`, `"decay"` |
| `peak_direction` | `"absolute"` | `"positive"`, `"negative"` or `"absolute"` (largest deflection either way, signed) |
| `decay_fraction` | `0.5` | decay time counts until the trace falls to this share of the peak |
| `correction` | `"holm"` | multiple-comparison correction across pairwise comparisons: `"holm"`, `"bonferroni"` or `"fdr_bh"` |
| `alpha` | `0.05` | significance level a corrected p-value is compared against |
| `regression_method` | `"ols"` | motion correction used when each recording's ΔF/F is recomputed |

The defaults follow TDT's own fiber photometry epoch-averaging example
(`TRANGE = [-10, 20]` there is start and *duration*: −10 s to +10 s; baseline
−10 to −6 s). There is no formal standard; GuPPy and pMAT leave the window
and baseline to the user.

`problems()` returns every reason the spec is unusable as plain sentences
(empty list = fine); `window_problems()` returns only the rules for the
window, baseline and response windows.

### `marker_index(scans) -> dict`

Which markers the recordings have. `scans` is a list of `{"subject",
"groups": {marker name: [event times]}}`. **Returns** `{marker:
{"recordings", "events", "per_subject": {subject: n}}}`; a recording with
no event of a marker is not counted for it.

### `common_markers(scans) -> list[str]`

The marker names every one of the recordings has at least one event of, sorted.

### `design_summary(scans, markers, pre, post) -> dict`

Trials per subject and marker, worked out from the event times alone. A
trial is *usable* if its whole window `[event − pre, event + post]` lies
inside the recording (`scans[i]["t_range"]`, or unknown = all usable);
trials *overlap* when their windows share samples (events less than
`pre + post` apart), so trials of one subject are not independent of each
other.

**Returns** `{"cells": {subject: {marker: {"events", "usable",
"overlapping"}}}, "markers": {marker: {"events", "usable", "overlapping",
"subjects_with_trials", "min_usable", "max_usable"}}, "n_subjects",
"warnings": [{"level": "warning" | "info", "text"}]}`. The warnings cover
subjects with no usable trials, heavy overlap, and too few subjects for a
mixed model (under 6) or for its p-values to be more than approximate
(under 20).

### `extract_group_trials(x, y, events, spec, subject, recording="", trace_step=0.04) -> dict`

Slices one recording into trials and measures each. `x`/`y` are the recording's
time axis and dF/F (smoothed here with `spec.smooth_seconds`); `events` is
`{marker name: [event times]}`, only the markers in `spec.markers` are used.

A trial is usable if its whole window lies inside the recording; each usable
trial's baseline-corrected trace (see `GroupSpec.signal`) is reduced to the
five measures (`measures_for_trial`) unless the baseline can't be scored (a
flat z-score baseline) or the window holds non-finite samples -- a trial that
can't be measured still gets a row, with `valid=False` and a `reason`.

**Returns** `{"trials": DataFrame (one row per usable trial; columns:
group, subject, recording, marker, trial, event_time, valid, reason, auc,
peak, mean, latency, decay, baseline_mean, baseline_sd, n_samples,
overlap), "traces": {marker: {"n", "mean"}} (the mean baseline-corrected
trace on `grid`), "grid": the common time axis (-pre..+post, spaced
`trace_step` apart), "excluded": {marker: events left out for not having a
full window}}`.

### `measures_for_trial(rel_t, y, spec) -> dict | None`

The five measures of one already baseline-corrected trial (`rel_t` seconds
relative to the event, `y` the trace). Uses `spec.response`, `spec.peak_direction`
and `spec.decay_fraction`. **Returns** `{"auc", "peak", "mean", "latency",
"decay"}`, or `None` if the response window holds fewer than 2 samples.

### `decay_time(t, y, peak_index, fraction) -> float`

Seconds from `y[peak_index]` until the trace has fallen to `fraction` of it
(by linear interpolation between the two samples either side of the
crossing) -- a positive peak falls, a negative one (a dip) climbs back.
`nan` if it never gets there, or the peak itself is 0.

### `fit_group_models(trials, spec, progress=None, traces=None, trace_grid=None, excluded=None) -> GroupResults`

Descriptives, linear mixed-effects models and pairwise comparisons for every
measure in `spec.metrics`, on the trial table from `extract_group_trials`
(every subject's trials concatenated).

For each measure, twice: **trials** -- every trial is a row; two or more
markers fit `measure ~ marker` with a random intercept for subject and one
for each subject x marker cell (statsmodels `MixedLM`, REML); one marker
fits `measure ~ 1` with a subject random intercept. **means** -- the same on
subject x marker means, which does not rely on trials being independent, as
a check. p-values use subject-based degrees of freedom (n - 1 for a
contrast, like a paired test; (k - 1)(n - 1) for the omnibus F test of
marker, as in repeated-measures ANOVA) rather than statsmodels' own Wald z,
which assumes many subjects. With two or more markers, every pairwise
comparison between markers is corrected (`spec.correction`); with one there
is nothing to compare it with, so AUC and mean amplitude are tested against
zero (a baseline-corrected trace has mean zero without a response) and the
others (peak, latency, decay -- no zero to test) are reported as estimates
with 95% intervals only.

Why the subject x marker term: trials of one subject and marker share an
effect of their own, and neighbouring trials share signal when their
windows overlap. Simulated cohorts with no true difference between markers
(three markers, 40 trials per cell, 6-20 subjects, with and without
correlated trials) found a random-intercept-only model called it
significant in 65-93% of runs at a nominal 5% whenever trials were
correlated within a cell; the model used here, and repeated-measures ANOVA
on the cell means, both landed at 4-8%.

**Returns** a `GroupResults` (a dataclass): `trials`, `traces`, `trace_grid`,
`excluded` (carried through unchanged), `descriptives`, `subject_means`,
`models`, `fixed_effects`, `omnibus`, `variance`, `estimated_means`,
`pairwise`, `diagnostics`, `random_effects` (all `pandas.DataFrame`),
`notes` (`[{"level", "measure", "text"}]` -- dropped markers, boundary
fits, undefined measures, a warning when the trial-level model and the
subject-means check disagree about significance), `software` (library
versions). `.frames()` returns `{name: DataFrame}` for every table.

### `write_group_results(results, directory) -> list[str]`

Writes every non-empty table in `results.frames()` (plus the mean traces
and `report_text(results)`) as UTF-8-with-BOM files into `directory`
(created if needed) -- CSVs readable by Excel and pandas alike, and
`analysis_report.txt`, and `analysis_notes.json` (the notes and software, which
no table holds: `read_group_notes(directory)` reads them back). The file each
table goes to is `analysis.group.report.TABLE_FILES`; the columns of each that are
always there are `GROUP_TABLES` -- both in the engine's contract (`outputs`).
**Returns** the paths written.

### `group_report_text(results) -> str`

The plain-text report `write_group_results` writes to
`analysis_report.txt`: the design, the results per measure (with the
notes), and an auto-written methods paragraph to adapt for a manuscript.

### `scan_recordings(folders, base_path, store_labels=None, progress=None) -> list[dict]`
### `scan_recording(folder, base_path, store_labels=None, progress=None) -> dict`

What each recording contains, without processing its signal (about a third of a
second per block): its events grouped by name (`group_markers_by_name`, with
its saved splices applied and its saved markers added), its length, and what it
saved. One dict per recording: `folder`, `name`, `label` (path from
`base_path`), `groups`, `t_range`, `n_events`, `n_splices` (applied),
`n_splices_saved`, `n_saved_markers`, `block_name`, `start_time`, `warnings`, and
`error` (`None`, or why it could not be read at all). A recording that fails comes
back with its `error` set; it never stops the others.

### `default_subject_names(folders, base_path) -> list[str]`

Each recording's folder name; two that share one get their path from `base_path`
on, so names start out unique.

### `load_group_recording(folder, regression_method="ols", store_labels=None, progress=None) -> dict`

One recording exactly as the single-recording analyses see it:
`load_tdt_recording`, reduced to `{"x", "y", "fs", "groups", "n_splices",
"warnings"}` — `y` the normalized dF/F, `groups` the events by name.

### `run_group_analysis(spec, out_dir=None, progress=None, load=load_group_recording, extra_writers=()) -> GroupResults`

The whole analysis, headless: each recording of `spec.subjects` is loaded (one at
a time, so a large group never holds more than one signal), its trials cut and
measured (`extract_group_trials`), the models fitted on all of them
(`fit_group_models`). With `out_dir` the tables and the report are written there
(`write_group_results`), then each of `extra_writers` — callables
`f(results, out_dir) -> list of paths written` — runs; the app draws and saves its
figures this way, since this library has no plotting. `load` is the hook that lets
the analysis run on stand-in recordings. Raises `RuntimeError` naming the subject
and folder if a recording cannot be loaded: leaving one out silently would change
the group.

### `write_group_settings(group_dir, spec, scans, app=None) -> str`
### `read_group_settings(path) -> (GroupSpec, dict)`

`analysis_settings.json` (`GROUP_SETTINGS_FILENAME`) is what makes a group analysis
repeatable: the spec, a summary of each recording used, when it was written, and
the program and library versions that wrote it. `app` names the program, e.g.
`{"name": "PyAT", "version": "3.2.0"}`. `read_group_settings` takes the file or
its folder and returns the `GroupSpec` and the whole document, so a saved analysis
runs again with `run_group_analysis(read_group_settings(folder)[0])`.

---

## Curve-fit models

All follow `f(x, *params) -> y`, ready to pass straight into
`fit_model_to_segment`.

|function|formula|
|-|-|
|`linear_model(x, m, b)`|`y = m*x + b`|
|`single_exponential_model(x, a, b, c)`|`y = a*exp(-b*x) + c`|
|`double_exponential_model(x, a, b, c, d, k)`|`y = a*exp(-b*x) + c*exp(-d*x) + k` — physical bleaching model|
|`exponential_rise_model(x, a, b, c)`|`y = a*(1 - exp(-b*x)) + c` — e.g. venous occlusion|
|`gaussian_model(x, a, mu, sigma)`|`y = a*exp(-(x-mu)^2 / (2*sigma^2))`|
|`sinusoidal_model(x, a, f, phi, c)`|`y = a*sin(2*pi*f*x + phi) + c`|
|`visibility_model(beta, a, v, beta_c, period)`|`y = (a/2) * (1 - v*sin((beta - beta_c) / period))` — photon-entanglement visibility|

### `CURVE_FIT_MODELS`

The registry PyAT's Curve Fit tool lists, and scripts can use the same way:
`{display name: (model_fn, p0_fn, [parameter names])}`, where `p0_fn(x_seg, y_seg)`
returns the starting guess for `curve_fit`.

```python
model_fn, p0_fn, names = pl.CURVE_FIT_MODELS["Exponential Decay  (a*e^(-bx) + c)"]
fit = pl.fit_model_to_segment(x_seg, y_seg, model_fn, p0_fn)
dict(zip(names, fit["popt"])), fit["r2"]
```

The entries are `Linear`, `Exponential Decay`, `Exponential Rise`, `Gaussian`,
`Sinusoidal` and `Double Exponential` (the display names carry their formulas; take
them from `CURVE_FIT_MODELS.keys()`). The starting guesses suit typical shapes: the
sinusoid's, for one, starts at 1 Hz, so it finds a signal only if the phase it drifts
from that guess across the segment stays small. `visibility_model` is not registered.

---

## Running analyses by name

`analyses.py` is every analysis PyAT offers, as a function of a recording and a plain
parameter dict. The app, a script, the [command line](#command-line) and a replay from a
[provenance record](#provenance-records) all call it, so all four give the same numbers.
Each result table has the same columns, in the same order, as the CSV the app exports for
that analysis, so a script's output can be compared with an export column for column.

```python
import PyATCLI as pl

cache = pl.load_recording("D:/rats/Rat7-250101")["cache"]
result = pl.run_analysis("event_peth", cache, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
result.tables["mean_sem"]       # a DataFrame: time_from_event_s, mean, sem
result.parameters               # the parameters as run, defaults filled in
result.summary                  # a few numbers (n_trials, group_peak, ...)
```

### `run_analysis(kind, cache, parameters=None, *, store_labels=None, progress=None) -> AnalysisResult`

Runs one analysis. `cache` is a recording as `recording.py` builds it
(`load_recording(...)["cache"]`, `make_oxysoft_cache`, ...). `parameters` is a dict;
anything left out takes its default. `store_labels` is `{store id: shown name}`, the
renames the app applies to event names. `ParameterError` (a `ValueError`) if a parameter
is missing, unknown or unusable; `AnalysisError` if the parameters were fine but the
recording does not hold enough to run the analysis (a window past the end, an event with
no usable trial).

### `AnalysisResult`

A dataclass: `kind`, `parameters` (as run, defaults filled in, with anything the analysis
resolved from the recording, such as an event's times, written out — this is what a
provenance record stores), `tables` (`{name: DataFrame}`), `summary` (`dict`) and `notes`
(`list[str]`, things worth knowing such as skipped events).

### `kinds() -> tuple` and `describe_kinds() -> dict`

`kinds()` lists the analyses; `describe_kinds()` returns
`{kind: {"summary", "parameters": [{"name", "type", "default", "required", "doc", "choices"}]}}`
for help text. `normalize_parameters(kind, parameters=None) -> dict` checks a parameter
dict and fills in the defaults without running anything.

| kind | what it is | tables |
|------|-----------|--------|
| `zscore_window` | z-score a window around one time against its own pre-event baseline | `window` |
| `auc_window` | area under the curve in a window, split by sign | `auc` |
| `fft_window` | power spectrum of a window and its top peaks | `peaks` |
| `curve_fit` | fit a model to the stretch between two times | `fit` |
| `event_peth` | every occurrence of one event aligned to it: trials, mean/SEM, AUC, peak, time bins | `zscore`, `mean_sem`, `trials`, `auc`, `peak`, `bins` |
| `peak_finder` | statistically significant peaks: whole recording (after a `start_buffer_s`), or near events (with the shuffle-null's p-values, `n_shuffles`) | `peaks` / `alignment` / `scan` by `scope` |
| `event_intervals` | the time between markers | `intervals` |
| `process` | the processed traces themselves (dF/F and the raw channels) | `trace` |

`python -m PyATCLI kinds` prints every parameter of each. The signal-taking
analyses take `signal` (`normalized` by default; for TDT also `main_driver`,
`isosbestic`).

Group analysis is the one analysis that is not a function of one recording; it runs from
a `GroupSpec` over several (see [Group analysis](#group-analysis),
`run_group_analysis`, and `python -m PyATCLI group`). It appears in provenance
records as the kind `group_analysis`.

---

## Provenance records

How a result was produced, as data. PyAT writes `<export name>.provenance.json` beside
every export (Options can turn that off), and `python -m PyATCLI run` writes one
beside its tables. With the record and the recording, the analysis can be run again
([Reproducing a result](#reproducing-a-result)).

A record is a plain JSON-able dict:

| key | contents |
|-----|----------|
| `schema` | `PROVENANCE_SCHEMA_VERSION`, the layout version |
| `kind` | what was produced: an analysis kind, `group_analysis`, `figure`, `table` |
| `created` | ISO date-time |
| `software` | `software_versions()`: this library, Python, platform, the numerical packages, the app (name, version) |
| `source` | `file_fingerprint()` of the recording's folder/file: names, sizes, SHA-256 of files up to 16 MB, a `digest` |
| `recording` | `recording_state()`: its type and length, motion-correction method, every splice in order, marker counts and digests, store renames, signal names |
| `pipeline` | `pipeline_parameters()`: what the processing does that no caller chose (filter, bleaching model, motion-correction settings, z-score baseline rule), read from the constants the code uses |
| `analysis` | `{"type": kind, "parameters": {...}}` — what was asked |
| `outputs` | each file written, with size and SHA-256 |
| `notes` | sentences: how a figure was exported, which labels were edited |

A record holds names, counts, event times, parameters and hashes — never recording
*data* (no samples, no trace values). File paths are in it, since they are what lets the
analysis be run again; `file_fingerprint(path, portable=True)` leaves the absolute path
out for a record that will leave the machine.

* `software_versions(app=None, packages=PACKAGES) -> dict` — what is running; never raises.
  `library_version()` is this library's running version (a source checkout reports its own
  `pyproject.toml` version rather than a stale editable-install one).
* `check_environment(record) -> list[str]` — sentences for the differences between the
  software a record was made with and what runs now that could matter: another library
  version, another Python minor version, another major.minor of numpy/scipy/pandas/
  statsmodels/scikit-learn. `[]` if none.
* `pipeline_parameters(regression_method="ols") -> dict`.
* `file_fingerprint(path, max_files=200, portable=False) -> dict`.
* `markers_summary(markers, detected_markers=None) -> dict` — counts and digests, so a
  record can say which markers an analysis saw without holding them.
* `recording_state(cache, regression_method=None, splices=(), store_labels=None, signal=None) -> dict`.
* `build_record(kind, parameters=None, *, source=None, recording=None, pipeline=None, app=None, outputs=None, notes=None, now=None) -> dict`
  — `source` is a path (fingerprinted here) or a fingerprint dict; `outputs` file paths or
  entries (`output_entry(path)`); empty sections are left out.
* `sidecar_path_for(export_path) -> str` — `Event PETH.png` -> `Event PETH.provenance.json`.
* `write_record(path, record) -> str` and `read_record(path) -> dict` (`ValueError` if the
  file is not a record or is from a newer layout than this library reads).

```python
from PyATCLI import provenance as pv

rec = pv.build_record("event_peth", result.parameters,
                      source="D:/rats/Rat7-250101",
                      recording=pv.recording_state(cache, regression_method="ols"),
                      pipeline=pv.pipeline_parameters("ols"),
                      outputs=["out/event_peth_zscore.csv"])
pv.write_record("out/event_peth.provenance.json", rec)
```

---

## Reproducing a result

`reproduce.py` runs an analysis again from its provenance record and says whether the
numbers come out the same. (`PyATCLI.reproduce` is the module; the function is
exported at top level as `reproduce_record`.)

```python
rep = pl.reproduce_record("Event PETH_L1P_103522.provenance.json",
                          out_dir="again", compare=["Event PETH_L1P_103522.csv"])
rep.ok            # True if every compared export matches
rep.warnings      # the recording changed since, a different numpy, ...
rep.result.tables # the analysis's tables, as run now
rep.comparisons   # per export: ok, message, max_abs_diff, cells, table
```

In order: it reads the record; notes whether the software differs from what produced it;
finds the recording (the path the record holds, or `recording=`) and checks it is the
same one by its fingerprint; rebuilds it in exactly the state recorded — motion-correction
method, every splice, store renames, signal names — which is not necessarily the state its
`JSON saves/` folder is in now; runs the analysis with exactly the recorded parameters (for
an event analysis that includes the event times themselves, so the markers no longer
matter); writes the tables and a new record if `out_dir` is given; and compares any exports
you name, column for column.

A record stores every parameter of its run, so one that lacks a parameter was written before the analysis had it. The
replay fills such a parameter with the value that means what the analysis did then (`Param.legacy`: the Peak Finder's
`start_buffer_s` and `n_shuffles` are `0` for records from before 2026.11.2), not with today's default, so an old
record still reproduces its own numbers. Running the analysis anew (`run_analysis`, `run`) uses today's defaults.

### `reproduce_record(record, recording=None, out_dir=None, compare=(), tolerance=1e-4, strict=False, progress=None, rounded=None) -> Reproduction`

`record` is the record dict or the path of its `.provenance.json`. `compare` lists
exports to check against: a path, or `(path, table name)`; which table is found from the
CSV's own header when not named. With `strict=True` a recording or software that differs
from the recorded one raises `ReproduceError` instead of being reported in `warnings`.
`ReproduceError` also covers a record that is not an analysis, or a recording that cannot
be found.

`rounded` says whether the exports were written with numbers rounded to a fixed number of
digits; `None` decides from the record (PyAT's exports are, the command line's are not).

### `compare_table(expected_path, table, tolerance=1e-4, rounded=False) -> dict`

Compares a CSV with a DataFrame, column for column. A number matches if
`|a - b| <= tolerance * (1 + |b|)`; with `rounded=True` it also matches if it is within the
rounding its own text shows (an export that wrote `13.8` for `13.846` agrees to the one
decimal it kept, while a whole number in it — a trial number, a count — must still be
exact). Everything else is compared as text, ignoring case; blank and missing are the same.
Returns `{"file", "ok", "message", "max_abs_diff", "cells"}`.

### `Reproduction`

A dataclass: `record`, `result` (an `AnalysisResult`), `warnings`, `written` (paths),
`comparisons`; the property `ok` is true if nothing was compared or everything matched.

---

## Command line

`python -m PyATCLI <command>` (and the `pyatcli` command pip installs) is a
thin wrapper over the functions above; the app calls the same ones, which is why the
numbers agree.

| command | does |
|---------|------|
| `info` | versions of this library, Python and the numerical packages — what a result would be stamped with |
| `kinds [KIND]` | the analyses and every parameter of each |
| `run KIND RECORDING [--param NAME=VALUE ...]` | run one analysis; writes CSV tables and a provenance record to `--out`, or under an Output folder with `--output-folder` |
| `reproduce RECORD [--compare EXPORT.csv ...]` | run an analysis again from its `.provenance.json` and check the numbers |
| `batch JOBS.json` | many analyses over many recordings (a folder of TDT blocks counts as many) |
| `group setup FOLDER... --name N [--marker M ...]` | set up a group analysis as the wizard does: find the recordings, list the markers they share (without `--marker`), check the design, write `analysis_settings.json` |
| `group run SETTINGS` | run a group analysis from its `analysis_settings.json` (or its folder): tables, a report and a record |
| `saved ACTION RECORDING ...` | what Save Changes saves, from a script: `show`, `marker`, `unmark`, `splice`, `unsplice`, `rename`, `highlight`, `text`, `clear` |
| `serve` | the engine: these operations as JSON lines on standard input and output, with recordings kept open (what PyAT's window talks to; see [The engine](#the-engine)) |
| `docs [--format markdown\|man\|contract]` | this command line's reference, generated from the program: `CLI_REFERENCE.md`, the man page `man/pyatcli.1`, or the engine's contract `ENGINE_CONTRACT.json` |

```bash
python -m PyATCLI run event_peth "D:/rats/Rat7-250101" --param event="L1P¹" --param pre_s=10 --param post_s=20 --out out
python -m PyATCLI run zscore_window "D:/rats/Rat7-250101" --param center_s=312.5 --param signal=main_driver
python -m PyATCLI reproduce "out/Rat7_event_peth.provenance.json" --compare "out/Event PETH.csv"
python -m PyATCLI batch jobs.json --out batch-out
```

Options shared by `run`, `batch`: `--regression {ransac,huber,ols}` (TDT motion
correction, default `ols`), `--no-saved` (ignore the recording's `JSON saves/`),
`--table`, `--x`, `--y` (generic files); `run` also takes `--params FILE.json`
(`--param` overrides it) and `--json` for a machine-readable summary. `reproduce` takes
`--recording` (where the data is now), `--tolerance`, `--strict`, and `--rounded` /
`--exact` (compare to the digits the export shows, or to the tolerance only).

**Every command** takes these, before or after its name: `--json` (the result as one JSON document on standard output),
`--plain` (tab-separated rows and nothing else, for `cut`, `awk` or a spreadsheet; the two are mutually exclusive),
`-q/--quiet` (only errors), `-v/--verbose` (repeat for more; also shows what libraries print), `-d/--debug` (the full
traceback of an error; also set by `DEBUG=1`), `--no-input` (never ask a question: an action that needs a yes fails
instead), `--events` (progress, warnings and notes as JSON lines on standard error, for a program driving this one) and
`--progress` / `--no-progress` (a progress bar on standard error for slow steps: shown by default when it is a terminal).
`-V/--version` prints the version. Run with no arguments, the program prints a short help (the commands and three
examples); `-h/--help` of a command lists its options and its examples. A `-` where a file is expected means standard input
(`run --params -`, `reproduce -`, `batch -`).

**Output.** Standard output is only ever the result. In `--json` it is one document,
`{"schema": 1, "ok": true, "command": "run", "result": {...}, "warnings": [...]}` (NaN is `null`); warnings, progress and
errors are on standard error, and anything a library prints on its own (the TDT SDK reports every block it opens) is dropped,
so `... --json | jq` always sees valid JSON. A failure under `--json` is one line on standard error,
`{"schema": 1, "ok": false, "error": {"code", "message", "hint", "type"}}`, with the code one of `bad_parameter`,
`invalid_input`, `not_found`, `io_error`, `analysis_failed`, `cannot_reproduce`, `unknown_operation`, `no_recording`,
`confirmation_required`, `cancelled`, `internal_error`; without it, `error: <what is wrong>` and, indented, what to do about
it. A bug (an exception nobody expected) says so and what to report; `--debug` adds the traceback. Colour is never used.

**Where results go.** `run`, `batch`, `group run` and `group setup` take `--out DIR` (exactly there) or `--output-folder
DIR` (the layout PyAT's exports use, see [Output folders](#output-folders): `DIR/<data type>/<recording>
(Analysis)/`, `DIR/Group Analysis/<group>`; a repeat run is a new set of files, never an overwrite; `batch`'s combined
tables go to `DIR/Batch/<NN_kind>/`). The two are mutually exclusive. `group run SETTINGS --out DIR` leaves the settings
beside the tables, so the folder says what it was made with and can be reproduced on its own.

**The command line draws nothing.** It computes and writes files — CSV tables, JSON records, the settings — and never makes a figure or opens a window; the
engine (`serve`) is the same. Drawing is PyAT's: it draws from the same tables (the columns each analysis always returns are in `ENGINE_CONTRACT.json`),
and saves its figures beside them. A script that wants a picture of a result plots the CSVs itself ([Drawing results](#drawing-results)).

`saved` writes the files in a recording's `JSON saves/` folder ([Writing saved state](#writing-saved-state)):
`show [--json]`; `marker --time T [--label --color --fontsize]`; `unmark --time T [--tolerance --label]` (exit 1 if
none was there); `splice --mode {cut_out,keep_inside} --start S --end E`; `unsplice N`; `rename KEY=NAME ...`;
`highlight --start S --end E [--color]`; `text --x X --y Y --text T [--fontsize]`; `clear --yes` (without `--yes`
nothing is deleted and it exits 2). A recording's own load options (`--regression`, `--table`, `--x`, `--y`) apply
where it has to be opened.

`group setup` is the Hypothesis Testing wizard (it was `group-setup`, and `group SETTINGS` was `group run SETTINGS`: the old
spellings still work and say so on standard error): `FOLDER...` are cohort folders (searched at any depth) or recordings;
without `--marker` it lists the markers the recordings have (`*` = in every one), needs no destination, and exits 2; with markers it takes the
wizard's choices (`--pre --post --baseline A B --response A B --signal --smooth --measure --peak-direction
--decay-fraction --correction --alpha --regression --store-label ID=NAME --exclude SUBJECT`, defaults as the
wizard's), prints the design warnings it shows, writes `analysis_settings.json` and, with `--run`, runs it. `group run
SETTINGS` runs a settings file (or its folder) and writes its tables, its report and its record (it draws nothing: the app draws and saves the figures).

A `batch` file is `{"defaults": {"regression_method", "use_saved", "parameters": {...}},
"jobs": [{"kind", "recording", "parameters"}, ...]}`; each job writes one folder per
recording and a combined `<kind>_<table>_all.csv` with a `recording` column.

Exit status: `0` done; `1` something failed (the message is on stderr); `2` bad usage (an unknown flag, a missing argument,
an action that needs a yes and was not given one: `saved clear` asks when it can, and without a terminal says to use `--yes`);
`3` `reproduce` ran but an export did not match; `130` interrupted (Ctrl-C).

**Documentation.** `CLI_REFERENCE.md` (every command and option, generated) and the man page `man/pyatcli.1` are made by the
program (`python -m PyATCLI docs --format markdown|man`), and a test regenerates both and fails if they differ from the
committed files, so they cannot drift from what the program accepts. `man ./man/pyatcli.1` reads the page.

**Environment.** `DEBUG` (anything but `0`/`false`/`no`/`off`: the same as `--debug`); `NO_COLOR` is accepted and unused (there is no colour to turn off).
The library keeps no configuration file and nothing between runs: no cache folder, no pointer to the last result.

---

## Operations and the engine

Every command above is a thin front over an **operation** (`PyATCLI.ops`): a function of keyword parameters that
returns a JSON-ready dict, registered with `@operation(name, summary)` and run with `ops.call(name, params, ctx)`. The command
line parses arguments and renders the result (human, `--plain`, `--json`); the engine serves the same operations to a program
over a pipe; because both are the same functions they cannot disagree. `ops.describe()` (or the `operations` operation) lists
them with their parameters.

| operation | does |
|-----------|------|
| `run`, `reproduce`, `batch` | the commands of the same name |
| `group.setup`, `group.run`, `group.scan` | the Hypothesis Testing wizard's steps; `group.scan` lists what each recording holds (markers, length, saved splices) without processing it |
| `saved.show`, `.marker`, `.unmark`, `.splice`, `.unsplice`, `.rename`, `.highlight`, `.text`, `.clear`, `.write` | what Save Changes saves (`write` replaces whole lists) |
| `recording.open`, `.view`, `.splice`, `.labels`, `.close`, `.list` | engine only: recordings held open between requests (below) |
| `compute.event_peth`, `compute.peaks` | the Event PETH window's trial matrix, and Peak Finder (`start_buffer_s`; `n_shuffles` and `seed` add a `null` per event type, with `p_count_holm` across the types, off by default here) |
| `files.inspect`, `pt2.load` | what a path is (a TDT block, a folder with blocks under it, an Oxysoft export, a table file); a Terranova `.pt2` image as an `.npy` file |
| `provenance.write` | the provenance record beside an export (`<export>.provenance.json`) |
| `fieldstudy.fields`, `.run`, `.validate` | the text field study and its statistical validation (they embed text with a language model) |
| `file.tables` | the tables of a generic file: names, columns, sizes and first rows |
| `results.open` | what a results folder, a record or a table file holds (`open_results`, below): its kind, parameters, tables, notes and the files they came from |
| `info`, `kinds`, `operations` | what is installed, the analyses, the operations |

An `OpError(message, code, hint)` is an expected failure; `ops.error_info(exc)` turns any exception into the record
`{code, message, hint, type}` (a bug is `internal_error`), and `ops.to_jsonable` makes a result JSON (NaN becomes `null`).
An operation receives an `OpContext` with `emit(event, **fields)` (progress, warnings, notes, files written), `cancelled()`,
and, in the engine, the open `store` and a `data_dir`.

### The engine

`python -m PyATCLI serve` is a long-lived process that speaks one JSON object per line on standard input and output.
It exists so that a window need not import the numerical libraries (a second or two, and a lot of memory) or freeze while an
analysis runs: PyAT starts it once and sends it requests. A request is `{"id": 7, "method": "recording.open", "params": {...}}`;
the engine answers with `progress` events (`{"id": 7, "event": "progress", "fraction": 0.4, "message": "..."}`, also `warning`
and `note`) and then one `{"id": 7, "event": "result", "ok": true, "result": {...}}` or `{"id": 7, "event": "error", "ok":
false, "error": {...}}`. On start it says `{"event": "ready", "protocol": 1, "library": ..., "pid": ..., "data_dir": ...,
"operations": [...]}`. `{"cancel": 7}` stops request 7 at its next progress report (or before it starts: the request then
fails with code `cancelled`); `{"method": "shutdown"}`, or closing standard input, ends the process. Requests run one at a
time, in the order they were sent. Standard output is protocol and nothing else (anything a library prints is dropped, or goes to
standard error with `-v`).

**Recordings stay open.** `recording.open` reads a recording as the app does (saved splices replayed, saved markers,
highlights, text boxes and signal names put on) and keeps it; `fresh: true` reads the disk again. Later requests name it as
`{"id": "r1", "splices": [...], "markers": [...], "signal_labels": {...}}` where an operation takes a `recording`: the
markers (and highlights, text boxes, names) in the reference are the caller's own and are put on the recording for that request,
so the engine never holds a second copy of a window's edits to keep in step. A **splice is state**: the engine keeps one variant
of the recording per list of splices (the most recent few), so `recording.view` with an earlier list — Ctrl+Z, removing a splice,
Restore Full Recording — costs nothing, and `recording.splice` stacks one more (a TDT splice re-fits the dF/F: seconds on a long
recording) and hands back the caller's markers moved through it.

**Arrays do not go through the pipe.** A recording is a few kilobytes of structure and tens of megabytes of arrays (a 17-minute,
three-signal TDT recording is 34 MB; as JSON text one trace is 44 MB and a second to parse). The structure is JSON; every array
is written once as a `.npy` file in the engine's data folder and replaced, in the structure, by
`{"__array__": "0003_signals.normalized.y.npy", "dtype": "float64", "shape": [1055744]}`. `PyATCLI.engine.wire.decode`
maps them back (`numpy.load(..., mmap_mode="c")`: nothing is read until it is touched and an edit stays in memory, never in the
file), so a window can plot a million samples that never passed through a pipe; an array that two places of the structure share
is one file and one array. Tables travel as their shapes (`tables: "shapes"`), inline (`"inline"`), or as CSV files (`"files"`).
Closing a recording removes its files; the rest go when the engine exits.

**The contract, and the protocol number.** What a program using the engine can count on is written down in `ENGINE_CONTRACT.json`
(generated: `python -m PyATCLI docs --format contract --out ENGINE_CONTRACT.json`): every operation with its parameters, and every
analysis with its parameters, the tables it returns and the columns of each that are always there, and the tables written besides (`outputs`: a
group analysis's folder, with the file each table goes to, and an Export View's tables). Read tables by column name, and count on those
columns; a table may have more (a column per time bin or per signal) and more can appear at any time. The same is written down for what an
operation returns (`result` in the contract: the keys of its answer that are always there; `PYATCLI_CHECK_RESULTS=1` makes the engine fail loudly
when one is missing) and for a loaded recording (`PyATCLI.recording.CACHE_KEYS`: the keys of its `cache`, for a TDT, an Oxysoft or a generic
recording). `ready` carries the protocol number
(`PyATCLI.ops.PROTOCOL_VERSION`), which goes up when something a caller could rely on is removed or renamed — an operation or a parameter, a
parameter that must now be given, an analysis, a choice of one, a table, a column that was always there — and not when something is added.
`PyATCLI.engine.contract.breaking_changes(old, new)` says which changes count, and `tests/test_engine_contract.py` fails until the protocol is
raised (and the new `tests/contracts/protocol_N.json` baseline added) when a change would break a caller. A program should compare the `protocol` of
`ready` with the ones it speaks, and stop with a clear message if they differ (PyAT does).

```python
import json, subprocess, sys
p = subprocess.Popen([sys.executable, "-m", "PyATCLI", "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
ready = json.loads(p.stdout.readline())
p.stdin.write(json.dumps({"id": 1, "method": "recording.open", "params": {"path": "D:/rats/Rat7-250101"}}) + "\n"); p.stdin.flush()
while (msg := json.loads(p.stdout.readline())).get("event") not in ("result", "error"):
    pass                                                # progress events
cache = msg["result"]["cache"]                          # structure; arrays are {"__array__": file, ...} in msg["result"]["directory"]
```

---

## Reading results back

### `open_results(source) -> OpenedResults`

`PyATCLI.results.open_results` finds the numbers of a finished analysis from the files it left, with no drawing and no plotting library. `source`
is a group analysis's folder (or its `analysis_settings.json`), a `.provenance.json` record, a folder holding records or tables (its newest record, or
else its `.csv` tables), or one `.csv`. A group's folder is recognised by its settings file, whatever has been exported into it since.

**Returns** an `OpenedResults` (a dataclass): `path` and `directory` (absolute), `kind` (what made them: `"group_analysis"`, an analysis kind such as
`"event_peth"`, or `None` for a bare table), `parameters` (what it was asked; a group's settings as a dict), `tables` (`{name: DataFrame}`, the numbers
read back exactly as written; a group's by the names in `TABLE_FILES`), `notes` (`[{"level", "measure", "text"}]`, a group's) and `software`, and
`files` (what was read). Raises `NoResults` (a `ValueError`) if there is nothing there to read.

The engine's `results.open` (`path`, `tables="shapes"|"inline"|"files"`) answers with the same, tables as `group.run` hands them back. PyAT's
Open Results… asks for it and draws what comes back.

`analysis.group.rebuild.results_from_tables(tables, parameters=None, notes=None, software=None)` puts a group analysis's `GroupResults` back together
from its tables (what the report and PyAT's figures are written from); `None` if they are not a group analysis's.

---

## Drawing results

The library draws nothing. It computes and writes numbers — tables, JSON records, settings — and has no plotting code and no plotting dependency (no
matplotlib, no `[plot]` extra). The figures are PyAT's: its analysis windows, a group analysis's figures, Export View, Edit Graph Attributes and the
writing of a figure to a PNG, TIFF, PDF or SVG file with the CSV of its data are `physicsanalysis_qt/figures`, which draws from the same tables the
library writes (the columns each one always has are in `ENGINE_CONTRACT.json`). Open Results… in PyAT draws a finished analysis again from its files.

A script that wants a picture reads the tables and draws them itself:

```python
result = pl.run_analysis("event_peth", cache, {"event": "L1P¹", "pre_s": 10, "post_s": 20})
result.tables["trials"]         # Trial, Trial Event Time (s), time_from_event_s, z_score — a row per trial and sample
result.tables["mean_sem"]       # time_from_event_s, mean, sem

opened = pl.open_results("out/Group Analysis/Sucrose")      # or from the files a run wrote (see Reading results back)
opened.tables["mean_traces"]    # marker, subject, n_trials, time, value
```

What a script that used to call `plotlatest`, `build_view`, `save_figure` or `edit_attributes` does now: take the tables (`run_analysis`,
`run_view(...).tables`, `open_results`) and plot them with matplotlib or any other tool. Export View's numbers (the lines, the range, the markers, the
data CSV) are `viewplan` ([Export View's numbers](#export-views-numbers)).

---

## Export View's numbers

`PyATCLI.viewplan` says what the main plot of a recording shows, as data: which lines it has (under what name, with what numbers), the range, the
markers in it, and the title and axis labels it would carry. The figure itself — colours, widths, the legend, the page — is PyAT's (Export View, and the
plot on screen, both engines).

```python
cache = pl.load_recording("D:/rats/Rat7-250101")["cache"]
plan = pl.viewplan.plan_view(cache, xlim=(40, 120), signals=["normalized", "main_driver"])
plan.data_table()          # time_s and a column per line, every sample in view
plan.write_data("fig1")    # fig1_data.csv, and fig1_markers.csv if any marker is in view
```

* `plan_view(cache, *, signals=None, active=None, xlim=None, ylim=None, show_grid=True, store_labels=None, attributes=None) -> ViewPlan`. `cache` is a
  recording as `load_recording` returns it. `xlim`, `ylim`: the range, in seconds and signal units (default: all of the recording with a little room;
  the y range then follows the data). `signals`: a TDT recording's signals to show — keys of `cache["signals"]` (`"normalized"` by default; more than
  one are overlaid, in the recording's order); `active` the one shown if only one is. `store_labels`: `{store id: shown name}` for marker names.
  `attributes`: whatever the viewer drew it with (a title, labels, colours), kept for `parameters()` and never interpreted.
* `ViewPlan`: `.traces`, `.xlim`, `.ylim`, `.markers` (the markers in view as `(time, shown name, colour)`), `.in_view`, `.signals`, `.title`, `.x_label`,
  `.y_label`, `.parameters()` (what it takes to make the view again, JSON-ready), `.data_table()` (one row per sample in view: `time_s` — or the file's own
  x name — and a column per line, at the recording's full resolution), `.markers_table()`, `.write_data(base)`.
* `view_traces(cache, *, signals=None, active=None) -> (traces, title, x_label, y_label)`: which lines a recording's main plot has. A `Trace` says what
  it is, never how it looks: `label` (`"_nolegend_"` for a channel line that follows its group's entry), `x`, `y`, `y_full` (the whole time axis, NaN
  where the line has no sample), `csv_name`, `key` (the name per-line choices are filed under), `role` (`"signal"`, `"corrected"`, `"o2hb_channel"`,
  `"hhb_channel"`, `"o2hb_mean"`, `"hhb_mean"`, `"thb_mean"`, `"column"`), `signal` (a TDT signal's key) and `color` (the colour the recording itself gives
  the signal, if any). A TDT recording shows its normalized (dF/F) trace, or the signals ticked; an Oxysoft export a line per channel and a mean per group
  (O2Hb, HHb, tHb), the fit factor in the mean's name; a generic table a line per column with its gaps left out.
* `padded_xlim(x)`: the "show everything" range, with a little room.

### Replaying a view

`run_view(cache, parameters=None, *, store_labels=None) -> AnalysisResult` gives a view's numbers: `tables["data"]` is the CSV beside the figure and
`tables["markers"]` the markers in view. `parameters` are a view's own (`ViewPlan.parameters()`): `x_range`, `y_range`, `signals`, `show_grid`,
`attributes` (carried as given: it changes no number). A record of kind `view` — PyAT's Export View writes one — is replayed by `reproduce_record`,
which compares a data CSV column for column like any analysis:

```python
rep = pl.reproduce_record("fig1.provenance.json", compare=["fig1_data.csv"])
rep.ok
```

### Naming signals

`recording.name_problem({signal key: name})` says why those names cannot be a recording's (a name must say something, and two signals cannot share one),
and `recording.plan_signal_renames(cache, renames)` returns `{signal key: new name}` for what would change (`ValueError`, nothing changed, if two signals
would share a name). A signal's name is saved state (`saved_state.rename_signals`, refusing what the app refuses) and applied when the recording is
opened, so it reaches every analysis, table and figure.

---

## Text Field Study

A pipeline for comparing free-text survey/response fields across
subjects via sentence-embedding similarity — e.g. "did this subject's
answer to field A resemble their answer to field B more than chance
pairing would predict?" Needs the optional `sentence-transformers`
package (see [Optional dependencies](#optional-dependencies)).

### `run_field_study_pipeline(folder_path, text_fields, delta_pair=None, paired_fields=None, model_name="all-MiniLM-L6-v2", n_null=200, rng_seed=None, file_glob="P-*.json", min_words=5, progress=None) -> pandas.DataFrame`

The single entry point — runs the whole pipeline (load → flag
low-quality → embed → optional delta vector → optional paired
similarity + permutation test + word-count confound check) and folds
everything into one DataFrame, one row per subject.

* `delta_pair` (`(field_from, field_to)`, optional): adds a
`delta_magnitude` column — `‖vec(field_to) - vec(field_from)‖`.
* `paired_fields` (`[(field_a, field_b, pair_name), ...]`, optional):
adds per pair `sim_<name>`, `null_mean_<name>`, `null_std_<name>`,
`pvalue_<name>`/`effect_size_<name>` (permutation test), and
`wc_confound_r_<name>`/`wc_confound_p_<name>` (word-count confound
check). The test-level values repeat identically across every row for
that pair — one number per pair, not per subject.

The stage functions below are also available individually for anyone
who wants an intermediate result.

### `load_field_study_folder(folder_path, text_fields, file_glob="P-*.json", progress=None) -> pandas.DataFrame`

Loads every file matching `file_glob` into one row-per-subject
DataFrame, adding a `wordcount_<field>` column per text field.

### `peek_fields(folder_path, file_glob="P-*.json") -> list[str]`

Field names (dict keys) found in the *first* matching file, without
loading the whole folder — for a GUI to show real field names to pick
from rather than requiring them typed blind.

### `flag_low_quality(df, text_fields, min_words=5) -> pandas.DataFrame`

Flags (doesn't drop) near-empty responses. Adds `low_quality_<field>`
per field and `any_low_quality` (OR across fields). Must run after
`load_field_study_folder` (needs the `wordcount_<field>` columns).

### `embed_text_fields(df, fields, model_name="all-MiniLM-L6-v2", progress=None) -> dict`

Embeds each field with a `sentence-transformers` model, L2-normalized
(so a plain dot product is cosine similarity). Returns
`{field: (n_subjects, dim) ndarray, ...}`. Raises a clear `ImportError`
with an install hint if `sentence-transformers` isn't installed.

### `compute_delta_vector(embeddings, field_from, field_to) -> (delta_vectors, delta_magnitude)`

Per-subject `vec(field_to) - vec(field_from)`, plus its magnitude.

### `compute_paired_similarity(embeddings, paired_fields, n_null=200, rng_seed=None, progress=None) -> dict`

Cosine similarity between each field pair for the same subject, plus a
null distribution built by repeatedly shuffling one field's vectors
across subjects. Returns, per pair name:

```python
{
    'same_subject_similarity': ndarray,   # (n_subjects,)
    'null_mean': float, 'null_std': float,
    'null_values': ndarray,               # (n_null * n_subjects,), pooled
    'null_shuffle_means': ndarray,        # (n_null,) — the actual null
                                           # distribution for testing the MEAN
}
```

### `permutation_test_similarity(same_subject_similarity, null_shuffle_means) -> dict`

One-tailed permutation test: is observed mean same-subject similarity
higher than chance pairing? Returns `{"observed_mean_similarity", "p_value", "effect_size", "null_shuffle_mean", "null_shuffle_std"}`.

### `wordcount_confound_check(df, paired_fields) -> dict`

Pearson correlation between each pair's combined word count and its
similarity, across subjects — a significant positive correlation would
mean "longer answers just look more similar," undercutting a similarity
claim. Returns `{pair_name: {"r", "p_value"}, ...}` (both `NaN` if fewer
than 2 subjects or word count has no variance).

---

## Text Field Study — Statistical Validation

A companion module that stress-tests a Text Field Study result: is the
similarity real, or could it be a word-count artifact, a single outlier
subject, or noise that wouldn't survive multiple-comparisons correction?

### `run_validation_pipeline(folder_path, text_fields, paired_fields, model_name="all-MiniLM-L6-v2", n_null=200, n_boot=1000, rng_seed=None, file_glob="P-*.json", id_field="participant_id", progress=None) -> pandas.DataFrame`

Full validation pipeline: load → embed → paired similarity → build the
summary table below. Self-contained (recomputes embeddings rather than
reusing a prior `run_field_study_pipeline` call). `paired_fields` is
required here — there's nothing to validate without at least one pair.

### `build_validation_summary(df, paired_fields, similarity_results, id_field="participant_id", n_boot=1000, rng_seed=None, progress=None) -> pandas.DataFrame`

One row per field pair:

|column|meaning|
|-|-|
|`pair`, `field_a`, `field_b`|which fields|
|`observed_mean_similarity`|mean same-subject similarity|
|`p_value`|permutation-test p-value (one-tailed)|
|`p_value_fdr`|**use this one**, not raw `p_value` — Benjamini-Hochberg corrected across all pairs in the table|
|`cohens_d`|effect size vs. the pooled chance-pairing null|
|`wc_coef_a`/`wc_pvalue_a`, `wc_coef_b`/`wc_pvalue_b`|word-count regression coefficient/p-value per field, controlling for the other — significant means part of the effect may just be verbosity|
|`regression_r_squared`|variance in similarity explained by both word counts (low = reassuring)|
|`ci_lower`, `ci_upper`|bootstrap 95% CI on `observed_mean_similarity`|
|`n_flagged_loo`|subjects flagged by leave-one-out sensitivity (0 is reassuring)|
|`flagged_participant_ids`|which ones|

### `cohens_d(group1, group2) -> float`

Standardized mean difference between two distributions (pooled std
units) — rough guide \~0.2 small, \~0.5 medium, \~0.8 large. Typically
called with `group1` = same-subject similarities, `group2` = pooled
null similarities.

### `benjamini_hochberg(p_values) -> ndarray`

False-discovery-rate correction across multiple hypothesis tests — less
conservative than Bonferroni while still controlling the *expected
proportion* of false positives among results called significant.
Returns corrected p-values ("q-values"), same order as input.

### `wordcount_controlled_regression(df, field_a, field_b, sim_col) -> dict`

OLS regression: `similarity \~ wordcount_<field_a> + wordcount_<field_b>`.
Tests whether either field's length predicts similarity independent of
the other. **Requires `statsmodels`** (hard import, no fallback).
Returns `{"r_squared", "coef_const", "coef_wordcount_<field_a>", "coef_wordcount_<field_b>", "pvalue_const", "pvalue_wordcount_<field_a>", "pvalue_wordcount_<field_b>", "model"}` (`model` is the full
`statsmodels` `RegressionResults` object).

### `bootstrap_mean_ci(values, n_boot=1000, ci=0.95, rng_seed=None) -> dict`

Bootstrap confidence interval on the mean (resample with replacement,
`n_boot` times) — doesn't assume normality, unlike a textbook
`mean ± 1.96·SE` interval. Returns `{"mean", "ci_lower", "ci_upper", "boot_means"}`.

### `leave_one_out_sensitivity(values, ids=None) -> dict`

Recomputes the mean with each subject dropped one at a time; flags any
subject whose removal shifts the mean by more than one standard
deviation of the resulting leave-one-out-means distribution. Catches a
single subject quietly dominating the result. A flagged subject isn't
automatically wrong/excludable — it's worth a manual look.

**Returns** `{"full_mean", "loo_means", "shifts", "threshold", "flagged_ids"}`.

---

## Progress reporting

Every slow function takes an optional `progress` argument (it is in the
signature of each one above), built on [tqdm](https://tqdm.github.io):

|`progress=`|what happens|
|-|-|
|`None` *(default)*|silent, and it costs nothing|
|`True`|a tqdm progress bar on the console (stderr), for scripts and notebooks|
|a callable|called as `progress(fraction, message)`: `fraction` runs 0.0 to 1.0 and never goes backwards, and `message` says what is happening right now (e.g. `"Fitting photobleaching baseline"`). It may be called from whatever thread the work runs on, so a GUI should hand each call over to its own thread (PyAT turns it into a Qt signal that updates a toast)|

```python
result = pl.process_tdt_folder(folder_path, progress=True)
result = pl.process_tdt_folder(folder_path, progress=lambda fraction, message: print(f"{fraction:4.0%}  {message}"))
```

Progress never changes a result, and a callback that raises is switched off
with a warning instead of aborting the work it was only watching.

What reports, and how: `process_tdt_folder` / `get_tdt_struct` / `compute_dff`
by stage (reading the streams, each event store, motion correction, the
photobleaching baseline), the Oxysoft loaders by bytes read, `load_any_file`
by sheet (Excel) or by row pass (CSV/TSV/text), `compute_event_zscore_peth`
and `find_peak_near_events` by event, and the Text Field Study pipelines
(`load_field_study_folder`, `embed_text_fields`, `compute_paired_similarity`,
`run_field_study_pipeline`, `build_validation_summary`, `run_validation_pipeline`)
by step, the language-model load and the embedding being the long ones.

### `Plan(progress, stages)`

Turns named, weighted stages into one 0-1 fraction: for your own slow code,
or to give a callee a slice of your bar.

```python
plan = pl.Plan(progress, [("Reading files", 30), ("Fitting", 70)])
for path in plan.track("Reading files", paths):      # a tqdm iterator, so updates are throttled
    ...
plan.begin("Fitting")
fit(data, progress=plan.sub("Fitting"))              # the callee's 0-1 is mapped into this stage
plan.done()
```

* `plan.track(stage, iterable, total=None)` iterates as that stage; `plan.bar(stage, total)` is the same for updates by hand (`bar.update(n)`); `plan.begin(stage)` marks a step with no loop to tick; `plan.sub(stage)` is a stage as a `progress` callback for another function; `plan.done()` reports 100%.
* With `progress=None` every one of these is a no-op: `track` hands the iterable straight back and `sub` returns `None`, so the callee stays silent too.

### `track(iterable, progress=None, description="Working", total=None)`

`Plan` for a single loop: `for f in pl.track(files, progress, "Reading files"): ...`

---

## Optional dependencies

`numpy`, `scipy`, `pandas`, `tdt`, `scikit-learn`, and `tqdm` are regular,
required dependencies (installed automatically by
`pip install PyATCLI`) — `scikit-learn` in particular backs
*all* of `process_tdt_folder`'s regression methods including `"ols"`,
since `sources/processing_TDT.py` imports it unconditionally at module level,
not just when RANSAC/Huber are actually selected.

The ones below are genuinely soft — each has a guarded import that fails
with a clear, actionable message instead of crashing obscurely, rather
than being unavailable outright:

|package|needed for|behavior if missing|
|-|-|-|
|`openpyxl`|`load_any_file` on `.xlsx`/`.xls`|`ImportError` at call time: "Install openpyxl to read .xlsx files: pip install openpyxl"|
|`sentence-transformers`|`embed_text_fields` (and everything downstream: `compute_delta_vector`, `compute_paired_similarity`, `permutation_test_similarity`, `wordcount_confound_check`, `run_field_study_pipeline`, `run_validation_pipeline`, `build_validation_summary`)|`ImportError` with an install hint (`pip install sentence-transformers`)|

`sentence-transformers` pulls in `torch`/`transformers` (hundreds of MB
to 1GB+) — it's actually listed as a normal, unconditional dependency in
`pyproject.toml` despite being guarded at the code level like a soft
one; if you're building a lightweight install and don't need Text Field
Study, it's worth installing everything else by hand rather than via
`pip install PyATCLI` directly.

`statsmodels` is also a required `pyproject.toml` dependency, imported
unguarded where it is used: `fit_group_models` (the mixed-effects models of
group analysis) and `wordcount_controlled_regression`. It is not touched by
loading, TDT processing or the single-recording analyses.

