"""
analysis/zscore_peth.py
------------------------
Single-click Z-Score PETH: extract and z-score a window around one
event. get_zscore_slice is also the shared building block event_peth.py
and peak_finder.py call per-event/per-trial, so a bug fix or behavior
change here (e.g. how the baseline is protected from artefacts) applies
everywhere a peri-event window gets z-scored, not just this one tool.
"""

import numpy as np

from .. import pipeline
from ..numeric import as_float64
from .shared import window_around

# How far (in robust SDs) a baseline sample may sit from the baseline's median before it is pulled in for the
# mean/SD — see _baseline_mean_std. Named so provenance can report it.
BASELINE_ROBUST_SD = 5.0
ZSCORE_STATISTICS = ("mean_sd", "median_mad")


def _check_baseline_window(values):
    start, end = values["zscore.baseline_start_s"], values["zscore.baseline_end_s"]
    if start is not None and not start < end:
        raise ValueError(f"zscore.baseline_start_s ({start:g}) must be before zscore.baseline_end_s ({end:g})")


pipeline.register_step("zscore", "Z-score baseline", 50, [
    pipeline.Option("zscore", "baseline_start_s", "seconds_or_none", None, "Baseline from", "Where the baseline a trial is z-scored against starts, in seconds from "
                    "the event (negative is before it). None: the start of the window.", unit="s"),
    pipeline.Option("zscore", "baseline_end_s", "float", 0.0, "Baseline until", "Where the baseline ends, in seconds from the event. 0: at the event (the samples "
                    "before it are the baseline).", unit="s"),
    pipeline.Option("zscore", "statistic", "choice", "mean_sd", "Z-score by", "mean_sd: (x - mean) / SD of the baseline, with the few samples far from its median "
                    "pulled in first; median_mad: (x - median) / (1.4826 x MAD), which large deviations in the baseline hardly move.", choices=ZSCORE_STATISTICS),
], check=_check_baseline_window)


@pipeline.sentence("zscore")
def _zscore_sentence(options, state):
    start, end = options["zscore.baseline_start_s"], options["zscore.baseline_end_s"]
    if start is None and end == 0:
        baseline = "the samples before the event"
    else:
        first = "the start of the window" if start is None else f"{abs(start):g} s {'before' if start < 0 else 'after'} the event"
        last = "the event" if end == 0 else f"{abs(end):g} s {'before' if end < 0 else 'after'} the event"
        baseline = f"the samples from {first} until {last}"
    if options["zscore.statistic"] == "median_mad":
        how = "the median and the median absolute deviation (scaled by 1.4826) of that baseline"
    else:
        how = (f"the mean and standard deviation of that baseline, after samples more than {BASELINE_ROBUST_SD:g} robust standard deviations "
               "(1.4826 x the median absolute deviation) from its median were pulled in to that limit")
    return f"Each trial was z-scored against its own baseline, {baseline}, using {how}"


def get_zscore_slice(time_array, signal, center_t, window=None, pre=None, post=None, options=None):
    """
    Extract and z-score a time window around an event.

    Supports either a symmetric `window` (split evenly before/after
    center_t, legacy behaviour) or an asymmetric `pre`/`post` pair — e.g.
    10s before the event and 20s after. If pre/post are given they take
    precedence over window.

    Parameters
    ----------
    time_array : array
    signal : array
    center_t : float
        Event time in seconds
    window : float or None
        Total window size in seconds, split evenly before/after center_t.
        Ignored if pre/post are given.
    pre : float or None
        Seconds before center_t to include. Defaults to window/2.
    post : float or None
        Seconds after center_t to include. Defaults to window/2.
    options : PipelineOptions, dict or None
        The pipeline's options (pipeline.py): where the baseline starts and ends
        relative to the event and whether the z-score is by mean and SD or by median
        and MAD. None: the defaults — the samples before the event, mean and SD.

    Returns
    -------
    (time segment, z-scored signal)
    """
    seg_x, seg_y = window_around(time_array, signal, center_t, window, pre, post)
    if seg_y.size == 0:                       # nothing in the window (an event past the recording's end): no statistics to take
        return seg_x, np.zeros_like(seg_y)

    # Baseline is the pre-event portion — the part of the window that
    # actually precedes the event, not just "the first half of the segment"
    # (those differ once pre != post).
    options = pipeline.PipelineOptions.from_dict(options)
    start, end = options["zscore.baseline_start_s"], options["zscore.baseline_end_s"]
    if start is None and end == 0.0:
        baseline_mask   = seg_x < center_t
    else:
        baseline_mask   = (seg_x < center_t + end) if start is None else ((seg_x >= center_t + start) & (seg_x < center_t + end))
    baseline_period = seg_y[baseline_mask] if baseline_mask.any() else seg_y
    mu, std = baseline_level_and_spread(baseline_period, options["zscore.statistic"])

    # "Flat" means the baseline's spread is down at floating-point rounding
    # level next to its own size — a relative test, so it doesn't depend on
    # the units the signal is in.
    scale = np.max(np.abs(baseline_period)) if baseline_period.size else 0.0
    if std == 0 or std <= 1e-12 * scale:
        return seg_x, np.zeros_like(seg_y)

    return seg_x, (seg_y - mu) / std


def baseline_level_and_spread(baseline, statistic="mean_sd"):
    """(level, spread) of a baseline: what a trial is z-scored against (or has subtracted). `statistic` is the pipeline option zscore.statistic: "mean_sd" (the
    mean and SD, with the few samples far from the median pulled in first: _baseline_mean_std) or "median_mad" (the median and 1.4826 x the median absolute
    deviation). The single-recording z-score and the group analysis take it from here, so they cannot disagree."""
    if statistic == "median_mad":
        level = np.median(baseline)
        return level, 1.4826 * np.median(np.abs(baseline - level))
    return _baseline_mean_std(baseline)


def _baseline_mean_std(baseline, n_robust_sd=BASELINE_ROBUST_SD):
    """
    Mean and SD of a baseline window, protected from artefact samples.

    Samples further than `n_robust_sd` robust standard deviations (1.4826 x
    the median absolute deviation) from the baseline's median are pulled in
    to that limit before the mean and SD are taken, so a brief motion spike
    inside the baseline can't inflate the SD and shrink every z-score in the
    trial. The limit is relative to the baseline's own spread, so the result
    is the same whatever units the signal is in, and only the baseline
    statistics are treated this way — the samples being scored are never
    altered, so a large real response keeps its real height.
    """
    med = np.median(baseline)
    robust_sd = 1.4826 * np.median(np.abs(baseline - med))
    if robust_sd > 0:
        baseline = np.clip(baseline, med - n_robust_sd * robust_sd, med + n_robust_sd * robust_sd)
    return np.mean(baseline), np.std(baseline)


def bin_for_heatmap(z_seg, num_bins=300):
    """
    Bin a signal into equal segments for heatmap plotting.

    Parameters
    ----------
    z_seg : array
    num_bins : int

    Returns
    -------
    array
        Binned signal
    """
    if z_seg is None or len(z_seg) == 0:
        return np.zeros(num_bins)
    z_seg = as_float64(z_seg)
    bin_edges = np.linspace(0, len(z_seg), num_bins + 1).astype(int)
    return np.array([np.mean(z_seg[bin_edges[i]:bin_edges[i+1]]) for i in range(num_bins)])
