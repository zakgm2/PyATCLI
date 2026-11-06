"""
analysis/peak_finder.py
-------------------------
Statistically significant transients: either auto-detected straight
from the signal (find_significant_peaks), or checked for near a given
set of event times (find_peak_near_events) rather than assuming an
event marker itself marks where the neural signal responds. How often
that second search finds a "response" by chance, and whether the real
events do better than that, is what shuffle_null_near_events measures.
"""

import numpy as np
from scipy.signal import find_peaks

from ..numeric import as_float64
from ..progress import Plan
from .shared import estimate_sample_rate
from .zscore_peth import get_zscore_slice

DEFAULT_START_BUFFER_S = 10.0    # seconds at the start of a recording in which find_significant_peaks reports nothing
DEFAULT_SHUFFLES = 1000          # shuffled event trains shuffle_null_near_events uses


def find_significant_peaks(time_array, signal, z_threshold=5.0, min_distance_sec=1.0,
                            include_troughs=False, start_buffer_s=DEFAULT_START_BUFFER_S):
    """
    Auto-detect statistically significant transients directly from the
    signal, rather than relying on externally-supplied event markers
    (TDT epocs, manual markers, ...) that may not actually line up with
    where the neural signal itself is doing something.

    The whole recording is z-scored against its own global mean/std
    (not a local baseline — this is a single-pass "how unusual is this
    point relative to the entire recording" measure, not per-event), and
    scipy.signal.find_peaks picks local maxima at or above z_threshold,
    at least min_distance_sec apart so a single transient's rising edge
    doesn't get counted as several peaks.

    Parameters
    ----------
    time_array : array
    signal : array
        Already-processed signal (e.g. bleach-corrected + smoothed) —
        this function does no filtering of its own.
    z_threshold : float
        Minimum z-score (standard deviations above the recording's own
        mean) for a peak to count as "statistically significant". The
        default of 5 is the usual five-sigma convention.
    min_distance_sec : float
        Minimum spacing between detected peaks, in seconds.
    include_troughs : bool
        Also detect significant negative-going deflections (z <=
        -z_threshold) — off by default since most fibre-photometry
        analyses care about excitatory transients specifically.
    start_buffer_s : float
        Seconds at the start of the recording in which no peak is reported.
        A trace is least settled right after the light comes on (the
        bleaching is steepest and the motion-correction fit has the least
        to go on there), and the whole-recording z-score reads the tail of
        that as unusual. It is only the search that skips the start: the
        mean and standard deviation still come from the whole recording,
        so no peak's z-score depends on the buffer. (A peak within
        min_distance_sec of the buffer's end that a taller peak inside the
        buffer would have hidden can show up.) The default is 10 s; 0
        searches everything.

    Returns
    -------
    list of dict, each {"time": float, "z_score": float, "kind": "peak"|"trough"},
    sorted by time.
    """
    time_array = as_float64(time_array)
    signal = as_float64(signal)
    if start_buffer_s < 0:
        raise ValueError(f"start_buffer_s cannot be negative (got {start_buffer_s!r})")
    fs = estimate_sample_rate(time_array)
    distance = max(1, int(min_distance_sec * fs))

    mu, std = np.mean(signal), np.std(signal)
    if std < 1e-9:
        return []
    z = (signal - mu) / std

    if start_buffer_s > 0:
        first = int(np.searchsorted(time_array, time_array[0] + start_buffer_s))
        time_array, z = time_array[first:], z[first:]
        if time_array.size < 3:
            return []

    results = []
    peak_idx, _ = find_peaks(z, height=z_threshold, distance=distance)
    for i in peak_idx:
        results.append({"time": float(time_array[i]), "z_score": float(z[i]), "kind": "peak"})

    if include_troughs:
        trough_idx, _ = find_peaks(-z, height=z_threshold, distance=distance)
        for i in trough_idx:
            results.append({"time": float(time_array[i]), "z_score": float(z[i]), "kind": "trough"})

    results.sort(key=lambda r: r["time"])
    return results


def _best_in_window(seg_z, include_troughs):
    """(index, z, kind) of the sample a response search takes in one z-scored window: the largest z, or with
    include_troughs whichever of the largest and the most negative is further from zero."""
    idx_max = int(np.argmax(seg_z))
    if include_troughs:
        idx_min = int(np.argmin(seg_z))
        if abs(seg_z[idx_min]) > seg_z[idx_max]:
            return idx_min, float(seg_z[idx_min]), "trough"
    return idx_max, float(seg_z[idx_max]), "peak"


def find_peak_near_events(time_array, signal, event_times, pre, post,
                           z_threshold=5.0, include_troughs=False, progress=None):
    """
    Check whether a statistically significant peak actually shows up near
    each given event time, rather than assuming the event marker itself
    marks where the neural signal responds. Works for a single event
    (event_times of length 1) or many occurrences of the same event type
    (checking consistency across all of them).

    Each event's window is baselined the same way as get_zscore_slice
    (pre-event portion), so "significant" means relative to that event's
    own local baseline, not the whole recording's.

    Parameters
    ----------
    time_array : array
    signal : array
    event_times : list of float
    pre, post : float
        Seconds before/after each event to search within.
    z_threshold : float
        Minimum |z-score| within the window for a peak to count as found.
        The default of 5 (the usual five-sigma convention) is deliberately
        strict: the search takes the LARGEST z over every sample in the
        window, so a low threshold reports a "response" by chance in most
        windows. On a real fibre-photometry recording searched the way PyAT
        does it (0.5 s-smoothed dF/F, 5 s baseline, 10 s response window),
        random pseudo-events (nothing to do with the animal) were reported
        as having a response 58% of the time at 2.5, 40% at 3, 21% at 4 and
        10% at 5; the exact rates depend on the window lengths. Even at 5,
        "found" is not proof of a response: shuffle_null_near_events
        measures the chance rate on the recording itself.
    include_troughs : bool
        Also consider negative-going deflections as candidate "peaks",
        keeping whichever (peak or trough) is more extreme.
    progress : None, True or callable(fraction, message)
        Progress reporting, one tick per event; see PyATCLI.progress.

    Returns
    -------
    list of dict, one per event_time (same order), each:
        {"event_time": float, "found": bool, "peak_time": float or None,
         "latency": float or None (peak_time - event_time),
         "z_score": float or None, "kind": "peak"|"trough"|None}
    "found" is False when the window was unusable (too close to the
    recording's edges) or nothing in it reached z_threshold.
    """
    time_array = as_float64(time_array)
    signal = as_float64(signal)
    results = []
    for t in Plan(progress, [("Searching for peaks", 1)]).track("Searching for peaks", event_times):
        seg_x, seg_z = get_zscore_slice(time_array, signal, t, pre=pre, post=post)
        if seg_x is None or len(seg_x) == 0:
            results.append({"event_time": t, "found": False, "peak_time": None,
                             "latency": None, "z_score": None, "kind": None})
            continue

        best_idx, best_z, kind = _best_in_window(seg_z, include_troughs)
        found = abs(best_z) >= z_threshold
        peak_time = float(seg_x[best_idx]) if found else None
        results.append({
            "event_time": t, "found": found,
            "peak_time": peak_time,
            "latency": (peak_time - t) if found else None,
            "z_score": best_z if found else None,
            "kind": kind if found else None,
        })
    return results


def holm_adjust(p_values):
    """Holm's step-down adjustment of a list of p-values for the number of them looked at together (the event types a scan
    tests). `None` entries stay `None` and are not counted. Gives the same numbers as statsmodels' multipletests(method="holm")."""
    tested = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(tested):
        running = max(running, min(1.0, (len(tested) - rank) * p))
        adjusted[i] = float(running)
    return adjusted


def _window_stat(time_array, signal, t, pre, post, include_troughs):
    """|z| of the sample find_peak_near_events takes in the window around t (what it compares with z_threshold), or NaN
    when the window holds no samples."""
    seg_x, seg_z = get_zscore_slice(time_array, signal, t, pre=pre, post=post)
    if seg_x is None or len(seg_x) == 0:
        return np.nan
    return abs(_best_in_window(seg_z, include_troughs)[1])


def shuffle_null_near_events(time_array, signal, event_times, pre, post, z_threshold=5.0, include_troughs=False,
                              n_shuffles=DEFAULT_SHUFFLES, seed=0, progress=None):
    """
    How often find_peak_near_events turns up a "response" by chance on this recording, and whether these events do
    better than that.

    The search takes the largest z over every sample of a window, so some windows reach the threshold with no response
    at all. This measures that on the recording itself. The whole train of events is moved as one by a random circular
    shift of the part of the recording where a complete window fits (so the spacing between events, and with it how
    much their windows overlap, is kept), `n_shuffles` times, and each shifted event is searched exactly as a real
    one is. From that:

    * `chance_hit_rate`: the share of shifted windows that reached z_threshold.
    * `p_count`: how often a shifted train had at least as many events with a response as the real one did
      ((1 + the shuffles that did) / (1 + n_shuffles)), for the events whose window is complete.
    * `p_event`: per event, the share of all shifted windows (counted the same way) whose largest z was at least that
      event's. Not corrected for how many events (or event types) are looked at.

    Events whose windows are cut by the recording's edge cannot be moved: they stay where they are in every shuffle,
    take no part in `p_count`, and have their `p_event` read against the complete shifted windows.

    If the recording really has responses, some shifted windows land on them and count as chance, so `chance_hit_rate`
    reads higher than it would without them and the p-values are conservative (a response is never made to look more
    significant than it is), the more so the denser the events.

    Parameters
    ----------
    time_array, signal, event_times, pre, post, z_threshold, include_troughs
        As for find_peak_near_events.
    n_shuffles : int
        Shifted trains; the smallest p-value possible is 1 / (1 + n_shuffles).
    seed : int
        Seed of the random shifts: the same inputs and seed give the same answer.
    progress : None, True or callable(fraction, message)
        One tick per shuffle; see PyATCLI.progress.

    Returns
    -------
    dict:
        {"available": bool, "reason": str or None (why not, when it is not available),
         "n_events", "n_tested" (events with a complete window), "n_shuffles", "seed",
         "found": events with a response among the tested ones, "expected_by_chance": the mean of that over the shuffles,
         "chance_hit_rate", "p_count", "p_event": list, one per event (None where the window holds no samples)}
    """
    time_array = as_float64(time_array)
    signal = as_float64(signal)
    n_shuffles = int(n_shuffles)
    if n_shuffles < 1:
        raise ValueError(f"n_shuffles must be at least 1 (got {n_shuffles!r})")
    times = np.asarray([float(t) for t in event_times], dtype=float)
    out = {"available": False, "reason": None, "n_events": int(times.size), "n_tested": 0, "n_shuffles": n_shuffles,
           "seed": int(seed), "found": None, "expected_by_chance": None, "chance_hit_rate": None, "p_count": None,
           "p_event": [None] * int(times.size)}

    lo = float(time_array[0]) + pre
    span = float(time_array[-1]) - post - lo               # how far the centre of a complete window can move
    if span <= 0:
        out["reason"] = "the recording is shorter than the window"
        return out
    movable = (times >= lo) & (times <= lo + span)
    if not movable.any():
        out["reason"] = "no event has a complete window inside the recording"
        return out

    observed = np.array([_window_stat(time_array, signal, t, pre, post, include_troughs) for t in times])
    base = times[movable]
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(0.0, span, n_shuffles)
    shifted_stats = np.empty((n_shuffles, base.size))
    for k in Plan(progress, [("Shuffling event times", 1)]).track("Shuffling event times", range(n_shuffles)):
        shifted = lo + np.mod(base - lo + offsets[k], span)
        shifted_stats[k] = [_window_stat(time_array, signal, t, pre, post, include_troughs) for t in shifted]

    hits = shifted_stats >= z_threshold                     # NaN (a window in a gap) compares False
    null_counts = hits.sum(axis=1)
    found = int(np.sum(observed[movable] >= z_threshold))
    pool = np.sort(shifted_stats[np.isfinite(shifted_stats)])
    if pool.size == 0:
        out["reason"] = "every shifted window was empty"
        return out
    p_event = [None if not np.isfinite(s) else float((1 + pool.size - np.searchsorted(pool, s, side="left")) / (1 + pool.size))
               for s in observed]
    out.update(available=True, n_tested=int(movable.sum()), found=found, expected_by_chance=float(null_counts.mean()),
               chance_hit_rate=float(np.mean(pool >= z_threshold)),
               p_count=float((1 + np.sum(null_counts >= found)) / (1 + n_shuffles)), p_event=p_event)
    return out
