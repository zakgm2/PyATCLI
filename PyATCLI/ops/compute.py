"""
ops/compute.py — the building blocks a window draws from, where `run` (analysis.py) gives the tables an export writes.

    compute.event_peth   every occurrence of one event aligned into a trial x time matrix of z-scores (the Event PETH window's data)
    compute.peaks        Peak Finder: significant peaks of the whole recording, or the one nearest each of some events

Both work on the recording's normalized signal (the corrected dF/F for TDT), smoothed, exactly as the app's windows do, and name the
recording as `run` does ({"id": ...} for one the engine holds, or a path). Arrays come back as .npy files to map (engine/wire.py): a
trial matrix is a result to look at and change the checked rows of, not text.
"""

import tempfile

from ..analyses import ParameterError
from ..analysis.event_peth import compute_event_zscore_peth
from ..analysis.peak_finder import find_peak_near_events, find_significant_peaks
from ..analysis.shared import smooth_signal
from ..progress import Plan
from ..recording import normalized_signal
from . import OpError, operation
from .common import open_recording


def arrays_wire(ctx, obj, prefix):
    """`obj` (dicts, lists, numbers and arrays) with its arrays written as .npy files in a new folder of the engine's data folder:
    {"directory": ..., "data": the structure with each array replaced by a reference} (decode with engine.wire.decode)."""
    from ..engine.wire import ArrayFiles, encode

    if ctx.data_dir is None:
        raise OpError("arrays need somewhere to go (the engine's data folder); this operation is for the engine", "invalid_input")
    directory = tempfile.mkdtemp(prefix=f"{prefix}-", dir=ctx.data_dir)
    return {"directory": directory, "data": encode(obj, ArrayFiles(directory), prefix)}


def _times(values, what):
    try:
        return [float(t) for t in values]
    except (TypeError, ValueError):
        raise ParameterError(f"{what} is a list of times in seconds") from None


def _window(pre, post):
    pre, post = float(pre), float(post)
    if pre <= 0 or post <= 0:
        raise ParameterError("pre and post are positive numbers of seconds")
    return pre, post


@operation("compute.event_peth", "z-score and align every occurrence of an event into a trial x time matrix (the Event PETH window's data)", result=["signal", "label", "result"])
def event_peth(ctx, *, recording, event_times, pre, post, num_bins=300, load=None):
    times = _times(event_times, "event_times")
    pre, post = _window(pre, post)
    loaded, _ = open_recording(ctx, recording, load)
    cache = loaded["cache"]
    key, label, y = normalized_signal(cache)
    plan = Plan(ctx.progress(), [("Smoothing the signal", 10), ("Aligning trials", 90)])
    plan.begin("Smoothing the signal")
    clean = smooth_signal(y, cache["fs"])
    result = compute_event_zscore_peth(cache["x"], clean, times, pre, post, num_bins=int(num_bins), progress=plan.sub("Aligning trials"))
    return {"signal": key, "label": label, "result": arrays_wire(ctx, result, "event_peth")}


@operation("compute.peaks", "Peak Finder: significant peaks of the whole recording (scope whole), or the peak nearest each event (scope events)", result=["signal", "label"])
def peaks(ctx, *, recording, scope, z_threshold, include_troughs=False, min_distance_s=1.0, events=None, pre=None, post=None, load=None):
    if scope not in ("whole", "events"):
        raise ParameterError(f"scope must be 'whole' or 'events', not {scope!r}")
    z_threshold = float(z_threshold)
    if scope == "events":
        if not isinstance(events, dict) or not events:
            raise ParameterError("events is {event name: [times]} for scope 'events'")
        groups = {str(name): _times(times, f"events[{name!r}]") for name, times in events.items()}
        if pre is None or post is None:
            raise ParameterError("pre and post (seconds around each event) are needed for scope 'events'")
        pre, post = _window(pre, post)
    loaded, _ = open_recording(ctx, recording, load)
    cache = loaded["cache"]
    key, label, y = normalized_signal(cache)
    plan = Plan(ctx.progress(), [("Smoothing the signal", 10), ("Searching for peaks", 90)])
    plan.begin("Smoothing the signal")
    clean = smooth_signal(y, cache["fs"])
    plan.begin("Searching for peaks")
    if scope == "whole":
        found = find_significant_peaks(cache["x"], clean, z_threshold=z_threshold, min_distance_sec=float(min_distance_s),
                                       include_troughs=bool(include_troughs))
        plan.done()
        return {"signal": key, "label": label, "peaks": found}
    if len(groups) == 1:                                                  # one event type: the progress follows its events
        (name, times), = groups.items()
        results = {name: find_peak_near_events(cache["x"], clean, times, pre, post, z_threshold=z_threshold,
                                               include_troughs=bool(include_troughs), progress=plan.sub("Searching for peaks"))}
    else:                                                                 # several: one tick per event type
        results = {name: find_peak_near_events(cache["x"], clean, times, pre, post, z_threshold=z_threshold, include_troughs=bool(include_troughs))
                   for name, times in plan.track("Searching for peaks", groups.items())}
    return {"signal": key, "label": label, "groups": results}
