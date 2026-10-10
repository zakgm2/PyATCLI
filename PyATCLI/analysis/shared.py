"""
analysis/shared.py
-------------------
Small helpers used by more than one analysis tool (or by a loader
outside analysis/ entirely, e.g. estimate_sample_rate for a generic
tabular file with no reported fs) — kept separate from any one tool's
own file so pulling in, say, curve_fit.py doesn't also drag in AUC- or
FFT-specific code that happens to need the same helper.
"""

import numpy as np

from ..numeric import as_float64

# Moving-average window every event-locked analysis smooths the dF/F with first; named so provenance can report it.
SMOOTH_WINDOW_SEC = 0.5


def estimate_sample_rate(time_array):
    """
    Robust sample-rate estimate from a timestamp array, for a source that
    doesn't report its own fs (e.g. a generic tabular file) or as a
    fallback wherever one's needed internally. Uses the median inter-
    sample interval rather than e.g. `1 / (time_array[1] - time_array[0])`
    so one irregular gap (a dropped sample, a rounding artifact at the
    very start) doesn't skew the whole estimate.

    Parameters
    ----------
    time_array : array
        Must have at least 2 samples.

    Returns
    -------
    float, sample rate in Hz.
    """
    return float(1.0 / np.median(np.diff(as_float64(time_array))))


def window_around(time_array, signal, center_t, window=None, pre=None, post=None):
    """
    (time segment, signal segment) of the samples from `pre` seconds before `center_t` to `post` seconds after it, both float64.

    The window is either asymmetric (`pre` and/or `post`) or a symmetric `window`, the total seconds split evenly before and after: a
    side that is not given takes half of `window`, which is 30 s when that is not given either.
    """
    if pre is None or post is None:
        half_win = (window if window is not None else 30) / 2
        pre  = pre  if pre  is not None else half_win
        post = post if post is not None else half_win

    time_array = as_float64(time_array)
    signal = as_float64(signal)
    start_idx = np.searchsorted(time_array, center_t - pre)
    end_idx   = np.searchsorted(time_array, center_t + post)
    return time_array[start_idx:end_idx], signal[start_idx:end_idx]


def mean_channels(arr):
    """
    Collapse a multi-channel array to its across-channel mean trace —
    e.g. Oxysoft's per-detector o2hb/hhb/thb arrays, shape
    (n_channels, n_samples), averaged down to one representative trace.
    A no-op for an already-1D array, so callers can use this unconditionally
    regardless of whether a given recording actually has multiple channels.

    Parameters
    ----------
    arr : array, 1D or 2D (n_channels, n_samples)

    Returns
    -------
    array, 1D — arr unchanged if already 1D, else arr.mean(axis=0).
    """
    arr = as_float64(arr)
    return arr.mean(axis=0) if arr.ndim > 1 else arr


def smooth_signal(data, fs, window_sec=SMOOTH_WINDOW_SEC):
    """
    Moving average smoothing filter.

    Parameters
    ----------
    data : array
    fs : float
        Sampling frequency in Hz
    window_sec : float
        Smoothing window in seconds

    Returns
    -------
    array
        Smoothed signal
    """
    data = as_float64(data)
    window_size = int(fs * window_sec)
    if window_size % 2 == 0:
        window_size += 1
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')
