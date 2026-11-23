"""The pipeline exactly as it was before the options existed (PyATCLI 2026.11.5): a frozen copy of the functions the pipeline options changed, kept so that
a test can say what "defaults reproduce today's behaviour exactly" means — the default options must give the same bits as these. Not imported by the library.
Do not edit these to match a change in the library: if a default's output changes, that is a breaking change to every saved recording."""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt

from PyATCLI.models import double_exponential_model as double_exponential
from PyATCLI.numeric import as_float64
from PyATCLI.progress import Plan
from PyATCLI.sources.processing_TDT import _robust_linear_fit
from PyATCLI.analysis.shared import window_around

DENOISE_CUTOFF_HZ = 5
DENOISE_ORDER = 2
SMOOTH_WINDOW_SEC = 0.5
BASELINE_ROBUST_SD = 5.0

def correct_bleaching(y, fs):
    """
    Estimate and correct photobleaching using masked curve fitting.

    Returns
    -------
    corrected : array
        Bleaching-corrected signal
    trend : array
        Estimated baseline trend
    """
    y = as_float64(y)
    x = np.arange(len(y)) / fs

    threshold = np.median(y)
    mask = y > threshold
    x_fit, y_fit = x[mask], y[mask]

    if len(y_fit) < 100:
        return y, np.zeros_like(y)

    # Fit in units of the signal's own size and scale the trend back
    # afterwards: curve_fit's optimiser isn't scale-free (it converged to a
    # baseline up to ~4.5% different for the same recording expressed x1000).
    scale = np.max(np.abs(y_fit))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    y_fit = y_fit / scale

    k_guess   = np.percentile(y_fit, 10)
    total_amp = np.max(y_fit) - k_guess
    p0        = [total_amp * 0.6, 0.05, total_amp * 0.4, 0.0001, k_guess]
    lower     = [0, 0, 0, 0, k_guess * 0.8]
    upper     = [np.inf, 1, np.inf, 0.1, np.max(y_fit)]

    try:
        popt, _ = curve_fit(double_exponential, x_fit, y_fit, p0=p0,
                            bounds=(lower, upper), maxfev=10000)
        trend = scale * double_exponential(x, *popt)
    except Exception:
        # curve_fit failed — fall back to log-linear fit as a rough trend estimate
        import warnings
        warnings.warn(
            "correct_bleaching: double-exponential fit failed; falling back to log-linear trend.",
            RuntimeWarning, stacklevel=2,
        )
        coeffs = np.polyfit(x_fit, np.log(np.maximum(y_fit, 1e-6)), 1)
        trend  = scale * np.exp(np.polyval(coeffs, x))

    corrected = y - trend + trend[0]
    return corrected, trend


def denoise_signal(signal, fs, cutoff=DENOISE_CUTOFF_HZ, order=DENOISE_ORDER):
    """
    Low-pass Butterworth filter for ΔF/F signals.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz
    order : int
        Filter order

    Returns
    -------
    array
        Filtered signal
    """
    signal        = as_float64(signal)
    nyquist       = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a          = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def compute_dff(y_465, y_415, fs, regression_method="ols", progress=None):
    """
    The actual motion-correction + bleaching-correction + ΔF/F + denoise
    pipeline (steps 3-6 of process_tdt_folder's docstring), factored out
    so it can be re-run on a signal that isn't a fresh TDT load — namely
    Splice's "Cut out this range": removing an artifact and re-running
    this on the stitched remainder can give a genuinely better motion
    correction (the artifact is gone from what the regression fits
    against), not just a shorter view of the original fit. See
    physicsanalysis_qt's analysis/splice.py for that caller.

    Parameters
    ----------
    y_465 : array
        Main driver (signal) fluorescence.
    y_415 : array or None
        Isosbestic (control) fluorescence, same length as y_465. None
        skips motion correction entirely (single-channel recording).
    fs : float
        Sampling rate, for correct_bleaching/denoise_signal.
    regression_method : {"ransac", "huber", "ols"}
        See _robust_linear_fit.
    progress : None, True or callable(fraction, message)
        Progress reporting; see PyATCLI.progress. The photobleaching
        fit dominates the run time (a single scipy call with no steps to
        tick), so it shows as one long stage.

    Returns
    -------
    dict with raw (motion-corrected fluorescence: with a 415 stream, the 465
    minus its isosbestic-predicted motion component, i.e. ΔF in the
    recording's own units), corr/dff (denoised ΔF/F), f0 (baseline
    fluorescence, the photobleaching trend of the 465 stream — the
    denominator of ΔF/F), motion_correction_inlier_fraction (None if y_415
    is None) — the same subset of process_tdt_folder's own return dict that
    actually depends on this computation.
    """
    # ΔF/F is (F - F0) / F0 with F0 the baseline FLUORESCENCE. With a 415
    # stream, the motion-corrected residual (465 minus its isosbestic-
    # predicted motion component) already IS F - F0, so it is divided by the
    # 465's own bleaching baseline. (Running correct_bleaching on the
    # residual itself and normalising by that, as this function used to,
    # divides by a near-zero number and gives a value that isn't a ΔF/F —
    # its scale and offset changed by orders of magnitude with the choice of
    # regression method.) Without a 415 stream there is no motion term to
    # remove, so the trace's own trend is F0.
    #
    # Everything is done in float64: TDT hands over float32, and the fit (see
    # _robust_linear_fit) is not safe in single precision.
    y_465 = as_float64(y_465)
    # Weights are roughly the measured share of the run time: on a 17-minute
    # 1 kHz recording the bleaching fit is ~70-90% of it, the regression 2-20%
    # depending on the method, the filtering almost nothing.
    stages = [("Fitting motion correction", 10)] if y_415 is not None else []
    plan = Plan(progress, stages + [("Fitting photobleaching baseline", 83),
                                    ("Normalising and filtering", 7)])
    if y_415 is not None:
        y_415 = as_float64(y_415)
        plan.begin("Fitting motion correction")
        a, b, inlier_fraction = _robust_linear_fit(y_415, y_465, method=regression_method)
        y_fit = a * y_415 + b
        y_final = y_465 - y_fit
        plan.begin("Fitting photobleaching baseline")
        _, trend = correct_bleaching(y_465, fs)
        f0  = np.maximum(trend, 1e-6)
        plan.begin("Normalising and filtering")
        dff = y_final / f0
    else:
        y_final = y_465
        inlier_fraction = None
        plan.begin("Fitting photobleaching baseline")
        _, trend = correct_bleaching(y_final, fs)
        f0  = np.maximum(trend, 1e-6)
        plan.begin("Normalising and filtering")
        dff = (y_final - f0) / f0

    dff = denoise_signal(dff, fs, cutoff=DENOISE_CUTOFF_HZ, order=DENOISE_ORDER)
    plan.done()

    return {
        "raw": y_final,
        "corr": dff,
        "dff": dff,
        "f0": f0,
        "motion_correction_inlier_fraction": inlier_fraction,
    }


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


def get_zscore_slice(time_array, signal, center_t, window=None, pre=None, post=None):
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
    baseline_mask   = seg_x < center_t
    baseline_period = seg_y[baseline_mask] if baseline_mask.any() else seg_y
    mu, std = _baseline_mean_std(baseline_period)

    # "Flat" means the baseline's spread is down at floating-point rounding
    # level next to its own size — a relative test, so it doesn't depend on
    # the units the signal is in.
    scale = np.max(np.abs(baseline_period)) if baseline_period.size else 0.0
    if std == 0 or std <= 1e-12 * scale:
        return seg_x, np.zeros_like(seg_y)

    return seg_x, (seg_y - mu) / std


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
