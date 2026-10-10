"""
models.py
---------
Parametric model functions for curve fitting via scipy.optimize.curve_fit.

Each function follows the signature  f(x, *params) -> y  so they can be
passed directly to fit_model_to_segment (analysis/curve_fit.py). CURVE_FIT_MODELS,
at the bottom, is the registry the app's Curve Fit tool (and scripts) pick from.
"""

import numpy as np


def visibility_model(beta, a, v, beta_c, period):
    """
    Photon-entanglement visibility model.

    Parameters
    ----------
    beta   : array  Polarizer angle (degrees or radians)
    a      : float  Amplitude of the sinusoid
    v      : float  Visibility (0–1)
    beta_c : float  Angular offset / centre
    period : float  Period of the oscillation

    Returns
    -------
    array
    """
    return (a / 2) * (1 - v * np.sin((beta - beta_c) / period))


def double_exponential_model(x, a, b, c, d, k):
    """
    Physical model of fluorophore bleaching.
    y = a*exp(-b*x) + c*exp(-d*x) + k
    """
    return a * np.exp(-b * x) + c * np.exp(-d * x) + k


def linear_model(x, m, b):
    """y = mx + b"""
    return m * x + b


def single_exponential_model(x, a, b, c):
    """y = a * exp(-b*x) + c"""
    return a * np.exp(-b * x) + c


def exponential_rise_model(x, a, b, c):
    """y = a * (1 - exp(-b*x)) + c   — rising exponential (e.g. venous occlusion)"""
    return a * (1 - np.exp(-b * x)) + c


def gaussian_model(x, a, mu, sigma):
    """y = a * exp(-(x-mu)^2 / (2*sigma^2))"""
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


def sinusoidal_model(x, a, f, phi, c):
    """y = a * sin(2*pi*f*x + phi) + c"""
    return a * np.sin(2 * np.pi * f * x + phi) + c


# Curve-fit registry: "Display Name": (model_fn, p0_fn, [param names]) where
#   model_fn  - f(x, *params) -> y
#   p0_fn     - f(x_seg, y_seg) -> list of initial guesses
#   param list- human-readable name for every parameter, in order
# To add a model, add an entry here — the app's Curve Fit dialog and the scripting API both read it.
CURVE_FIT_MODELS = {
    "Linear  (y = mx + b)": (
        linear_model,
        lambda x, y: [(y[-1] - y[0]) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 0, y[0]],
        ["m (slope)", "b (intercept)"],
    ),
    "Exponential Decay  (a*e^(-bx) + c)": (
        single_exponential_model,
        lambda x, y: [y.max() - y.min(), 0.1, y.min()],
        ["a (amplitude)", "b (decay rate)", "c (offset)"],
    ),
    "Exponential Rise  (a*(1-e^(-bx)) + c)": (
        exponential_rise_model,
        lambda x, y: [y.max() - y.min(), 0.1, y.min()],
        ["a (amplitude)", "b (rise rate)", "c (offset)"],
    ),
    "Gaussian  (a*exp(-(x-u)^2/2s^2))": (
        gaussian_model,
        lambda x, y: [y.max(), x[y.argmax()], (x[-1] - x[0]) / 4],
        ["a (amplitude)", "u (centre)", "s (width)"],
    ),
    "Sinusoidal  (a*sin(2*pi*f*x + phi) + c)": (
        sinusoidal_model,
        lambda x, y: [(y.max() - y.min()) / 2, 1.0, 0.0, y.mean()],
        ["a (amplitude)", "f (frequency Hz)", "phi (phase)", "c (offset)"],
    ),
    "Double Exponential  (a*e^(-bx) + c*e^(-dx) + k)": (
        double_exponential_model,
        lambda x, y: [y.max() * 0.6, 0.05, y.max() * 0.4, 0.001, y.min()],
        ["a", "b (fast rate)", "c", "d (slow rate)", "k (offset)"],
    ),
}
