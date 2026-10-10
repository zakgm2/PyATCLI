"""
numeric.py
----------
The library's one numeric-precision rule: signals, time axes and images are float64, always.

Why it is a rule and not a per-function judgement call: TDT hands its streams over as float32, and
numpy functions are not safe on single-precision data of a realistic length. np.polyfit on a float32
regressor silently drops the slope term once n * float32-eps exceeds the ratio of the regressor's
spread to its mean (~200,000 samples, ~3 minutes at 1 kHz, for a channel of 25 +/- 1), returning a line
at roughly half the true slope with only a RankWarning; sums, means and FFTs lose digits the same way,
just less visibly. Nothing here relies on a caller remembering to widen first.

Where it is enforced
    - Every file load: get_tdt_struct / get_plot_data (TDT), the Oxysoft parser, the generic table
      parser (GenericTable), the .pt2 image parser, and the Dataset container every loader returns.
      Whatever a load hands back is float64, whatever the file stores.
    - Every public function that takes a signal, a time axis or a trial matrix (smoothing, z-scoring,
      FFT, AUC, curve fits, peak finding, PETH, splicing, motion correction, ...), so arrays that did
      not come through a loader — straight from the tdt SDK, or from a script's own pipeline — are
      widened on the way in too.

as_float64 is a no-op (returns the same object, no copy) for an array that is already float64, so the
rule costs nothing on the normal path, and widening float32 to float64 is exact: the values do not change,
only what is computed from them stops losing precision.
"""

import numpy as np

FLOAT = np.float64


def as_float64(a):
    """`a` as a float64 ndarray: the same object if it already is one (no copy), otherwise a widened
    copy (float32 / float16 / integer arrays, lists, ...). NaN and inf are kept as they are."""
    return np.asarray(a, dtype=np.float64)
