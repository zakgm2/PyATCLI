"""
sources/pt2.py
----------------
Terranova / Prospa .pt2 EFNMR/MRI image parser.
"""

import numpy as np

from ..numeric import as_float64


def load_pt2(path: str) -> np.ndarray:
    """
    Parse a Terranova Prospa .pt2 2D NMR/MRI image file.

    The reconstructed magnitude image is stored after the 4-byte marker
    b'LAER' ('REAL' reversed) as little-endian float32 values (widened to
    float64 on the way out, like every load — see numeric.py).

    Returns
    -------
    np.ndarray
        2D float64 array shaped (n, n).
    """
    with open(path, 'rb') as f:
        raw = f.read()

    pos = raw.find(b'LAER')
    if pos == -1:
        raise ValueError("Not a recognised .pt2 file — LAER image marker not found.")
    pos += 4

    arr = as_float64(np.frombuffer(raw[pos:], dtype='<f4'))
    n_total = len(arr)

    for n in [16, 32, 64, 128, 256]:
        if n_total == n * n and np.isfinite(arr).all() and arr.max() > 0:
            return arr.reshape(n, n)

    sq = int(np.sqrt(n_total))
    if sq * sq == n_total and np.isfinite(arr).all() and arr.max() > 0:
        return arr.reshape(sq, sq)

    raise ValueError(
        f"Could not determine image dimensions ({n_total} floats after LAER)."
    )
