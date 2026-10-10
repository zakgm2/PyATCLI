"""
sources/load.py
---------------
Reads a recording from disk into the universal Dataset: detects the format inside a selected folder or file, hands it to the reader for
that format, and packages what the reader returned.

  - TDT  (Tucker-Davis Technologies) – processing_TDT.py reads and corrects the tank; load_tdt() wraps the result
  - Oxysoft / Artinis (Oxymon, OctaMon, PortaMon …) – processing_oxysoft.py reads the export; load_oxysoft() wraps the result

Format detection and the Dataset struct live in dataset.py. The .pt2 EFNMR/MRI image format is a standalone parser in pt2.py (it returns a
raw image array, not a Dataset).

Usage
-----
    from PyATCLI import detect_format, load_dataset, DataFormat

    fmt = detect_format(folder_path)
    dataset = load_dataset(folder_path, fmt)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from . import processing_TDT, processing_oxysoft
from .dataset import Dataset, DataFormat, detect_format, detect_format_file


def load_dataset_file(file_path: str, progress=None) -> Dataset:
    """Load a single file and return a Dataset.

    progress : None, True or callable(fraction, message)
        Progress reporting; see PyATCLI.progress.
    """
    fmt = detect_format_file(file_path)
    if fmt == DataFormat.OXYSOFT:
        return load_oxysoft_file(file_path, progress=progress)
    raise ValueError(f"Unrecognised file format: {file_path}")


def load_dataset(folder_path: str, fmt: Optional[DataFormat] = None,
                  regression_method: str = "ols", progress=None) -> Dataset:
    """
    Load a recording folder and return a Dataset.
    If *fmt* is None, detect_format() is called automatically.

    regression_method : {"ransac", "huber", "ols"}
        TDT-only — forwarded to load_tdt()/process_tdt_folder() for the
        isosbestic-vs-signal motion-correction regression. Ignored for
        Oxysoft, which has no such correction step.
    progress : None, True or callable(fraction, message)
        Progress reporting; see PyATCLI.progress.
    """
    folder_name = os.path.basename(folder_path.rstrip('/\\'))

    if fmt is None:
        fmt = detect_format(folder_path)

    if fmt is DataFormat.TDT:
        return load_tdt(folder_path, folder_name, regression_method=regression_method, progress=progress)
    elif fmt is DataFormat.OXYSOFT:
        return load_oxysoft(folder_path, folder_name, progress=progress)
    else:
        raise ValueError(
            f"Could not identify a supported data format in: {folder_path}\n"
            "Expected TDT proprietary files (.Tbk/.tev/…) or an Oxysoft .txt export."
        )


# ---------------------------------------------------------------------------
# TDT
# ---------------------------------------------------------------------------

def load_tdt(folder_path: str, folder_name: str, regression_method: str = "ols", progress=None) -> Dataset:
    """Load a TDT tank using the existing validate + process pipeline.

    `progress` (None, True or callable(fraction, message); see
    PyATCLI.progress) is handed to process_tdt_folder.
    """
    valid, msg = processing_TDT.validate_tdt_folder(folder_path)
    if not valid:
        raise ValueError(f"TDT validation failed: {msg}")

    result     = processing_TDT.process_tdt_folder(folder_path, regression_method=regression_method,
                                                    progress=progress)
    signals    = result["corr"]
    signals_2d = signals[np.newaxis, :]
    fs         = float(result["fs"])
    num_samp   = signals.shape[0]

    events = []
    for marker in result["markers"]:
        events.append({
            "label":  marker["label"],
            "sample": int(round(marker["time"] * fs)),
        })

    return Dataset(
        source_format = "TDT",
        folder_path   = folder_path,
        folder_name   = folder_name,
        sample_rate   = fs,
        num_samples   = num_samp,
        duration_s    = num_samp / fs if fs > 0 else 0.0,
        signals       = signals_2d,
        channel_names = [result["store"]],
        num_channels  = 1,
        events        = events,
        metadata      = {
            "f0":  result["f0"],
            "raw": result["raw"],
            "x":   result["x"],
        },
    )


# ---------------------------------------------------------------------------
# Oxysoft
# ---------------------------------------------------------------------------

def load_oxysoft(folder_path: str, folder_name: str, progress=None) -> Dataset:
    """
    Parse an Oxysoft TXT export folder into a Dataset.
    Multiple .txt files are concatenated in alphabetical order.

    `progress` (None, True or callable(fraction, message); see
    PyATCLI.progress) is weighted by each file's size in bytes.
    """
    result = processing_oxysoft.process_oxysoft_folder(folder_path, progress=progress)
    return _oxysoft_dataset(result, folder_path=folder_path, folder_name=folder_name)


def load_oxysoft_file(file_path: str, progress=None) -> Dataset:
    """Parse a single Oxysoft .txt export into a Dataset.

    `progress` (None, True or callable(fraction, message); see
    PyATCLI.progress) follows the file as it is read.
    """
    result = processing_oxysoft.process_oxysoft_file(file_path, progress=progress)
    return _oxysoft_dataset(result, folder_path=os.path.dirname(file_path),
                            folder_name=os.path.splitext(os.path.basename(file_path))[0])


def _oxysoft_dataset(result: dict, folder_path: str, folder_name: str) -> Dataset:
    o2hb, hhb = result["o2hb"], result["hhb"]
    sample_rate = result["sample_rate"]
    n_ch, n_samp = o2hb.shape

    # Stacked as (2 * n_channels, n_samples): first all O2Hb, then all HHb
    signals = np.concatenate([o2hb, hhb], axis=0)

    # Channel names: e.g. ['Tx1 O2Hb', 'Tx2 O2Hb', 'Tx3 O2Hb', 'Tx1 HHb', ...]
    labels = result["channel_labels"]
    return Dataset(
        source_format = "Oxysoft",
        folder_path   = folder_path,
        folder_name   = folder_name,
        sample_rate   = sample_rate,
        num_samples   = n_samp,
        duration_s    = n_samp / sample_rate if sample_rate > 0 else 0.0,
        signals       = signals,
        channel_names = [f"{l} O2Hb" for l in labels] + [f"{l} HHb" for l in labels],
        num_channels  = 2 * n_ch,
        events        = result["events"],
        metadata      = result["metadata"],
    )
