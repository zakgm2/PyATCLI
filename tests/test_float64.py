"""The float64 rule (numeric.py): every file load hands back float64 whatever the file stores, and every public
function that takes a signal widens float32 on the way in — returning float64, and exactly the numbers the
float64 call gives (widening float32 -> float64 is lossless, so the two must agree bit for bit)."""

import os
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

import tdt

import PyATCLI as pl
from PyATCLI.sources import processing_TDT as P
from PyATCLI.analysis import auc, curve_fit, event_peth, fft
from PyATCLI.analysis.group import trials as group_trials
from PyATCLI.analysis import peak_finder, shared, zscore_peth
from PyATCLI.analysis.group import GroupSpec
from PyATCLI.sources.dataset import Dataset
from PyATCLI.sources.tables import GenericTable, load_any_file
from PyATCLI.numeric import as_float64
from PyATCLI.splice import splice_cut_out, splice_keep_inside


TAB, NL = chr(9), chr(10)


def write_export(path, n_rows, rate=10.0):
    """A minimal Oxysoft export: two channels (Tx1, Tx2), O2Hb and HHb each, and an event column."""
    lines = [f"Datafile sample rate{TAB}{rate:g} Hz", "Legend:", f"1{TAB}sample number", f"2{TAB}Rx1 - Tx1 O2Hb (file)",
             f"3{TAB}Rx1 - Tx1 HHb (file)", f"4{TAB}Rx1 - Tx2 O2Hb (file)", f"5{TAB}Rx1 - Tx2 HHb (file)",
             f"6{TAB}Event (Event)", TAB.join("123456"), TAB.join("000000")]
    for i in range(1, n_rows + 1):
        lines.append(TAB.join([str(i), f"{0.1 * i:.6f}", f"{-0.1 * i:.6f}", f"{2.0 * i:.6f}", f"{-2.0 * i:.6f}", "0"]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(NL.join(lines) + NL)



def float_dtypes(obj, path="result"):
    """(path, dtype) for every floating ndarray, numpy float scalar and DataFrame column inside obj."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "f":
            yield path, obj.dtype
    elif isinstance(obj, np.floating):
        yield path, obj.dtype
    elif isinstance(obj, pd.DataFrame):
        for col, dt in obj.dtypes.items():
            if dt.kind == "f":
                yield f"{path}[{col}]", dt
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from float_dtypes(v, f"{path}[{k!r}]")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from float_dtypes(v, f"{path}[{i}]")


def assert_all_float64(test, obj):
    bad = [(p, str(dt)) for p, dt in float_dtypes(obj) if dt != np.float64]
    test.assertEqual(bad, [], f"not float64: {bad}")


def assert_same(test, a, b, path="result"):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    elif isinstance(a, np.ndarray):
        assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys(), path)
        for k in a:
            assert_same(test, a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b), path)
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(test, x, y, f"{path}[{i}]")
    elif isinstance(a, float) and np.isnan(a):
        test.assertTrue(np.isnan(b), path)
    else:
        test.assertEqual(a, b, path)


class TestAsFloat64(unittest.TestCase):
    def test_float32_is_widened_without_changing_a_value(self):
        a = np.array([0.1, 1 / 3, 25.123456], dtype=np.float32)
        out = as_float64(a)
        self.assertEqual(out.dtype, np.float64)
        assert_array_equal(out, a.astype(np.float64))

    def test_an_array_that_is_already_float64_is_returned_as_is(self):
        a = np.arange(5, dtype=np.float64)
        self.assertIs(as_float64(a), a)

    def test_integers_lists_and_nan_are_handled(self):
        self.assertEqual(as_float64(np.arange(3)).dtype, np.float64)
        assert_array_equal(as_float64([1, 2, 3]), [1.0, 2.0, 3.0])
        out = as_float64(np.array([1.0, np.nan, np.inf], dtype=np.float32))
        self.assertTrue(np.isnan(out[1]) and np.isinf(out[2]))

    def test_the_helper_is_exported_with_the_package_numeric_module(self):
        from PyATCLI import numeric
        self.assertIs(numeric.as_float64, as_float64)
        self.assertEqual(numeric.FLOAT, np.float64)


# --------------------------------------------------------------------------------------------------
# Every file load
# --------------------------------------------------------------------------------------------------

def fake_tdt_block(n=4000, fs=1000.0):
    """(read_block, data) standing in for the tdt SDK, whose streams are float32 like the real thing."""
    rng = np.random.default_rng(3)
    t = np.arange(n) / fs
    base = 60 + 30 * np.exp(-t / 2)
    f415 = (0.5 * base + rng.normal(0, 0.1, n)).astype(np.float32)
    f465 = (base * (1 + 0.1 * np.exp(-0.5 * ((t - 2) / 0.2) ** 2)) + rng.normal(0, 0.1, n)).astype(np.float32)

    heads = tdt.StructType()
    heads.stores = {"L1P_": types.SimpleNamespace(type_str="epocs", name="L1P/"),
                    "x465A": types.SimpleNamespace(type_str="streams", name="x465A")}
    data = tdt.StructType()
    data.streams = {"x465A": types.SimpleNamespace(data=f465, fs=fs), "x415A": types.SimpleNamespace(data=f415, fs=fs)}
    data.scalars = tdt.StructType()
    data.epocs = tdt.StructType()

    def read_block(path, headers=None, evtype=None, store=None, verbose=None):
        if isinstance(headers, int) and headers == 1:
            return heads
        if store is not None:
            block = tdt.StructType()
            block.epocs = tdt.StructType()
            block.epocs["L1P_"] = types.SimpleNamespace(name="L1P", onset=np.array([1.0, 3.0]), offset=np.array([1.5, 3.5]))
            return block
        return data

    return read_block, data


class TestEveryFileLoadIsFloat64(unittest.TestCase):
    def test_tdt_streams_are_widened_when_the_block_is_read(self):
        read_block, data = fake_tdt_block()
        raw32 = data.streams["x465A"].data.copy()
        self.assertEqual(raw32.dtype, np.float32)                           # the SDK's own dtype
        with mock.patch.object(tdt, "read_block", read_block):
            struct = P.get_tdt_struct("ignored")
        for name, stream in struct.streams.items():
            self.assertEqual(stream.data.dtype, np.float64, name)
        assert_array_equal(struct.streams["x465A"].data, raw32.astype(np.float64))      # values untouched

    def test_get_plot_data_returns_float64_even_for_a_struct_that_did_not_come_from_the_loader(self):
        _, data = fake_tdt_block()
        x, y, fs = P.get_plot_data(data, "x465A")
        self.assertEqual((x.dtype, y.dtype), (np.float64, np.float64))
        self.assertIsInstance(fs, float)
        x, y, _ = P.get_plot_data(data, "x465A", max_points=500)
        self.assertEqual((x.dtype, y.dtype), (np.float64, np.float64))

    def test_a_processed_tdt_recording_is_float64_throughout(self):
        read_block, _ = fake_tdt_block()
        with mock.patch.object(tdt, "read_block", read_block):
            result = P.process_tdt_folder("ignored")
        assert_all_float64(self, {k: v for k, v in result.items() if k != "markers"})
        self.assertTrue(result["channels"])
        for ch in result["channels"]:                                       # the raw per-wavelength traces too
            self.assertEqual(ch["y"].dtype, np.float64, ch["key"])

    def test_the_tdt_dataset_is_float64(self):
        read_block, _ = fake_tdt_block()
        with mock.patch.object(tdt, "read_block", read_block), \
                mock.patch("PyATCLI.sources.load.processing_TDT.validate_tdt_folder", return_value=(True, "ok")):
            ds = pl.load_dataset("ignored", fmt=pl.DataFormat.TDT)
        self.assertEqual(ds.signals.dtype, np.float64)
        for key in ("f0", "raw", "x"):
            self.assertEqual(ds.metadata[key].dtype, np.float64, key)

    def test_oxysoft_file_folder_and_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            write_export(os.path.join(d, "a.txt"), 30)
            r = pl.process_oxysoft_file(os.path.join(d, "a.txt"))
            self.assertEqual((r["o2hb"].dtype, r["hhb"].dtype), (np.float64, np.float64))
            r = pl.process_oxysoft_folder(d)
            self.assertEqual((r["o2hb"].dtype, r["hhb"].dtype), (np.float64, np.float64))
            self.assertEqual(pl.load_dataset_file(os.path.join(d, "a.txt")).signals.dtype, np.float64)
            self.assertEqual(pl.load_dataset(d).signals.dtype, np.float64)

    def test_generic_tables_are_float64(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.csv")
            with open(path, "w") as fh:
                fh.write("time,a,b\n" + "\n".join(f"{i * 0.1:.1f},{i},{i * 2}" for i in range(20)) + "\n")
            tables = load_any_file(path)
        self.assertTrue(tables)
        for t in tables:
            self.assertEqual(t.data.dtype, np.float64)

    def test_a_table_built_by_hand_is_widened_too(self):
        t = GenericTable(name="t", headers=["x", "y"], data=np.ones((3, 2), dtype=np.float32))
        self.assertEqual(t.data.dtype, np.float64)

    def test_a_dataset_built_by_hand_is_widened_too(self):
        ds = Dataset(source_format="X", folder_path="p", folder_name="n", signals=np.ones((1, 4), dtype=np.float32))
        self.assertEqual(ds.signals.dtype, np.float64)
        self.assertIsNone(Dataset(source_format="X", folder_path="p", folder_name="n").signals)

    def test_a_pt2_image_is_float64_with_the_values_the_file_holds(self):
        img = (np.arange(256, dtype=np.float32).reshape(16, 16) + 1.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scan.pt2")
            with open(path, "wb") as fh:
                fh.write(b"header-bytes" + b"LAER" + img.astype("<f4").tobytes())
            out = pl.load_pt2(path)
        self.assertEqual(out.dtype, np.float64)
        self.assertEqual(out.shape, (16, 16))
        assert_array_equal(out, img.astype(np.float64))


# --------------------------------------------------------------------------------------------------
# Every public function that takes a signal
# --------------------------------------------------------------------------------------------------

class TestEveryEntryPointWidensItsInput(unittest.TestCase):
    N, FS = 3000, 100.0

    def setUp(self):
        rng = np.random.default_rng(0)
        self.x32 = (np.arange(self.N) / self.FS).astype(np.float32)
        self.y32 = (5 + np.sin(self.x32 * 2) + 0.2 * rng.standard_normal(self.N)).astype(np.float32)
        self.x64, self.y64 = self.x32.astype(np.float64), self.y32.astype(np.float64)
        self.matrix32 = rng.standard_normal((6, 120)).astype(np.float32)
        self.axis32 = np.linspace(-5, 5, 120).astype(np.float32)
        self.f415_32 = (0.5 * self.y32 + 0.1 * rng.standard_normal(self.N)).astype(np.float32)
        self.events = [10.0, 15.0, 20.0]
        self.spec = GroupSpec(group_name="G", markers=["A"], pre=5, post=5, baseline=(-5, -2), response=(0, 5),
                              metrics=["auc", "peak", "mean", "latency", "decay"])

    def cases(self):
        """name -> (callable on a dtype-converter, i.e. f(conv) -> result)."""
        yield "smooth_signal", lambda c: shared.smooth_signal(c(self.y32), self.FS)
        yield "mean_channels", lambda c: shared.mean_channels(c(self.matrix32))
        yield "estimate_sample_rate", lambda c: shared.estimate_sample_rate(c(self.x32))
        yield "get_zscore_slice", lambda c: zscore_peth.get_zscore_slice(c(self.x32), c(self.y32), 15.0, pre=5, post=5)
        yield "bin_for_heatmap", lambda c: zscore_peth.bin_for_heatmap(c(self.y32[:1000]))
        yield "compute_fft_slice", lambda c: fft.compute_fft_slice(c(self.x32), c(self.y32), 15.0, self.FS, pre=10, post=10)
        freqs32 = (self.axis32[60:] + 5).astype(np.float32)
        power32 = (np.abs(self.matrix32[0, 60:]) + 0.1).astype(np.float32)      # built first: the same values on both paths
        yield "find_fft_peaks", lambda c: fft.find_fft_peaks(c(freqs32), c(power32))
        yield "compute_slope_segment", lambda c: curve_fit.compute_slope_segment(c(self.x32), c(self.y32), 100, 400)
        yield "fit_model_to_segment", lambda c: curve_fit.fit_model_to_segment(
            c(self.x32[:200]), c(self.y32[:200]), *pl.CURVE_FIT_MODELS["Linear  (y = mx + b)"][:2])
        yield "find_significant_peaks", lambda c: peak_finder.find_significant_peaks(c(self.x32), c(self.y32), z_threshold=2.0)
        yield "find_peak_near_events", lambda c: peak_finder.find_peak_near_events(
            c(self.x32), c(self.y32), self.events, 5, 5, z_threshold=1.0)
        yield "shuffle_null_near_events", lambda c: peak_finder.shuffle_null_near_events(
            c(self.x32), c(self.y32), self.events, 5, 5, z_threshold=1.0, n_shuffles=20)
        yield "compute_auc_from_trace", lambda c: auc.compute_auc_from_trace(c(self.axis32), c(self.matrix32[0]))
        yield "compute_auc_window", lambda c: auc.compute_auc_window(c(self.x32), c(self.y32), 15.0, 5, 5)
        yield "compute_auc_matrix", lambda c: event_peth.compute_auc_matrix(c(self.axis32), c(self.matrix32))
        yield "compute_group_stats", lambda c: event_peth.compute_group_stats(c(self.matrix32))
        yield "compute_peri_event_from_trace", lambda c: event_peth.compute_peri_event_from_trace(
            c(self.axis32), c(self.matrix32[0]), 5.0, 1.0)
        yield "compute_peri_event_matrix", lambda c: event_peth.compute_peri_event_matrix(
            c(self.axis32), c(self.matrix32), 5.0, 1.0)
        yield "compute_event_zscore_peth", lambda c: event_peth.compute_event_zscore_peth(
            c(self.x32), c(self.y32), self.events, 5, 5)
        yield "denoise_signal", lambda c: P.denoise_signal(c(self.y32), self.FS)
        yield "correct_bleaching", lambda c: P.correct_bleaching(c(self.y32), self.FS)
        yield "compute_dff", lambda c: P.compute_dff(c(self.y32), c(self.f415_32), self.FS, "ols")
        yield "compute_dff (single channel)", lambda c: P.compute_dff(c(self.y32), None, self.FS)
        yield "splice_keep_inside", lambda c: splice_keep_inside(
            c(self.x32), c(self.y32), c(self.y32), [], [], 5.0, 20.0, extra_channels={"a": c(self.f415_32)})
        yield "splice_cut_out", lambda c: splice_cut_out(
            c(self.x32), c(self.y32), c(self.y32), [], [], 5.0, 20.0, extra_channels={"a": c(self.f415_32)})
        yield "extract_group_trials", lambda c: group_trials.extract_group_trials(
            c(self.x32), c(self.y32), {"A": self.events}, self.spec, subject="S")

    def test_float32_in_gives_float64_out_and_the_same_numbers_as_float64_in(self):
        import warnings
        for name, run in self.cases():
            with self.subTest(name), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r32 = run(lambda a: a)                                       # the float32 arrays as they are
                r64 = run(lambda a: a.astype(np.float64))                    # the same values, widened by the caller
                assert_all_float64(self, r32)
                assert_same(self, r32, r64, name)

    def test_the_slope_of_a_long_float32_regressor_is_not_lost(self):
        """The failure this whole rule exists for: np.polyfit on a float32 regressor returns half the slope once
        the recording is long (see numeric.py). The public slope function must not inherit it."""
        n = 300_000
        rng = np.random.default_rng(1)
        x = (25 + rng.standard_normal(n)).astype(np.float32)
        y = (1.9 * x.astype(np.float64) + 0.01 * rng.standard_normal(n)).astype(np.float32)
        out = curve_fit.compute_slope_segment(x, y, 0, n - 1)
        self.assertAlmostEqual(out["slope"], 1.9, places=2)


if __name__ == "__main__":
    unittest.main()
