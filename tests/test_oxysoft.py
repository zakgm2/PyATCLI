"""Oxysoft / Artinis export reading (processing_oxysoft): the Legend block, the data rows, the event column,
the fit factor, a folder of exports concatenated in order, and the loader that wraps it all into a Dataset."""

import os
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from PyATCLI.sources import processing_oxysoft as po
from PyATCLI.sources import load as sources_load


def write_export(path, n_rows, rate=10.0, offset=0.0, events=(), with_fit_factor=True):
    """A small, fully known Oxysoft export: Tx1 O2Hb = offset + 0.1 * i, Tx1 HHb = -that, Tx2 O2Hb = 2 * i,
    Tx2 HHb = -2 * i, fit factor alternating 90 / 95, and an event label in the event column at the 1-based
    sample numbers in `events` ({sample number: code}); the parser reads every column as a number, so an
    event code is numeric (Oxysoft writes 1, 2, ...) and its label is that code as text. Row 0 is the
    absolute baseline the parser skips."""
    lines = [
        "Some header\tvalue",
        f"Datafile sample rate\t{rate:g} Hz",
        "Legend:",
        "1\tsample number",
        "2\tRx1 - Tx1 O2Hb (file)",
        "3\tRx1 - Tx1 HHb (file)",
        "4\tRx1 - Tx2 O2Hb (file)",
        "5\tRx1 - Tx2 HHb (file)",
        "6\tEvent (Event)",
    ]
    n_cols = 6
    if with_fit_factor:
        lines.append("7\tFit Factor")
        n_cols = 7
    lines.append("\t".join(str(i) for i in range(1, n_cols + 1)))
    lines.append("\t".join(["0"] * n_cols))                         # row 0: absolute baseline, skipped
    for i in range(1, n_rows + 1):
        v1 = offset + 0.1 * i
        row = [str(i), f"{v1:.6f}", f"{-v1:.6f}", f"{2.0 * i:.6f}", f"{-2.0 * i:.6f}", str(events.get(i, 0)) if events else "0"]
        if with_fit_factor:
            row.append("90" if i % 2 else "95")
        lines.append("\t".join(row))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


class TestParser(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "nirs.txt")

    def test_channels_come_out_in_label_order_with_their_known_values(self):
        write_export(self.path, 8)
        o2hb, hhb, events, meta, labels, fs = po._parse_oxysoft_txt(self.path)
        self.assertEqual(labels, ["Tx1", "Tx2"])
        self.assertEqual(fs, 10.0)
        self.assertEqual(o2hb.shape, (2, 8))
        i = np.arange(1, 9)
        assert_allclose(o2hb[0], 0.1 * i)
        assert_allclose(hhb[0], -0.1 * i)
        assert_allclose(o2hb[1], 2.0 * i)
        assert_allclose(hhb[1], -2.0 * i)

    def test_the_baseline_row_zero_is_skipped(self):
        write_export(self.path, 5)
        o2hb, *_ = po._parse_oxysoft_txt(self.path)
        self.assertEqual(o2hb.shape[1], 5)                           # 5 data rows, not 6
        self.assertAlmostEqual(o2hb[0, 0], 0.1)

    def test_events_are_labelled_and_located_by_sample_index(self):
        write_export(self.path, 10, events={3: 1, 7: 2})
        _, _, events, *_ = po._parse_oxysoft_txt(self.path)
        self.assertEqual(events, [{"label": "1", "sample": 2}, {"label": "2", "sample": 6}])   # 0-based

    def test_the_fit_factor_mean_is_reported_when_the_file_has_that_column(self):
        write_export(self.path, 10)
        _, _, _, meta, *_ = po._parse_oxysoft_txt(self.path)
        self.assertAlmostEqual(meta["fit_factor_mean"], 92.5)
        write_export(self.path, 10, with_fit_factor=False)
        _, _, _, meta, *_ = po._parse_oxysoft_txt(self.path)
        self.assertNotIn("fit_factor_mean", meta)

    def test_a_file_with_no_data_rows_is_an_error(self):
        write_export(self.path, 0)
        with self.assertRaisesRegex(ValueError, "No data rows parsed"):
            po._parse_oxysoft_txt(self.path)

    def test_progress_runs_from_zero_to_a_hundred_percent(self):
        write_export(self.path, 50)
        seen = []
        po._parse_oxysoft_txt(self.path, progress=lambda f, m: seen.append(f))
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))

    def test_the_parser_lives_in_processing_oxysoft_and_the_loader_only_wraps_it(self):
        self.assertTrue(callable(po._parse_oxysoft_txt))
        self.assertFalse(hasattr(sources_load, "_parse_oxysoft_txt"))


class TestFileAndFolderReaders(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def test_a_single_file_comes_back_as_arrays_and_metadata(self):
        path = os.path.join(self.dir.name, "one.txt")
        write_export(path, 12)
        r = po.process_oxysoft_file(path)
        self.assertEqual(r["o2hb"].shape, (2, 12))
        self.assertEqual(r["hhb"].shape, (2, 12))
        self.assertEqual(r["channel_labels"], ["Tx1", "Tx2"])
        self.assertEqual(r["sample_rate"], 10.0)
        self.assertEqual(r["metadata"]["n_channels"], 2)             # physical channels, not O2Hb + HHb

    def test_a_file_without_o2hb_channels_says_which_one_and_why(self):
        path = os.path.join(self.dir.name, "odd.txt")
        with open(path, "w") as fh:
            fh.write("Datafile sample rate\t10 Hz\nLegend:\n1\tsample number\n2\tTSI%\n1\t2\n1\t5\n2\t6\n")
        with self.assertRaisesRegex(ValueError, r"No O2Hb channels recognized.*odd\.txt"):
            po.process_oxysoft_file(path)

    def test_a_folder_concatenates_its_exports_alphabetically(self):
        write_export(os.path.join(self.dir.name, "b.txt"), 4, offset=100.0)
        write_export(os.path.join(self.dir.name, "a.txt"), 6, offset=0.0)
        r = po.process_oxysoft_folder(self.dir.name)
        self.assertEqual(r["o2hb"].shape, (2, 10))
        assert_allclose(r["o2hb"][0, :6], 0.1 * np.arange(1, 7))                    # a.txt first
        assert_allclose(r["o2hb"][0, 6:], 100.0 + 0.1 * np.arange(1, 5))             # then b.txt
        self.assertEqual(r["metadata"]["n_channels"], 2)

    def test_a_folder_without_exports_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            po.process_oxysoft_folder(self.dir.name)

    def test_folder_progress_ends_at_a_hundred_percent(self):
        write_export(os.path.join(self.dir.name, "a.txt"), 300)
        write_export(os.path.join(self.dir.name, "b.txt"), 30)
        seen = []
        po.process_oxysoft_folder(self.dir.name, progress=lambda f, m: seen.append(f))
        self.assertEqual(seen[-1], 1.0)
        self.assertEqual(seen, sorted(seen))

    @unittest.expectedFailure
    def test_events_in_a_later_file_are_placed_after_the_earlier_files(self):
        """KNOWN DEFECT. Each file's events carry a sample index counted from the start of THAT file, and the
        folder reader concatenates the arrays but does not shift those indices by the length of the files
        before it — so an event in the second export lands at the same place as the corresponding event in
        the first. Here b.txt's event is its 2nd sample (global sample 6 + 2 - 1 = 7, 0-based) but is reported
        as 1."""
        write_export(os.path.join(self.dir.name, "a.txt"), 6)
        write_export(os.path.join(self.dir.name, "b.txt"), 4, events={2: 1})
        r = po.process_oxysoft_folder(self.dir.name)
        self.assertEqual(r["events"], [{"label": "1", "sample": 7}])


class TestLoaderWrapsIntoADataset(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def check(self, ds, n):
        self.assertEqual(ds.source_format, "Oxysoft")
        self.assertEqual(ds.signals.shape, (4, n))                    # all O2Hb first, then all HHb
        self.assertEqual(ds.num_channels, 4)
        self.assertEqual(ds.channel_names, ["Tx1 O2Hb", "Tx2 O2Hb", "Tx1 HHb", "Tx2 HHb"])
        self.assertEqual(ds.num_samples, n)
        self.assertAlmostEqual(ds.duration_s, n / 10.0)
        self.assertEqual(ds.sample_rate, 10.0)
        self.assertEqual(ds.metadata["n_channels"], 2)
        assert_allclose(ds.signals[0], 0.1 * np.arange(1, n + 1))

    def test_load_oxysoft_file(self):
        path = os.path.join(self.dir.name, "session.txt")
        write_export(path, 20)
        ds = sources_load.load_oxysoft_file(path)
        self.check(ds, 20)
        self.assertEqual(ds.folder_name, "session")
        self.assertEqual(ds.folder_path, self.dir.name)

    def test_load_oxysoft_folder(self):
        write_export(os.path.join(self.dir.name, "a.txt"), 20)
        ds = sources_load.load_oxysoft(self.dir.name, "mystudy")
        self.check(ds, 20)
        self.assertEqual(ds.folder_name, "mystudy")
        self.assertEqual(ds.folder_path, self.dir.name)

    def test_the_wrapped_data_is_what_the_reader_returned(self):
        path = os.path.join(self.dir.name, "x.txt")
        write_export(path, 15, events={4: 1})
        r = po.process_oxysoft_file(path)
        ds = sources_load.load_oxysoft_file(path)
        assert_array_equal(ds.signals, np.concatenate([r["o2hb"], r["hhb"]]))
        self.assertEqual(ds.events, r["events"])
        self.assertEqual(ds.metadata, r["metadata"])


if __name__ == "__main__":
    unittest.main()
