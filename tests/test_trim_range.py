"""trim_range and describe_duration (splice.py): what is left of a recording once its ends are trimmed, and when that is refused. (PyAT's Trim ends asks
the library for it; the arithmetic and the words of its refusals are here, where a script can use them too.)"""

import unittest

import numpy as np

import PyATCLI as pl


class TestTrimRange(unittest.TestCase):
    X = np.arange(0, 40.0, 0.001)                            # a 40 s recording

    def test_it_is_the_range_from_the_new_first_sample_to_the_new_last(self):
        self.assertEqual(pl.trim_range(self.X, 5.0, 3.0), (5.0, self.X[-1] - 3.0))
        self.assertEqual(pl.trim_range(self.X, 5.0, 0.0), (5.0, self.X[-1]))
        self.assertEqual(pl.trim_range(self.X, 0.0, 3.0), (0.0, self.X[-1] - 3.0))

    def test_it_counts_from_where_the_time_axis_starts_now(self):
        """A recording already trimmed (or a table whose x does not start at 0) is measured from its own first time."""
        shifted = self.X + 44.0
        self.assertEqual(pl.trim_range(shifted, 10.0, 0.0), (54.0, shifted[-1]))

    def test_nothing_to_trim_a_negative_amount_and_trimming_it_all_are_refused(self):
        for first, last, words in ((0.0, 0.0, "how much"), (-1.0, 2.0, "zero or more"), (20.0, 20.0, "whole recording"), (39.999, 5.0, "whole recording")):
            with self.assertRaises(ValueError, msg=(first, last)) as caught:
                pl.trim_range(self.X, first, last)
            self.assertIn(words, str(caught.exception))

    def test_the_refusal_says_how_long_the_recording_is(self):
        with self.assertRaises(ValueError) as caught:
            pl.trim_range(np.arange(0, 300.0, 1.0), 200.0, 200.0)
        self.assertIn("(4.983 min long)", str(caught.exception))

    def test_the_range_is_what_a_keep_inside_splice_keeps(self):
        x = np.arange(0, 40.0, 0.001)
        start, end = pl.trim_range(x, 5.0, 3.0)
        kept = pl.splice_keep_inside(x, x.copy(), x.copy(), [], [], start, end)
        self.assertAlmostEqual(kept["x"][0], 5.0, places=3)
        self.assertAlmostEqual(kept["x"][-1], x[-1] - 3.0, places=3)

    def test_durations_are_said_in_seconds_then_minutes(self):
        self.assertEqual(pl.describe_duration(44.0), "44 s")
        self.assertEqual(pl.describe_duration(2.5), "2.5 s")
        self.assertEqual(pl.describe_duration(119.0), "119 s")
        self.assertEqual(pl.describe_duration(150.0), "2.5 min")
        self.assertEqual(pl.describe_duration(3600.0), "60 min")


if __name__ == "__main__":
    unittest.main()
