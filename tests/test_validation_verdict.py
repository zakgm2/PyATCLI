"""validation_verdict (fieldstudy/validation.py): the plain-language read of a pair's numbers, and the "verdict" column of the validation summary — the
rule lives here, so a script and PyAT's window give the same read."""

import unittest

import numpy as np
import pandas as pd

import PyATCLI as pl


class TestTheVerdict(unittest.TestCase):
    def test_a_small_corrected_p_and_an_interval_clear_of_zero_is_a_real_effect(self):
        self.assertEqual(pl.validation_verdict(0.01, 0.12), "Likely a real effect")

    def test_a_small_corrected_p_with_an_interval_that_reaches_zero_is_only_borderline(self):
        self.assertEqual(pl.validation_verdict(0.01, -0.02), "Weak/borderline — worth more subjects")
        self.assertEqual(pl.validation_verdict(0.01, 0.0), "Weak/borderline — worth more subjects")

    def test_between_five_and_ten_percent_is_borderline_and_beyond_it_is_no_evidence(self):
        self.assertEqual(pl.validation_verdict(0.07, 0.3), "Weak/borderline — worth more subjects")
        self.assertEqual(pl.validation_verdict(0.05, 0.3), "Weak/borderline — worth more subjects")      # (the limits are strict: 0.05 itself is not under it)
        self.assertEqual(pl.validation_verdict(0.10, 0.3), "No evidence of an effect")
        self.assertEqual(pl.validation_verdict(0.6, 0.3), "No evidence of an effect")

    def test_no_p_is_not_enough_data(self):
        self.assertEqual(pl.validation_verdict(float("nan"), 0.2), "Not enough data yet")


class TestTheSummaryHasIt(unittest.TestCase):
    def test_every_row_of_the_summary_carries_its_verdict(self):
        rng = np.random.default_rng(5)
        n = 14
        df = pd.DataFrame({"wordcount_a": rng.integers(20, 80, n), "wordcount_b": rng.integers(20, 80, n), "sim_ab": rng.uniform(0.4, 0.9, n)})
        results = {"ab": {"same_subject_similarity": rng.uniform(0.5, 0.9, n), "null_values": rng.uniform(0.1, 0.5, 80),
                          "null_shuffle_means": rng.uniform(0.2, 0.4, 60)}}
        summary = pl.build_validation_summary(df, [("a", "b", "ab")], results, n_boot=60, rng_seed=1)
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["verdict"], pl.validation_verdict(row["p_value_fdr"], row["ci_lower"]))
        self.assertIn("verdict", summary.columns)


if __name__ == "__main__":
    unittest.main()
