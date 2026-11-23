"""analysis/group/figure_stats.py: the numbers a group analysis's figures are made of — the pooled value of each subject for a group of markers, a group's mean,
standard error and quartiles, the points of a Q-Q plot, and the stars of a p-value."""

import unittest

import numpy as np
import pandas as pd
import scipy.stats as st

import PyATCLI as pl


class TestPooledSubjectValues(unittest.TestCase):
    MEANS = pd.DataFrame([
        {"measure": "auc", "subject": "s1", "marker": "A", "mean": 1.0},
        {"measure": "auc", "subject": "s1", "marker": "B", "mean": 3.0},
        {"measure": "auc", "subject": "s2", "marker": "A", "mean": 5.0},
        {"measure": "auc", "subject": "s3", "marker": "C", "mean": 7.0},
        {"measure": "peak", "subject": "s1", "marker": "A", "mean": 99.0},
    ])

    def test_a_subject_gives_the_average_of_the_markers_of_a_group_it_has_data_for(self):
        out = pl.pooled_subject_values(self.MEANS, "auc", {"AB": ["A", "B"], "C": ["C"]})
        self.assertEqual(out["AB"], {"s1": 2.0, "s2": 5.0})                                      # s2 has only A: its value is A's
        self.assertEqual(out["C"], {"s3": 7.0})

    def test_only_the_measure_asked_about_counts(self):
        self.assertEqual(pl.pooled_subject_values(self.MEANS, "peak", {"A": ["A"]}), {"A": {"s1": 99.0}})

    def test_every_label_is_in_the_answer_even_with_nobody_in_it(self):
        out = pl.pooled_subject_values(self.MEANS, "auc", {"A": ["A"], "Z": ["Z"]})
        self.assertEqual(out["Z"], {})
        self.assertEqual(list(out), ["A", "Z"])                                                   # (in the order given)

    def test_an_empty_table_gives_empty_groups(self):
        empty = pd.DataFrame(columns=["measure", "subject", "marker", "mean"])
        self.assertEqual(pl.pooled_subject_values(empty, "auc", {"A": ["A"]}), {"A": {}})

    def test_the_values_are_plain_floats(self):
        out = pl.pooled_subject_values(self.MEANS, "auc", {"AB": ["A", "B"]})
        self.assertTrue(all(type(v) is float for v in out["AB"].values()))


class TestSummaryStats(unittest.TestCase):
    def test_the_mean_sem_and_quartiles(self):
        values = [1.0, 2.0, 4.0, 8.0, 16.0]
        s = pl.summary_stats(values)
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["mean"], 6.2)
        self.assertAlmostEqual(s["sem"], np.std(values, ddof=1) / np.sqrt(5))
        self.assertEqual((s["q1"], s["median"], s["q3"]), tuple(np.percentile(values, [25, 50, 75])))

    def test_one_value_has_no_sem(self):
        s = pl.summary_stats([3.0])
        self.assertEqual((s["n"], s["mean"], s["median"]), (1, 3.0, 3.0))
        self.assertIsNone(s["sem"])

    def test_no_values_have_no_statistics(self):
        self.assertEqual(pl.summary_stats([]), {"n": 0, "mean": None, "sem": None, "q1": None, "median": None, "q3": None})

    def test_it_takes_an_array_or_a_list_and_gives_floats_that_travel_as_json(self):
        import json
        json.dumps(pl.summary_stats(np.array([1.0, 2.0, 3.0])))


class TestNormalQQ(unittest.TestCase):
    def test_it_is_scipys_probplot(self):
        values = np.random.default_rng(2).normal(size=40)
        got = pl.normal_qq(values)
        (osm, osr), (slope, intercept, _r) = st.probplot(values, dist="norm")
        np.testing.assert_array_equal(got["theoretical"], osm)
        np.testing.assert_array_equal(got["ordered"], osr)
        self.assertEqual((got["slope"], got["intercept"]), (float(slope), float(intercept)))

    def test_normal_values_lie_near_the_reference_line(self):
        values = np.random.default_rng(1).normal(size=400)
        got = pl.normal_qq(values)
        self.assertGreater(np.corrcoef(got["theoretical"], got["ordered"])[0, 1], 0.99)              # (a straight line)
        self.assertAlmostEqual(got["slope"], 1.0, delta=0.15)                                         # (of about the values' own spread, here one)
        self.assertAlmostEqual(got["intercept"], 0.0, delta=0.15)


class TestSignificanceStars(unittest.TestCase):
    def test_the_thresholds(self):
        for p, stars in ((0.0004, "***"), (0.001, "**"), (0.009, "**"), (0.01, "*"), (0.049, "*"), (0.05, "ns"), (0.7, "ns")):
            with self.subTest(p=p):
                self.assertEqual(pl.significance_stars(p), stars)

    def test_a_p_that_is_not_a_number_has_none(self):
        self.assertEqual(pl.significance_stars(float("nan")), "")
        self.assertEqual(pl.significance_stars(float("inf")), "")


class TestTheTraceFigureAveragesAsTheLibraryDoes(unittest.TestCase):
    def test_compute_group_stats_is_the_mean_and_the_sem_across_subjects(self):
        rows = np.random.default_rng(4).normal(size=(7, 30))
        mean, sem = pl.compute_group_stats(rows)
        np.testing.assert_array_equal(mean, rows.mean(axis=0))
        np.testing.assert_array_equal(sem, rows.std(axis=0, ddof=1) / np.sqrt(7))
        one_mean, one_sem = pl.compute_group_stats(rows[:1])
        np.testing.assert_array_equal(one_sem, np.zeros(30))                                      # (a single subject: no spread to speak of)


if __name__ == "__main__":
    unittest.main()
