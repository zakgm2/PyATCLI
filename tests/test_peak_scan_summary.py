"""peak_scan_summary (analysis/peak_finder.py): the one-row-per-event-type table of a Peak Finder scan — counts, hit rate, average z of what was found,
what chance would do — which the engine's compute.peaks hands to PyAT's results window and its CSV, and a script reads the same way."""

import unittest

import PyATCLI as pl


def result(found, z=0.0):
    return {"found": found, "z_score": z}


class TestTheSummary(unittest.TestCase):
    GROUPS = {
        "Lick": [result(True, 4.0), result(True, 6.0), result(False)],
        "Tone": [result(True, 3.0), result(False), result(False), result(False)],
        "Door": [result(False), result(False)],
        "Reward": [result(True, 5.0), result(True, 7.0)],
    }

    def rows(self, **kwargs):
        return {r["event"]: r for r in pl.peak_scan_summary(self.GROUPS, **kwargs)}

    def test_it_counts_the_events_the_peaks_found_and_the_rate(self):
        rows = self.rows()
        self.assertEqual((rows["Lick"]["occurrences"], rows["Lick"]["found"]), (3, 2))
        self.assertAlmostEqual(rows["Lick"]["hit_rate"], 2 / 3)
        self.assertEqual((rows["Tone"]["occurrences"], rows["Tone"]["found"]), (4, 1))
        self.assertEqual(rows["Door"]["hit_rate"], 0.0)

    def test_the_average_z_is_of_the_peaks_that_were_found_only(self):
        rows = self.rows()
        self.assertEqual(rows["Lick"]["avg_z"], 5.0)
        self.assertEqual(rows["Tone"]["avg_z"], 3.0)
        self.assertIsNone(rows["Door"]["avg_z"])                                                  # none found: no average

    def test_the_most_often_aligned_come_first_and_ties_keep_their_order(self):
        order = [r["event"] for r in pl.peak_scan_summary(self.GROUPS)]
        self.assertEqual(order, ["Reward", "Lick", "Tone", "Door"])
        tied = {"A": [result(True, 1.0), result(False)], "B": [result(True, 2.0), result(False)], "C": [result(True, 3.0)]}
        self.assertEqual([r["event"] for r in pl.peak_scan_summary(tied)], ["C", "A", "B"])

    def test_a_type_with_no_events_has_a_zero_rate_and_no_error(self):
        (row,) = pl.peak_scan_summary({"None": []})
        self.assertEqual((row["occurrences"], row["found"], row["hit_rate"], row["avg_z"]), (0, 0, 0.0, None))

    def test_without_a_null_the_chance_fields_are_none_but_there(self):
        for row in pl.peak_scan_summary(self.GROUPS):
            self.assertEqual([row["chance_hit_rate"], row["p_shuffle"], row["p_holm"]], [None, None, None])

    def test_with_a_null_each_type_gets_its_own_chance_level_and_ps(self):
        nulls = {"Lick": {"available": True, "chance_hit_rate": 0.25, "p_count": 0.01, "p_count_holm": 0.04},
                 "Tone": {"available": False, "reason": "too few events"},
                 "Reward": {"available": True, "chance_hit_rate": 0.5, "p_count": 0.3}}
        rows = self.rows(nulls=nulls)
        self.assertEqual((rows["Lick"]["chance_hit_rate"], rows["Lick"]["p_shuffle"], rows["Lick"]["p_holm"]), (0.25, 0.01, 0.04))
        self.assertEqual((rows["Tone"]["chance_hit_rate"], rows["Tone"]["p_shuffle"], rows["Tone"]["p_holm"]), (None, None, None))
        self.assertEqual((rows["Reward"]["chance_hit_rate"], rows["Reward"]["p_shuffle"], rows["Reward"]["p_holm"]), (0.5, 0.3, None))
        self.assertEqual(rows["Door"]["chance_hit_rate"], None)                                   # (no null for it at all)

    def test_the_rows_are_plain_numbers_and_text_so_they_travel_as_json(self):
        import json
        json.dumps(pl.peak_scan_summary(self.GROUPS))


if __name__ == "__main__":
    unittest.main()
