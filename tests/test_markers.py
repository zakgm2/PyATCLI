"""How a marker is named and how markers are grouped by name (markers.py) — the rules every per-event analysis
shares, including the store renames a group analysis saves with its settings."""

import unittest

import PyATCLI as pl
from PyATCLI import markers as mk


def phase_marker(t, store, phase):
    return {"time": float(t), "label": store, "color": "black", "store": store, "phase": phase}


def note(t, text):
    return {"time": float(t), "label": text, "color": "black", "store": "Note"}


class TestNames(unittest.TestCase):
    def test_a_store_keeps_its_id_unless_it_was_renamed(self):
        self.assertEqual(mk.store_display_name("L1P"), "L1P")
        self.assertEqual(mk.store_display_name("L1P", {}), "L1P")
        self.assertEqual(mk.store_display_name("L1P", {"L1P": "Left Lever"}), "Left Lever")
        self.assertEqual(mk.store_display_name("R1P", {"L1P": "Left Lever"}), "R1P")
        self.assertIsNone(mk.store_display_name(None, {"L1P": "x"}))

    def test_onset_and_offset_markers_show_the_store_name_with_a_superscript(self):
        self.assertEqual(mk.marker_display_label(phase_marker(1, "L1P", "high")), "L1P¹")
        self.assertEqual(mk.marker_display_label(phase_marker(1, "L1P", "low")), "L1P⁰")
        self.assertEqual(mk.marker_display_label(phase_marker(1, "L1P", "high"), {"L1P": "Left Lever"}), "Left Lever¹")

    def test_renaming_a_store_never_touches_a_note_that_shares_its_store_id(self):
        self.assertEqual(mk.marker_display_label(note(5, "Clap"), {"Note": "Renamed"}), "Clap")

    def test_a_hand_placed_marker_with_no_store_keeps_its_label(self):
        self.assertEqual(mk.marker_display_label({"time": 1.0, "label": "Mine", "color": "green"}), "Mine")

    def test_an_unnamed_marker_has_an_empty_label(self):
        self.assertEqual(mk.marker_display_label({"time": 1.0, "label": "", "color": "green"}), "")


class TestGrouping(unittest.TestCase):
    def test_events_are_grouped_by_display_name_and_sorted(self):
        cache = {"detected_markers": [phase_marker(30, "L1P", "high"), phase_marker(10, "L1P", "high"),
                                      phase_marker(11, "L1P", "low")],
                 "markers": [note(20, "Clap")]}
        self.assertEqual(mk.group_markers_by_name(cache), {"L1P¹": [10.0, 30.0], "L1P⁰": [11.0], "Clap": [20.0]})

    def test_a_marker_both_detected_and_placed_is_counted_once(self):
        m = phase_marker(10, "L1P", "high")
        cache = {"detected_markers": [m], "markers": [dict(m)]}
        self.assertEqual(mk.group_markers_by_name(cache), {"L1P¹": [10.0]})

    def test_same_name_at_different_times_is_kept(self):
        cache = {"detected_markers": [], "markers": [note(1, "Clap"), note(2, "Clap")]}
        self.assertEqual(mk.group_markers_by_name(cache), {"Clap": [1.0, 2.0]})

    def test_a_recording_with_no_detected_markers_key_still_groups_its_own(self):
        self.assertEqual(mk.group_markers_by_name({"markers": [note(3, "A")]}), {"A": [3.0]})

    def test_two_stores_renamed_to_one_name_pool_their_events(self):
        cache = {"detected_markers": [phase_marker(5, "L1P", "high"), phase_marker(2, "L2P", "high")], "markers": []}
        self.assertEqual(mk.group_markers_by_name(cache, {"L1P": "Lever", "L2P": "Lever"}), {"Lever¹": [2.0, 5.0]})

    def test_nothing_is_left_when_there_are_no_markers(self):
        self.assertEqual(mk.group_markers_by_name({"detected_markers": [], "markers": []}), {})


class TestPublicNames(unittest.TestCase):
    def test_the_package_exports_them(self):
        for name in ("store_display_name", "marker_display_label", "group_markers_by_name", "PHASE_SUFFIX"):
            self.assertTrue(hasattr(pl, name), name)


if __name__ == "__main__":
    unittest.main()
