"""saved_state.py: writing the state Save Changes writes (markers, splices, signal names, highlights, text boxes) and the edits a
script makes with it — the same files the app reads back when it opens the recording, so an edit made in either shows in both."""

import json
import os
import unittest
from pathlib import Path

import numpy as np

import PyATCLI as pl
from PyATCLI import saved_state as S

from tests.test_cli import CliCase, make_tdt_folder


class SavedCase(CliCase):
    def setUp(self):
        super().setUp()
        self.saves = os.path.join(self.tmp, "JSON saves")                       # beside session.txt (a file source)

    def reopen(self, path=None):
        return pl.load_recording(path or self.oxy)

    def saved_files(self):
        return sorted(os.listdir(self.saves)) if os.path.isdir(self.saves) else []


class TestWriters(SavedCase):
    def test_markers_are_written_where_the_app_looks_and_read_back_as_written(self):
        markers = [{"time": 12.5, "label": "Lever", "color": "red", "fontsize": 8}, {"time": 3.0, "label": "Clap"}]
        path = S.write_saved_markers(self.oxy, markers)
        self.assertEqual(path, pl.marker_sidecar_path(self.oxy))
        self.assertEqual(self.saved_files(), ["session.txt.markers.json"])
        self.assertEqual(pl.read_saved_markers(self.oxy), markers)
        self.assertEqual(self.reopen()["cache"]["markers"], markers)                 # and the app's own load restores them

    def test_a_tdt_folders_markers_are_markers_json_inside_it(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        S.write_saved_markers(folder, [{"time": 1.0, "label": "x"}])
        self.assertTrue(os.path.isfile(os.path.join(folder, "JSON saves", "markers.json")))

    def test_what_the_app_could_not_read_back_is_refused_and_nothing_is_written(self):
        for bad in ([{"label": "no time"}], [{"time": "soon"}], [{"time": float("nan")}], ["marker"], [{"time": True}]):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "numeric 'time'"):
                S.write_saved_markers(self.oxy, bad)
        self.assertEqual(self.saved_files(), [])

    def test_splices_are_written_in_order_and_with_none_the_file_goes(self):
        splices = [{"mode": "cut_out", "start": 10.0, "end": 20.0}, {"mode": "keep_inside", "start": 1.0, "end": 30.0, "extra": "dropped"}]
        S.write_saved_splices(self.oxy, splices)
        self.assertEqual(pl.read_saved_splices(self.oxy), [splices[0], {"mode": "keep_inside", "start": 1.0, "end": 30.0}])
        self.assertEqual(S.write_saved_splices(self.oxy, []), None)
        self.assertEqual(self.saved_files(), [])
        self.assertEqual(S.write_saved_splices(self.oxy, []), None)                   # nothing to remove is fine

    def test_a_splice_the_app_could_not_replay_is_refused(self):
        for bad in ({"mode": "trim", "start": 1, "end": 2}, {"mode": "cut_out", "start": 5, "end": 5}, {"mode": "cut_out", "start": 9, "end": 2},
                    {"mode": "cut_out", "start": "a", "end": 2}, {"mode": "cut_out", "start": 1}):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "a splice is"):
                S.write_saved_splices(self.oxy, [bad])
        self.assertEqual(self.saved_files(), [])

    def test_signal_names_keep_their_characters_and_none_removes_the_file(self):
        S.write_saved_signal_labels(self.oxy, {"main_driver": "GCaMP µ", "isosbestic": "  ", "normalized": "dF/F"})
        self.assertEqual(pl.read_saved_signal_labels(self.oxy), {"main_driver": "GCaMP µ", "normalized": "dF/F"})   # a blank name is no name
        self.assertIn("GCaMP µ", Path(pl.signal_labels_path(self.oxy)).read_text(encoding="utf-8"))               # written as the letter
        S.write_saved_signal_labels(self.oxy, {})
        self.assertEqual(self.saved_files(), [])

    def test_highlights_and_text_boxes_have_their_own_files_and_load_into_the_recording(self):
        S.write_saved_annotations(self.oxy, pl.recording.HIGHLIGHTS_FILE, [{"start": 1.0, "end": 2.0, "color": "#FFEB3B"}])
        S.write_saved_annotations(self.oxy, pl.recording.TEXT_ANNOTATIONS_FILE, [{"x": 5.0, "y": 0.5, "text": "peak", "fontsize": 10}])
        self.assertEqual(self.saved_files(), ["session.txt.highlights.json", "session.txt.text_annotations.json"])
        cache = self.reopen()["cache"]
        self.assertEqual(cache["highlights"], [{"start": 1.0, "end": 2.0, "color": "#FFEB3B"}])
        self.assertEqual(cache["text_annotations"], [{"x": 5.0, "y": 0.5, "text": "peak", "fontsize": 10}])
        with self.assertRaisesRegex(ValueError, "list of dicts"):
            S.write_saved_annotations(self.oxy, "highlights.json", ["x"])

    def test_a_recording_with_none_saved_has_no_highlight_or_text_keys(self):
        cache = self.reopen()["cache"]
        self.assertNotIn("highlights", cache)
        self.assertNotIn("text_annotations", cache)

    def test_an_unreadable_highlights_file_is_a_warning_and_the_recording_still_opens(self):
        os.makedirs(self.saves)
        Path(self.saves, "session.txt.highlights.json").write_text("{not json", encoding="utf-8")
        loaded = self.reopen()
        self.assertTrue(any("highlights could not be read" in w for w in loaded["warnings"]))
        self.assertNotIn("highlights", loaded["cache"])

    def test_clearing_removes_the_files_in_the_saves_folder_and_nothing_else(self):
        S.write_saved_markers(self.oxy, [{"time": 1.0, "label": "x"}])
        S.write_saved_splices(self.oxy, [{"mode": "cut_out", "start": 1.0, "end": 2.0}])
        os.makedirs(os.path.join(self.saves, "kept folder"))
        outside = Path(self.tmp, "session.txt.notes")
        outside.write_text("mine", encoding="utf-8")
        removed = S.clear_saved_state(self.oxy)
        self.assertEqual(removed, ["session.txt.markers.json", "splice.json"])
        self.assertEqual(self.saved_files(), ["kept folder"])
        self.assertTrue(outside.is_file())
        self.assertEqual(S.clear_saved_state(self.oxy), [])
        self.assertEqual(S.clear_saved_state(os.path.join(self.tmp, "nowhere.txt")), [])

    def test_the_summary_says_what_is_saved(self):
        empty = S.saved_state_summary(self.oxy)
        self.assertEqual({k: v for k, v in empty.items() if k != "folder"}, {"markers": [], "splices": [], "signal_labels": {}, "highlights": [], "text_annotations": []})
        S.add_marker(self.oxy, 5.0, label="Lever")
        S.add_highlight(self.oxy, 2.0, 1.0, "pink")
        summary = S.saved_state_summary(self.oxy)
        self.assertEqual(summary["folder"], self.saves)
        self.assertEqual([m["label"] for m in summary["markers"]], ["Lever"])
        self.assertEqual(summary["highlights"], [{"start": 1.0, "end": 2.0, "color": "#FF4D94"}])

    def test_a_file_that_cannot_be_read_is_named(self):
        os.makedirs(self.saves)
        Path(self.saves, "session.txt.markers.json").write_text('{"not": "a list"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "saved markers of .* cannot be read"):
            S.saved_state_summary(self.oxy)


class TestMarkers(SavedCase):
    def test_a_marker_is_stamped_as_add_marker_stamps_it(self):
        self.assertEqual(S.new_marker(4), {"time": 4.0, "label": "Marker", "color": "green", "fontsize": 8})
        self.assertEqual(S.new_marker(4, "Lever", "red", 12), {"time": 4.0, "label": "Lever", "color": "red", "fontsize": 12})

    def test_adding_one_keeps_the_others_and_the_next_open_has_it(self):
        S.add_marker(self.oxy, 10.0, label="First")
        marker = S.add_marker(self.oxy, 20.0, label="Second", color="blue")
        self.assertEqual(marker, {"time": 20.0, "label": "Second", "color": "blue", "fontsize": 8})
        self.assertEqual([m["label"] for m in self.reopen()["cache"]["markers"]], ["First", "Second"])

    def test_a_time_outside_the_recording_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"999 s is outside the recording, which runs from 0\.000 s to 59\.900 s"):
            S.add_marker(self.oxy, 999)
        self.assertEqual(self.saved_files(), [])

    def test_removing_takes_the_ones_near_a_time_and_nothing_else(self):
        for t, label in ((10.0, "A"), (10.02, "B"), (30.0, "A")):
            S.add_marker(self.oxy, t, label=label)
        removed = S.remove_markers(self.oxy, 10.0, tolerance=0.05)
        self.assertEqual([m["label"] for m in removed], ["A", "B"])
        self.assertEqual([m["time"] for m in self.reopen()["cache"]["markers"]], [30.0])

    def test_removing_can_name_the_marker_too(self):
        S.add_marker(self.oxy, 10.0, label="A")
        S.add_marker(self.oxy, 10.0, label="B")
        self.assertEqual([m["label"] for m in S.remove_markers(self.oxy, 10.0, label="B")], ["B"])
        self.assertEqual([m["label"] for m in self.reopen()["cache"]["markers"]], ["A"])

    def test_removing_what_is_not_there_removes_nothing_and_leaves_the_file(self):
        S.add_marker(self.oxy, 10.0)
        before = Path(pl.marker_sidecar_path(self.oxy)).read_bytes()
        self.assertEqual(S.remove_markers(self.oxy, 40.0), [])
        self.assertEqual(Path(pl.marker_sidecar_path(self.oxy)).read_bytes(), before)


class TestSplices(SavedCase):
    def test_a_cut_out_is_saved_and_the_recording_opens_without_those_seconds(self):
        full = self.reopen()["cache"]
        splices = S.add_splice(self.oxy, "cut_out", 10.0, 20.0)
        self.assertEqual(splices, [{"mode": "cut_out", "start": 10.0, "end": 20.0}])
        loaded = self.reopen()
        self.assertEqual(loaded["splices"], splices)
        self.assertAlmostEqual(len(loaded["cache"]["x"]), len(full["x"]) - 100, delta=2)             # 10 s at 10 Hz
        self.assertIn("[cut", loaded["cache"]["store"])

    def test_a_second_splice_stacks_on_the_first_and_is_on_the_time_axis_as_it_now_stands(self):
        S.add_splice(self.oxy, "cut_out", 10.0, 20.0)
        splices = S.add_splice(self.oxy, "keep_inside", 5.0, 30.0)
        self.assertEqual([s["mode"] for s in splices], ["cut_out", "keep_inside"])
        x = self.reopen()["cache"]["x"]
        self.assertAlmostEqual(float(x[-1] - x[0]), 25.0, delta=0.2)

    def test_a_range_that_leaves_too_little_is_refused_and_nothing_is_written(self):
        with self.assertRaisesRegex(ValueError, "leaves too little of the recording"):
            S.add_splice(self.oxy, "keep_inside", 10.0, 10.05)
        self.assertEqual(self.saved_files(), [])

    def test_a_bad_mode_or_range_is_refused_before_the_recording_is_opened(self):
        for args in (("trim", 1, 2), ("cut_out", 5, 2), ("cut_out", "a", 2)):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "a splice is"):
                S.add_splice(os.path.join(self.tmp, "does-not-exist.txt"), *args)

    def test_a_saved_splice_can_be_taken_out_again(self):
        S.add_splice(self.oxy, "cut_out", 10.0, 20.0)
        S.add_splice(self.oxy, "cut_out", 5.0, 8.0)
        self.assertEqual(S.remove_splice(self.oxy, 1), [{"mode": "cut_out", "start": 5.0, "end": 8.0}])
        self.assertEqual(S.remove_splice(self.oxy, 1), [])
        self.assertEqual(self.saved_files(), [])
        with self.assertRaisesRegex(ValueError, "no saved splice 1; the recording has 0"):
            S.remove_splice(self.oxy, 1)


class TestSignalNames(SavedCase):
    def tdt(self):
        return make_tdt_folder(self.tmp, "Rat7-250101-101010")

    def test_a_signal_is_renamed_and_the_next_open_calls_it_that(self):
        folder = self.tdt()
        saved = S.rename_signals(folder, {"main_driver": "GCaMP"})
        self.assertEqual(saved, {"main_driver": "GCaMP"})
        self.assertEqual(pl.load_recording(folder)["cache"]["signals"]["main_driver"]["label"], "GCaMP")

    def test_renames_add_up_and_an_empty_name_takes_one_away(self):
        folder = self.tdt()
        S.rename_signals(folder, {"main_driver": "GCaMP"})
        self.assertEqual(S.rename_signals(folder, {"isosbestic": "Control"}), {"main_driver": "GCaMP", "isosbestic": "Control"})
        self.assertEqual(S.rename_signals(folder, {"main_driver": ""}), {"isosbestic": "Control"})
        S.rename_signals(folder, {"isosbestic": ""})
        self.assertFalse(os.path.exists(pl.signal_labels_path(folder)))

    def test_a_signal_the_recording_does_not_have_lists_the_ones_it_does(self):
        with self.assertRaisesRegex(ValueError, "no signal 'red' in this recording; its signals are: normalized, main_driver, isosbestic"):
            S.rename_signals(self.tdt(), {"red": "x"})

    def test_a_recording_with_no_named_signals_says_so(self):
        with self.assertRaisesRegex(ValueError, "has no named signals"):
            S.rename_signals(self.oxy, {"o2hb": "x"})


class TestHighlightsAndText(SavedCase):
    def test_a_highlight_is_sorted_and_takes_the_apps_colours_by_name_or_any_hex(self):
        self.assertEqual(S.new_highlight(9, 3, "Blue"), {"start": 3.0, "end": 9.0, "color": "#29B6F6"})
        self.assertEqual(S.new_highlight(1, 2, "yellow")["color"], "#FFEB3B")
        self.assertEqual(S.new_highlight(1, 2, "#abc")["color"], "#abc")
        with self.assertRaisesRegex(ValueError, "not a highlight colour: use Yellow, Green, Pink, Orange, Blue"):
            S.new_highlight(1, 2, "chartreuse")
        with self.assertRaisesRegex(ValueError, "two different times"):
            S.new_highlight(2, 2)

    def test_highlights_stack(self):
        S.add_highlight(self.oxy, 1, 2)
        S.add_highlight(self.oxy, 1.5, 3, "Green")
        self.assertEqual([h["color"] for h in self.reopen()["cache"]["highlights"]], ["#FFEB3B", "#8BC34A"])

    def test_a_text_box_is_placed_as_the_text_tool_places_it(self):
        self.assertEqual(S.new_text_annotation(5, 0.5, "  peak  "), {"x": 5.0, "y": 0.5, "text": "peak", "fontsize": 10})
        for args in ((5, 0.5, "  "), (5, 0.5, "x", 5), (5, 0.5, "x", 73), ("a", 0.5, "x")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                S.new_text_annotation(*args)
        S.add_text(self.oxy, 5, 0.5, "peak", 14)
        self.assertEqual(self.reopen()["cache"]["text_annotations"], [{"x": 5.0, "y": 0.5, "text": "peak", "fontsize": 14}])


class TestWhatTheAnalysesSee(SavedCase):
    """An edit made here changes what a run on the recording sees, exactly as the same edit made in the app does."""

    def test_a_cut_out_changes_the_window_an_analysis_reads_and_use_saved_false_ignores_it(self):
        S.add_splice(self.oxy, "cut_out", 10.0, 20.0)
        spliced = pl.run_analysis("process", self.reopen()["cache"])
        as_recorded = pl.run_analysis("process", pl.load_recording(self.oxy, use_saved=False)["cache"])
        self.assertLess(len(spliced.tables["trace"]), len(as_recorded.tables["trace"]))
        self.assertTrue(np.isfinite(spliced.tables["trace"].iloc[:, 1:].to_numpy()).all())


class TestTheSavedCommand(SavedCase):
    def saved(self, *argv):
        from tests.test_cli import run_cli
        return run_cli("saved", *argv)

    def test_every_edit_is_a_command_and_show_lists_them(self):
        steps = (("marker", self.oxy, "--time", "12.5", "--label", "Lever", "--color", "red"),
                 ("splice", self.oxy, "--mode", "cut_out", "--start", "30", "--end", "40"),
                 ("highlight", self.oxy, "--start", "2", "--end", "4", "--color", "Pink"),
                 ("text", self.oxy, "--x", "5", "--y", "0.25", "--text", "peak", "--fontsize", "12"))
        for step in steps:
            status, out, err = self.saved(*step)
            self.assertEqual((status, err), (0, ""), (step, err))
        status, out, _ = self.saved("show", self.oxy)
        self.assertEqual(status, 0)
        for line in ("marker      12.500 s  Lever  (red)", "splice 1    cut_out 30.000..40.000 s", "highlight   2.000..4.000 s  #FF4D94",
                     "text        (5.000 s, 0.25)  peak"):
            self.assertIn(line, out)
        status, out, _ = self.saved("show", self.oxy, "--json")
        summary = json.loads(out)["result"]
        self.assertEqual((len(summary["markers"]), len(summary["splices"]), len(summary["highlights"]), len(summary["text_annotations"])), (1, 1, 1, 1))

    def test_what_the_command_line_saved_is_what_the_library_reads(self):
        self.saved("marker", self.oxy, "--time", "12.5", "-q")
        self.assertEqual(pl.read_saved_markers(self.oxy), [{"time": 12.5, "label": "Marker", "color": "green", "fontsize": 8}])

    def test_unmark_and_unsplice_take_edits_away(self):
        self.saved("marker", self.oxy, "--time", "12.5", "-q")
        self.saved("splice", self.oxy, "--mode", "cut_out", "--start", "30", "--end", "40", "-q")
        status, out, _ = self.saved("unmark", self.oxy, "--time", "12.51")
        self.assertEqual((status, out.strip()), (0, "removed 1 marker"))
        status, out, _ = self.saved("unsplice", self.oxy, "1")
        self.assertEqual((status, out.strip()), (0, "splice 1 removed (0 left)"))
        self.assertEqual(self.saved_files(), ["session.txt.markers.json"])                  # the markers file stays, now empty
        self.assertEqual(pl.read_saved_markers(self.oxy), [])

    def test_unmark_with_nothing_there_fails_and_says_where_it_looked(self):
        status, _, err = self.saved("unmark", self.oxy, "--time", "12.5", "--label", "Lever")
        self.assertEqual(status, 1)
        self.assertIn("no saved marker within 0.05 s of 12.5 s named 'Lever'", err)

    def test_rename_takes_key_equals_name(self):
        folder = make_tdt_folder(self.tmp, "Rat7-250101-101010")
        status, out, err = self.saved("rename", folder, "main_driver=GCaMP", "isosbestic=Control")
        self.assertEqual((status, err, out.strip()), (0, "", "2 signal names saved"))
        self.assertEqual(pl.read_saved_signal_labels(folder), {"main_driver": "GCaMP", "isosbestic": "Control"})
        status, _, err = self.saved("rename", folder, "just words")
        self.assertEqual(status, 1)
        self.assertIn("a rename is KEY=NAME", err)

    def test_clear_wants_to_be_told_and_then_removes_the_saved_files(self):
        self.saved("marker", self.oxy, "--time", "12.5", "-q")
        status, _, err = self.saved("clear", self.oxy)
        self.assertEqual(status, 2)
        self.assertIn("say --yes", err)
        self.assertEqual(self.saved_files(), ["session.txt.markers.json"])                  # nothing was deleted
        status, out, _ = self.saved("clear", self.oxy, "--yes")
        self.assertEqual((status, out.strip()), (0, "removed 1 file: session.txt.markers.json"))
        self.assertEqual(self.saved_files(), [])

    def test_a_recording_that_is_not_there_is_an_error(self):
        status, _, err = self.saved("show", os.path.join(self.tmp, "nope.txt"))
        self.assertEqual(status, 1)
        self.assertIn("does not exist", err)

    def test_a_bad_value_is_refused_by_the_parser_or_the_library_with_what_is_wrong(self):
        status, _, err = self.saved("splice", self.oxy, "--mode", "trim", "--start", "1", "--end", "2")
        self.assertEqual(status, 2)
        self.assertIn("invalid choice: 'trim'", err)
        status, _, err = self.saved("highlight", self.oxy, "--start", "1", "--end", "2", "--color", "chartreuse")
        self.assertEqual(status, 1)
        self.assertIn("not a highlight colour", err)
        status, _, err = self.saved("splice", self.oxy, "--mode", "keep_inside", "--start", "10", "--end", "10.05")
        self.assertEqual(status, 1)
        self.assertIn("leaves too little", err)
        self.assertEqual(self.saved_files(), [])


if __name__ == "__main__":
    unittest.main()
