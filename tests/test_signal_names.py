"""recording.py: the rules for naming a recording's signals — a name must say something and two signals cannot share one (name_problem), what a set of
renames would change (plan_signal_renames) — and that Save Changes' saved names obey them too."""

import unittest

import numpy as np

import PyATCLI as pl
from PyATCLI.recording import name_problem, plan_signal_renames

from tests.test_analyses import make_cache

NORMALIZED, MAIN = "Normalized (dF/F)", "Main Driver (x465A)"


def three_signal_cache():
    cache = make_cache()
    cache["markers"] = [{"time": 60.0, "label": "lever", "color": "green", "fontsize": 8}]
    cache["signals"]["isosbestic"] = {"label": "Isosbestic (x415A)", "y": cache["corr"] * 0.5}
    return cache


class TestTheRules(unittest.TestCase):

    def test_a_name_must_say_something_and_two_signals_cannot_share_one(self):
        self.assertIsNone(name_problem({"a": "x", "b": "y"}))
        self.assertEqual(name_problem({"a": "x", "b": "  "}), "A signal needs a name.")
        self.assertEqual(name_problem({"a": "x", "b": " x "}), "Two signals can't both be called \"x\".")


    def test_what_would_change(self):
        cache = three_signal_cache()
        self.assertEqual(plan_signal_renames(cache, {"normalized": " GCaMP ", "main_driver": MAIN, "isosbestic": "", "nope": "x"}), {"normalized": "GCaMP"})
        self.assertEqual(plan_signal_renames({"x": np.arange(3.0)}, {"normalized": "a"}), {})
        with self.assertRaisesRegex(ValueError, "Two signals can't both be called \"Same\""):
            plan_signal_renames(cache, {"normalized": "Same", "main_driver": "Same"})
        with self.assertRaisesRegex(ValueError, "both be called"):
            plan_signal_renames(cache, {"normalized": MAIN})                                   # already another signal's name


class TestNamesTheAppRefusesAreRefusedWhenSaved(unittest.TestCase):
    def test_saved_state_rename_checks_for_a_shared_name(self):
        import os
        import tempfile
        from unittest import mock

        import tdt
        from tests.test_cli import fake_block, make_tdt_folder

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(tdt, "read_block", fake_block()):
            folder = make_tdt_folder(tmp, "Rat7-250101-101010")
            with self.assertRaisesRegex(ValueError, "Two signals can't both be called \"Same\""):
                pl.saved_state.rename_signals(folder, {"main_driver": "Same", "isosbestic": "Same"})
            self.assertFalse(os.path.exists(pl.signal_labels_path(folder)))
            with self.assertRaisesRegex(ValueError, "both be called"):
                pl.saved_state.rename_signals(folder, {"main_driver": "Normalized (dF/F)"})                     # another signal's name
            self.assertEqual(pl.saved_state.rename_signals(folder, {"main_driver": "GCaMP"}), {"main_driver": "GCaMP"})


if __name__ == "__main__":
    unittest.main()
