"""Provenance records (provenance.py) and the version resolver (_version.py): which software ran, what the pipeline
does that nobody chose, which recording went in and in what state, and the sidecar file that carries it all."""

import json
import os
import re
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import numpy as np

import PyATCLI as pl
from PyATCLI import _version, provenance as pv
from PyATCLI.sources import processing_TDT as P

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TempDirCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name


class TestVersionResolver(TempDirCase):
    def write_project(self, name="PyATCLI", version="1.2.3"):
        pkg = os.path.join(self.tmp, "PyATCLI")
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(self.tmp, "pyproject.toml"), "w") as f:
            f.write(f'[project]\nname = "{name}"\nversion = "{version}"\n')
        return os.path.join(pkg, "_version.py")

    def test_a_source_checkout_reports_its_own_version(self):
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
            expected = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M).group(1)
        self.assertEqual(pl.__version__, expected)
        self.assertEqual(_version.resolve_version(), expected)

    def test_it_reads_the_pyproject_beside_the_package(self):
        with mock.patch.object(_version, "__file__", self.write_project(version="1.2.3")):
            self.assertEqual(_version.source_tree_version(), "1.2.3")

    def test_another_projects_pyproject_does_not_count(self):
        with mock.patch.object(_version, "__file__", self.write_project(name="SomethingElse")):
            self.assertIsNone(_version.source_tree_version())

    def test_an_installed_copy_has_no_pyproject_and_reports_the_distributions_version(self):
        nowhere = os.path.join(self.tmp, "site-packages", "PyATCLI", "_version.py")
        with mock.patch.object(_version, "__file__", nowhere), mock.patch.object(_version, "installed_version", return_value="9.9.9"):
            self.assertIsNone(_version.source_tree_version())
            self.assertEqual(_version.resolve_version(), "9.9.9")

    def test_a_checkout_wins_over_stale_installed_metadata(self):
        with mock.patch.object(_version, "__file__", self.write_project(version="2.0.0")), \
                mock.patch.object(_version, "installed_version", return_value="1.0.0"):
            self.assertEqual(_version.resolve_version(), "2.0.0")

    def test_with_neither_it_says_unknown(self):
        nowhere = os.path.join(self.tmp, "x", "_version.py")
        with mock.patch.object(_version, "__file__", nowhere), mock.patch.object(_version, "installed_version", return_value=None):
            self.assertEqual(_version.resolve_version(), "unknown")


class TestSoftwareVersions(unittest.TestCase):
    def test_it_names_python_the_os_the_library_and_the_packages(self):
        sv = pv.software_versions()
        for key in ("python", "implementation", "bits", "platform", "frozen", "library", "packages"):
            self.assertIn(key, sv)
        self.assertEqual(sv["library"], pl.__version__)
        self.assertEqual(sv["packages"]["numpy"], np.__version__)
        self.assertIs(sv["frozen"], False)
        self.assertNotIn("app", sv)

    def test_the_program_using_the_library_is_recorded(self):
        self.assertEqual(pv.software_versions({"name": "PyAT", "version": "3.2.0"})["app"], {"name": "PyAT", "version": "3.2.0"})

    def test_an_uninstalled_package_is_just_absent_and_nothing_raises(self):
        sv = pv.software_versions(packages=("numpy", "definitely-not-a-real-package-xyz"))
        self.assertEqual(list(sv["packages"]), ["numpy"])
        with mock.patch("importlib.metadata.version", side_effect=OSError("broken metadata")):
            self.assertEqual(pv.software_versions()["packages"], {})

    def test_it_is_json_serialisable(self):
        json.dumps(pv.software_versions({"name": "x", "version": "1"}))


class TestCheckEnvironment(unittest.TestCase):
    def record(self, **changes):
        sv = pv.software_versions()
        for k, v in changes.items():
            if k in sv["packages"] or k in ("numpy", "scipy"):
                sv["packages"][k] = v
            else:
                sv[k] = v
        return {"software": sv}

    def test_the_same_software_reports_nothing(self):
        self.assertEqual(pv.check_environment(self.record()), [])

    def test_a_different_library_version_is_named(self):
        out = pv.check_environment(self.record(library="2020.1.1"))
        self.assertEqual(len(out), 1)
        self.assertIn("2020.1.1", out[0])

    def test_a_different_python_minor_version_is_named_but_a_patch_is_not(self):
        out = pv.check_environment(self.record(python="3.9.1"))
        self.assertTrue(any("Python" in o for o in out))
        major, minor = pv.software_versions()["python"].split(".")[:2]
        self.assertEqual(pv.check_environment(self.record(python=f"{major}.{minor}.0")), [])

    def test_a_numerical_package_changing_minor_version_is_named_a_patch_is_not(self):
        out = pv.check_environment(self.record(numpy="1.26.4"))
        self.assertTrue(any("numpy" in o and "last digits" in o for o in out))
        major, minor = np.__version__.split(".")[:2]
        self.assertEqual(pv.check_environment(self.record(numpy=f"{major}.{minor}.99")), [])


class TestPipelineParameters(unittest.TestCase):
    def test_it_reports_what_the_code_uses(self):
        p = pv.pipeline_parameters("huber")
        self.assertEqual(p["dtype"], "float64")
        self.assertEqual(p["motion_correction"]["method"], "huber")
        self.assertEqual(p["denoise"]["cutoff_hz"], P.DENOISE_CUTOFF_HZ)
        self.assertEqual(p["denoise"]["order"], P.DENOISE_ORDER)
        self.assertEqual(p["smoothing"]["window_s"], 0.5)
        self.assertEqual(p["zscore"]["robust_sd_limit"], 5.0)

    def test_it_follows_the_constants_rather_than_a_copy_of_them(self):
        from PyATCLI.analysis import shared, zscore_peth
        with mock.patch.object(P, "DENOISE_CUTOFF_HZ", 3), mock.patch.object(shared, "SMOOTH_WINDOW_SEC", 0.25), \
                mock.patch.object(zscore_peth, "BASELINE_ROBUST_SD", 4.0):
            p = pv.pipeline_parameters()
        self.assertEqual((p["denoise"]["cutoff_hz"], p["smoothing"]["window_s"], p["zscore"]["robust_sd_limit"]), (3, 0.25, 4.0))
        self.assertIn("4 robust SDs", p["zscore"]["baseline_artifact_guard"])

    def test_the_constants_are_what_the_pipeline_really_uses(self):
        """The record is only honest if the defaults it names are the ones in force."""
        import inspect
        from PyATCLI.analysis import shared, zscore_peth
        d = inspect.signature(P.denoise_signal).parameters
        self.assertEqual((d["cutoff"].default, d["order"].default), (P.DENOISE_CUTOFF_HZ, P.DENOISE_ORDER))
        self.assertEqual(inspect.signature(shared.smooth_signal).parameters["window_sec"].default, shared.SMOOTH_WINDOW_SEC)
        self.assertEqual(inspect.signature(zscore_peth._baseline_mean_std).parameters["n_robust_sd"].default,
                         zscore_peth.BASELINE_ROBUST_SD)

    def test_an_unknown_regression_method_is_refused(self):
        with self.assertRaises(ValueError):
            pv.pipeline_parameters("magic")

    def test_it_is_json_serialisable(self):
        json.dumps(pv.pipeline_parameters())


class TestFingerprint(TempDirCase):
    def make_block(self):
        d = os.path.join(self.tmp, "Rat7-250101-101010")
        os.makedirs(os.path.join(d, "JSON saves"))
        for name, content in (("a.Tbk", b"aaaa"), ("b.Tdx", b"bbbbbb"), ("c.tev", b"c" * 100)):
            with open(os.path.join(d, name), "wb") as f:
                f.write(content)
        with open(os.path.join(d, "JSON saves", "markers.json"), "w") as f:
            f.write("[]")
        return d

    def test_a_folders_files_are_listed_with_sizes_and_hashes(self):
        d = self.make_block()
        fp = pv.file_fingerprint(d)
        self.assertEqual((fp["kind"], fp["n_files"], fp["size_bytes"]), ("folder", 3, 110))
        self.assertEqual([f["name"] for f in fp["files"]], ["a.Tbk", "b.Tdx", "c.tev"])
        self.assertEqual(fp["files"][0]["sha256"], "61be55a8e2f6b4e172338bddf184d6dbee29c98853e0a0485ecee7f27b9af0b4")
        self.assertEqual(fp["name"], "Rat7-250101-101010")
        self.assertEqual(fp["path"], os.path.abspath(d))
        datetime.fromisoformat(fp["modified"])

    def test_the_apps_own_saves_folder_is_not_part_of_the_recording(self):
        d = self.make_block()
        before = pv.file_fingerprint(d)["digest"]
        with open(os.path.join(d, "JSON saves", "markers.json"), "w") as f:
            f.write('[{"time": 1}]')
        self.assertEqual(pv.file_fingerprint(d)["digest"], before)

    def test_a_changed_byte_changes_the_digest(self):
        d = self.make_block()
        before = pv.file_fingerprint(d)["digest"]
        with open(os.path.join(d, "b.Tdx"), "wb") as f:
            f.write(b"bbbbbX")
        self.assertNotEqual(pv.file_fingerprint(d)["digest"], before)

    def test_a_file_too_big_to_hash_is_identified_by_name_and_size_only(self):
        d = self.make_block()
        with mock.patch.object(pv, "HASH_LIMIT_BYTES", 10):
            fp = pv.file_fingerprint(d)
        self.assertIsNone(fp["files"][2]["sha256"])                                     # 100 bytes > 10
        self.assertIsNotNone(fp["files"][0]["sha256"])
        self.assertEqual(fp["files"][2]["size"], 100)

    def test_portable_leaves_the_absolute_path_out(self):
        fp = pv.file_fingerprint(self.make_block(), portable=True)
        self.assertNotIn("path", fp)
        self.assertEqual(fp["name"], "Rat7-250101-101010")

    def test_a_single_file(self):
        p = os.path.join(self.tmp, "session.txt")
        with open(p, "wb") as f:
            f.write(b"hello")
        fp = pv.file_fingerprint(p)
        self.assertEqual((fp["kind"], fp["n_files"], fp["size_bytes"], fp["name"]), ("file", 1, 5, "session.txt"))

    def test_a_missing_path_says_so(self):
        fp = pv.file_fingerprint(os.path.join(self.tmp, "nope"))
        self.assertEqual((fp["kind"], fp["n_files"]), ("missing", 0))
        self.assertNotIn("digest", fp)

    def test_a_huge_folder_is_capped_not_walked_forever(self):
        d = os.path.join(self.tmp, "many")
        os.makedirs(d)
        for i in range(30):
            open(os.path.join(d, f"f{i:02d}"), "wb").close()
        self.assertEqual(pv.file_fingerprint(d, max_files=10)["n_files"], 10)


class TestMarkersSummary(unittest.TestCase):
    def test_counts_and_an_order_independent_digest(self):
        a = [{"time": 1.0, "label": "x"}, {"time": 2.0, "label": "y", "store": "S", "phase": "high"}]
        s = pv.markers_summary(a)
        self.assertEqual(s["n_user"], 2)
        self.assertEqual(s["user_digest"], pv.markers_summary(list(reversed(a)))["user_digest"])
        self.assertNotIn("n_detected", s)

    def test_a_moved_marker_changes_the_digest(self):
        a = [{"time": 1.0, "label": "x"}]
        b = [{"time": 1.000001, "label": "x"}]
        self.assertNotEqual(pv.markers_summary(a)["user_digest"], pv.markers_summary(b)["user_digest"])

    def test_detected_markers_are_summarised_separately(self):
        s = pv.markers_summary([], [{"time": 3.0, "label": "L1P", "store": "L1P", "phase": "high"}])
        self.assertEqual((s["n_user"], s["n_detected"]), (0, 1))
        self.assertIn("detected_digest", s)


class TestRecordingState(unittest.TestCase):
    def cache(self, n=1_000_000):
        return {"source": "TDT", "store": "Rat7 [cut 5.0s-8.0s]", "x": np.arange(n) / 1000.0, "fs": 1000.0,
                "corr": np.zeros(n), "markers": [{"time": 9.0, "label": "mine"}],
                "detected_markers": [{"time": 3.0, "label": "L1P", "store": "L1P", "phase": "high"}]}

    def test_it_states_what_decides_what_an_analysis_sees(self):
        splices = [{"mode": "cut_out", "start": 5.0, "end": 8.0}]
        s = pv.recording_state(self.cache(), regression_method="ransac", splices=splices, store_labels={"L1P": "Lever"},
                               signal="normalized")
        self.assertEqual((s["source_type"], s["n_samples"], s["fs"], s["regression_method"]), ("TDT", 1_000_000, 1000.0, "ransac"))
        self.assertAlmostEqual(s["duration_s"], 999.999)
        self.assertEqual(s["splices"], splices)
        self.assertEqual(s["store_labels"], {"L1P": "Lever"})
        self.assertEqual(s["signal"], "normalized")
        self.assertEqual((s["markers"]["n_user"], s["markers"]["n_detected"]), (1, 1))

    def test_a_million_sample_recording_leaves_no_data_in_the_record(self):
        s = pv.recording_state(self.cache(), regression_method="ols")
        text = json.dumps(s)
        self.assertLess(len(text), 1500)                                                # counts and hashes, not samples
        for key in ("x", "y", "corr", "raw", "signals"):
            self.assertNotIn(key, s)

    def test_optional_parts_are_left_out(self):
        s = pv.recording_state(self.cache(10))
        for key in ("regression_method", "store_labels", "signal"):
            self.assertNotIn(key, s)
        self.assertEqual(s["splices"], [])


class TestRecords(TempDirCase):
    def test_a_record_has_the_sections_that_were_given(self):
        rec = pv.build_record("event_peth", {"pre": 10.0, "post": 20.0}, recording={"n_samples": 5},
                              pipeline={"dtype": "float64"}, app={"name": "PyAT", "version": "3.2.0"}, notes=["hello"],
                              now=datetime(2026, 10, 6, 12, 0, 5))
        self.assertEqual((rec["schema"], rec["kind"], rec["created"]), (pv.SCHEMA_VERSION, "event_peth", "2026-10-06T12:00:05"))
        self.assertEqual(rec["analysis"], {"type": "event_peth", "parameters": {"pre": 10.0, "post": 20.0}})
        self.assertEqual(rec["recording"], {"n_samples": 5})
        self.assertEqual(rec["software"]["app"]["name"], "PyAT")
        self.assertEqual(rec["notes"], ["hello"])
        self.assertNotIn("source", rec)
        self.assertNotIn("outputs", rec)

    def test_a_source_path_is_fingerprinted_and_a_ready_made_fingerprint_is_kept(self):
        f = os.path.join(self.tmp, "x.txt")
        with open(f, "w") as fh:
            fh.write("data")
        self.assertEqual(pv.build_record("k", source=f)["source"]["kind"], "file")
        ready = {"name": "n", "digest": "d"}
        self.assertIs(pv.build_record("k", source=ready)["source"], ready)

    def test_outputs_are_listed_with_their_hashes(self):
        f = os.path.join(self.tmp, "fig.csv")
        with open(f, "wb") as fh:                                    # bytes: a text-mode write would make it 10 on Windows
            fh.write(b"a,b" + bytes([10]) + b"1,2" + bytes([10]))
        rec = pv.build_record("k", outputs=[f])
        self.assertEqual(rec["outputs"], [pv.output_entry(f)])
        self.assertEqual((rec["outputs"][0]["name"], rec["outputs"][0]["size"]), ("fig.csv", 8))

    def test_the_sidecar_sits_beside_its_export_named_after_it(self):
        self.assertEqual(pv.sidecar_path_for(os.path.join("a", "b", "Event PETH.png")), os.path.join("a", "b", "Event PETH.provenance.json"))
        self.assertEqual(pv.sidecar_path_for("x.tar.csv"), "x.tar.provenance.json")
        self.assertEqual(pl.PROVENANCE_SUFFIX, ".provenance.json")

    def test_a_record_survives_a_json_round_trip_numpy_values_included(self):
        rec = pv.build_record("k", {"times": np.array([1.5, 2.5]), "n": np.int64(3), "x": np.float64(0.25), "s": {1, 2}})
        path = pv.write_record(os.path.join(self.tmp, "deep", "er", "r.provenance.json"), rec)
        back = pv.read_record(path)
        self.assertEqual(back["analysis"]["parameters"]["times"], [1.5, 2.5])
        self.assertEqual((back["analysis"]["parameters"]["n"], back["analysis"]["parameters"]["x"]), (3, 0.25))
        self.assertEqual(sorted(back["analysis"]["parameters"]["s"]), [1, 2])

    def test_non_ascii_is_kept_readable(self):
        path = pv.write_record(os.path.join(self.tmp, "r.json"), pv.build_record("k", {"event": "L1P¹"}))
        with open(path, encoding="utf-8") as f:
            self.assertIn("L1P¹", f.read())

    def test_a_file_that_is_not_a_record_is_refused(self):
        p = os.path.join(self.tmp, "x.json")
        with open(p, "w") as f:
            json.dump({"hello": 1}, f)
        with self.assertRaisesRegex(ValueError, "not a provenance record"):
            pv.read_record(p)

    def test_a_record_from_a_newer_layout_is_refused_with_advice(self):
        p = os.path.join(self.tmp, "x.json")
        with open(p, "w") as f:
            json.dump({"schema": pv.SCHEMA_VERSION + 1, "kind": "k"}, f)
        with self.assertRaisesRegex(ValueError, "update PyATCLI"):
            pv.read_record(p)

    def test_something_that_cannot_be_serialised_is_an_error_not_a_silent_loss(self):
        with self.assertRaises(TypeError):
            pv.write_record(os.path.join(self.tmp, "x.json"), pv.build_record("k", {"bad": object()}))


class TestPublicNames(unittest.TestCase):
    def test_the_package_exports_the_provenance_api(self):
        for name in ("software_versions", "library_version", "check_environment", "pipeline_parameters", "file_fingerprint",
                     "markers_summary", "recording_state", "build_record", "output_entry", "sidecar_path_for",
                     "write_record", "read_record", "PROVENANCE_SCHEMA_VERSION", "PROVENANCE_SUFFIX"):
            self.assertTrue(hasattr(pl, name), name)


if __name__ == "__main__":
    unittest.main()
