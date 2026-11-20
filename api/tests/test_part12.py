"""Protocol safety tests; synthetic fixtures never count as benchmark results."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("part12", ROOT / "api/part12.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class Part12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        driver.bootstrap(ROOT)
        from baseline.cli import demo_run
        demo_run(cls.base / "demo")
        cls.run_dir = cls.base / "demo/searchqa"
        cls.manifest = cls.base / "baseline_freeze.json"
        driver.freeze(cls.run_dir, cls.run_dir / "data", cls.manifest, ROOT)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_baseline_only_evaluation_and_no_memory_write(self):
        before = driver.sha(self.run_dir / "C/memory.json")
        with contextlib.redirect_stdout(io.StringIO()):
            result = driver.evaluate(self.manifest, self.base / "test")
        self.assertTrue(result["demo"])
        self.assertEqual(set(result["conditions"]), {"N", "I", "B"})
        self.assertTrue(all(v["n"] == 6 for v in result["conditions"].values()))
        self.assertEqual(before, driver.sha(self.run_dir / "C/memory.json"))
        pairs = driver.read(self.base / "test/comparison.json")
        self.assertEqual(set(pairs), {"N_to_I", "N_to_B", "I_to_B"})
        self.assertTrue(all(sum(x["counts"].values()) == 6 for x in pairs.values()))
        self.assertTrue(driver.read(self.manifest)["future_C"]["files"])

    def test_modified_manifest_rejected(self):
        f = driver.read(self.manifest)
        f["test_ids"] = list(reversed(f["test_ids"]))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            driver.verify(f)

    def test_modified_skill_rejected(self):
        p = self.run_dir / "B/best_skill.md"
        original = p.read_bytes()
        try:
            p.write_bytes(original + b"\nchanged")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                driver.verify(driver.read(self.manifest))
        finally:
            p.write_bytes(original)

    def test_no_overwrite(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            driver.freeze(self.run_dir, self.run_dir / "data", self.manifest, ROOT)

    def test_pending_test_not_replaced_with_dev(self):
        p = self.run_dir / "data/manifest.json"
        original = p.read_bytes()
        try:
            data = json.loads(original)
            data["test_pending"] = True
            driver.write(p, data)
            with self.assertRaisesRegex(ValueError, "Missing course held-out"):
                driver.freeze(self.run_dir, self.run_dir / "data", self.base / "pending.json", ROOT)
        finally:
            p.write_bytes(original)

    def test_driver_change_rejected(self):
        f = driver.read(self.manifest)
        f["driver_sha256"] = "changed"
        p = driver.bootstrap(ROOT)
        f["freeze_id"] = p.digest({k: v for k, v in f.items() if k != "freeze_id"})
        with self.assertRaisesRegex(ValueError, "driver changed"):
            driver.verify(f)

    def test_public_transport_rejected(self):
        with self.assertRaisesRegex(ValueError, "private HTTP"):
            driver.install_source_transport("10.1.2.3", "http://8.8.8.8:80/v1")


if __name__ == "__main__":
    unittest.main()
