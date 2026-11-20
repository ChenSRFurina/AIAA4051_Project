"""Atomic result persistence under transient Windows file locks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.io import write_json


class AtomicJSONTests(unittest.TestCase):
    def test_transient_lock_preserves_old_file_until_replace(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "metrics.json"
            target.write_text('{"old": true}', encoding="utf-8")
            original_replace = Path.replace
            attempts = []

            def replace(source, destination):
                attempts.append(source)
                self.assertEqual(json.loads(target.read_text()), {"old": True})
                if len(attempts) < 3:
                    raise PermissionError("simulated Windows sharing violation")
                return original_replace(source, destination)

            with patch.object(Path, "replace", replace), patch("baseline.io.time.sleep") as sleep:
                write_json(target, {"correct": 23})
            self.assertEqual(json.loads(target.read_text()), {"correct": 23})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_permanent_lock_fails_bounded_and_keeps_both_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "metrics.json"
            target.write_text('{"old": true}', encoding="utf-8")
            with patch.object(Path, "replace", side_effect=PermissionError("locked")) as replace, \
                 patch("baseline.io.time.sleep") as sleep:
                with self.assertRaises(PermissionError):
                    write_json(target, {"new": True})
            self.assertEqual(replace.call_count, 8)
            self.assertEqual(sleep.call_count, 7)
            self.assertEqual(json.loads(target.read_text()), {"old": True})
            files = list(Path(folder).glob("*.tmp"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text()), {"new": True})

    def test_other_os_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Path, "replace", side_effect=OSError("disk failure")) as replace, \
                 patch("baseline.io.time.sleep") as sleep:
                with self.assertRaises(OSError):
                    write_json(Path(folder) / "metrics.json", {"new": True})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
