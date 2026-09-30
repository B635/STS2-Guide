import json
from pathlib import Path
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

from jsonschema import Draft202012Validator
from realtime.runtime_status import RuntimeStatusPublisher


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "a" * 64


class RuntimeStatusTests(unittest.TestCase):
    def test_published_progress_matches_contract_and_exit_is_owner_scoped(self):
        schema = json.loads((ROOT / "protocol/runtime-status.schema.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime-status.json"
            first = RuntimeStatusPublisher(path, FINGERPRINT)
            first.update("RUNNING", True, True)
            one = json.loads(path.read_text())
            Draft202012Validator(schema).validate(one)
            first.update("RUNNING", True, True)
            two = json.loads(path.read_text())
            self.assertEqual(one["session_id"], two["session_id"])
            self.assertGreater(two["heartbeat_sequence"], one["heartbeat_sequence"])
            second = RuntimeStatusPublisher(path, FINGERPRINT)
            second.update("RUNNING", True, True)
            self.assertEqual(json.loads(path.read_text())["session_id"], first.session_id)
            second.close()
            self.assertTrue(path.exists())
            first.close()
            self.assertFalse(path.exists())
            second.update("RUNNING", True, True)
            self.assertTrue(path.exists())
            second.close()
            self.assertFalse(path.exists())

    def test_waiting_tray_yields_to_worker_and_cannot_overwrite_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime-status.json"
            tray = RuntimeStatusPublisher(path, FINGERPRINT)
            worker = RuntimeStatusPublisher(path, FINGERPRINT)
            try:
                tray.update("IDLE", False, False)
                worker.update("RUNNING", None, True)
                for _ in range(100):
                    tray.update("WORKER_FAILED", True, False)
                    worker.update("RUNNING", None, True)
                    self.assertEqual(json.loads(path.read_text())["session_id"], worker.session_id)
                tray.close()
                self.assertTrue(path.exists())
                worker.close()
                tray.update("RUNNING", True, True)
                self.assertEqual(json.loads(path.read_text())["session_id"], tray.session_id)
            finally:
                tray.close()
                worker.close()

    def test_concurrent_publishers_keep_one_live_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime-status.json"
            publishers = [RuntimeStatusPublisher(path, FINGERPRINT) for _ in range(2)]
            def publish(publisher):
                for _ in range(100):
                    publisher.update("RUNNING", True, True)
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(publish, publishers))
                current = json.loads(path.read_text())
                owner = next(p for p in publishers if p.session_id == current["session_id"])
                self.assertEqual(current["heartbeat_sequence"], 100)
                for publisher in publishers:
                    if publisher is not owner:
                        publisher.close()
                self.assertTrue(path.exists())
            finally:
                for publisher in publishers:
                    publisher.close()

    def test_example_is_valid(self):
        schema = json.loads((ROOT / "protocol/runtime-status.schema.json").read_text(encoding="utf-8"))
        example = json.loads((ROOT / "protocol/runtime-status.example.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(example)
