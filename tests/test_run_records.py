import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from realtime.run_records import (
    ARCHIVE_FORMAT,
    collector_id,
    export_run_records,
    find_run_files,
    unpack_collected,
)

STEAM_ID = "76561198000000001"


class RunRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.history = root / "steam"
        self.exchange = root / "exchange"
        for profile in ("profile1", "modded/profile1"):
            folder = self.history / STEAM_ID / profile / "saves" / "history"
            folder.mkdir(parents=True)
            (folder / "1782982719.run").write_text(json.dumps({"seed": profile}), encoding="utf-8")
        # Not a run file, or not in a history folder: left out.
        (self.history / STEAM_ID / "profile1" / "saves" / "current_run.save").write_text("{}", encoding="utf-8")
        (self.history / STEAM_ID / "profile1" / "saves" / "history" / "notes.txt").write_text("x", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def export(self):
        return export_run_records(
            Path(self.temp.name) / "out", self.exchange, "0.3.14-alpha.0",
            history_root=self.history, now=datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc),
        )

    def test_the_archive_has_every_run_and_no_steam_id(self):
        output, count = self.export()
        self.assertEqual(count, 2)
        with zipfile.ZipFile(output) as archive:
            names = sorted(archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
        self.assertEqual(names, [
            "manifest.json",
            "runs/modded/profile1/saves/history/1782982719.run",
            "runs/profile1/saves/history/1782982719.run",
        ])
        self.assertNotIn(STEAM_ID, output.name + "".join(names) + json.dumps(manifest))
        self.assertEqual((manifest["format"], manifest["runs"]), (ARCHIVE_FORMAT, 2))
        self.assertEqual(manifest["collector_id"], collector_id(self.exchange))   # stable

    def test_collected_archives_unpack_by_collector_and_twice_is_the_same(self):
        output, _ = self.export()
        target = Path(self.temp.name) / "collected"
        first = unpack_collected([output, output], target)
        self.assertEqual(len(set(first)), 2)
        collector = collector_id(self.exchange)
        self.assertTrue(all(path.relative_to(target).parts[0] == collector for path in first))
        self.assertEqual(json.loads(first[0].read_text(encoding="utf-8"))["seed"] in ("profile1", "modded/profile1"), True)

    def test_unknown_or_crafted_archives_write_nothing_outside(self):
        target = Path(self.temp.name) / "collected"
        crafted = Path(self.temp.name) / "crafted.zip"
        with zipfile.ZipFile(crafted, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"format": ARCHIVE_FORMAT, "collector_id": "a" * 32}))
            archive.writestr("runs/../../escape.run", "{}")
            archive.writestr("runs/C:/escape.run", "{}")
            archive.writestr("other/x.run", "{}")
            archive.writestr("runs/profile1/saves/history/ok.run", "{}")
        foreign = Path(self.temp.name) / "foreign.zip"
        with zipfile.ZipFile(foreign, "w") as archive:
            archive.writestr("runs/profile1/saves/history/x.run", "{}")    # no manifest
        written = unpack_collected([crafted, foreign], target)
        self.assertEqual([p.relative_to(target).as_posix() for p in written],
                         ["a" * 32 + "/profile1/saves/history/ok.run"])
        self.assertFalse((Path(self.temp.name) / "escape.run").exists())

    def test_no_game_folder_exports_an_empty_archive(self):
        self.assertEqual(find_run_files(Path(self.temp.name) / "missing"), [])


if __name__ == "__main__":
    unittest.main()
