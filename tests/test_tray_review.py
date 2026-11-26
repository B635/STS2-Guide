from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from realtime.controller import PystrayTrayBackend
from realtime.installation import DiscoveryResult
from realtime.post_run_review import (
    BUSY_MESSAGE,
    ERROR_MESSAGE,
    NO_REVIEWABLE_RUN_MESSAGE,
    NO_RUNS_MESSAGE,
    PostRunReviewAction,
    find_latest_run_file,
    generate_and_open_latest_review,
    list_recent_run_files,
    review_output_path,
)
from realtime.public_beta import build_public_beta_controller
from review import html_report
from review.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
DEV_RUN_SHOP = ROOT / "coach" / "eval" / "runs" / "dev" / "1773034032.run"
DEV_RUN_ONE_CARD = ROOT / "coach" / "eval" / "runs" / "dev" / "1773317604.run"
DEV_RUN_WIN = ROOT / "coach" / "eval" / "runs" / "dev" / "1772816067.run"
UNSCORED_SHOP_RUN = ROOT / "coach" / "eval" / "runs" / "test" / "1791474137.run"
COLLECTED_ROOT = ROOT / "build" / "collected-runs"
STEAM_ID = "76561198000000042"


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_fake_pystray():
    module = types.ModuleType("pystray")

    class FakeMenuItem:
        def __init__(
            self,
            text,
            action,
            checked=None,
            radio=False,
            default=False,
            visible=True,
            enabled=True,
        ):
            self.text = text
            self.action = action
            self.visible = visible
            self.enabled = enabled

        def resolve_text(self) -> str:
            return self.text(self) if callable(self.text) else str(self.text)

        def is_visible(self) -> bool:
            return bool(self.visible(self)) if callable(self.visible) else bool(self.visible)

    class FakeMenu:
        SEPARATOR = object()

        def __init__(self, *items):
            self.items = list(items)

    class FakeIcon:
        def __init__(self, name, icon, title, menu):
            self.name = name
            self.icon = icon
            self.title = title
            self.menu = menu
            self.started = False
            self.stopped = False
            self.updated = 0
            self.notifications = []

        def run_detached(self):
            self.started = True

        def update_menu(self):
            self.updated += 1

        def notify(self, message, title=None):
            self.notifications.append((message, title))

        def stop(self):
            self.stopped = True

    module.MenuItem = FakeMenuItem
    module.Menu = FakeMenu
    module.Icon = FakeIcon
    return module


class HtmlReportEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.engine = Engine(Path(cls.temp.name) / "release.db")

    @classmethod
    def tearDownClass(cls):
        del cls.engine
        cls.temp.cleanup()

    def test_dev_run_renders_complete_html_with_6_svgs_tier1_whatif_and_faithful_shop(self):
        before_hash = _sha256_bytes(DEV_RUN_SHOP)
        doc = html_report.render_run_review_html(DEV_RUN_SHOP, engine=self.engine)
        after_hash = _sha256_bytes(DEV_RUN_SHOP)
        self.assertEqual(before_hash, after_hash)

        # Self-contained HTML with zero external CDN/network references.
        self.assertTrue(doc.startswith("<!DOCTYPE html>"))
        self.assertIn('<html lang="zh-CN">', doc)
        self.assertIn("</html>", doc)
        self.assertEqual(re.findall(r"https?://", doc), [])
        self.assertIn("prefers-color-scheme: dark", doc)
        self.assertIn('data-theme="dark"', doc)

        # Overview hero card facts.
        self.assertIn("铁甲战士", doc)
        self.assertIn("IRONCLAD", doc)
        self.assertIn("进阶 2", doc)
        self.assertIn("v0.98.2", doc)
        self.assertIn("知识恶魔", doc)
        self.assertIn("到达第 2 幕", doc)
        self.assertIn("12 张", doc)
        self.assertIn("商店“买或离开”一致性", doc)
        self.assertIn("同为消费但具体项不同", doc)

        # All 6 capability SVG charts from deck_trend.
        self.assertEqual(doc.count('<svg class="dim-svg"'), 6)
        for dim in ("frontload", "scaling", "aoe", "defense", "cycle", "energy"):
            self.assertIn(f'data-dim="{dim}"', doc)
        self.assertIn("<polyline", doc)
        self.assertIn("<circle", doc)

        # Tier-1 & Tier-2 disagreement cards + deterministic what_if probes.
        for fid in ("F14-card", "F08-card", "F06-card", "F02-card", "F12-card", "F16-rest"):
            self.assertIn(f'id="decision-{fid}"', doc)
        self.assertGreaterEqual(doc.count('class="subblock whatif-box"'), 3)

        # Faithful shop display: F15-shop bought removal only while engine top is OFFERING (祭品).
        f15_match = re.search(
            r'<tr[^>]*data-kind="shop"[^>]*>.*?F15.*?</tr>',
            doc,
            re.DOTALL,
        )
        self.assertIsNotNone(f15_match)
        f15_row = f15_match.group(0)
        self.assertIn('<td class="choice-cell">删牌</td>', f15_row)
        self.assertIn("<td>祭品</td>", f15_row)
        self.assertIn("一致（均选择消费，具体购买项不同）", f15_row)
        self.assertNotIn('<td class="choice-cell">祭品</td>', f15_row)

        # F05-shop bought both SHRUG_IT_OFF (耸肩无视) and removal (删牌).
        f05_match = re.search(
            r'<tr[^>]*data-kind="shop"[^>]*>.*?F05.*?</tr>',
            doc,
            re.DOTALL,
        )
        self.assertIsNotNone(f05_match)
        self.assertIn("耸肩无视（共购：耸肩无视、删牌）", f05_match.group(0))

    def test_dev_single_card_reward_and_zero_tier1_run_handles_edge_cases(self):
        before_hash = _sha256_bytes(DEV_RUN_ONE_CARD)
        doc = html_report.render_run_review_html(DEV_RUN_ONE_CARD, engine=self.engine)
        self.assertEqual(before_hash, _sha256_bytes(DEV_RUN_ONE_CARD))
        self.assertEqual(doc.count('<svg class="dim-svg"'), 6)
        self.assertIn("静默猎手", doc)
        self.assertIn("本局未发现引擎高把握且分差明显的 Tier-1 分歧", doc)
        self.assertIn("status-skipped", doc)

    def test_dev_win_run_with_skipped_decisions_and_unscored_disclosure(self):
        before_hash = _sha256_bytes(DEV_RUN_WIN)
        doc = html_report.render_run_review_html(DEV_RUN_WIN, engine=self.engine)
        self.assertEqual(before_hash, _sha256_bytes(DEV_RUN_WIN))
        self.assertIn("胜利通关", doc)
        self.assertIn("VICTORY", doc)
        self.assertEqual(doc.count('<svg class="dim-svg"'), 6)
        self.assertIn("未计分效果诚实披露", doc)
        self.assertIn("Boss 不在当前数据目录中", doc)

    def test_unscored_shop_purchase_is_not_misreported_as_engine_top(self):
        before_hash = _sha256_bytes(UNSCORED_SHOP_RUN)
        doc = html_report.render_run_review_html(UNSCORED_SHOP_RUN, engine=self.engine)
        self.assertEqual(before_hash, _sha256_bytes(UNSCORED_SHOP_RUN))
        f03_match = re.search(
            r'<tr[^>]*data-kind="shop"[^>]*>.*?F03.*?</tr>',
            doc,
            re.DOTALL,
        )
        self.assertIsNotNone(f03_match)
        f03_row = f03_match.group(0)
        self.assertIn("已购买（1 项未计分商品）", f03_row)
        self.assertIn("<td>播种</td>", f03_row)
        self.assertNotIn('<td class="choice-cell">播种</td>', f03_row)

    def test_collected_single_player_run_renders_end_to_end_readonly(self):
        if not COLLECTED_ROOT.is_dir():
            self.skipTest("build/collected-runs not present")
        candidates = sorted(
            COLLECTED_ROOT.glob("**/saves/history/*.run"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else -1,
            reverse=True,
        )
        chosen = None
        for path in candidates:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not raw.get("was_abandoned") and len(raw.get("players") or ()) == 1:
                chosen = path
                break
        if chosen is None:
            self.skipTest("no finished single-player run in build/collected-runs")

        before_hash = _sha256_bytes(chosen)
        doc = html_report.render_run_review_html(chosen, engine=self.engine)
        self.assertEqual(before_hash, _sha256_bytes(chosen))
        self.assertTrue(doc.startswith("<!DOCTYPE html>"))
        self.assertEqual(doc.count('<svg class="dim-svg"'), 6)
        self.assertEqual(re.findall(r"https?://", doc), [])
        self.assertIn('id="decision-ledger"', doc)

    def test_cli_entrypoint_writes_html_atomically_and_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as tempdir:
            out_path = Path(tempdir) / "nested" / "report.html"
            with patch.object(html_report, "_get_default_engine", return_value=self.engine):
                code = html_report.main([str(DEV_RUN_SHOP), "-o", str(out_path)])
            self.assertEqual(code, 0)
            self.assertTrue(out_path.is_file())
            self.assertEqual(list(out_path.parent.glob("*.tmp")), [])
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("<!DOCTYPE html>", text)
            self.assertEqual(text.count('<svg class="dim-svg"'), 6)

            missing_code = html_report.main([str(Path(tempdir) / "does-not-exist.run")])
            self.assertEqual(missing_code, 1)


class PostRunReviewServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.engine = Engine(Path(cls.temp.name) / "release.db")

    @classmethod
    def tearDownClass(cls):
        del cls.engine
        cls.temp.cleanup()

    def _history_dir(self, root: Path, profile: str = "modded/profile1") -> Path:
        folder = root / "steam" / STEAM_ID / profile / "saves" / "history"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def test_find_latest_run_file_orders_by_mtime_and_timestamp_and_skips_unreviewable(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            history = root / "steam"
            p1 = self._history_dir(root, "profile1")
            p2 = self._history_dir(root, "modded/profile1")

            older = p1 / "1772000001.run"
            newer_same_sec = p2 / "1773034032.run"
            shutil.copy2(DEV_RUN_ONE_CARD, older)
            shutil.copy2(DEV_RUN_SHOP, newer_same_sec)
            os.utime(older, (1_700_000_000, 1_700_000_000))
            os.utime(newer_same_sec, (1_700_000_000, 1_700_000_000))
            self.assertEqual(find_latest_run_file(history), newer_same_sec)

            # Explicit newer mtime overrides filename timestamp.
            os.utime(older, (1_700_000_500, 1_700_000_500))
            self.assertEqual(find_latest_run_file(history), older)

            # Newer abandoned and multiplayer runs are skipped when skip_unreviewable=True.
            raw = json.loads(DEV_RUN_SHOP.read_text(encoding="utf-8"))
            abandoned = p2 / "1779999998.run"
            abandoned.write_text(json.dumps(dict(raw, was_abandoned=True)), encoding="utf-8")
            multiplayer = p2 / "1779999999.run"
            multiplayer.write_text(json.dumps(dict(raw, players=raw["players"] * 2)), encoding="utf-8")
            os.utime(abandoned, (1_700_001_000, 1_700_001_000))
            os.utime(multiplayer, (1_700_002_000, 1_700_002_000))

            self.assertEqual(list_recent_run_files(history)[0], multiplayer)
            self.assertEqual(find_latest_run_file(history, skip_unreviewable=False), multiplayer)
            self.assertEqual(find_latest_run_file(history, skip_unreviewable=True), older)

    def test_empty_or_missing_history_root_notifies_friendly_message(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            notices = []
            opened = []
            result = generate_and_open_latest_review(
                exchange_dir=root / "exchange",
                history_root=root / "missing-steam",
                engine=self.engine,
                opener=opened.append,
                notify=notices.append,
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.status, "no_runs")
            self.assertEqual(result.message, NO_RUNS_MESSAGE)
            self.assertEqual(notices, [NO_RUNS_MESSAGE])
            self.assertEqual(opened, [])

    def test_only_abandoned_runs_degrades_gracefully(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            folder = self._history_dir(root)
            raw = json.loads(DEV_RUN_SHOP.read_text(encoding="utf-8"))
            (folder / "1790000001.run").write_text(
                json.dumps(dict(raw, was_abandoned=True)),
                encoding="utf-8",
            )
            notices = []
            opened = []
            result = generate_and_open_latest_review(
                exchange_dir=root / "exchange",
                history_root=root / "steam",
                engine=self.engine,
                opener=opened.append,
                notify=notices.append,
            )
            self.assertFalse(result.ok)
            self.assertEqual(result.status, "unreviewable")
            self.assertEqual(result.message, NO_REVIEWABLE_RUN_MESSAGE)
            self.assertEqual(notices, [NO_REVIEWABLE_RUN_MESSAGE])
            self.assertEqual(opened, [])

    def test_generate_and_open_latest_review_writes_atomically_without_modifying_run(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            folder = self._history_dir(root)
            run_copy = folder / DEV_RUN_SHOP.name
            shutil.copy2(DEV_RUN_SHOP, run_copy)
            before_hash = _sha256_bytes(run_copy)
            before_files = sorted(p.name for p in folder.iterdir())

            exchange = root / "exchange"
            opened = []
            notices = []
            with patch("realtime.post_run_review.subprocess.Popen") as popen_mock:
                result = generate_and_open_latest_review(
                    exchange_dir=exchange,
                    history_root=root / "steam",
                    engine=self.engine,
                    opener=opened.append,
                    notify=notices.append,
                )

            self.assertTrue(result.ok)
            self.assertEqual(result.status, "opened")
            self.assertEqual(result.run_id, DEV_RUN_SHOP.stem)
            expected_out = review_output_path(exchange, DEV_RUN_SHOP.stem)
            self.assertEqual(result.report_path, expected_out)
            self.assertTrue(expected_out.is_file())
            self.assertEqual(list(expected_out.parent.glob("*.tmp")), [])
            self.assertEqual(opened, [expected_out])
            self.assertEqual(len(notices), 1)
            self.assertIn(DEV_RUN_SHOP.stem, notices[0])
            popen_mock.assert_not_called()

            # Game history folder & .run bytes remain strictly untouched.
            self.assertEqual(_sha256_bytes(run_copy), before_hash)
            self.assertEqual(sorted(p.name for p in folder.iterdir()), before_files)

            html_text = expected_out.read_text(encoding="utf-8")
            self.assertEqual(html_text.count('<svg class="dim-svg"'), 6)
            self.assertIn("铁甲战士", html_text)

    def test_optional_ai_coach_cli_only_runs_when_explicitly_enabled(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            folder = self._history_dir(root)
            run_copy = folder / DEV_RUN_ONE_CARD.name
            shutil.copy2(DEV_RUN_ONE_CARD, run_copy)
            exchange = root / "exchange"
            opened = []
            env = {
                "STS2_GUIDE_ENABLE_AI_COACH": "1",
                "STS2_GUIDE_COACH_CMD": "sts2-coach review",
            }
            with (
                patch.dict(os.environ, env, clear=False),
                patch("realtime.post_run_review.subprocess.Popen") as popen_mock,
            ):
                result = generate_and_open_latest_review(
                    exchange_dir=exchange,
                    history_root=root / "steam",
                    engine=self.engine,
                    opener=opened.append,
                )
            self.assertTrue(result.ok)
            self.assertEqual(len(opened), 1)
            popen_mock.assert_called_once()
            argv = popen_mock.call_args.args[0]
            self.assertEqual(argv[:2], ["sts2-coach", "review"])
            self.assertEqual(argv[2], str(run_copy))
            self.assertEqual(argv[3], "-o")
            self.assertTrue(argv[4].endswith("-coach.html"))

    def test_post_run_review_action_runs_async_guards_concurrency_and_isolates_errors(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            folder = self._history_dir(root)
            shutil.copy2(DEV_RUN_ONE_CARD, folder / DEV_RUN_ONE_CARD.name)
            opened = []
            notices = []
            action = PostRunReviewAction(
                exchange_dir=root / "exchange",
                history_root=root / "steam",
                engine=self.engine,
                opener=opened.append,
                notify=notices.append,
                background=True,
                allow_optional_coach=False,
            )
            action()
            self.assertTrue(action.wait(timeout=15.0))
            self.assertIsNotNone(action.last_result)
            self.assertTrue(action.last_result.ok)
            self.assertEqual(len(opened), 1)

            # Concurrency guard: while lock is held, clicking again notifies busy and returns immediately.
            busy_notices = []
            blocker = threading.Event()
            release = threading.Event()

            def slow_generate(**_kwargs):
                blocker.set()
                release.wait(timeout=5.0)
                return action.last_result

            guarded = PostRunReviewAction(
                exchange_dir=root / "exchange",
                history_root=root / "steam",
                notify=busy_notices.append,
                background=True,
            )
            with patch("realtime.post_run_review.generate_and_open_latest_review", side_effect=slow_generate):
                guarded()
                self.assertTrue(blocker.wait(timeout=2.0))
                guarded()
                release.set()
                self.assertTrue(guarded.wait(timeout=5.0))
            self.assertIn(BUSY_MESSAGE, busy_notices)

            # Unexpected error is caught, logged, and reported via notify without raising.
            err_notices = []
            with patch("review.html_report.render_run_review_html", side_effect=RuntimeError("boom")):
                err_res = generate_and_open_latest_review(
                    exchange_dir=root / "exchange",
                    history_root=root / "steam",
                    engine=self.engine,
                    notify=err_notices.append,
                )
            self.assertFalse(err_res.ok)
            self.assertEqual(err_res.status, "error")
            self.assertEqual(err_notices, [ERROR_MESSAGE])


class TrayMenuIntegrationTests(unittest.TestCase):
    def test_pystray_menu_mounts_review_item_next_to_export_and_handles_click_and_notify(self):
        fake_pystray = _make_fake_pystray()
        clicked = []

        def broken_action():
            raise RuntimeError("isolated tray error")

        tray = PystrayTrayBackend(
            open_logs=lambda: clicked.append("logs"),
            open_latest_review=lambda: clicked.append("review"),
            export_run_records=lambda: clicked.append("records"),
            export_diagnostics=broken_action,
        )
        with patch.dict(sys.modules, {"pystray": fake_pystray}):
            tray.start(lambda: clicked.append("exit"))
            self.assertIsNotNone(tray._icon)
            self.assertTrue(tray._icon.started)

            items = [item for item in tray._icon.menu.items if item is not fake_pystray.Menu.SEPARATOR]
            labels = [item.resolve_text() for item in items]
            self.assertIn("打开最近一局赛后复盘", labels)
            self.assertIn("导出对局记录", labels)
            self.assertIn("导出脱敏诊断", labels)
            review_idx = labels.index("打开最近一局赛后复盘")
            records_idx = labels.index("导出对局记录")
            self.assertEqual(abs(review_idx - records_idx), 1)

            # Click review menu item.
            review_item = items[review_idx]
            review_item.action(tray._icon, review_item)
            self.assertEqual(clicked, ["review"])

            # Failing callback is isolated and does not crash tray.
            diag_item = items[labels.index("导出脱敏诊断")]
            diag_item.action(tray._icon, diag_item)

            # Tray notify updates last_notice, calls icon.notify, and reveals notice status line.
            notice_items = [item for item in items if callable(item.visible)]
            self.assertEqual(len(notice_items), 1)
            self.assertFalse(notice_items[0].is_visible())

            tray.notify(NO_RUNS_MESSAGE)
            self.assertEqual(tray.last_notice, NO_RUNS_MESSAGE)
            self.assertEqual(tray._icon.notifications[-1], (NO_RUNS_MESSAGE, "STS2 Guide"))
            self.assertTrue(notice_items[0].is_visible())
            self.assertIn(NO_RUNS_MESSAGE, notice_items[0].resolve_text())
            tray.stop()

    def test_public_beta_controller_wires_post_run_review_menu_safely(self):
        fake_pystray = _make_fake_pystray()
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            args = SimpleNamespace(
                compatibility_manifest=ROOT / "packaging" / "compatibility.json",
                install_receipt=root / "missing-install.json",
                game_dir=None,
                output=root / "advice-event.json",
                log_file=root / "controller.log",
                worker_log_file=root / "worker.log",
                runtime_status=root / "runtime-status.json",
                checkpoint=root / "active-run.json",
                poll_interval=0.1,
                drain_seconds=2.0,
                worker_stop_timeout=1.0,
            )
            with (
                patch(
                    "realtime.public_beta.discover_game_directory",
                    return_value=DiscoveryResult(None, None, ("game_directory_not_found",)),
                ),
                patch("realtime.public_beta.CtypesKernelApi", return_value=object()),
            ):
                controller = build_public_beta_controller(args)

            graphical = controller.tray.backend
            self.assertIsInstance(graphical._open_latest_review, PostRunReviewAction)
            graphical._open_latest_review.history_root = root / "empty-steam"
            graphical._open_latest_review.background = False

            with patch.dict(sys.modules, {"pystray": fake_pystray}):
                graphical.start(lambda: None)
                items = [
                    item
                    for item in graphical._icon.menu.items
                    if item is not fake_pystray.Menu.SEPARATOR
                ]
                review_item = next(
                    item for item in items if item.resolve_text() == "打开最近一局赛后复盘"
                )
                review_item.action(graphical._icon, review_item)
                self.assertEqual(graphical.last_notice, NO_RUNS_MESSAGE)
                self.assertEqual(
                    graphical._icon.notifications[-1],
                    (NO_RUNS_MESSAGE, "STS2 Guide"),
                )
                graphical.stop()


if __name__ == "__main__":
    unittest.main()
