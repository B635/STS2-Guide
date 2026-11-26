"""One-click local post-run HTML review discovery, generation, and opening.

Strictly read-only against the game's ``saves/history/*.run`` folder: it reuses
``realtime.run_records.default_history_root`` and ``find_run_files`` to locate
the newest finished single-player ``.run`` file, renders a self-contained HTML
report locally (zero network by default), atomically writes it under
``<exchange_dir>/reviews/STS2-赛后复盘-<run_id>.html``, and opens it with the
system browser/handler.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from realtime.file_bridge import default_exchange_dir
from realtime.run_records import default_history_root, find_run_files

LOGGER = logging.getLogger("sts2-guide")

REVIEWS_SUBDIR = "reviews"
NO_RUNS_MESSAGE = "尚未发现已结束的对局记录"
NO_REVIEWABLE_RUN_MESSAGE = "未找到可复盘的完整单人对局（最近记录可能为中途放弃、多人对局或旧版无法回放）"
BUSY_MESSAGE = "正在生成赛后复盘报告，请稍候…"
ERROR_MESSAGE = "生成赛后复盘失败，详情请查看日志目录。"

_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_MAX_CANDIDATES_SCAN = 25
_MAX_REPLAY_ATTEMPTS = 5


@dataclass(frozen=True)
class PostRunReviewResult:
    ok: bool
    status: str  # "opened" | "no_runs" | "unreviewable" | "busy" | "error"
    message: str
    run_path: Optional[Path] = None
    report_path: Optional[Path] = None
    run_id: Optional[str] = None
    elapsed_ms: float = 0.0


def _filename_timestamp(path: Path) -> int:
    stem = path.stem.strip()
    if stem.isdigit():
        try:
            return int(stem)
        except ValueError:
            return -1
    return -1


def _sanitize_run_id(raw: str) -> str:
    cleaned = _UNSAFE_FILENAME_RE.sub("_", str(raw or "").strip())
    cleaned = cleaned.replace("..", "_").strip(". ")
    if not cleaned:
        return "latest"
    return cleaned[:80]


def review_output_path(exchange_dir: Path, run_id: str) -> Path:
    safe_id = _sanitize_run_id(run_id)
    return Path(exchange_dir) / REVIEWS_SUBDIR / f"STS2-赛后复盘-{safe_id}.html"


def list_recent_run_files(history_root: Optional[Path] = None) -> List[Path]:
    """Return all ``.run`` files under ``history_root`` sorted newest-first."""
    root = Path(history_root) if history_root is not None else default_history_root()
    discovered = find_run_files(root)
    ranked: List[Tuple[int, int, int, str, Path]] = []
    for path, _archive_name in discovered:
        try:
            stat = path.stat()
        except OSError:
            continue
        ranked.append((
            int(stat.st_mtime),
            _filename_timestamp(path),
            int(stat.st_mtime_ns),
            path.as_posix(),
            path,
        ))
    ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return [item[4] for item in ranked]


def _quick_reviewable_check(path: Path) -> Tuple[bool, Optional[str]]:
    """Cheap read-only check before invoking full replay/scoring."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return False, "unreadable"
    if not isinstance(raw, dict):
        return False, "invalid_payload"
    if raw.get("was_abandoned"):
        return False, "abandoned"
    players = raw.get("players")
    if not isinstance(players, list) or len(players) != 1:
        return False, "multiplayer"
    return True, None


def find_latest_run_file(
    history_root: Optional[Path] = None,
    *,
    skip_unreviewable: bool = False,
) -> Optional[Path]:
    """Locate the newest ``.run`` file (pure read-only scan)."""
    recent = list_recent_run_files(history_root)
    if not recent:
        return None
    if not skip_unreviewable:
        return recent[0]
    for candidate in recent[:_MAX_CANDIDATES_SCAN]:
        ok, _reason = _quick_reviewable_check(candidate)
        if ok:
            return candidate
    return None


def _atomic_write_html(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        deadline = time.monotonic() + 1.0
        last_exc: Optional[PermissionError] = None
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as exc:
                last_exc = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def open_report_file(path: Path) -> None:
    """Open the generated local HTML report in the user's default browser."""
    resolved = path.resolve()
    if os.name == "nt" and hasattr(os, "startfile"):
        try:
            os.startfile(str(resolved))  # type: ignore[attr-defined]
            return
        except OSError:
            pass
    webbrowser.open(resolved.as_uri())


def _maybe_launch_optional_coach_cli(run_path: Path, report_path: Path) -> bool:
    """Opt-in non-blocking hook for an external AI Coach CLI.

    Disabled by default to preserve the real-time host's zero-network boundary.
    Only activates when both ``STS2_GUIDE_ENABLE_AI_COACH=1`` and
    ``STS2_GUIDE_COACH_CMD`` are explicitly set in the environment.
    """
    enabled = os.getenv("STS2_GUIDE_ENABLE_AI_COACH", "").strip().lower()
    if enabled not in ("1", "true", "yes"):
        return False
    raw_cmd = os.getenv("STS2_GUIDE_COACH_CMD", "").strip()
    if not raw_cmd:
        return False
    try:
        argv = shlex.split(raw_cmd, posix=(os.name != "nt"))
        if not argv:
            return False
        coach_out = report_path.with_name(f"{report_path.stem}-coach.html")
        argv.extend([str(run_path), "-o", str(coach_out)])
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        LOGGER.info("Launched optional AI Coach CLI in background for %s", run_path.name)
        return True
    except Exception:
        LOGGER.exception("Optional AI Coach CLI launch failed; local HTML report remains unaffected.")
        return False


def _emit_notice(notify: Optional[Callable[[str], None]], message: str) -> None:
    if notify is None:
        return
    try:
        notify(message)
    except Exception:
        LOGGER.exception("Tray notification callback failed.")


def generate_and_open_latest_review(
    *,
    exchange_dir: Optional[Path] = None,
    history_root: Optional[Path] = None,
    engine: Any = None,
    opener: Optional[Callable[[Path], None]] = None,
    notify: Optional[Callable[[str], None]] = None,
    allow_optional_coach: bool = True,
) -> PostRunReviewResult:
    """Find the newest finished ``.run`` file, render HTML atomically, and open it."""
    started = time.perf_counter()
    target_exchange = Path(exchange_dir) if exchange_dir is not None else default_exchange_dir()
    try:
        recent = list_recent_run_files(history_root)
        if not recent:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
            LOGGER.info("Post-run review skipped: no .run files found.")
            _emit_notice(notify, NO_RUNS_MESSAGE)
            return PostRunReviewResult(
                ok=False,
                status="no_runs",
                message=NO_RUNS_MESSAGE,
                elapsed_ms=elapsed_ms,
            )

        # Import review renderer lazily so realtime tray startup stays lean.
        from review.html_report import render_run_review_html
        from review.run_review import ReviewError

        selected_path: Optional[Path] = None
        html_doc: Optional[str] = None
        last_reason: Optional[str] = None
        replay_attempts = 0

        for candidate in recent[:_MAX_CANDIDATES_SCAN]:
            quick_ok, quick_reason = _quick_reviewable_check(candidate)
            if not quick_ok:
                if last_reason is None:
                    last_reason = quick_reason
                continue
            replay_attempts += 1
            try:
                html_doc = render_run_review_html(candidate, engine=engine)
                selected_path = candidate
                break
            except ReviewError as exc:
                last_reason = str(exc)
                LOGGER.warning(
                    "Skipping unreviewable run %s: %s",
                    candidate.name,
                    exc,
                )
                if replay_attempts >= _MAX_REPLAY_ATTEMPTS:
                    break

        if selected_path is None or html_doc is None:
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
            LOGGER.warning(
                "Post-run review found %d run files but none were reviewable (last_reason=%s).",
                len(recent),
                last_reason,
            )
            _emit_notice(notify, NO_REVIEWABLE_RUN_MESSAGE)
            return PostRunReviewResult(
                ok=False,
                status="unreviewable",
                message=NO_REVIEWABLE_RUN_MESSAGE,
                elapsed_ms=elapsed_ms,
            )

        run_id = selected_path.stem
        out_path = review_output_path(target_exchange, run_id)
        _atomic_write_html(out_path, html_doc)
        (opener or open_report_file)(out_path)

        if allow_optional_coach:
            _maybe_launch_optional_coach_cli(selected_path, out_path)

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        msg = f"已打开最近一局赛后复盘（{run_id}）"
        LOGGER.info(
            "Generated post-run HTML review run_id=%s path=%s elapsed_ms=%.1f",
            run_id,
            out_path,
            elapsed_ms,
        )
        _emit_notice(notify, msg)
        return PostRunReviewResult(
            ok=True,
            status="opened",
            message=msg,
            run_path=selected_path,
            report_path=out_path,
            run_id=run_id,
            elapsed_ms=elapsed_ms,
        )
    except Exception:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        LOGGER.exception("Failed to generate or open post-run HTML review.")
        _emit_notice(notify, ERROR_MESSAGE)
        return PostRunReviewResult(
            ok=False,
            status="error",
            message=ERROR_MESSAGE,
            elapsed_ms=elapsed_ms,
        )


class PostRunReviewAction:
    """Non-blocking, concurrency-safe tray action for one-click post-run review."""

    def __init__(
        self,
        *,
        exchange_dir: Optional[Path] = None,
        history_root: Optional[Path] = None,
        engine: Any = None,
        opener: Optional[Callable[[Path], None]] = None,
        notify: Optional[Callable[[str], None]] = None,
        background: bool = True,
        allow_optional_coach: bool = True,
    ) -> None:
        self.exchange_dir = exchange_dir
        self.history_root = history_root
        self.engine = engine
        self.opener = opener
        self.notify = notify
        self.background = bool(background)
        self.allow_optional_coach = bool(allow_optional_coach)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.last_result: Optional[PostRunReviewResult] = None

    def run_sync(self) -> PostRunReviewResult:
        if not self._lock.acquire(blocking=False):
            _emit_notice(self.notify, BUSY_MESSAGE)
            return PostRunReviewResult(ok=False, status="busy", message=BUSY_MESSAGE)
        try:
            result = generate_and_open_latest_review(
                exchange_dir=self.exchange_dir,
                history_root=self.history_root,
                engine=self.engine,
                opener=self.opener,
                notify=self.notify,
                allow_optional_coach=self.allow_optional_coach,
            )
            self.last_result = result
            return result
        finally:
            self._lock.release()

    def __call__(self) -> None:
        if not self.background:
            self.run_sync()
            return
        if not self._lock.acquire(blocking=False):
            _emit_notice(self.notify, BUSY_MESSAGE)
            return

        def _worker() -> None:
            try:
                result = generate_and_open_latest_review(
                    exchange_dir=self.exchange_dir,
                    history_root=self.history_root,
                    engine=self.engine,
                    opener=self.opener,
                    notify=self.notify,
                    allow_optional_coach=self.allow_optional_coach,
                )
                self.last_result = result
            finally:
                self._lock.release()

        thread = threading.Thread(
            target=_worker,
            name="STS2GuidePostRunReview",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def wait(self, timeout: Optional[float] = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()
