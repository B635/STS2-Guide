"""Windows tray controller and the one realtime Worker it owns.

This module is intentionally independent from ``realtime.host``.  The
production shell supplies one cooperative in-process Worker while lifecycle
and failure semantics remain unit-testable here.
"""
from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping, Protocol

from realtime.windows_runtime import NamedInstanceCoordinator


LOGGER = logging.getLogger("sts2-guide")

_STATE_LABELS = {
    "IDLE": "待机",
    "CHECKING_COMPATIBILITY": "正在核验兼容性",
    "RUNNING": "推荐服务运行中",
    "DRAINING": "正在停止推荐服务",
    "INCOMPATIBLE": "不兼容（已停止推荐）",
    "WORKER_FAILED": "推荐服务异常",
    "EXITING": "正在退出",
}


class ControllerState(str, Enum):
    IDLE = "IDLE"
    CHECKING_COMPATIBILITY = "CHECKING_COMPATIBILITY"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    INCOMPATIBLE = "INCOMPATIBLE"
    WORKER_FAILED = "WORKER_FAILED"
    EXITING = "EXITING"


class StartDisposition(str, Enum):
    PRIMARY = "primary"
    ACTIVATED_EXISTING = "activated_existing"
    FAILED_CLOSED = "failed_closed"


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.compatible, bool):
            raise ValueError("compatible must be a boolean")
        if any(not isinstance(code, str) or not code.strip() for code in self.reason_codes):
            raise ValueError("compatibility reason codes must be non-empty strings")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("compatibility reason codes must be unique")
        if self.compatible and self.reason_codes:
            raise ValueError("compatible result cannot contain blocking reasons")


@dataclass(frozen=True)
class ControllerSnapshot:
    state: ControllerState
    game_running: bool
    worker_running: bool
    compatibility_reasons: tuple[str, ...]
    detail: str | None
    activation_count: int


class GameDetector(Protocol):
    def is_running(self) -> bool: ...


class CompatibilityChecker(Protocol):
    def check(self) -> CompatibilityResult: ...


class Worker(Protocol):
    def start(self) -> None: ...

    def is_running(self) -> bool: ...

    def stop(self, timeout_seconds: float) -> None: ...


class TrayBackend(Protocol):
    def start(self, request_exit: Callable[[], None]) -> None: ...

    def update(self, snapshot: ControllerSnapshot) -> None: ...

    def activate(self) -> None: ...

    def stop(self) -> None: ...


class ThreadWorker:
    """One cooperative in-process Worker with an explicit stop contract."""

    def __init__(
        self,
        run_once: Callable[[], object],
        *,
        poll_interval: float = 0.1,
        on_stopped: Callable[[], None] | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("worker poll interval must be positive")
        self._run_once = run_once
        self._poll_interval = float(poll_interval)
        self._on_stopped = on_stopped
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._released = False

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("worker cannot be started more than once")
        self._thread = threading.Thread(
            target=self._run,
            name="STS2GuideWorker",
            daemon=False,
        )
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._run_once()
                self._stop.wait(self._poll_interval)
        except Exception:
            LOGGER.exception("Realtime Worker stopped unexpectedly.")
        finally:
            self._release()

    def _release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._on_stopped is not None:
            try:
                self._on_stopped()
            except Exception:
                LOGGER.exception("Realtime Worker cleanup failed.")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout_seconds: float) -> None:
        thread = self._thread
        if thread is None:
            self._release()
            return
        self._stop.set()
        thread.join(max(0.0, float(timeout_seconds)))
        if thread.is_alive():
            raise RuntimeError("worker thread did not stop within timeout")
        self._release()


class NullTrayBackend:
    """Test/development tray adapter with no graphical side effects."""

    def start(self, request_exit: Callable[[], None]) -> None:
        self.request_exit = request_exit

    def update(self, snapshot: ControllerSnapshot) -> None:
        self.snapshot = snapshot

    def activate(self) -> None:
        return

    def stop(self) -> None:
        return


class PystrayTrayBackend:
    """Optional pystray implementation; imports UI dependencies lazily."""

    def __init__(
        self,
        *,
        title: str = "STS2 Guide",
        open_logs: Callable[[], None] | None = None,
        export_diagnostics: Callable[[], None] | None = None,
        version_lines: Mapping[str, str] | None = None,
        icon_path: str | Path | None = None,
    ) -> None:
        self.title = title
        self._open_logs = open_logs
        self._export_diagnostics = export_diagnostics
        self._version_lines = tuple(
            (str(key), str(value))
            for key, value in (version_lines or {}).items()
        )
        self._icon_path = None if icon_path is None else Path(icon_path)
        self._icon = None
        self._snapshot = ControllerSnapshot(
            ControllerState.IDLE,
            False,
            False,
            (),
            None,
            0,
        )

    def start(self, request_exit: Callable[[], None]) -> None:
        try:
            import pystray
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise RuntimeError("pystray and Pillow are required for tray mode") from exc

        if self._icon_path is not None and self._icon_path.is_file():
            with Image.open(self._icon_path) as source:
                image = source.convert("RGBA").copy()
        else:
            image = Image.new("RGBA", (64, 64), (20, 28, 38, 255))
            draw = ImageDraw.Draw(image)
            draw.polygon(
                ((32, 5), (55, 53), (9, 53)),
                fill=(234, 145, 45, 255),
            )

        def status_text(_item=None) -> str:
            value = self._snapshot.state.value
            return f"状态：{_STATE_LABELS.get(value, value)}"

        def game_text(_item=None) -> str:
            return "游戏：已检测" if self._snapshot.game_running else "游戏：未运行"

        def worker_text(_item=None) -> str:
            return "Worker：运行中" if self._snapshot.worker_running else "Worker：已停止"

        def compatibility_text(_item=None) -> str:
            reasons = self._snapshot.compatibility_reasons
            if reasons:
                return "原因：" + ", ".join(reasons)
            if not self._snapshot.game_running:
                return "兼容：等待游戏"
            if self._snapshot.state == ControllerState.CHECKING_COMPATIBILITY:
                return "兼容：正在核验"
            return "兼容：通过"

        def invoke(callback: Callable[[], None] | None):
            def handler(_icon, _item) -> None:
                if callback is not None:
                    callback()
            return handler

        def exit_handler(_icon, _item) -> None:
            request_exit()

        menu_items = [
            pystray.MenuItem(
                status_text,
                lambda _icon, _item: None,
                enabled=False,
            ),
            pystray.MenuItem(game_text, lambda _icon, _item: None, enabled=False),
            pystray.MenuItem(worker_text, lambda _icon, _item: None, enabled=False),
            pystray.MenuItem(
                compatibility_text,
                lambda _icon, _item: None,
                enabled=False,
            ),
            *(
                pystray.MenuItem(
                    f"{key}: {value}",
                    lambda _icon, _item: None,
                    enabled=False,
                )
                for key, value in self._version_lines
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("打开日志目录", invoke(self._open_logs)),
            pystray.MenuItem("导出脱敏诊断", invoke(self._export_diagnostics)),
            pystray.MenuItem("完全退出", exit_handler),
        ]
        menu = pystray.Menu(*menu_items)
        self._icon = pystray.Icon("sts2-guide", image, self.title, menu)
        self._icon.run_detached()

    def update(self, snapshot: ControllerSnapshot) -> None:
        self._snapshot = snapshot
        if self._icon is not None:
            state = snapshot.state.value
            self._icon.title = (
                f"{self.title} — {_STATE_LABELS.get(state, state)}"
            )
            self._icon.update_menu()

    def activate(self) -> None:
        # pystray has no portable "open tray flyout" primitive. Refreshing
        # title/menu provides an observable wake-up without creating a window.
        if self._icon is not None:
            self._icon.update_menu()
            try:
                self._icon.notify("STS2 Guide 已在后台运行。", self.title)
            except (NotImplementedError, OSError):
                pass

    def stop(self) -> None:
        if self._icon is not None:
            self._icon.stop()
            self._icon = None


class WindowsGuideController:
    """Deterministic owner of one tray instance and at most one Worker."""

    def __init__(
        self,
        *,
        instances: NamedInstanceCoordinator,
        game_detector: GameDetector,
        compatibility: CompatibilityChecker,
        worker_factory: Callable[[], Worker],
        tray: TrayBackend,
        clear_advice: Callable[[], None] = lambda: None,
        initial_blocking_reasons: tuple[str, ...] = (),
        drain_seconds: float = 2.0,
        worker_stop_timeout: float = 1.0,
        worker_retry_seconds: float = 5.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if (
            drain_seconds < 0
            or worker_stop_timeout < 0
            or worker_retry_seconds < 0
        ):
            raise ValueError("controller timeouts cannot be negative")
        self.instances = instances
        self.game_detector = game_detector
        self.compatibility = compatibility
        self.worker_factory = worker_factory
        self.tray = tray
        self.clear_advice = clear_advice
        self.drain_seconds = float(drain_seconds)
        self.worker_stop_timeout = float(worker_stop_timeout)
        self.worker_retry_seconds = float(worker_retry_seconds)
        self._monotonic = monotonic
        self._sleep = sleep
        self._initial_blocking_reasons = tuple(initial_blocking_reasons)
        self.state = (
            ControllerState.INCOMPATIBLE
            if self._initial_blocking_reasons
            else ControllerState.IDLE
        )
        self._worker: Worker | None = None
        self._started = False
        self._primary = False
        self._exit_requested = threading.Event()
        self._drain_deadline: float | None = None
        self._retry_deadline: float | None = None
        self._game_running = False
        self._recheck_requested = False
        self._reasons: tuple[str, ...] = self._initial_blocking_reasons
        self._detail: str | None = None
        self._activation_count = 0

    @property
    def worker(self) -> Worker | None:
        return self._worker

    def snapshot(self) -> ControllerSnapshot:
        return ControllerSnapshot(
            state=self.state,
            game_running=self._game_running,
            worker_running=self._worker_is_running(),
            compatibility_reasons=self._reasons,
            detail=self._detail,
            activation_count=self._activation_count,
        )

    def start(self) -> StartDisposition:
        if self._started:
            raise RuntimeError("controller was already started")
        self._started = True
        try:
            primary = self.instances.acquire_or_activate()
        except Exception as exc:
            self._fail_closed("windows_instance_api_unavailable", exc)
            return StartDisposition.FAILED_CLOSED
        if not primary:
            self.state = ControllerState.EXITING
            return StartDisposition.ACTIVATED_EXISTING
        self._primary = True
        try:
            self.tray.start(self.request_exit)
        except Exception as exc:
            self._fail_closed("tray_initialization_failed", exc)
            self._release_primary()
            return StartDisposition.FAILED_CLOSED
        if not self._clear_advice_or_fail():
            self._publish()
            return StartDisposition.PRIMARY
        self._publish()
        return StartDisposition.PRIMARY

    def request_exit(self) -> None:
        self._exit_requested.set()

    def tick(self) -> ControllerSnapshot:
        if not self._started:
            raise RuntimeError("controller must be started before polling")
        if not self._primary:
            return self.snapshot()
        try:
            if self.instances.consume_shutdown():
                self.request_exit()
        except Exception as exc:
            self._stop_worker()
            self._fail_closed("windows_shutdown_api_failed", exc)
            self._publish()
            return self.snapshot()
        if self._exit_requested.is_set():
            self.shutdown()
            return self.snapshot()

        try:
            if self.instances.consume_activation():
                self._activation_count += 1
                self._recheck_requested = True
                self.tray.activate()
        except Exception as exc:
            self._stop_worker()
            self._fail_closed("windows_activation_api_failed", exc)
            self._publish()
            return self.snapshot()

        try:
            self._game_running = bool(self.game_detector.is_running())
        except Exception as exc:
            self._stop_worker()
            self._fail_closed("windows_process_api_failed", exc)
            self._publish()
            return self.snapshot()

        if self._initial_blocking_reasons:
            self.state = ControllerState.INCOMPATIBLE
            self._reasons = self._initial_blocking_reasons
            self._detail = None
            self._publish()
            return self.snapshot()

        if self.state == ControllerState.IDLE and self._game_running:
            self.state = ControllerState.CHECKING_COMPATIBILITY

        if self.state == ControllerState.CHECKING_COMPATIBILITY:
            if not self._game_running:
                self._become_idle()
            else:
                self._check_and_start_worker()

        elif self.state == ControllerState.RUNNING:
            if not self._worker_is_running():
                self._worker = None
                if self._clear_advice_or_fail():
                    self.state = ControllerState.WORKER_FAILED
                    self._reasons = ("worker_exited_unexpectedly",)
                    self._detail = "worker_exited_unexpectedly"
                    self._retry_deadline = (
                        self._monotonic() + self.worker_retry_seconds
                    )
            elif not self._game_running:
                if self._clear_advice_or_fail():
                    self.state = ControllerState.DRAINING
                    self._drain_deadline = self._monotonic() + self.drain_seconds

        elif self.state == ControllerState.DRAINING:
            if self._game_running:
                if self._worker_is_running():
                    self.state = ControllerState.RUNNING
                    self._drain_deadline = None
                else:
                    self._worker = None
                    if self._clear_advice_or_fail():
                        self.state = ControllerState.WORKER_FAILED
                        self._reasons = ("worker_exited_while_draining",)
                        self._detail = "worker_exited_while_draining"
                        self._retry_deadline = (
                            self._monotonic() + self.worker_retry_seconds
                        )
            elif not self._worker_is_running():
                self._worker = None
                self._become_idle()
            elif (
                self._drain_deadline is not None
                and self._monotonic() >= self._drain_deadline
            ):
                self._stop_worker()
                self._become_idle()

        elif self.state in {
            ControllerState.INCOMPATIBLE,
            ControllerState.WORKER_FAILED,
        }:
            if not self._game_running:
                self._become_idle()
            elif self._recheck_requested or (
                self.state == ControllerState.WORKER_FAILED
                and self._retry_deadline is not None
                and self._monotonic() >= self._retry_deadline
            ):
                self._recheck_requested = False
                self._retry_deadline = None
                self.state = ControllerState.CHECKING_COMPATIBILITY
                self._check_and_start_worker()

        self._publish()
        return self.snapshot()

    def run(self, poll_interval: float = 0.25) -> int:
        disposition = self.start()
        if disposition == StartDisposition.ACTIVATED_EXISTING:
            return 0
        if disposition == StartDisposition.FAILED_CLOSED:
            return 2
        try:
            while self.state != ControllerState.EXITING:
                self.tick()
                if self.state != ControllerState.EXITING:
                    self._sleep(max(0.05, float(poll_interval)))
            return 0
        finally:
            if self._primary:
                self.shutdown()

    def shutdown(self) -> None:
        if self.state == ControllerState.EXITING and not self._primary:
            return
        self.state = ControllerState.EXITING
        self._drain_deadline = None
        self._retry_deadline = None
        self._clear_advice_or_fail()
        try:
            self._stop_worker()
        except Exception as exc:
            # A non-daemon Worker still owns the process.  Keep the tray and
            # mutex alive so the user sees a recoverable failure instead of a
            # half-exited controller with an orphaned execution thread.
            self.state = ControllerState.WORKER_FAILED
            self._reasons = ("worker_stop_timeout",)
            self._detail = f"{type(exc).__name__}: {exc}"
            self._exit_requested.clear()
            self._publish()
            return
        try:
            self.tray.update(self.snapshot())
        finally:
            try:
                self.tray.stop()
            finally:
                self._release_primary()

    def _check_and_start_worker(self) -> None:
        try:
            result = self.compatibility.check()
        except Exception as exc:
            self._fail_closed("compatibility_check_failed", exc)
            return
        if not isinstance(result, CompatibilityResult):
            self._fail_closed(
                "compatibility_check_invalid",
                TypeError("compatibility checker returned an invalid result"),
            )
            return
        if not result.compatible:
            if not self._clear_advice_or_fail():
                return
            self._reasons = result.reason_codes or ("incompatible",)
            self._detail = None
            self.state = ControllerState.INCOMPATIBLE
            return
        self._reasons = ()
        self._detail = None
        if self._worker is not None:
            if self._worker_is_running():
                self.state = ControllerState.RUNNING
                return
            self._worker = None
        try:
            worker = self.worker_factory()
            worker.start()
            if not worker.is_running():
                raise RuntimeError("worker did not remain running after start")
        except Exception as exc:
            LOGGER.exception("Realtime Worker initialization failed.")
            try:
                if "worker" in locals():
                    worker.stop(0.0)
            except Exception:
                pass
            self._worker = None
            if not self._clear_advice_or_fail():
                return
            self.state = ControllerState.WORKER_FAILED
            self._reasons = ("worker_start_failed",)
            self._detail = f"worker_start_failed:{type(exc).__name__}"
            self._retry_deadline = (
                self._monotonic() + self.worker_retry_seconds
            )
            return
        self._worker = worker
        self.state = ControllerState.RUNNING
        self._reasons = ()
        self._drain_deadline = None
        self._retry_deadline = None

    def _worker_is_running(self) -> bool:
        if self._worker is None:
            return False
        try:
            return bool(self._worker.is_running())
        except Exception:
            return False

    def _stop_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        worker.stop(self.worker_stop_timeout)
        if worker.is_running():
            raise RuntimeError("worker remained alive after stop")
        self._worker = None

    def _become_idle(self) -> None:
        if not self._clear_advice_or_fail():
            return
        self.state = ControllerState.IDLE
        self._drain_deadline = None
        self._reasons = ()
        self._detail = None
        self._recheck_requested = False
        self._retry_deadline = None

    def _fail_closed(self, reason: str, exc: Exception) -> None:
        self._clear_advice_or_fail(record_failure=False)
        self.state = ControllerState.INCOMPATIBLE
        self._reasons = (reason,)
        self._detail = f"{type(exc).__name__}: {exc}"
        self._retry_deadline = None

    def _clear_advice_or_fail(self, *, record_failure: bool = True) -> bool:
        """Remove every visible recommendation before leaving RUNNING.

        Clearing is deliberately owned by the controller rather than only by
        the compatibility checker: game exit, Worker failure and complete
        application exit are all fail-closed transitions too.
        """

        try:
            self.clear_advice()
        except Exception as exc:
            LOGGER.exception("Visible advice could not be cleared.")
            if record_failure:
                self.state = ControllerState.INCOMPATIBLE
                self._reasons = ("advice_clear_failed",)
                self._detail = f"{type(exc).__name__}: {exc}"
                self._retry_deadline = None
            return False
        return True

    def _publish(self) -> None:
        self.tray.update(self.snapshot())

    def _release_primary(self) -> None:
        if not self._primary:
            return
        self._primary = False
        self.instances.close()
