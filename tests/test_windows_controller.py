from __future__ import annotations

import subprocess
import sys
import threading
import unittest
import uuid

from realtime.controller import (
    CompatibilityResult,
    ControllerState,
    StartDisposition,
    ThreadWorker,
    WindowsGuideController,
)
from realtime.windows_runtime import (
    ExactGameProcessDetector,
    CtypesKernelApi,
    NamedInstanceCoordinator,
    ProcessInfo,
    WindowsRuntimeUnavailable,
    canonical_windows_path,
)


GAME_PATH = r"D:\SteamLibrary\steamapps\common\Slay the Spire 2\SlayTheSpire2.exe"


class FakeKernelApi:
    def __init__(self):
        self._next_handle = 1
        self._handles = {}
        self._objects = {}
        self.fail = False

    def _open(self, kind, name):
        key = (kind, name)
        already = key in self._objects
        obj = self._objects.setdefault(key, {"refs": 0, "signaled": False})
        obj["refs"] += 1
        handle = self._next_handle
        self._next_handle += 1
        self._handles[handle] = key
        return handle, already

    def create_mutex(self, name):
        if self.fail:
            raise WindowsRuntimeUnavailable("unavailable")
        return self._open("mutex", name)

    def create_event(self, name):
        if self.fail:
            raise WindowsRuntimeUnavailable("unavailable")
        return self._open("event", name)[0]

    def signal_event(self, handle):
        self._objects[self._handles[handle]]["signaled"] = True

    def event_is_signaled(self, handle):
        obj = self._objects[self._handles[handle]]
        result = obj["signaled"]
        obj["signaled"] = False
        return result

    def close_handle(self, handle):
        key = self._handles.pop(handle)
        obj = self._objects[key]
        obj["refs"] -= 1
        if obj["refs"] == 0:
            self._objects.pop(key)


class FakeProcesses:
    def __init__(self, processes=()):
        self.processes = list(processes)
        self.error = None

    def iter_processes(self):
        if self.error is not None:
            raise self.error
        return list(self.processes)


class FakeDetector:
    def __init__(self, running=False):
        self.running = running
        self.error = None

    def is_running(self):
        if self.error is not None:
            raise self.error
        return self.running


class FakeCompatibility:
    def __init__(self, result=None):
        self.result = result or CompatibilityResult(True)
        self.calls = 0
        self.error = None

    def check(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class FakeWorker:
    def __init__(self, *, fail_start=False):
        self.running = False
        self.fail_start = fail_start
        self.starts = 0
        self.stops = []
        self.fail_stop = False

    def start(self):
        self.starts += 1
        if self.fail_start:
            raise RuntimeError("boom")
        self.running = True

    def is_running(self):
        return self.running

    def stop(self, timeout_seconds):
        self.stops.append(timeout_seconds)
        if self.fail_stop:
            raise RuntimeError("worker is busy")
        self.running = False


class WorkerFactory:
    def __init__(self):
        self.workers = []
        self.fail_next = False

    def __call__(self):
        worker = FakeWorker(fail_start=self.fail_next)
        self.fail_next = False
        self.workers.append(worker)
        return worker


class FakeTray:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.activations = 0
        self.snapshots = []
        self.request_exit = None

    def start(self, request_exit):
        self.started = True
        self.request_exit = request_exit

    def update(self, snapshot):
        self.snapshots.append(snapshot)

    def activate(self):
        self.activations += 1

    def stop(self):
        self.stopped = True


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def make_controller(
    *,
    api=None,
    detector=None,
    compatibility=None,
    factory=None,
    tray=None,
    clock=None,
    clear_advice=lambda: None,
    initial_blocking_reasons=(),
):
    api = api or FakeKernelApi()
    detector = detector or FakeDetector()
    compatibility = compatibility or FakeCompatibility()
    factory = factory or WorkerFactory()
    tray = tray or FakeTray()
    clock = clock or Clock()
    controller = WindowsGuideController(
        instances=NamedInstanceCoordinator(api),
        game_detector=detector,
        compatibility=compatibility,
        worker_factory=factory,
        tray=tray,
        clear_advice=clear_advice,
        initial_blocking_reasons=initial_blocking_reasons,
        drain_seconds=2.0,
        worker_stop_timeout=0.25,
        monotonic=clock,
        sleep=lambda _seconds: None,
    )
    return controller, api, detector, compatibility, factory, tray, clock


class WindowsRuntimeTests(unittest.TestCase):
    def test_named_mutex_activates_primary_without_second_owner(self):
        api = FakeKernelApi()
        first = NamedInstanceCoordinator(api)
        second = NamedInstanceCoordinator(api)
        self.assertTrue(first.acquire_or_activate())
        self.assertFalse(second.acquire_or_activate())
        self.assertTrue(first.consume_activation())
        self.assertFalse(first.consume_activation())
        self.assertTrue(first.is_primary)
        self.assertFalse(second.is_primary)
        first.close()
        third = NamedInstanceCoordinator(api)
        self.assertTrue(third.acquire_or_activate())
        third.close()

    def test_named_shutdown_event_requests_primary_exit_idempotently(self):
        api = FakeKernelApi()
        primary = NamedInstanceCoordinator(api)
        self.assertTrue(primary.acquire_or_activate())
        requester = NamedInstanceCoordinator(api)
        self.assertTrue(requester.request_existing_shutdown())
        self.assertTrue(primary.consume_shutdown())
        self.assertFalse(primary.consume_shutdown())
        primary.close()
        self.assertFalse(
            NamedInstanceCoordinator(api).request_existing_shutdown()
        )

    @unittest.skipUnless(sys.platform == "win32", "requires Win32 named objects")
    def test_twenty_concurrent_launches_cannot_become_second_primary(self):
        suffix = uuid.uuid4().hex
        mutex = rf"Local\STS2Guide.Test.{suffix}"
        event = rf"Local\STS2Guide.Test.Activate.{suffix}"
        primary = NamedInstanceCoordinator(
            CtypesKernelApi(),
            mutex_name=mutex,
            activation_event_name=event,
        )
        self.assertTrue(primary.acquire_or_activate())
        code = (
            "import sys; "
            "from realtime.windows_runtime import CtypesKernelApi,NamedInstanceCoordinator; "
            "c=NamedInstanceCoordinator(CtypesKernelApi(),mutex_name=sys.argv[1],activation_event_name=sys.argv[2]); "
            "print('primary' if c.acquire_or_activate() else 'secondary')"
        )
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", code, mutex, event],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for _ in range(20)
        ]
        try:
            results = [process.communicate(timeout=15) for process in processes]
            self.assertTrue(all(process.returncode == 0 for process in processes))
            self.assertEqual(
                [stdout.strip() for stdout, _ in results],
                ["secondary"] * 20,
            )
            self.assertTrue(primary.consume_activation())
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            primary.close()

    def test_exact_game_detector_uses_full_case_insensitive_path(self):
        processes = FakeProcesses([
            ProcessInfo(1, r"C:\Fake\SlayTheSpire2.exe"),
            ProcessInfo(2, r"D:\SteamLibrary\steamapps\common\Other\Game.exe"),
        ])
        detector = ExactGameProcessDetector(GAME_PATH, processes)
        self.assertFalse(detector.is_running())
        processes.processes.append(
            ProcessInfo(
                3,
                r"d:/steamlibrary/steamapps/common/slay the spire 2/SLAYTHESPIRE2.EXE",
            )
        )
        self.assertTrue(detector.is_running())
        self.assertEqual(detector.matching_processes()[0].pid, 3)

    def test_game_path_must_be_absolute_and_exact_executable_name(self):
        with self.assertRaises(ValueError):
            canonical_windows_path("SlayTheSpire2.exe")
        with self.assertRaises(ValueError):
            ExactGameProcessDetector(r"C:\Games\Other.exe", FakeProcesses())


class ControllerLifecycleTests(unittest.TestCase):
    def test_all_non_running_transitions_clear_visible_advice(self):
        cleared = []
        controller, _, detector, _, factory, _, clock = make_controller(
            clear_advice=lambda: cleared.append("clear")
        )
        controller.start()
        self.assertEqual(cleared, ["clear"])

        detector.running = True
        controller.tick()
        factory.workers[0].running = False
        controller.tick()
        self.assertGreaterEqual(len(cleared), 2)

        clock.advance(5.0)
        controller.tick()
        detector.running = False
        controller.tick()
        self.assertEqual(controller.state, ControllerState.DRAINING)
        self.assertGreaterEqual(len(cleared), 3)

        controller.request_exit()
        controller.tick()
        self.assertGreaterEqual(len(cleared), 4)

    def test_advice_clear_failure_blocks_worker_and_remains_visible(self):
        def fail_clear():
            raise PermissionError("locked")

        controller, _, detector, _, factory, tray, _ = make_controller(
            clear_advice=fail_clear
        )
        self.assertEqual(controller.start(), StartDisposition.PRIMARY)
        self.assertEqual(controller.state, ControllerState.INCOMPATIBLE)
        self.assertEqual(
            controller.snapshot().compatibility_reasons,
            ("advice_clear_failed",),
        )
        detector.running = True
        controller.tick()
        self.assertEqual(factory.workers, [])
        self.assertTrue(tray.started)

    def test_bootstrap_blocker_keeps_tray_alive_without_game_directory(self):
        controller, _, detector, _, factory, tray, _ = make_controller(
            initial_blocking_reasons=("game_directory_not_found",)
        )
        self.assertEqual(controller.start(), StartDisposition.PRIMARY)
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.INCOMPATIBLE)
        self.assertFalse(snapshot.game_running)
        self.assertEqual(
            snapshot.compatibility_reasons,
            ("game_directory_not_found",),
        )
        self.assertEqual(factory.workers, [])
        self.assertTrue(tray.started)
        detector.running = True
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.INCOMPATIBLE)
        self.assertEqual(factory.workers, [])

    def test_game_start_runs_exactly_one_worker_and_game_exit_drains(self):
        controller, _, detector, _, factory, tray, clock = make_controller()
        self.assertEqual(controller.start(), StartDisposition.PRIMARY)
        self.assertEqual(controller.state, ControllerState.IDLE)
        detector.running = True
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.RUNNING)
        self.assertEqual(len(factory.workers), 1)
        controller.tick()
        self.assertEqual(len(factory.workers), 1)

        detector.running = False
        self.assertEqual(controller.tick().state, ControllerState.DRAINING)
        clock.advance(1.99)
        self.assertEqual(controller.tick().state, ControllerState.DRAINING)
        self.assertTrue(factory.workers[0].running)
        clock.advance(0.01)
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.IDLE)
        self.assertFalse(snapshot.worker_running)
        self.assertEqual(factory.workers[0].stops, [0.25])
        self.assertTrue(tray.started)
        self.assertFalse(tray.stopped)

    def test_game_return_during_drain_reuses_same_worker(self):
        controller, _, detector, _, factory, _, clock = make_controller()
        controller.start()
        detector.running = True
        controller.tick()
        worker = factory.workers[0]
        detector.running = False
        controller.tick()
        clock.advance(1.0)
        detector.running = True
        self.assertEqual(controller.tick().state, ControllerState.RUNNING)
        self.assertIs(controller.worker, worker)
        self.assertEqual(len(factory.workers), 1)

    def test_worker_failure_is_visible_and_activation_allows_retry(self):
        controller, api, detector, _, factory, tray, _ = make_controller()
        controller.start()
        detector.running = True
        controller.tick()
        factory.workers[0].running = False
        failed = controller.tick()
        self.assertEqual(failed.state, ControllerState.WORKER_FAILED)
        self.assertEqual(
            failed.compatibility_reasons,
            ("worker_exited_unexpectedly",),
        )
        secondary = NamedInstanceCoordinator(api)
        self.assertFalse(secondary.acquire_or_activate())
        self.assertEqual(controller.tick().state, ControllerState.RUNNING)
        self.assertEqual(len(factory.workers), 2)
        self.assertEqual(tray.activations, 1)

    def test_worker_failure_retries_once_after_bounded_backoff(self):
        controller, _, detector, _, factory, _, clock = make_controller()
        controller.start()
        detector.running = True
        controller.tick()
        factory.workers[0].running = False
        self.assertEqual(controller.tick().state, ControllerState.WORKER_FAILED)
        clock.advance(4.99)
        self.assertEqual(controller.tick().state, ControllerState.WORKER_FAILED)
        self.assertEqual(len(factory.workers), 1)
        clock.advance(0.01)
        self.assertEqual(controller.tick().state, ControllerState.RUNNING)
        self.assertEqual(len(factory.workers), 2)

    def test_incompatible_never_starts_worker_and_rechecks_on_activation(self):
        compatibility = FakeCompatibility(
            CompatibilityResult(False, ("game_version_mismatch",))
        )
        controller, api, detector, _, factory, _, _ = make_controller(
            compatibility=compatibility
        )
        controller.start()
        detector.running = True
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.INCOMPATIBLE)
        self.assertEqual(snapshot.compatibility_reasons, ("game_version_mismatch",))
        self.assertEqual(factory.workers, [])
        compatibility.result = CompatibilityResult(True)
        self.assertFalse(NamedInstanceCoordinator(api).acquire_or_activate())
        self.assertEqual(controller.tick().state, ControllerState.RUNNING)

    def test_game_close_after_failure_returns_to_idle(self):
        factory = WorkerFactory()
        factory.fail_next = True
        controller, _, detector, _, _, tray, _ = make_controller(factory=factory)
        controller.start()
        detector.running = True
        failed = controller.tick()
        self.assertEqual(failed.state, ControllerState.WORKER_FAILED)
        self.assertEqual(
            failed.compatibility_reasons,
            ("worker_start_failed",),
        )
        self.assertFalse(controller.snapshot().worker_running)
        detector.running = False
        self.assertEqual(controller.tick().state, ControllerState.IDLE)
        self.assertFalse(tray.stopped)

    def test_second_controller_only_activates_existing_instance(self):
        api = FakeKernelApi()
        first, _, _, _, first_factory, first_tray, _ = make_controller(api=api)
        second, _, _, _, second_factory, second_tray, _ = make_controller(api=api)
        self.assertEqual(first.start(), StartDisposition.PRIMARY)
        self.assertEqual(second.start(), StartDisposition.ACTIVATED_EXISTING)
        self.assertEqual(second.state, ControllerState.EXITING)
        self.assertEqual(second_factory.workers, [])
        self.assertFalse(second_tray.started)
        first.tick()
        self.assertEqual(first_tray.activations, 1)
        self.assertEqual(first_factory.workers, [])
        first.shutdown()

    def test_windows_api_failures_fail_closed_without_worker(self):
        api = FakeKernelApi()
        api.fail = True
        controller, _, _, _, factory, tray, _ = make_controller(api=api)
        self.assertEqual(controller.start(), StartDisposition.FAILED_CLOSED)
        self.assertEqual(controller.state, ControllerState.INCOMPATIBLE)
        self.assertEqual(controller.snapshot().compatibility_reasons, (
            "windows_instance_api_unavailable",
        ))
        self.assertEqual(factory.workers, [])
        self.assertFalse(tray.started)

        controller, _, detector, _, factory, _, _ = make_controller()
        controller.start()
        detector.error = WindowsRuntimeUnavailable("process API unavailable")
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.INCOMPATIBLE)
        self.assertEqual(snapshot.compatibility_reasons, ("windows_process_api_failed",))
        self.assertEqual(factory.workers, [])

    def test_complete_exit_stops_worker_tray_and_releases_mutex(self):
        controller, api, detector, _, factory, tray, _ = make_controller()
        controller.start()
        detector.running = True
        controller.tick()
        controller.request_exit()
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.EXITING)
        self.assertFalse(snapshot.worker_running)
        self.assertFalse(factory.workers[0].running)
        self.assertTrue(tray.stopped)
        replacement = NamedInstanceCoordinator(api)
        self.assertTrue(replacement.acquire_or_activate())
        replacement.close()

    def test_external_shutdown_stops_worker_tray_and_releases_mutex(self):
        controller, api, detector, _, factory, tray, _ = make_controller()
        controller.start()
        detector.running = True
        controller.tick()
        self.assertTrue(
            NamedInstanceCoordinator(api).request_existing_shutdown()
        )
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.EXITING)
        self.assertFalse(snapshot.worker_running)
        self.assertFalse(factory.workers[0].running)
        self.assertTrue(tray.stopped)
        replacement = NamedInstanceCoordinator(api)
        self.assertTrue(replacement.acquire_or_activate())
        replacement.close()

    def test_exit_does_not_orphan_worker_when_stop_times_out(self):
        controller, api, detector, _, factory, tray, _ = make_controller()
        controller.start()
        detector.running = True
        controller.tick()
        worker = factory.workers[0]
        worker.fail_stop = True
        controller.request_exit()
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.WORKER_FAILED)
        self.assertTrue(snapshot.worker_running)
        self.assertFalse(tray.stopped)
        contender = NamedInstanceCoordinator(api)
        self.assertFalse(contender.acquire_or_activate())
        worker.fail_stop = False
        controller.request_exit()
        self.assertEqual(controller.tick().state, ControllerState.EXITING)
        self.assertTrue(tray.stopped)

    def test_bad_compatibility_checker_fails_closed(self):
        compatibility = FakeCompatibility()
        compatibility.error = RuntimeError("cannot read manifest")
        controller, _, detector, _, factory, _, _ = make_controller(
            compatibility=compatibility
        )
        controller.start()
        detector.running = True
        snapshot = controller.tick()
        self.assertEqual(snapshot.state, ControllerState.INCOMPATIBLE)
        self.assertEqual(snapshot.compatibility_reasons, ("compatibility_check_failed",))
        self.assertEqual(factory.workers, [])


class ThreadWorkerTests(unittest.TestCase):
    def test_cooperative_worker_stops_and_runs_cleanup_once(self):
        called = threading.Event()
        cleanup = []

        def run_once():
            called.set()

        worker = ThreadWorker(
            run_once,
            poll_interval=0.01,
            on_stopped=lambda: cleanup.append("released"),
        )
        worker.start()
        self.assertTrue(called.wait(1.0))
        self.assertTrue(worker.is_running())
        worker.stop(1.0)
        worker.stop(1.0)
        self.assertFalse(worker.is_running())
        self.assertEqual(cleanup, ["released"])

    def test_worker_exception_releases_owner_and_becomes_not_running(self):
        cleanup = threading.Event()

        def fail():
            raise RuntimeError("expected test failure")

        worker = ThreadWorker(
            fail,
            poll_interval=0.01,
            on_stopped=cleanup.set,
        )
        worker.start()
        self.assertTrue(cleanup.wait(1.0))
        self.assertFalse(worker.is_running())


if __name__ == "__main__":
    unittest.main()
