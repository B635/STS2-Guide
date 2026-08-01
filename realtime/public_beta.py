"""Production wiring for the Windows Public Beta tray controller."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable

from realtime.compatibility import CompatibilityManifest, load_compatibility_manifest
from realtime.controller import (
    CompatibilityResult,
    ControllerSnapshot,
    PystrayTrayBackend,
    ThreadWorker,
    TrayBackend,
    WindowsGuideController,
)
from realtime.diagnostics import DiagnosticSnapshot, export_diagnostics_zip
from realtime.file_bridge import _atomic_write_json, default_exchange_dir
from realtime.installation import (
    InstallReceipt,
    InstallationPreflight,
    default_install_receipt_path,
    discover_game_directory,
    expected_mod_hashes,
    preflight_installation,
    read_install_receipt,
    sha256_file,
)
from realtime.windows_runtime import (
    CtypesKernelApi,
    CtypesWindowsProcessApi,
    ExactGameProcessDetector,
    NamedInstanceCoordinator,
)
from storage.relational import SCHEMA_VERSION


LOGGER = logging.getLogger("sts2-guide")


class _UnavailableGameDetector:
    """Keep the tray alive when installation discovery cannot identify STS2."""

    def is_running(self) -> bool:
        return False


class _BootstrapCompatibilityChecker:
    """Diagnostic surface for failures discovered before process polling."""

    def __init__(self, reason_codes: tuple[str, ...]) -> None:
        self.reason_codes = reason_codes
        self.last_preflight = None
        self.guide_executable_hash = None

    def check(self) -> CompatibilityResult:
        return CompatibilityResult(False, self.reason_codes)


class InstallationCompatibilityChecker:
    """Revalidate the exact install receipt and every production artifact."""

    def __init__(
        self,
        game_directory: Path,
        manifest: CompatibilityManifest,
        receipt_path: Path,
        *,
        clear_advice: Callable[[], None],
        guide_executable: Path | None = None,
    ) -> None:
        self.game_directory = game_directory
        self.manifest = manifest
        self.receipt_path = receipt_path
        self.clear_advice = clear_advice
        self.guide_executable = guide_executable
        self.guide_executable_hash: str | None = None
        self.last_preflight: InstallationPreflight | None = None

    def _receipt(self) -> tuple[InstallReceipt | None, tuple[str, ...]]:
        receipt = read_install_receipt(self.receipt_path)
        if receipt is None:
            return None, ("install_receipt_missing",)
        reasons: list[str] = []
        if receipt.guide_version != self.manifest.guide.version:
            reasons.append("install_receipt_guide_mismatch")
        if receipt.release_fingerprint != self.manifest.release_fingerprint:
            reasons.append("install_receipt_release_mismatch")
        if self.guide_executable is not None:
            try:
                executable_hash = sha256_file(self.guide_executable)
            except OSError:
                reasons.append("guide_executable_unreadable")
            else:
                self.guide_executable_hash = executable_hash
                if executable_hash != receipt.guide_executable_sha256:
                    reasons.append("guide_executable_hash_mismatch")
        try:
            same_game = (
                receipt.game_directory.resolve()
                == self.game_directory.resolve()
            )
        except OSError:
            same_game = False
        if not same_game:
            reasons.append("install_receipt_game_mismatch")
        if expected_mod_hashes(receipt) is None:
            reasons.append("install_receipt_artifacts_invalid")
        return receipt, tuple(reasons)

    def check(self) -> CompatibilityResult:
        receipt, receipt_reasons = self._receipt()
        if receipt_reasons or receipt is None:
            self.last_preflight = None
            self.clear_advice()
            return CompatibilityResult(False, receipt_reasons)
        hashes = expected_mod_hashes(receipt)
        if hashes is None:  # guarded above, retained for type safety
            self.clear_advice()
            return CompatibilityResult(
                False,
                ("install_receipt_artifacts_invalid",),
            )
        result = preflight_installation(
            self.game_directory,
            self.manifest,
            hashes,
        )
        self.last_preflight = result
        if not result.compatible:
            self.clear_advice()
        return CompatibilityResult(result.compatible, result.reason_codes)


class StatusTrayBackend:
    """Publish a strict local status file alongside the actual tray UI."""

    def __init__(self, backend: TrayBackend, status_path: Path) -> None:
        self.backend = backend
        self.status_path = status_path
        self.last_snapshot: ControllerSnapshot | None = None

    def start(self, request_exit: Callable[[], None]) -> None:
        self.backend.start(request_exit)

    def update(self, snapshot: ControllerSnapshot) -> None:
        self.last_snapshot = snapshot
        _atomic_write_json(
            self.status_path,
            {
                "schema_version": 1,
                "state": snapshot.state.value,
                "game_running": snapshot.game_running,
                "worker_running": snapshot.worker_running,
                "compatibility_reasons": list(
                    snapshot.compatibility_reasons
                ),
                "activation_count": snapshot.activation_count,
            },
        )
        self.backend.update(snapshot)

    def activate(self) -> None:
        self.backend.activate()

    def stop(self) -> None:
        try:
            self.backend.stop()
        finally:
            self.status_path.unlink(missing_ok=True)
            self.status_path.with_name(
                f"{self.status_path.name}.tmp"
            ).unlink(missing_ok=True)


def _open_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]


def build_public_beta_controller(args) -> WindowsGuideController:
    manifest = load_compatibility_manifest(args.compatibility_manifest)
    receipt_path = Path(args.install_receipt)
    discovery = discover_game_directory(
        receipt_path=receipt_path,
        manual_directory=args.game_dir,
    )
    blocking_reasons: tuple[str, ...] = ()
    game_directory = discovery.game_directory
    if not discovery.found or game_directory is None:
        blocking_reasons = discovery.reason_codes or (
            "game_directory_not_found",
        )
        detector = _UnavailableGameDetector()
    else:
        expected_executable = game_directory / "SlayTheSpire2.exe"
        detector = ExactGameProcessDetector(
            expected_executable,
            CtypesWindowsProcessApi(),
        )

    def clear_advice() -> None:
        args.output.unlink(missing_ok=True)
        args.output.with_name(f"{args.output.name}.tmp").unlink(
            missing_ok=True
        )

    checker = (
        _BootstrapCompatibilityChecker(blocking_reasons)
        if blocking_reasons
        else InstallationCompatibilityChecker(
            game_directory,
            manifest,
            receipt_path,
            clear_advice=clear_advice,
            guide_executable=(
                Path(sys.executable)
                if getattr(sys, "frozen", False)
                else None
            ),
        )
    )
    status_holder: dict[str, StatusTrayBackend] = {}

    def export_diagnostics() -> None:
        tray = status_holder.get("tray")
        snapshot = None if tray is None else tray.last_snapshot
        reasons = () if snapshot is None else snapshot.compatibility_reasons
        state = "IDLE" if snapshot is None else snapshot.state.value
        hashes = dict(
            {}
            if checker.last_preflight is None
            else checker.last_preflight.artifact_hashes
        )
        if checker.guide_executable_hash is not None:
            hashes["guide_executable"] = checker.guide_executable_hash
        diagnostic = DiagnosticSnapshot(
            versions={
                "guide": manifest.guide.version,
                "game": manifest.game.version,
                "mod": manifest.mod.version,
                "protocol": str(
                    manifest.protocol.state_event_schema_version
                ),
                "sqlite": str(SCHEMA_VERSION),
                "policy": manifest.policy.bundle_version,
            },
            release_fingerprint=manifest.release_fingerprint,
            capabilities=manifest.capabilities,
            status=state,
            reason_codes=reasons,
            artifact_hashes=hashes,
        )
        lines: list[str] = []
        for log_path in (args.log_file, args.worker_log_file):
            try:
                lines.extend(
                    log_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()[-200:]
                )
            except OSError:
                continue
        output_dir = default_exchange_dir() / "diagnostics"
        output = output_dir / "STS2-Guide-diagnostics.zip"
        sensitive_values = [
            str(receipt_path),
            str(default_exchange_dir()),
        ]
        if game_directory is not None:
            sensitive_values.append(str(game_directory))
        export_diagnostics_zip(
            output,
            diagnostic,
            lines,
            sensitive_values=sensitive_values,
        )
        _open_directory(output_dir)

    graphical = PystrayTrayBackend(
        title="STS2 Guide",
        open_logs=lambda: _open_directory(args.log_file.parent),
        export_diagnostics=export_diagnostics,
        version_lines={
            "Guide": manifest.guide.version,
            "Game": manifest.game.version,
            "Mod": manifest.mod.version,
            "Protocol": str(manifest.protocol.state_event_schema_version),
            "Data": str(SCHEMA_VERSION),
        },
        icon_path=(
            Path(getattr(sys, "_MEIPASS"))
            / "packaging"
            / "sts2-guide.png"
            if getattr(sys, "_MEIPASS", None)
            else Path(__file__).resolve().parents[1]
            / "build"
            / "release"
            / "sts2-guide.png"
        ),
    )
    tray = StatusTrayBackend(graphical, args.runtime_status)
    status_holder["tray"] = tray

    def create_worker() -> ThreadWorker:
        # Import lazily to avoid an entry-point cycle while retaining the
        # development worker command in realtime.host.
        from realtime.host import (
            _acquire_instance_lock,
            _release_instance_lock,
            build_bridge,
        )

        exchange_dir = args.checkpoint.parent
        _acquire_instance_lock(exchange_dir)
        try:
            bridge = build_bridge(args)
        except Exception:
            _release_instance_lock(exchange_dir)
            raise
        def cleanup_worker() -> None:
            try:
                clear_advice()
            finally:
                _release_instance_lock(exchange_dir)

        return ThreadWorker(
            bridge.run_once,
            poll_interval=args.poll_interval,
            on_stopped=cleanup_worker,
        )

    return WindowsGuideController(
        instances=NamedInstanceCoordinator(CtypesKernelApi()),
        game_detector=detector,
        compatibility=checker,
        worker_factory=create_worker,
        tray=tray,
        clear_advice=clear_advice,
        initial_blocking_reasons=blocking_reasons,
        drain_seconds=args.drain_seconds,
        worker_stop_timeout=args.worker_stop_timeout,
    )


def run_public_beta_controller(args) -> int:
    controller = build_public_beta_controller(args)
    LOGGER.info("Public Beta controller started.")
    return controller.run(args.controller_poll_interval)


__all__ = [
    "InstallationCompatibilityChecker",
    "StatusTrayBackend",
    "build_public_beta_controller",
    "default_install_receipt_path",
    "run_public_beta_controller",
]
