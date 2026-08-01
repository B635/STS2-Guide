"""P0 background host: file bridge + checkpoint + local advisor only.

This entry point deliberately does not import or start FastAPI, Vue, RAG, or
an LLM client.  It is suitable for a no-console PyInstaller executable.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from advisor.data_sources import load_local_card_tiers
from advisor.versions import (
    CAMPFIRE_POLICY_VERSION,
    CARD_REWARD_POLICY_VERSION,
    DECK_EDIT_POLICY_VERSION,
    EVENT_POLICY_VERSION,
    MERCHANT_POLICY_VERSION,
    NEOW_POLICY_VERSION,
    POLICY_BUNDLE_VERSION,
    ROUTE_POLICY_VERSION,
)
from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.compatibility import (
    RuntimeComponents,
    default_manifest_path,
    load_compatibility_manifest,
)
from realtime.file_bridge import (
    GameStateFileBridge,
    default_checkpoint_path,
    default_exchange_dir,
    default_events_dir,
    default_input_path,
    default_output_path,
)
from realtime.processor import RealtimeEventProcessor
from realtime.protocol import CURRENT_SCHEMA_VERSION
from realtime.version import GUIDE_VERSION
from storage.relational import RelationalRepository
from storage.release_database import ensure_runtime_database


LOGGER = logging.getLogger("sts2-guide")


def _ensure_windowed_stdio() -> None:
    """Give argparse safe sinks inside a no-console PyInstaller process.

    PyInstaller sets stdout/stderr to None for a windowed executable.
    argparse writes its help and error text to those streams; without sinks,
    ``--help`` raises before it can exit and the hidden error dialog makes the
    process appear hung.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _project_or_bundle_root() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)
    return Path(__file__).resolve().parents[1]


def _default_catalog_path() -> Path:
    return _project_or_bundle_root() / "data" / "knowledge.json"


def _default_community_path() -> Path:
    return _project_or_bundle_root() / "data" / "community_scores.json"


def _default_local_tier_path() -> Path:
    configured = os.getenv("STS2_LOCAL_CARD_TIERS_FILE")
    if configured:
        return Path(configured).expanduser()
    executable_root = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else _project_or_bundle_root()
    )
    return (
        executable_root
        / "data"
        / "local"
        / "mobalytics_card_tiers.json"
    )


def _default_runtime_database_path() -> Path:
    configured = os.getenv("STS2_RELATIONAL_DB_FILE")
    if configured:
        return Path(configured).expanduser()
    return default_input_path().parent / "sts2-guide.db"


def _bundled_release_database_path() -> Path:
    return _project_or_bundle_root() / "data" / "sts2-guide-template.db"


def _sha256_path(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _observe_local_components(
    repository: RelationalRepository,
    *,
    catalog_path: Path,
    community_path: Path,
) -> RuntimeComponents:
    """Read independent Host/data/policy identities for manifest comparison."""

    metadata: dict[str, str] = {}
    try:
        with repository.connect() as connection:
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    """
                    SELECT key, value FROM schema_metadata
                    WHERE key IN (
                        'schema_version',
                        'catalog_sha256',
                        'community_scores_sha256'
                    )
                    """
                ).fetchall()
            }
    except Exception:
        # Missing observations are deliberately represented as None and will
        # fail closed in the compatibility assessment.
        metadata = {}

    stored_catalog_hash = metadata.get("catalog_sha256")
    try:
        sqlite_schema_version = int(metadata["schema_version"])
    except (KeyError, TypeError, ValueError):
        sqlite_schema_version = None
    return RuntimeComponents(
        guide_version=GUIDE_VERSION,
        protocol_schema_version=CURRENT_SCHEMA_VERSION,
        sqlite_schema_version=sqlite_schema_version,
        sqlite_snapshot_id=(
            f"knowledge-{stored_catalog_hash[:16]}"
            if stored_catalog_hash
            else None
        ),
        knowledge_sha256=(
            stored_catalog_hash or _sha256_path(catalog_path)
        ),
        community_scores_sha256=(
            metadata.get("community_scores_sha256")
            or _sha256_path(community_path)
        ),
        policy_bundle_version=POLICY_BUNDLE_VERSION,
        card_reward_policy_version=CARD_REWARD_POLICY_VERSION,
        route_policy_version=ROUTE_POLICY_VERSION,
        merchant_policy_version=MERCHANT_POLICY_VERSION,
        campfire_policy_version=CAMPFIRE_POLICY_VERSION,
        neow_policy_version=NEOW_POLICY_VERSION,
        event_policy_version=EVENT_POLICY_VERSION,
        deck_edit_policy_version=DECK_EDIT_POLICY_VERSION,
    )


def _clear_visible_advice(path: Path) -> None:
    """Fail closed before polling when Host compatibility is unavailable."""

    path.unlink(missing_ok=True)
    path.with_name(f"{path.name}.tmp").unlink(missing_ok=True)


def _configure_logging(log_path: Path, console: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            log_path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
    ]
    if console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


def _acquire_instance_lock(exchange_dir: Path) -> None:
    """Atomically create an exclusive lock file for the exchange directory."""
    import ctypes
    import ctypes.wintypes

    exchange_dir.mkdir(parents=True, exist_ok=True)
    lock_path = exchange_dir / ".host.lock"
    # Use O_CREAT | O_EXCL for atomic create-or-fail (no check-then-write race).
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # File exists — check if the holder is still alive.
        try:
            stale_pid = int(lock_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            stale_pid = 0
        if stale_pid and stale_pid != os.getpid():
            SYNCHRONIZE = 0x00100000
            PROCESS_QUERY_LIMITED_INFO = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFO | SYNCHRONIZE,
                False,
                stale_pid,
            )
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                raise RuntimeError(
                    f"Another host instance (PID {stale_pid}) is already "
                    f"using this exchange directory: {exchange_dir}"
                )
        # Stale lock — remove and retry atomically.
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode("utf-8"))
    os.close(fd)


def _release_instance_lock(exchange_dir: Path) -> None:
    """Release only the lock owned by this process."""
    lock_path = exchange_dir / ".host.lock"
    try:
        owner = int(lock_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return
    if owner == os.getpid():
        lock_path.unlink(missing_ok=True)


def build_bridge(args: argparse.Namespace) -> GameStateFileBridge:
    compatibility_manifest = load_compatibility_manifest(
        args.compatibility_manifest
    )
    repository = RelationalRepository(str(args.database))
    release_database = getattr(args, "release_database", None)
    if release_database is None and getattr(sys, "frozen", False):
        release_database = _bundled_release_database_path()
    if release_database is not None:
        ensure_runtime_database(release_database, args.database)
    else:
        repository.ensure_schema()
        if not args.catalog.exists():
            raise FileNotFoundError(
                f"Structured catalog not found: {args.catalog}"
            )
        repository.sync_catalog(str(args.catalog))
        if args.community_scores.exists():
            repository.sync_entity_statistics(str(args.community_scores))

    local_tiers = load_local_card_tiers(args.local_card_tiers)
    LOGGER.info(
        "Local tier source status=%s path=%s warning=%s",
        local_tiers.status,
        local_tiers.path,
        local_tiers.warning,
    )
    processor = RealtimeEventProcessor(
        repository,
        checkpoint=ActiveRunCheckpointStore(args.checkpoint),
        local_tiers=local_tiers,
        compatibility_manifest=compatibility_manifest,
        runtime_components=_observe_local_components(
            repository,
            catalog_path=args.catalog,
            community_path=args.community_scores,
        ),
    )
    if (
        processor.local_compatibility is not None
        and not processor.local_compatibility.compatible
    ):
        _clear_visible_advice(args.output)
        LOGGER.error(
            "Local compatibility gate is closed: %s",
            ",".join(processor.local_compatibility.reason_codes),
        )
    return GameStateFileBridge(
        processor,
        input_path=args.input,
        output_path=args.output,
        events_dir=args.events_dir,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="STS2 Guide P0 local background advisor."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output", type=Path, default=default_output_path())
    parser.add_argument(
        "--events-dir",
        type=Path,
        default=default_events_dir(),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=_default_runtime_database_path(),
    )
    parser.add_argument(
        "--release-database",
        type=Path,
        default=None,
        help=(
            "Use an immutable release SQLite template; frozen builds always "
            "use their bundled template."
        ),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=_default_catalog_path(),
    )
    parser.add_argument(
        "--community-scores",
        type=Path,
        default=_default_community_path(),
    )
    parser.add_argument(
        "--local-card-tiers",
        type=Path,
        default=_default_local_tier_path(),
    )
    parser.add_argument(
        "--compatibility-manifest",
        type=Path,
        default=default_manifest_path(),
    )
    parser.add_argument("--poll-interval", type=float, default=0.1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tray", action="store_true")
    mode.add_argument("--worker", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--startup-check",
        action="store_true",
        help="Initialize the packaged host and exit without polling.",
    )
    parser.add_argument("--console-log", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument(
        "--worker-log-file",
        type=Path,
        default=default_exchange_dir() / "sts2-guide-worker.log",
    )
    parser.add_argument(
        "--runtime-status",
        type=Path,
        default=default_exchange_dir() / "runtime-status.json",
    )
    parser.add_argument("--game-dir", type=Path, default=None)
    from realtime.installation import default_install_receipt_path

    parser.add_argument(
        "--install-receipt",
        type=Path,
        default=default_install_receipt_path(),
    )
    parser.add_argument(
        "--controller-poll-interval",
        type=float,
        default=0.25,
    )
    parser.add_argument("--drain-seconds", type=float, default=2.0)
    parser.add_argument("--worker-stop-timeout", type=float, default=1.0)
    parser.add_argument(
        "--shutdown-existing",
        action="store_true",
        help="Request graceful shutdown of the existing tray controller.",
    )
    return parser


def _runtime_mode(args: argparse.Namespace) -> str:
    if args.tray:
        return "tray"
    if args.worker or args.once or args.startup_check:
        return "worker"
    return "tray" if getattr(sys, "frozen", False) else "worker"


def main() -> int:
    _ensure_windowed_stdio()
    args = _parser().parse_args()
    if args.shutdown_existing:
        try:
            from realtime.windows_runtime import (
                CtypesKernelApi,
                NamedInstanceCoordinator,
            )

            NamedInstanceCoordinator(
                CtypesKernelApi()
            ).request_existing_shutdown()
            return 0
        except Exception:
            return 2
    mode = _runtime_mode(args)
    if args.log_file is None:
        args.log_file = default_exchange_dir() / (
            "sts2-guide-controller.log"
            if mode == "tray"
            else "sts2-guide-worker.log"
        )
    _configure_logging(
        args.log_file,
        args.console_log,
    )
    if mode == "tray":
        try:
            from realtime.public_beta import run_public_beta_controller

            return run_public_beta_controller(args)
        except Exception:
            _clear_visible_advice(args.output)
            LOGGER.exception("STS2 Guide controller failed to initialize.")
            return 2
    # Acquire instance lock *before* database/bridge initialization so a
    # second host never opens the SQLite file or syncs the catalog.
    lock_acquired = False
    if not args.once:
        try:
            _acquire_instance_lock(args.checkpoint.parent)
            lock_acquired = True
        except RuntimeError:
            LOGGER.exception("Cannot start: another host instance is running.")
            return 1

    try:
        try:
            bridge = build_bridge(args)
        except Exception:
            _clear_visible_advice(args.output)
            LOGGER.exception("STS2 Guide background host failed to initialize.")
            return 1

        LOGGER.info(
            "P0 host started queue=%s output=%s checkpoint=%s",
            args.events_dir,
            args.output,
            args.checkpoint,
        )
        if args.startup_check:
            LOGGER.info("Packaged host startup check passed.")
            return 0
        if args.once:
            result = bridge.run_once()
            LOGGER.info(
                "One-shot result=%s",
                None if result is None else result.get("status"),
            )
            return 0
        try:
            bridge.run_forever(args.poll_interval)
        except KeyboardInterrupt:
            LOGGER.info("P0 host stopped.")
            return 0
        except Exception:
            LOGGER.exception("P0 host stopped unexpectedly.")
            return 1
    finally:
        if lock_acquired:
            _release_instance_lock(args.checkpoint.parent)


if __name__ == "__main__":
    raise SystemExit(main())
