"""P0 background host: file bridge + checkpoint + local advisor only.

This entry point deliberately does not import or start FastAPI, Vue, RAG, or
an LLM client.  It is suitable for a no-console PyInstaller executable.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from advisor.data_sources import load_local_card_tiers
from realtime.checkpoint import ActiveRunCheckpointStore
from realtime.file_bridge import (
    GameStateFileBridge,
    default_checkpoint_path,
    default_events_dir,
    default_input_path,
    default_output_path,
)
from realtime.processor import RealtimeEventProcessor
from storage.relational import RelationalRepository


LOGGER = logging.getLogger("sts2-guide")


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


def build_bridge(args: argparse.Namespace) -> GameStateFileBridge:
    repository = RelationalRepository(str(args.database))
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
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--console-log", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    _configure_logging(
        args.input.parent / "sts2-guide.log",
        args.console_log,
    )
    # Acquire instance lock *before* database/bridge initialization so a
    # second host never opens the SQLite file or syncs the catalog.
    if not args.once:
        try:
            _acquire_instance_lock(args.checkpoint.parent)
        except RuntimeError:
            LOGGER.exception("Cannot start: another host instance is running.")
            return 1

    try:
        bridge = build_bridge(args)
    except Exception:
        LOGGER.exception("STS2 Guide background host failed to initialize.")
        return 1

    LOGGER.info(
        "P0 host started queue=%s output=%s checkpoint=%s",
        args.events_dir,
        args.output,
        args.checkpoint,
    )
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


if __name__ == "__main__":
    raise SystemExit(main())
