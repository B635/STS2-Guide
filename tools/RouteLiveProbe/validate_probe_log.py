"""Validate and summarize RouteLiveProbe JSONL without third-party packages."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_KEYS = {
    "schema_version",
    "session_id",
    "sequence",
    "observed_at_utc",
    "event_name",
    "assembly_version",
    "assembly_mvid",
    "snapshot",
    "details",
}


def iter_records(paths: Iterable[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                missing = REQUIRED_KEYS.difference(record)
                if missing:
                    raise ValueError(
                        f"{path}:{line_number}: missing keys: {sorted(missing)}"
                    )
                if record["schema_version"] != 1:
                    raise ValueError(
                        f"{path}:{line_number}: unsupported schema_version "
                        f"{record['schema_version']!r}"
                    )
                yield path, line_number, record


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    event_counts: Counter[str] = Counter()
    sequences: dict[str, list[int]] = defaultdict(list)
    travel_state_counts: Counter[str] = Counter()
    top_bar_open_counts: Counter[str] = Counter()
    visual_point_counts: list[int] = []
    coordinate_samples = 0
    selected_node_ids: list[str] = []
    model_visual_candidate_mismatches = 0

    for path, line_number, record in iter_records(paths):
        session_id = record["session_id"]
        sequence = record["sequence"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"{path}:{line_number}: invalid session_id")
        if not isinstance(sequence, int) or sequence <= 0:
            raise ValueError(f"{path}:{line_number}: invalid sequence")
        sequences[session_id].append(sequence)
        event_name = str(record["event_name"])
        event_counts[event_name] += 1

        details = record.get("details") or {}
        if event_name == "map_open_postfix":
            top_bar_open_counts[str(details.get("is_opened_from_top_bar"))] += 1
        if event_name == "map_point_selected_postfix":
            selected = details.get("selected_node_id")
            if isinstance(selected, str):
                selected_node_ids.append(selected)

        snapshot = record.get("snapshot")
        if not isinstance(snapshot, dict):
            continue
        screen = snapshot.get("screen")
        run = snapshot.get("run")
        points = snapshot.get("visual_points") or []
        if isinstance(points, list):
            visual_point_counts.append(len(points))
            coordinate_samples += sum(
                1
                for point in points
                if isinstance(point, dict)
                and point.get("global_rect_center") is not None
                and point.get("net_position_from_center") is not None
                and point.get("screen_position_round_trip") is not None
            )
        if isinstance(screen, dict):
            key = "/".join(
                str(screen.get(name))
                for name in ("is_open", "is_travel_enabled", "is_traveling")
            )
            travel_state_counts[key] += 1
        if isinstance(run, dict):
            model = run.get("model_next_node_ids")
            visual = run.get("visual_travelable_node_ids")
            if isinstance(model, list) and isinstance(visual, list):
                if set(model) != set(visual):
                    model_visual_candidate_mismatches += 1

    for session_id, values in sequences.items():
        expected = list(range(1, len(values) + 1))
        if values != expected:
            raise ValueError(
                f"session {session_id}: sequences are not contiguous: {values}"
            )

    return {
        "sessions": len(sequences),
        "records": sum(event_counts.values()),
        "event_counts": dict(sorted(event_counts.items())),
        "top_bar_open_counts": dict(sorted(top_bar_open_counts.items())),
        "travel_state_counts": dict(sorted(travel_state_counts.items())),
        "selected_node_ids": selected_node_ids,
        "visual_point_count_min": min(visual_point_counts, default=0),
        "visual_point_count_max": max(visual_point_counts, default=0),
        "coordinate_samples": coordinate_samples,
        "model_visual_candidate_mismatch_records": model_visual_candidate_mismatches,
        "gate_evidence": {
            "opened": event_counts["map_open_postfix"] > 0,
            "closed": event_counts["map_close_postfix"] > 0,
            "selected": event_counts["map_point_selected_postfix"] > 0,
            "saved_setup": event_counts["saved_singleplayer_setup_postfix"] > 0,
            "coordinates": coordinate_samples > 0,
        },
    }


def expand_log_arguments(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        matches = [Path(match) for match in glob.glob(value)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(Path(value))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args(argv)
    try:
        result = summarize(expand_log_arguments(args.logs))
    except (OSError, ValueError) as exc:
        print(f"route probe validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
