"""Extract real-combat HP-loss records and calibrate monotone combat weights.

Strictly reads ONLY ``build/collected-runs/`` and ``coach/eval/runs/dev/``,
explicitly excluding every run listed under ``test`` in ``coach/eval/split.json``
so there is zero leakage from the held-out blind test split.

Usage:
    python scripts/calibrate_combat_model.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from advisor.combat_calibration import (  # noqa: E402
    CALIBRATION_DIMENSIONS,
    DEFAULT_CALIBRATION_PATH,
    calibrate_from_records,
    capability_feature_vector,
)
from advisor.deck_fitness import DeckFitnessPolicy, PARAMETERS_KEY  # noqa: E402
from advisor.deck_mechanics import Situation, deck_capabilities, enchant  # noqa: E402
from advisor.encounter_turns import encounter_turns  # noqa: E402
from advisor.history_requests import starting_decks  # noqa: E402
from advisor.monster_turns import mean  # noqa: E402
from advisor.run_history import RunHistoryError, bare_id, load_run, replay_run  # noqa: E402
from storage.relational import RelationalRepository  # noqa: E402
from storage.release_database import build_release_database  # noqa: E402

DEFAULT_COLLECTED_DIR = ROOT_DIR / "build" / "collected-runs"
DEFAULT_DEV_DIR = ROOT_DIR / "coach" / "eval" / "runs" / "dev"
DEFAULT_SPLIT_PATH = ROOT_DIR / "coach" / "eval" / "split.json"
DEFAULT_KNOWLEDGE_PATH = ROOT_DIR / "data" / "knowledge.json"
DEFAULT_COMMUNITY_PATH = ROOT_DIR / "data" / "community_scores.json"


def _load_excluded_test_stems(split_path: Path) -> set:
    if not split_path.exists():
        return set()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return {str(stem) for stem in split.get("test") or ()}


def extract_combat_records(
    *,
    collected_dir: Path = DEFAULT_COLLECTED_DIR,
    dev_dir: Path = DEFAULT_DEV_DIR,
    split_path: Path = DEFAULT_SPLIT_PATH,
    knowledge_path: Path = DEFAULT_KNOWLEDGE_PATH,
    community_path: Path = DEFAULT_COMMUNITY_PATH,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract clean pre-combat feature vectors and actual HP loss from runs."""

    excluded_test = _load_excluded_test_stems(split_path)
    decks_by_char = starting_decks(knowledge_path)
    counts: collections.Counter = collections.Counter()

    raw_files: List[Tuple[Path, str]] = []
    if collected_dir.exists():
        for path in sorted(collected_dir.rglob("*.run")):
            if path.stem in excluded_test:
                counts["excluded_test_split"] += 1
                continue
            raw_files.append((path, f"cr-{path.parent.name[:8]}-{path.stem}"))
    if dev_dir.exists():
        for path in sorted(dev_dir.rglob("*.run")):
            if path.stem in excluded_test:
                counts["excluded_test_split"] += 1
                continue
            raw_files.append((path, f"dev-{path.stem}"))

    # Defensive guard: never touch coach/eval/runs/test.
    for path, _ in raw_files:
        parts = [p.lower() for p in path.parts]
        if "runs" in parts and "test" in parts:
            raise RuntimeError(f"Refusing to read held-out test run during calibration: {path}")

    seen_identity: set = set()
    fight_payloads: List[Dict[str, Any]] = []
    for path, run_key in raw_files:
        counts["files_scanned"] += 1
        try:
            run = load_run(path)
        except RunHistoryError:
            counts["unreadable_run"] += 1
            continue
        identity = (run.get("seed"), run.get("start_time"))
        if None not in identity:
            if identity in seen_identity:
                counts["duplicate_run"] += 1
                continue
            seen_identity.add(identity)
        if len(run.get("players") or ()) != 1:
            counts["multiplayer_excluded"] += 1
            continue
        character = bare_id(run["players"][0]["character"])
        if character not in decks_by_char:
            counts["unknown_character"] += 1
            continue
        try:
            replayed = replay_run(run, decks_by_char[character], run_key)
        except (AttributeError, KeyError, TypeError, ValueError, RunHistoryError):
            counts["replay_error"] += 1
            continue
        if replayed["problems"]:
            counts["deck_replay_mismatch"] += 1
            continue

        was_abandoned = bool(run.get("was_abandoned"))
        total_floors = sum(len(act_pts) for act_pts in run.get("map_point_history") or ())
        ascension = int(run.get("ascension") or 0)
        counts["valid_runs"] += 1
        if was_abandoned:
            counts["valid_abandoned_runs"] += 1

        for fight in replayed["fights"]:
            turns_taken = fight.get("turns_taken")
            damage_taken = fight.get("damage_taken")
            if not turns_taken or int(turns_taken) <= 0 or damage_taken is None:
                counts["fight_missing_outcome"] += 1
                continue
            # Mid-fight abandon sets hp_after=0 and truncates damage/turns.
            if was_abandoned and int(fight["floor"]) == total_floors:
                counts["abandoned_final_fight_excluded"] += 1
                continue
            fight_payloads.append({
                "run_id": run_key,
                "character": character,
                "ascension": ascension,
                "act": int(fight["act"]),
                "floor": int(fight["floor"]),
                "fight_kind": str(fight["room_type"]).upper(),
                "encounter_id": str(fight["encounter"]).upper(),
                "hp_lost": max(0.0, float(damage_taken)),
                "turns_taken": int(turns_taken),
                "hp_after": fight.get("hp_after"),
                "max_hp": fight.get("max_hp"),
                "deck": fight["deck"],
                "relics": fight["relics"],
            })

    records: List[Dict[str, Any]] = []
    by_stratum_counts: collections.Counter = collections.Counter()
    by_char_counts: collections.Counter = collections.Counter()

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "release.db"
        build_release_database(db_path, knowledge_path, community_path)
        repo = RelationalRepository(str(db_path))
        policy = DeckFitnessPolicy(repo)
        params = repo.mechanic_constant(PARAMETERS_KEY)
        cap = float(params["satisfaction_cap"])
        surplus_slope = float(params["surplus_slope"])
        vuln_bonus = float(params["vulnerable_damage_bonus"])
        block_share = float(params["block_share"])
        scaling_onset = float(params["scaling_onset_turns"])
        cycle_need = float(params["cycle_need"])
        orbs = policy._orb_facts()
        sovereign_blade = policy._card("SOVEREIGN_BLADE", 0)[0]

        def monster_fact(monster_id: str) -> Optional[Mapping]:
            fact = repo.find_assembly_fact("monsters", monster_id)
            return fact if fact is not None and fact.get("status") == "converted" else None

        for fight in fight_payloads:
            enc_fact = repo.find_assembly_fact("encounters", fight["encounter_id"])
            if enc_fact is None or enc_fact.get("status") != "converted":
                counts["encounter_unconverted"] += 1
                continue
            kind = fight["fight_kind"]
            if kind not in {"MONSTER", "ELITE", "BOSS"}:
                counts["unknown_room_type"] += 1
                continue
            nominal_turns = float(params["fight_turns"][kind])
            walked = encounter_turns(
                enc_fact, monster_fact, max(1, int(round(nominal_turns))), fight["ascension"]
            )
            if walked is None or any(g.startswith("monster_fact_missing") for g in walked.gaps):
                counts["monster_fact_missing"] += 1
                continue

            cards = []
            for entry in fight["deck"]:
                prof, _ = policy._card(str(entry["card"]).upper(), int(entry.get("upgrades") or 0))
                if prof is not None and entry.get("enchantment"):
                    prof, _ = enchant(
                        prof,
                        repo.find_assembly_fact("enchantments", str(entry["enchantment"]).upper()),
                        float(entry.get("enchantment_amount") or 1.0),
                    )
                cards.extend([prof] * int(entry.get("count") or 1))
            relics = []
            for relic_id in fight["relics"]:
                rprof, _ = policy._relic(str(relic_id).upper())
                if rprof is not None:
                    relics.append(rprof)

            enemies = max(1.0, float(walked.enemies))
            player_vuln = mean([min(1.0, v) for v in walked.debuffs["Vulnerable"]])
            player_weak = mean([min(1.0, v) for v in walked.debuffs["Weak"]])
            player_frail = mean([min(1.0, v) for v in walked.debuffs["Frail"]])
            incoming_attack = mean(walked.attack) * (1.0 + vuln_bonus * player_vuln)
            hp_before = (
                float(fight["hp_after"]) + float(fight["hp_lost"])
                if fight["hp_after"] is not None else float(fight["max_hp"] or 80)
            )
            max_hp = max(1.0, float(fight["max_hp"] or 80))
            hp_ratio = min(1.0, max(0.05, hp_before / max_hp))

            character = fight["character"]
            situation = Situation(
                fight_turns=nominal_turns,
                extra_enemies=max(0.0, enemies - 1.0),
                incoming_attack=incoming_attack,
                fight_mix={kind: 1.0},
                hp_ratio=hp_ratio,
                potion_count=0,
                orb_slots=float((params.get("base_orb_slots") or {}).get(character, 0.0)),
                arrow_slots=1.0 if character == "FOOL" else 0.0,
                orbs=orbs,
                enemy_hits=mean(walked.hits),
                player_weak=player_weak,
                player_frail=player_frail,
                status_cards=mean(walked.cards),
                enemy_hp=float(walked.hp),
                sovereign_blade=sovereign_blade,
            )
            capabilities, _ = deck_capabilities(cards, relics, situation, params)

            damage_need = float(walked.hp) / nominal_turns
            needs_map = {
                "frontload": damage_need,
                "aoe": damage_need * (enemies - 1.0) / enemies,
                "defense": incoming_attack * block_share,
                "scaling": damage_need * max(0.0, nominal_turns - scaling_onset) / nominal_turns,
                "cycle": cycle_need,
                "energy": 1.0,
            }
            features = capability_feature_vector(
                capabilities, needs_map, cap=cap, surplus_slope=surplus_slope
            )
            attack_load = incoming_attack * nominal_turns
            records.append({
                "run_id": fight["run_id"],
                "character": character,
                "ascension": fight["ascension"],
                "act": fight["act"],
                "floor": fight["floor"],
                "fight_kind": kind,
                "encounter_id": fight["encounter_id"],
                "hp_lost": fight["hp_lost"],
                "turns_taken": fight["turns_taken"],
                "attack_load": attack_load,
                "capabilities": dict(capabilities),
                "needs": needs_map,
                "features": features,
            })
            by_stratum_counts[f"act{fight['act']}:{kind}"] += 1
            by_char_counts[character] += 1

    distinct_runs = len({r["run_id"] for r in records})
    summary = {
        "sources": [str(collected_dir.relative_to(ROOT_DIR)), str(dev_dir.relative_to(ROOT_DIR))],
        "excluded_test_runs": len(excluded_test),
        "test_split_leakage": 0,
        "valid_runs": distinct_runs,
        "valid_fights": len(records),
        "counts": dict(counts),
        "fights_by_stratum": dict(sorted(by_stratum_counts.items())),
        "fights_by_character": dict(sorted(by_char_counts.items())),
    }
    return records, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--l2-mean", type=float, default=0.05)
    parser.add_argument("--l2-q75", type=float, default=0.05)
    parser.add_argument("--l2-attack", type=float, default=0.20)
    parser.add_argument("--prior-kind", type=float, default=15.0)
    parser.add_argument("--prior-stratum", type=float, default=25.0)
    args = parser.parse_args()

    records, summary = extract_combat_records()
    model = calibrate_from_records(
        records,
        n_splits=args.splits,
        dataset_summary=summary,
        l2_mean=args.l2_mean,
        l2_q75=args.l2_q75,
        l2_attack=args.l2_attack,
        prior_strength_kind=args.prior_kind,
        prior_strength_stratum=args.prior_stratum,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    ov = model.cv_metrics["overall"]
    print(f"Calibrated combat model written to {args.output}")
    print(
        f"Dataset: {summary['valid_runs']} runs, {summary['valid_fights']} fights "
        f"(excluded test runs: {summary['excluded_test_runs']}, leakage: 0)"
    )
    print(f"By stratum: {summary['fights_by_stratum']}")
    print(
        f"5-fold GroupKFold OOF: MAE={ov['oof_mae']:.3f} (naive={ov['naive_stratum_mae']:.3f}), "
        f"Pinball@0.75={ov['oof_pinball_q75']:.3f} (naive={ov['naive_stratum_pinball_q75']:.3f}), "
        f"Spearman(mean)={ov['oof_spearman_mean']:.3f}, Q75-cov={ov['oof_q75_coverage']:.3f}"
    )
    print("Global HP-saved weights (mean):", model.global_weights.to_dict()["mean"]["hp_saved_weights"])
    print("Global HP-saved weights (q75): ", model.global_weights.to_dict()["q75"]["hp_saved_weights"])
    for key, sw in sorted(model.by_stratum.items()):
        d = sw.to_dict()
        print(f"  {key:12} n={sw.sample_count:3} mean_w={d['mean']['hp_saved_weights']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
