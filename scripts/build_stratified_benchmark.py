"""Build a stratified 240+ card-reward benchmark from non-test finished runs.

Sources:
- ``coach/eval/runs/dev/``
- ``build/collected-runs/`` (strictly excluding every run in ``coach/eval/split.json["test"]``
  and never touching ``coach/eval/runs/test/``).

Stratification axes:
  character (IRONCLAD / SILENT / DEFECT / REGENT / NECROBINDER)
  x act (1 / 2 / 3)
  x decision_type (pick / skip)
  x margin_bucket (high_margin >= 2.0 / close_call < 2.0)

Preserves the full ``scoring_view`` (deck, relics, potions, floor, candidates) plus
player choice and baseline engine reference without modifying the 79-case golden set.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from advisor.golden_set import card_reward_request
from advisor.history_requests import starting_decks, unscorable_reason
from advisor.run_history import RunHistoryError, bare_id, load_run, replay_run
from review.benchmark_stats import bootstrap_rate_ci, cohens_kappa
from review.engine import Engine, KNOWLEDGE_PATH
from review.run_review import POLICY_LEANINGS, STRICT_GAP

DEFAULT_SPLIT_PATH = ROOT_DIR / "coach" / "eval" / "split.json"
DEFAULT_DEV_RUNS_DIR = ROOT_DIR / "coach" / "eval" / "runs" / "dev"
DEFAULT_COLLECTED_RUNS_DIR = ROOT_DIR / "build" / "collected-runs"
DEFAULT_OUT_PATH = ROOT_DIR / "data" / "card_reward_stratified_bench.json"


def _deck_score(cand: Any) -> Optional[float]:
    if cand.score is None:
        return None
    leaning = sum(
        float(f.get("delta") or 0.0)
        for f in cand.factors
        if f.get("code") in POLICY_LEANINGS
    )
    return round(float(cand.score) - leaning, 2)


def discover_non_test_runs(
    *,
    split_path: Path = DEFAULT_SPLIT_PATH,
    dev_dir: Path = DEFAULT_DEV_RUNS_DIR,
    collected_dir: Path = DEFAULT_COLLECTED_RUNS_DIR,
) -> Tuple[List[Tuple[str, Path, str]], Set[str]]:
    """Return deduplicated (run_id, path, origin) tuples strictly excluding test runs."""
    split = json.loads(split_path.read_text(encoding="utf-8"))
    excluded_test_ids: Set[str] = set(split.get("test") or ())
    dev_ids: Set[str] = set(split.get("dev") or ())

    seen: Dict[str, Tuple[Path, str]] = {}
    if dev_dir.is_dir():
        for path in sorted(dev_dir.rglob("*.run")):
            rid = path.stem
            if rid in excluded_test_ids:
                continue
            seen[rid] = (path, "coach_dev")

    if collected_dir.is_dir():
        for path in sorted(collected_dir.rglob("*.run")):
            # Belt-and-suspenders guard against any test directory path.
            if "runs/test" in path.as_posix().lower():
                continue
            rid = path.stem
            if rid in excluded_test_ids or rid in seen:
                continue
            origin = "coach_dev" if rid in dev_ids else "collected_non_test"
            seen[rid] = (path, origin)

    ordered = [(rid, seen[rid][0], seen[rid][1]) for rid in sorted(seen.keys())]
    return ordered, excluded_test_ids


def extract_candidate_pool(
    runs: Sequence[Tuple[str, Path, str]],
    *,
    engine: Optional[Engine] = None,
    margin_threshold: float = STRICT_GAP,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Replay eligible non-test runs and collect every scorable card-reward decision."""
    eng = engine or Engine()
    decks = starting_decks(KNOWLEDGE_PATH)
    pool: List[Dict[str, Any]] = []
    run_stats: Dict[str, int] = collections.Counter()

    for rid, path, origin in runs:
        try:
            raw = load_run(path)
        except Exception:
            run_stats["unreadable"] += 1
            continue
        players = raw.get("players") or ()
        if len(players) != 1:
            run_stats["multiplayer_skipped"] += 1
            continue
        if raw.get("was_abandoned"):
            run_stats["abandoned_skipped"] += 1
            continue
        character = bare_id(players[0]["character"])
        if character not in decks:
            run_stats["unknown_character_skipped"] += 1
            continue
        try:
            replay = replay_run(raw, decks[character], rid)
        except (AttributeError, KeyError, TypeError, ValueError, RunHistoryError):
            run_stats["replay_error_skipped"] += 1
            continue
        if replay["problems"]:
            run_stats["replay_mismatch_skipped"] += 1
            continue

        run_stats["eligible_runs"] += 1
        seen_floor_counts: Dict[int, int] = collections.Counter()
        for sit in replay["situations"]:
            view = sit["scoring_view"]
            if unscorable_reason(view, eng.known_cards, eng.known_encounters) is not None:
                run_stats["unscorable_card_rewards"] += 1
                continue

            floor = int(sit["source"]["floor"])
            seen_floor_counts[floor] += 1
            suffix = "" if seen_floor_counts[floor] == 1 else f"#{seen_floor_counts[floor]}"
            state = view["state"]
            act = int(state["act"])
            best_list = sit.get("best") or []
            if not best_list:
                continue
            player_choice = str(best_list[0])
            decision_type = "skip" if player_choice == "skip" else "pick"

            request = card_reward_request(view)
            rec = eng.card_rewards.recommend(request)
            ranked = sorted(
                rec.candidates,
                key=lambda c: (c.rank is None, c.rank or 999, c.display_index),
            )
            scored_pairs: List[Tuple[str, float]] = []
            for c in ranked:
                ds = _deck_score(c)
                if ds is not None:
                    scored_pairs.append((c.candidate_id, ds))
            if len(scored_pairs) < 2 or rec.recommended_candidate_id is None:
                run_stats["insufficient_scores_skipped"] += 1
                continue

            deck_scores_sorted = sorted((ds for _, ds in scored_pairs), reverse=True)
            top1_top2_margin = round(deck_scores_sorted[0] - deck_scores_sorted[1], 2)
            margin_bucket = "high_margin" if top1_top2_margin >= margin_threshold else "close_call"

            score_by_id = dict(scored_pairs)
            engine_top = str(rec.recommended_candidate_id)
            top_ds = score_by_id.get(engine_top)
            chosen_ds = score_by_id.get(player_choice)
            gap_to_player = (
                round(top_ds - chosen_ds, 2)
                if top_ds is not None and chosen_ds is not None
                else None
            )

            stratum = f"{character}|act{act}|{decision_type}|{margin_bucket}"
            entry_id = f"{character.lower()}-{rid}-f{floor:02d}{suffix}"
            candidate_summaries = [
                {
                    "candidate_id": c["candidate_id"],
                    "entity_id": c.get("entity_id"),
                    "upgraded": bool(c.get("upgraded", False)),
                }
                for c in view.get("candidates") or ()
            ]
            pool.append(
                {
                    "id": entry_id,
                    "character": character,
                    "act": act,
                    "floor": floor,
                    "decision_type": decision_type,
                    "margin_bucket": margin_bucket,
                    "stratum": stratum,
                    "player_choice": player_choice,
                    "best": [player_choice],
                    "label_status": "observed_player_choice",
                    "source": {
                        "kind": "run_history",
                        "run": rid,
                        "split_origin": origin,
                        "build_id": sit["source"].get("build_id"),
                        "win": bool(sit["source"].get("win", False)),
                        "floor": floor,
                    },
                    "deck_summary": {
                        "deck_size": sum(int(x.get("count", 1)) for x in state.get("deck") or ()),
                        "relic_count": len(state.get("relics") or ()),
                        "relics": list(state.get("relics") or ()),
                        "hp": state.get("hp"),
                        "max_hp": state.get("max_hp"),
                        "gold": state.get("gold"),
                    },
                    "candidates": candidate_summaries,
                    "baseline_reference": {
                        "engine_top": engine_top,
                        "agreed": engine_top == player_choice,
                        "top1_top2_margin": top1_top2_margin,
                        "gap_to_player": gap_to_player,
                        "confidence": rec.confidence,
                    },
                    "scoring_view": view,
                }
            )
    return pool, dict(run_stats)


def stratified_sample(
    pool: Sequence[Mapping[str, Any]],
    *,
    target_size: int = 260,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Water-filling stratified sample maximizing balance across Character x Act x Type x Margin."""
    if target_size >= len(pool):
        return [dict(x) for x in pool]

    rng = random.Random(seed)
    by_stratum: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for item in pool:
        by_stratum[str(item["stratum"])].append(dict(item))

    # Shuffle deterministically within each stratum while balancing across distinct runs.
    for key in sorted(by_stratum.keys()):
        items = by_stratum[key]
        rng.shuffle(items)
        # Round-robin by run inside the stratum so no single run dominates a stratum.
        by_run: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for it in items:
            by_run[str(it["source"]["run"])].append(it)
        run_keys = sorted(by_run.keys())
        rng.shuffle(run_keys)
        interleaved: List[Dict[str, Any]] = []
        while any(by_run[rk] for rk in run_keys):
            for rk in run_keys:
                if by_run[rk]:
                    interleaved.append(by_run[rk].pop())
        by_stratum[key] = interleaved

    # Water-filling round-robin across strata ensures minority strata (SILENT, Act 3, skip,
    # close_call) keep 100% representation while dominant strata are proportionally capped.
    selected: List[Dict[str, Any]] = []
    strata_keys = sorted(by_stratum.keys())
    cursors = {k: 0 for k in strata_keys}
    while len(selected) < target_size:
        progressed = False
        for k in strata_keys:
            if len(selected) >= target_size:
                break
            idx = cursors[k]
            if idx < len(by_stratum[k]):
                selected.append(by_stratum[k][idx])
                cursors[k] = idx + 1
                progressed = True
        if not progressed:
            break

    selected.sort(key=lambda x: (x["character"], x["act"], x["source"]["run"], x["floor"], x["id"]))
    return selected


def build_benchmark(
    *,
    target_size: int = 260,
    seed: int = 42,
    split_path: Path = DEFAULT_SPLIT_PATH,
    dev_dir: Path = DEFAULT_DEV_RUNS_DIR,
    collected_dir: Path = DEFAULT_COLLECTED_RUNS_DIR,
) -> Dict[str, Any]:
    runs, excluded_test_ids = discover_non_test_runs(
        split_path=split_path,
        dev_dir=dev_dir,
        collected_dir=collected_dir,
    )
    pool, run_stats = extract_candidate_pool(runs)
    selected = stratified_sample(pool, target_size=target_size, seed=seed)

    # Hard assertion: zero test-run contamination.
    selected_run_ids = {s["source"]["run"] for s in selected}
    leaked = selected_run_ids & excluded_test_ids
    if leaked:
        raise RuntimeError(f"Test run leakage detected: {sorted(leaked)}")

    by_character = dict(sorted(collections.Counter(s["character"] for s in selected).items()))
    by_act = {f"act{k}": v for k, v in sorted(collections.Counter(s["act"] for s in selected).items())}
    by_decision_type = dict(sorted(collections.Counter(s["decision_type"] for s in selected).items()))
    by_margin_bucket = dict(sorted(collections.Counter(s["margin_bucket"] for s in selected).items()))
    by_stratum = dict(sorted(collections.Counter(s["stratum"] for s in selected).items()))

    hits = [bool(s["baseline_reference"]["agreed"]) for s in selected]
    player_actions = [s["decision_type"] for s in selected]
    engine_actions = [
        "skip" if s["baseline_reference"]["engine_top"] == "skip" else "pick"
        for s in selected
    ]
    rate_ci = bootstrap_rate_ci(hits, n_resamples=5000, seed=seed)
    action_kappa = cohens_kappa(player_actions, engine_actions, n_bootstrap=2000, seed=seed + 1)

    return {
        "schema_version": 1,
        "seed": seed,
        "note": (
            "大规模分层真实选牌评测集（来自 coach/eval/runs/dev 与 build/collected-runs 非测试对局，"
            "严格排除全部 15 局 coach/eval/runs/test 对局）。不替代 data/golden_set/card_reward.json "
            "79 题人工标注金样本；用于高功效配对策略比较、Bootstrap 95% 置信区间、McNemar 显著性检验与 Cohen's kappa。"
        ),
        "excluded_test_runs": sorted(excluded_test_ids),
        "summary": {
            "total_situations": len(selected),
            "candidate_pool_size": len(pool),
            "unique_runs": len(selected_run_ids),
            "run_filtering": run_stats,
            "by_character": by_character,
            "by_act": by_act,
            "by_decision_type": by_decision_type,
            "by_margin_bucket": by_margin_bucket,
            "unique_strata": len(by_stratum),
            "baseline_agreement_ci": rate_ci.to_dict(),
            "baseline_pick_vs_skip_kappa": action_kappa.to_dict(),
        },
        "strata_counts": by_stratum,
        "situations": selected,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build stratified 240+ card-reward benchmark JSON.")
    parser.add_argument("--target-size", type=int, default=260)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args(argv)

    payload = build_benchmark(target_size=args.target_size, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")

    s = payload["summary"]
    ci = s["baseline_agreement_ci"]
    kp = s["baseline_pick_vs_skip_kappa"]
    print(
        f"Wrote {args.out}: {s['total_situations']} situations from {s['unique_runs']} non-test runs "
        f"(pool={s['candidate_pool_size']}, strata={s['unique_strata']})"
    )
    print(f"  by_character: {s['by_character']}")
    print(f"  by_act: {s['by_act']}")
    print(f"  by_decision_type: {s['by_decision_type']}")
    print(f"  by_margin_bucket: {s['by_margin_bucket']}")
    print(
        f"  baseline agreement: {ci['mean']:.4f} 95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}] | "
        f"pick/skip kappa={kp['kappa']:.4f} [{kp['ci_low']:.4f}, {kp['ci_high']:.4f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
