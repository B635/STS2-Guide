"""When does the player rest (HEAL) rather than smith (SMITH) at a rest site?

Input: the decisions file written by ``scripts/import_run_history.py
--all-runs`` (default ``build/history-all/decisions.json``; entries with
kind "rest"). Prints the heal rate by HP ratio and, in the 0.5-0.7 band, by
what the player could see at that moment, then checks the campfire rule
(``campfire_rest_rule_v1`` in data/guide_mechanics.json) and simpler and
richer alternatives on halves of the runs the thresholds were not read from.
Analysis only: it changes nothing.

Thresholds are read by hand from a training half's table: the lower edge of
the first 0.1-wide HP band whose heal rate is at most one half.
"""

import collections
import io
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from advisor.campfire import REST_RULE_KEY, rest_rule_parameter_gaps, rest_rule_prefers_heal  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DIR / "build" / "history-all" / "decisions.json"
decisions = [d for d in json.loads(path.read_text(encoding="utf-8"))["decisions"] if d["kind"] == "rest"]
params = json.loads((ROOT_DIR / "data" / "guide_mechanics.json").read_text(encoding="utf-8")).get(REST_RULE_KEY)
if rest_rule_parameter_gaps(params):
    raise SystemExit(f"{REST_RULE_KEY} is invalid: {rest_rule_parameter_gaps(params)}")

rows = []
for d in decisions:
    view = d["scoring_view"]
    state, world_map = view["state"], view["map_context"]
    if state.get("hp") is None or not state.get("max_hp"):
        continue
    nodes = world_map["nodes"]
    by_id = {node["node_id"]: node for node in nodes}
    here = by_id[world_map["current_node_id"]]
    boss_rows = {by_id[b]["row"] for b in world_map["boss_node_ids"] if b in by_id}
    # The map's structure: is the boss the very next row?
    boss_next = here["row"] + 1 in boss_rows
    # The walked path up to the next rest site or the boss is the player's own
    # plan; when the run ended before reaching either, it is unknown.
    ahead, reached = [], False
    for node in nodes[nodes.index(here) + 1:]:
        if node["kind"] == "CAMPFIRE":
            reached = True
            break
        ahead.append(node["kind"])
        if node["kind"] == "BOSS":
            reached = True
            break
    rows.append({
        "run": d["run"], "heal": d["chosen"] == "HEAL", "hp": state["hp"], "max_hp": state["max_hp"],
        "ratio": state["hp"] / state["max_hp"], "act": d["act"], "boss_next": boss_next,
        "fights": sum(kind in ("MONSTER", "ELITE") for kind in ahead) if reached else None,
        "elites": ahead.count("ELITE") if reached else None,
        "potions": len(state.get("potions") or ()),
    })


def band(ratio):
    return min(9, int(ratio * 10 + 1e-9)) / 10


def table(title, key, subset):
    groups = collections.defaultdict(list)
    for r in subset:
        k = key(r)
        if k is not None:
            groups[k].append(r["heal"])
    print(f"\n{title}")
    for k in sorted(groups, key=str):
        g = groups[k]
        print(f"  {str(k):24} heal {sum(g):3}/{len(g):3} = {sum(g) / len(g):4.0%}")


print(f"rest decisions {len(rows)} in {len({r['run'] for r in rows})} runs, heal {sum(r['heal'] for r in rows)}")
table("all: HP ratio band", lambda r: f"{band(r['ratio']):.1f}", rows)
middle = [r for r in rows if 0.5 <= r["ratio"] < 0.7]
table("HP 0.5-0.7: boss is the next row", lambda r: r["boss_next"], middle)
table("HP 0.5-0.7: fights before the next rest site or boss (walked path)",
      lambda r: None if r["fights"] is None else min(r["fights"], 2), middle)
table("HP 0.5-0.7: elites before the next rest site or boss (walked path)",
      lambda r: None if r["elites"] is None else min(r["elites"], 1), middle)
table("HP 0.5-0.7: act", lambda r: r["act"], middle)
table("HP 0.5-0.7: potions held", lambda r: min(r["potions"], 3), middle)
table("HP 0.5-0.7: max HP", lambda r: "<70" if r["max_hp"] < 70 else ("70-79" if r["max_hp"] < 80 else "80+"), middle)
table("all: HP band x boss next", lambda r: (f"{band(r['ratio']):.1f}", r["boss_next"]), rows)


def read_threshold(subset):
    """Lower edge of the first band (from the bottom) healed at most half the time."""
    rates = collections.defaultdict(list)
    for r in subset:
        rates[band(r["ratio"])].append(r["heal"])
    for edge in sorted(rates):
        if sum(rates[edge]) * 2 <= len(rates[edge]):
            return edge
    return 1.0


def accuracy(rule, subset):
    hits = sum(rule(r) == r["heal"] for r in subset)
    return f"{hits:3}/{len(subset):3} = {hits / len(subset):5.1%}"


# Runs are split by run id: older/newer half (run ids are start times) and
# alternating ids. Thresholds come from one half and are checked on the other.
runs = sorted({r["run"] for r in rows}, key=int)
splits = {
    "older/newer": (set(runs[: len(runs) // 2]), set(runs[len(runs) // 2:])),
    "alternating": (set(runs[0::2]), set(runs[1::2])),
}
shipped = lambda r: rest_rule_prefers_heal(r["hp"], r["max_hp"], params)  # noqa: E731
print(f"\nshipped rule: heal below {params['heal_below_hp_ratio']:.0%} of max HP")
for name, halves in splits.items():
    for train_index, test_index in ((0, 1), (1, 0)):
        train = [r for r in rows if r["run"] in halves[train_index]]
        test = [r for r in rows if r["run"] in halves[test_index]]
        hp_only = read_threshold(train)
        normal = read_threshold([r for r in train if not r["boss_next"]])
        boss = read_threshold([r for r in train if r["boss_next"]])
        print(f"\n{name}: thresholds read from half {train_index + 1} ({len(train)} decisions), "
              f"checked on half {test_index + 1} ({len(test)} decisions)")
        for label, rule in (
            ("always SMITH", lambda r: False),
            ("always HEAL", lambda r: True),
            ("HP only, heal below 0.5", lambda r: r["ratio"] < 0.5),
            (f"HP only, read: below {hp_only:.1f}", lambda r: r["ratio"] < hp_only),
            (f"HP + boss next, read: {normal:.1f} / boss {boss:.1f}",
             lambda r: r["ratio"] < (boss if r["boss_next"] else normal)),
            ("shipped rule", shipped),
        ):
            print(f"  {label:38} {accuracy(rule, test)}")
