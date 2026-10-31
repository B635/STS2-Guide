"""Gold, HP and max HP in deck-fitness points (TASK-027 ticket 03).

One scale for every decision that trades resources against the deck: a
relic that gives gold, an event that costs HP, a shop. The values are set by
hand from two anchors measured with the engine on the run history (see
``scripts/analyze_resource_exchange.py``), never fitted to what was chosen:

* gold: worth what it buys at the shops still ahead, best deals first. Each
  shop's marginal value of gold falls with the gold spent there (measured by
  band of gold, ``scripts/analyze_gold_curve.py``); the shops ahead are the
  rest of this act and the later acts, weighted by how many shops an act has
  and how often a run reaches it. So a little gold is worth a lot, a pile of
  gold much less per coin, and spending is cheap when rich;
* HP: the campfire rule rests below 60% HP, so at 60% a rest heal (30% of max
  HP) is worth about the best upgrade of the deck. Lower HP makes each point
  worth more, higher HP less (``threshold / ratio``, within set bounds);
* max HP: ``CreatureCmd.GainMaxHp`` raises the maximum then heals the same
  amount (0.111.0 IL); it also lasts: every rest heal is 30% of max HP
  (``HealRestSiteOption.GetBaseHealAmount``), so each point adds 0.3 HP to
  every rest heal still ahead in the run, priced at the threshold rate.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

PARAMETERS_KEY = "resource_exchange_v1"
CAMPFIRE_KEY = "campfire_rest_rule_v1"


def parameter_gaps(params, campfire) -> tuple[str, ...]:
    if not isinstance(params, Mapping):
        return (f"parameter_missing:{PARAMETERS_KEY}",)
    gaps = []
    curve = params.get("gold_marginal_points_per_100_by_act")
    if not isinstance(curve, Mapping) or not all(
        isinstance(curve.get(act), list) and curve[act]
        and all(_number(v) is not None and v >= 0 for v in curve[act]) for act in ("1", "2", "3")
    ):
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.gold_marginal_points_per_100_by_act")
    band = _number(params.get("gold_band"))
    if band is None or band <= 0:
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.gold_band")
    shops = params.get("shops_per_act")
    if not isinstance(shops, Mapping) or not all(
        _number(shops.get(act)) is not None and shops[act] >= 0 for act in ("1", "2", "3")
    ):
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.shops_per_act")
    reach = params.get("reach_next_act")
    if not isinstance(reach, Mapping) or not all(
        _number(reach.get(act)) is not None and 0 <= reach[act] <= 1 for act in ("1", "2")
    ):
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.reach_next_act")
    floors = _number(params.get("act_floors"))
    if floors is None or floors < 1:
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.act_floors")
    for name in ("hp_points_at_threshold", "rest_heal_share"):
        value = _number(params.get(name))
        if value is None or value < 0:
            gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.{name}")
    rests = params.get("rest_heals_per_act")
    if not isinstance(rests, Mapping) or not all(
        _number(rests.get(act)) is not None and rests[act] >= 0 for act in ("1", "2", "3")
    ):
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.rest_heals_per_act")
    scale = params.get("hp_value_scale")
    low = _number((scale or {}).get("min")) if isinstance(scale, Mapping) else None
    high = _number((scale or {}).get("max")) if isinstance(scale, Mapping) else None
    if low is None or high is None or not 0 < low <= high:
        gaps.append(f"parameter_invalid:{PARAMETERS_KEY}.hp_value_scale")
    threshold = _number((campfire or {}).get("heal_below_hp_ratio")) if isinstance(campfire, Mapping) else None
    if threshold is None or not 0 < threshold < 1:
        gaps.append(f"parameter_invalid:{CAMPFIRE_KEY}.heal_below_hp_ratio")
    return tuple(gaps)


def resource_points(
    resources: Mapping[str, float],
    world,
    params: Mapping,
    campfire: Mapping,
) -> tuple[float, List[dict], tuple[str, ...]]:
    """``(points, reasons, gaps)`` for gold / hp / max_hp changes.

    A reason is ``{"code", "delta", "message", "dimension"}``. A change that
    needs the current HP or gold when it is unknown becomes a gap.
    """

    total, reasons, gaps = 0.0, [], []
    gold = resources.get("gold") or 0.0
    if gold:
        if world.gold is None:
            return total, reasons, ("resource:gold_missing",)
        held = float(world.gold)
        if gold < 0:
            gold = -min(-gold, held)   # only the gold there is can be lost
        points = gold_points(held, gold, world, params)
        total += points
        reasons.append({
            "code": "resource:gold",
            "delta": round(points, 2),
            "message": f"{'获得' if gold > 0 else '失去'} {abs(gold):g} 金币（现有 {held:g}）：按之后的商店能买到的东西换算，"
                       "金币越多每枚越不值钱",
            "dimension": "resource_efficiency",
        })
    forgone = resources.get("gold_forgone") or 0.0
    if forgone:
        if world.gold is None:
            return total, reasons, ("resource:gold_missing",)
        points = -gold_points(float(world.gold), forgone, world, params)
        total += points
        reasons.append({
            "code": "resource:gold_income",
            "delta": round(points, 2),
            "message": f"之后约 {forgone:.0f} 金币收入拿不到（按每幕平均收入与走到下一幕的比例估计）",
            "dimension": "resource_efficiency",
        })
    hp_change = resources.get("hp") or 0.0
    max_change = resources.get("max_hp") or 0.0
    if hp_change or max_change:
        if world.hp is None or not world.max_hp:
            return total, reasons, ("resource:hp_missing",)
        hp, max_hp = float(world.hp), float(world.max_hp)
        if max_change:
            new_max = max(1.0, max_hp + max_change)
            # GainMaxHp heals the same amount; LoseMaxHp only caps current HP.
            healed = max_change if max_change > 0 else min(0.0, new_max - hp)
            points = _hp_points(hp, healed, new_max, params, campfire)
            lasting = (max_change * float(params["rest_heal_share"]) * rests_ahead(world, params)
                       * float(params["hp_points_at_threshold"]))
            total += points + lasting
            reasons.append({
                "code": "resource:max_hp",
                "delta": round(points + lasting, 2),
                "message": f"最大生命 {max_change:+g}（同时{'回复' if max_change > 0 else '失去'}生命；"
                           f"之后每次休息回复的生命也{'更多' if max_change > 0 else '更少'}）",
                "dimension": "survival",
            })
            hp, max_hp = max(0.0, min(new_max, hp + healed)), new_max
        if hp_change:
            change = min(hp_change, max_hp - hp) if hp_change > 0 else hp_change
            points = _hp_points(hp, change, max_hp, params, campfire)
            total += points
            reasons.append({
                "code": "resource:hp",
                "delta": round(points, 2),
                "message": f"生命 {change:+g}（当前 {hp:g}/{max_hp:g}；生命越低每点越值钱）",
                "dimension": "survival",
            })
    return total, reasons, tuple(gaps)


def share_left(world, params: Mapping) -> float:
    """Share of the current act still ahead, by floor."""

    act = min(3, max(1, int(world.act or 1)))
    floors = float(params["act_floors"])
    into = max(0.0, float(world.floor or 0) - floors * (act - 1) - 1.0)
    return max(0.0, 1.0 - into / floors)


def rests_ahead(world, params: Mapping) -> float:
    """Expected rest heals left in the run (later acts weighted by reaching them)."""

    act = min(3, max(1, int(world.act or 1)))
    total = share_left(world, params) * float(params["rest_heals_per_act"][str(act)])
    reach = 1.0
    for later in range(act + 1, 4):
        reach *= float(params["reach_next_act"][str(later - 1)])
        total += reach * float(params["rest_heals_per_act"][str(later)])
    return total


def gold_points(held: float, change: float, world, params: Mapping) -> float:
    """Points of gaining (``change`` > 0) or spending (< 0) gold when holding ``held``.

    The shops ahead each offer their deals best first: the marginal value of
    the g-th gold is the rate of the band covering g once every shop's bands
    (widened by how many such shops are expected) are sorted best first.
    """

    act = min(3, max(1, int(world.act or 1)))
    weights = {act: float(params["shops_per_act"][str(act)]) * share_left(world, params)}
    reach = 1.0
    for later in range(act + 1, 4):
        reach *= float(params["reach_next_act"][str(later - 1)])
        weights[later] = reach * float(params["shops_per_act"][str(later)])
    band = float(params["gold_band"])
    bands = sorted(
        ((float(rate) / 100.0, band * weight)
         for a, weight in weights.items() if weight > 0
         for rate in params["gold_marginal_points_per_100_by_act"][str(a)]),
        key=lambda b: -b[0],
    )

    def value(gold: float) -> float:
        """Points of holding ``gold``: the best bands it can buy."""
        total, left = 0.0, max(0.0, gold)
        for rate, width in bands:
            if left <= 0:
                break
            used = min(left, width)
            total += rate * used
            left -= used
        return total

    return value(held + change) - value(held)


def _hp_points(hp: float, change: float, max_hp: float, params: Mapping, campfire: Mapping) -> float:
    """Points of ``change`` HP starting from ``hp``, valued at the midpoint ratio."""

    if not change:
        return 0.0
    threshold = float(campfire["heal_below_hp_ratio"])
    scale = params["hp_value_scale"]
    middle = max(1.0, hp + change / 2.0) / max_hp
    factor = max(float(scale["min"]), min(float(scale["max"]), threshold / middle))
    return change * float(params["hp_points_at_threshold"]) * factor


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
