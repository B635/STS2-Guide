"""Compare the engine's fight expectations with what really happened.

Input: ``build/history/fights.json`` from ``scripts/import_run_history.py``
(every fight of the imported runs with its encounter, the turns it took and
the damage taken). For each fight the deck and relics of that moment go
through the deck engine against that encounter's facts, and the script prints
predicted against actual fight length and damage taken. Analysis only: it
changes nothing.
"""

import sys, json, tempfile, collections, io, statistics, math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from storage.release_database import build_release_database
from storage.relational import RelationalRepository
from advisor.deck_fitness import DeckFitnessPolicy, PARAMETERS_KEY
from advisor.deck_mechanics import Situation, deck_capabilities, enchant
from advisor.encounter_turns import encounter_turns
from advisor.monster_turns import mean

source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DIR / 'build' / 'history' / 'fights.json'
fights = json.loads(source.read_text(encoding='utf-8'))['fights']
runs_meta = {}
rows = []; skipped = collections.Counter()
with tempfile.TemporaryDirectory() as t:
    db = Path(t) / 'r.db'
    build_release_database(db, ROOT_DIR / 'data' / 'knowledge.json', ROOT_DIR / 'data' / 'community_scores.json')
    repo = RelationalRepository(str(db))
    policy = DeckFitnessPolicy(repo)
    params = repo.mechanic_constant(PARAMETERS_KEY)

    def monster_fact(mid):
        f = repo.find_assembly_fact("monsters", mid)
        return f if f is not None and f.get("status") == "converted" else None

    for f in fights:
        T = f.get('turns_taken')
        if not T or f.get('damage_taken') is None:
            skipped['no_turns'] += 1; continue
        enc = repo.find_assembly_fact("encounters", f['encounter'])
        if enc is None or enc.get('status') != 'converted':
            skipped['encounter_unknown'] += 1; continue
        cards = []; gaps = 0
        for e in f['deck']:
            p, g = policy._card(e['card'], e['upgrades'])
            if p is not None and e.get('enchantment'):
                p, g2 = enchant(p, repo.find_assembly_fact("enchantments", e['enchantment']), float(e.get('enchantment_amount') or 1))
                g = (*g, *g2)
            gaps += bool(g) * e['count'] + (p is None) * e['count']
            cards.extend([p] * e['count'])
        relics = []
        for r in f['relics']:
            p, g = policy._relic(r)
            gaps += bool(g) or p is None
            if p is not None:
                relics.append(p)
        kind = {'monster': 'MONSTER', 'elite': 'ELITE', 'boss': 'BOSS'}[f['room_type']]
        nominal = float(params['fight_turns'][kind])

        def caps(turns):
            walked = encounter_turns(enc, monster_fact, max(1, int(round(turns))), 0)
            if walked is None or any(g.startswith('monster_fact_missing') for g in walked.gaps):
                return None, None
            sit = Situation(
                float(turns), max(0.0, walked.enemies - 1.0), mean(walked.attack), {kind: 1.0},
                (f['hp_after'] or 1) / (f['max_hp'] or 1), 0,
                orb_slots=3.0 if 'DEFECT' in str(f['deck']) else 0.0, orbs=policy._orb_facts(),
                enemy_hits=mean(walked.hits), enemy_hp=walked.hp,
                player_weak=mean([min(1.0, v) for v in walked.debuffs['Weak']]),
                player_frail=mean([min(1.0, v) for v in walked.debuffs['Frail']]),
                status_cards=mean(walked.cards), sovereign_blade=policy._card('SOVEREIGN_BLADE', 0)[0],
            )
            return deck_capabilities(cards, relics, sit, params)[0], walked

        c, w = caps(nominal)
        if c is None:
            skipped['monster_unknown'] += 1; continue
        dmg = c['frontload'] + c['aoe'] + c['scaling']
        t_pred = w.hp / dmg if dmg > 0 else 99.0
        # One refinement at the predicted length.
        c2, w2 = caps(min(30.0, max(1.0, t_pred)))
        dmg2 = c2['frontload'] + c2['aoe'] + c2['scaling']
        t_pred2 = w2.hp / dmg2 if dmg2 > 0 else 99.0
        # Damage taken over the turns the fight really lasted.
        ca, wa = caps(float(T))
        taken_pred = sum(max(0.0, a - ca['defense']) for a in wa.attack)
        # The same block spent only where the attacks are (no block wasted).
        taken_mean = max(0.0, sum(wa.attack) - ca['defense'] * float(T))
        rows.append({
            'kind': kind, 'act': f['act'], 'T': T, 'T_pred': t_pred2, 'T_nominal': nominal,
            'taken': f['damage_taken'], 'taken_pred': taken_pred, 'taken_mean': taken_mean,
            'attack_total': sum(wa.attack), 'gaps': gaps, 'deck': len(cards),
            'hp': w.hp, 'attack': mean(wa.attack), 'block': ca['defense'], 'enc': f['encounter'],
        })
    del policy, repo

print('fights', len(fights), 'used', len(rows), 'skipped', dict(skipped))


def corr(xs, ys):
    if len(xs) < 3:
        return float('nan')
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float('nan')


def report(title, sel):
    if not sel:
        return
    T = [r['T'] for r in sel]; P = [r['T_pred'] for r in sel]; N = [r['T_nominal'] for r in sel]
    D = [r['taken'] for r in sel]; DP = [r['taken_pred'] for r in sel]; DM = [r['taken_mean'] for r in sel]
    print(f"{title:22} n={len(sel):3} | turns actual {statistics.fmean(T):4.1f} predicted {statistics.fmean(P):5.1f} "
          f"(median ratio {statistics.median(p / t for p, t in zip(P, T)):4.2f}, corr {corr(P, T):4.2f}) nominal {statistics.fmean(N):3.1f}"
          f" | damage actual {statistics.fmean(D):4.1f} per-turn block {statistics.fmean(DP):5.1f} (corr {corr(DP, D):4.2f})"
          f" block where needed {statistics.fmean(DM):5.1f} (corr {corr(DM, D):4.2f})")


report('all', rows)
for kind in ('MONSTER', 'ELITE', 'BOSS'):
    report(kind, [r for r in rows if r['kind'] == kind])
for act in (1, 2, 3):
    report(f'act {act}', [r for r in rows if r['act'] == act])
report('fully modeled decks', [r for r in rows if r['gaps'] == 0])
report('decks with gaps', [r for r in rows if r['gaps'] > 0])
print()
print('mean engine inputs: enemy hp', round(statistics.fmean(r['hp'] for r in rows), 1),
      'attack/turn', round(statistics.fmean(r['attack'] for r in rows), 1),
      'block/turn', round(statistics.fmean(r['block'] for r in rows), 1))
worst = sorted(rows, key=lambda r: -abs(r['T_pred'] - r['T']))[:8]
print('largest turn errors:')
for r in worst:
    print(f"  {r['enc']:32} act{r['act']} deck {r['deck']:2} gaps {r['gaps']:2} actual {r['T']:2} predicted {r['T_pred']:5.1f} hp {r['hp']:.0f}")

print()
print("share of the enemies' attack that got through, by block per turn / attack per turn:")
edges = [0.0, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 99.0]
for lo, hi in zip(edges, edges[1:]):
    sel = [r for r in rows if r['attack'] > 0 and lo <= r['block'] / r['attack'] < hi]
    if sel:
        through = sum(r['taken'] for r in sel) / sum(r['attack_total'] for r in sel)
        print(f"  block/attack {lo:3.1f}..{hi:4.1f}  n={len(sel):3}  got through {through:.0%}  mean turns {statistics.fmean(r['T'] for r in sel):.1f}")

print()
print("the same share by how fast the engine expects the deck to kill (predicted turns / nominal turns):")
edges = [0.0, 0.7, 1.0, 1.3, 1.8, 99.0]
for lo, hi in zip(edges, edges[1:]):
    sel = [r for r in rows if lo <= r['T_pred'] / r['T_nominal'] < hi]
    if sel:
        through = sum(r['taken'] for r in sel) / max(1e-9, sum(r['attack_total'] for r in sel))
        print(f"  predicted/nominal {lo:3.1f}..{hi:4.1f}  n={len(sel):3}  got through {through:.0%}"
              f"  damage taken {statistics.fmean(r['taken'] for r in sel):5.1f}  actual turns {statistics.fmean(r['T'] for r in sel):.1f}")
print()
print("by room: attack aimed at the player over the fight, and what got through")
for kind in ('MONSTER', 'ELITE', 'BOSS'):
    sel = [r for r in rows if r['kind'] == kind]
    print(f"  {kind:8} attack over the fight {statistics.fmean(r['attack_total'] for r in sel):6.1f}"
          f"  taken {statistics.fmean(r['taken'] for r in sel):5.1f}"
          f"  got through {sum(r['taken'] for r in sel) / sum(r['attack_total'] for r in sel):.0%}"
          f"  corr(attack over the fight, taken) {corr([r['attack_total'] for r in sel], [r['taken'] for r in sel]):.2f}")

