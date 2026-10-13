"""When does the player skip a card reward, and do the policies see it coming?

Input: a situation file in the golden-set format (default
``build/history/card_rewards.json`` from ``scripts/import_run_history.py``).
Prints the skip rate by act, deck size, room, offered rarity and by each
policy version's own scores. Analysis only: it changes nothing.
"""

import sys, json, tempfile, collections, io
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from advisor.golden_set import CARD_REWARD_VERSIONS, PolicyRegistry, card_reward_request, load_golden_set
from storage.release_database import build_release_database
from storage.relational import RelationalRepository

path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT_DIR / 'build' / 'history' / 'card_rewards.json'
gold = load_golden_set(path)
know = json.loads((ROOT_DIR / 'data' / 'knowledge.json').read_text(encoding='utf-8'))
rarity = {c['id']: c.get('rarity_key') for c in know['cards']}
rows = []
with tempfile.TemporaryDirectory() as t:
    db = Path(t) / 'r.db'
    build_release_database(db, ROOT_DIR / 'data' / 'knowledge.json', ROOT_DIR / 'data' / 'community_scores.json')
    repo = RelationalRepository(str(db))
    regs = {v: PolicyRegistry([CARD_REWARD_VERSIONS[v](repo)]) for v in ('v1', 'v2')}
    for s in gold:
        view = s['scoring_view']; st = view['state']
        sc = {}
        for v, reg in regs.items():
            rec = reg.recommend(card_reward_request(view))
            sc[v] = {c.candidate_id: c.score for c in rec.candidates if c.score is not None}
        cards = [c for c in view['candidates'] if c['entity_id']]
        node = next(n for n in view['map_context']['nodes'] if n['node_id'] == view['map_context']['current_node_id'])
        v1c = [x for k, x in sc['v1'].items() if k != 'skip']
        v2c = [x for k, x in sc['v2'].items() if k != 'skip']
        rows.append({
            'skip': s['best'] == ['skip'], 'char': s['character'], 'act': st['act'], 'floor': st['floor'],
            'deck': sum(r['count'] for r in st['deck']), 'room': node['kind'],
            'hp': (st['hp'] or 0) / (st['max_hp'] or 1),
            'rar': sorted(str(rarity.get(c['entity_id'])) for c in cards),
            'v1best': max(v1c) if v1c else None,
            'v2gain': (max(v2c) - sc['v2']['skip']) if v2c and 'skip' in sc['v2'] else None,
            'win': s['source'].get('win'), 'run': s['source'].get('run'),
        })
    del regs, repo


def table(title, key, order=None):
    groups = collections.defaultdict(list)
    for r in rows:
        k = key(r)
        if k is not None:
            groups[k].append(r['skip'])
    print(f"\n{title}")
    for k in (order or sorted(groups)):
        if k in groups:
            g = groups[k]
            print(f"  {str(k):22} skip {sum(g):3}/{len(g):3} = {sum(g)/len(g):.0%}")


def bucket(value, edges, fmt="{}"):
    if value is None:
        return None
    for lo, hi in zip(edges, edges[1:]):
        if lo <= value < hi:
            return f"{fmt.format(lo)}..{fmt.format(hi)}"
    return None


print('situations', len(rows), 'skip', sum(r['skip'] for r in rows))
table('by act', lambda r: r['act'])
table('by deck size', lambda r: bucket(r['deck'], [0, 12, 15, 18, 22, 26, 99]),
      ['0..12', '12..15', '15..18', '18..22', '22..26', '26..99'])
table('by room', lambda r: r['room'])
table('by character', lambda r: r['char'])
table('by hp ratio', lambda r: bucket(r['hp'], [0, 0.3, 0.5, 0.7, 1.01], "{:.1f}"))
table('by best offered rarity', lambda r: 'has Rare' if 'Rare' in r['rar'] else ('has Uncommon' if 'Uncommon' in r['rar'] else 'all Common'))
table('by v1 best card score', lambda r: bucket(r['v1best'], [0, 50, 55, 60, 65, 70, 200]),
      ['0..50', '50..55', '55..60', '60..65', '65..70', '70..200'])
table('by v2 best gain over skip', lambda r: bucket(r['v2gain'], [-99, -1, 0, 1, 2, 4, 99]),
      ['-99..-1', '-1..0', '0..1', '1..2', '2..4', '4..99'])
table('act x deck size', lambda r: f"act{r['act']} " + ('<15' if r['deck'] < 15 else '15-21' if r['deck'] < 22 else '22+'))
runs = collections.defaultdict(list)
for r in rows:
    runs[r['run']].append(r)
print('\nper run: skips/rewards, final deck size at last reward')
for k, g in runs.items():
    print(f"  {k} {g[0]['char']:12} skip {sum(x['skip'] for x in g):2}/{len(g):2} deck at last reward {g[-1]['deck']}")

# How well would simple rules predict skip?
def rule_eval(name, pred):
    tp = sum(pred(r) and r['skip'] for r in rows); fp = sum(pred(r) and not r['skip'] for r in rows)
    fn = sum((not pred(r)) and r['skip'] for r in rows)
    print(f"  {name:44} says skip {tp+fp:3}  right {tp:3}  wrong {fp:3}  missed {fn:3}")
print('\nrules for "skip"')
rule_eval('v2 gain <= 0', lambda r: r['v2gain'] is not None and r['v2gain'] <= 0)
rule_eval('v2 gain <= 1', lambda r: r['v2gain'] is not None and r['v2gain'] <= 1)
rule_eval('v2 gain <= 2', lambda r: r['v2gain'] is not None and r['v2gain'] <= 2)
rule_eval('v1 best < 55', lambda r: r['v1best'] is not None and r['v1best'] < 55)
rule_eval('v1 best < 60', lambda r: r['v1best'] is not None and r['v1best'] < 60)
rule_eval('act >= 2', lambda r: r['act'] >= 2)
rule_eval('deck >= 18', lambda r: r['deck'] >= 18)
rule_eval('act >= 2 and v1 best < 60', lambda r: r['act'] >= 2 and r['v1best'] is not None and r['v1best'] < 60)
rule_eval('act >= 2 and v2 gain <= 2', lambda r: r['act'] >= 2 and r['v2gain'] is not None and r['v2gain'] <= 2)
