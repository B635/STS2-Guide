import requests
import json
import time
import re
import os

BASE_URL = "https://spire-codex.com/api"
LANG = "zhs"
RATE_LIMIT_DELAY = 1.1
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_FILE = os.path.join(ROOT_DIR, "data", "knowledge.json")


def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\[/?[^\]]+\]', '', text)
    return text.strip()


def fetch_with_delay(endpoint, params=None):
    if params is None:
        params = {}
    params["lang"] = LANG
    response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
    time.sleep(RATE_LIMIT_DELAY)
    return response.json()


def pick(d, keys):
    return {k: d.get(k) for k in keys}


def build_character(raw):
    fields = pick(raw, [
        "id", "name", "description",
        "starting_hp", "starting_gold", "max_energy", "orb_slots",
        "starting_relics", "starting_deck", "unlocks_after",
    ])
    fields["description"] = clean_text(fields.get("description") or "")

    name = fields["name"]
    desc = fields["description"]
    base = f"角色{name}：{desc}初始血量{fields['starting_hp']}点，初始能量{fields['max_energy']}点"
    if fields.get("orb_slots"):
        base += f"，球槽数量{fields['orb_slots']}"
    if fields.get("starting_relics"):
        base += f"，初始遗物：{'、'.join(fields['starting_relics'])}"
    fields["embed_text"] = base
    return fields


def _clean_powers_applied(powers):
    if not powers:
        return None
    cleaned = []
    for p in powers:
        cleaned.append({
            "id": p.get("id"),
            "name": p.get("name"),
            "amount": p.get("amount"),
            "target": p.get("target"),
        })
    return cleaned


def build_card(raw):
    fields = pick(raw, [
        "id", "name", "description",
        "cost", "is_x_cost", "is_x_star_cost", "star_cost",
        "type", "type_key", "rarity", "rarity_key", "color", "target",
        "damage", "block", "hit_count",
        "cards_draw", "energy_gain", "hp_loss",
        "keywords", "keywords_key", "tags", "spawns_cards",
        "vars", "upgrade", "upgrade_description", "type_variants",
    ])
    fields["description"] = clean_text(fields.get("description") or "")
    fields["upgrade_description"] = clean_text(fields.get("upgrade_description") or "") or None
    fields["powers_applied"] = _clean_powers_applied(raw.get("powers_applied"))

    name = fields["name"]
    cost = fields["cost"]
    color = fields.get("color") or ""
    rarity = fields.get("rarity") or ""
    ctype = fields.get("type") or ""
    desc = fields["description"]

    header_bits = [b for b in [color, rarity, ctype, f"费用{cost}"] if b]
    parts = [f"卡牌{name}（{'，'.join(header_bits)}）：{desc}"]
    if fields.get("damage") is not None:
        parts.append(f"伤害{fields['damage']}")
    if fields.get("block") is not None:
        parts.append(f"格挡{fields['block']}")
    if fields.get("hit_count"):
        parts.append(f"攻击次数{fields['hit_count']}")
    if fields.get("keywords"):
        parts.append(f"关键词：{'、'.join(fields['keywords'])}")
    if fields.get("tags"):
        parts.append(f"标签：{'、'.join(fields['tags'])}")
    if fields.get("upgrade_description"):
        parts.append(f"升级后：{fields['upgrade_description']}")
    fields["embed_text"] = "，".join(parts)
    return fields


def build_relic(raw):
    fields = pick(raw, [
        "id", "name", "description",
        "rarity", "rarity_key", "pool", "flavor",
    ])
    fields["description"] = clean_text(fields.get("description") or "")
    fields["flavor"] = clean_text(fields.get("flavor") or "")

    name = fields["name"]
    rarity = fields.get("rarity") or ""
    pool = fields.get("pool") or ""
    desc = fields["description"]
    header = "，".join([b for b in [rarity, pool] if b])
    base = f"遗物{name}（{header}）：{desc}" if header else f"遗物{name}：{desc}"
    flavor = fields["flavor"]
    if flavor and "细节将在未来揭晓" not in flavor:
        base += f"。{flavor}"
    fields["embed_text"] = base
    return fields


def build_potion(raw):
    fields = pick(raw, [
        "id", "name", "description",
        "rarity", "rarity_key", "pool",
    ])
    fields["description"] = clean_text(fields.get("description") or "")

    name = fields["name"]
    rarity = fields.get("rarity") or ""
    pool = fields.get("pool") or ""
    desc = fields["description"]
    header = "，".join([b for b in [rarity, pool] if b])
    fields["embed_text"] = f"药水{name}（{header}）：{desc}" if header else f"药水{name}：{desc}"
    return fields


def _clean_moves(moves):
    if not moves:
        return []
    cleaned = []
    for m in moves:
        cleaned.append({
            "id": m.get("id"),
            "name": clean_text(m.get("name") or ""),
            "intent": m.get("intent"),
            "damage": m.get("damage"),
            "block": m.get("block"),
            "heal": m.get("heal"),
            "powers": m.get("powers"),
        })
    return cleaned


def _clean_encounters(encounters):
    if not encounters:
        return []
    cleaned = []
    for e in encounters:
        cleaned.append({
            "encounter_id": e.get("encounter_id"),
            "encounter_name": clean_text(e.get("encounter_name") or ""),
            "room_type": e.get("room_type"),
            "act": e.get("act"),
            "is_weak": e.get("is_weak"),
        })
    return cleaned


def build_monster(raw):
    fields = pick(raw, [
        "id", "name", "type",
        "min_hp", "max_hp", "min_hp_ascension", "max_hp_ascension",
        "damage_values", "block_values",
    ])
    fields["moves"] = _clean_moves(raw.get("moves"))
    fields["encounters"] = _clean_encounters(raw.get("encounters"))
    fields["innate_powers"] = raw.get("innate_powers")
    fields["attack_pattern"] = raw.get("attack_pattern")

    name = fields["name"]
    mtype = fields.get("type") or ""
    min_hp = fields.get("min_hp")
    max_hp = fields.get("max_hp")

    parts = [f"怪物{name}（{mtype}）"]
    if min_hp is not None:
        hp_str = f"血量{min_hp}" if max_hp in (None, min_hp) else f"血量{min_hp}-{max_hp}"
        parts.append(hp_str)

    move_names = [m["name"] for m in fields["moves"] if m["name"] and m["name"] != "无"]
    if move_names:
        parts.append(f"技能：{'、'.join(move_names)}")

    encounter_names = sorted({
        e["encounter_name"] for e in fields["encounters"] if e["encounter_name"]
    })
    if encounter_names:
        parts.append(f"遭遇：{'、'.join(encounter_names)}")

    fields["embed_text"] = "，".join(parts)
    return fields


def fetch_all():
    print("正在拉取角色数据...")
    characters = [build_character(c) for c in fetch_with_delay("characters")]
    print(f"获取到 {len(characters)} 个角色")

    print("正在拉取卡牌数据...")
    cards = [build_card(c) for c in fetch_with_delay("cards")]
    print(f"获取到 {len(cards)} 张卡牌")

    print("正在拉取遗物数据...")
    relics = [build_relic(r) for r in fetch_with_delay("relics")]
    print(f"获取到 {len(relics)} 个遗物")

    print("正在拉取药水数据...")
    potions = [build_potion(p) for p in fetch_with_delay("potions")]
    print(f"获取到 {len(potions)} 个药水")

    print("正在拉取怪物数据...")
    monsters = [build_monster(m) for m in fetch_with_delay("monsters")]
    print(f"获取到 {len(monsters)} 个怪物")

    return {
        "characters": characters,
        "cards": cards,
        "relics": relics,
        "potions": potions,
        "monsters": monsters,
    }


def main():
    knowledge = fetch_all()
    total = sum(len(v) for v in knowledge.values())
    print(f"\n共生成 {total} 条结构化条目")

    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge, f, ensure_ascii=False, indent=2)

    print(f"知识库已更新：{KNOWLEDGE_FILE}")


if __name__ == "__main__":
    main()
