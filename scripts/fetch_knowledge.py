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


def clean_text(text: str) -> str:
    if not text:
        return ""
    # 清理格式标签如 [gold], [blue], [sine] 等
    text = re.sub(r'\[/?[^\]]+\]', '', text)
    return text.strip()


def fetch_with_delay(endpoint, params=None):
    if params is None:
        params = {}
    params["lang"] = LANG
    response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
    time.sleep(RATE_LIMIT_DELAY)
    return response.json()


def fetch_characters():
    print("正在拉取角色数据...")
    data = fetch_with_delay("characters")
    print(f"获取到 {len(data)} 个角色")
    return data


def fetch_cards():
    print("正在拉取卡牌数据...")
    data = fetch_with_delay("cards")
    print(f"获取到 {len(data)} 张卡牌")
    return data


def fetch_relics():
    print("正在拉取遗物数据...")
    data = fetch_with_delay("relics")
    print(f"获取到 {len(data)} 个遗物")
    return data


def fetch_potions():
    print("正在拉取药水数据...")
    data = fetch_with_delay("potions")
    print(f"获取到 {len(data)} 个药水")
    return data


def fetch_monsters():
    print("正在拉取怪物数据...")
    data = fetch_with_delay("monsters")
    print(f"获取到 {len(data)} 个怪物")
    return data


def build_knowledge(characters, cards, relics, potions, monsters):
    docs = []

    # 角色总览
    char_names = [c.get("name", "") for c in characters if c.get("name")]
    docs.append(f"杀戮尖塔2共有{len(char_names)}个角色：{'、'.join(char_names)}")

    # 角色详情
    for char in characters:
        name = char.get("name", "")
        description = clean_text(char.get("description", ""))
        hp = char.get("starting_hp", "")
        energy = char.get("max_energy", "")
        orb_slots = char.get("orb_slots", "")
        starting_relics = char.get("starting_relics", [])

        if name and description:
            base = f"角色{name}：{description}初始血量{hp}点，初始能量{energy}点"
            if orb_slots:
                base += f"，球槽数量{orb_slots}"
            if starting_relics:
                base += f"，初始遗物：{'、'.join(starting_relics)}"
            docs.append(base)

    # 卡牌
    for card in cards:
        name = card.get("name", "")
        description = clean_text(card.get("description", ""))
        cost = card.get("cost", "")
        card_type = card.get("type", "")
        rarity = card.get("rarity", "")
        color = card.get("color", "")
        damage = card.get("damage")
        block = card.get("block")

        if name and description:
            base = f"卡牌{name}（{color}，{rarity}，{card_type}，费用{cost}）：{description}"
            if damage:
                base += f"，伤害{damage}"
            if block:
                base += f"，格挡{block}"
            docs.append(base)

    # 遗物
    for relic in relics:
        name = relic.get("name", "")
        description = clean_text(relic.get("description", ""))
        rarity = relic.get("rarity", "")
        pool = relic.get("pool", "")
        flavor = clean_text(relic.get("flavor", ""))

        if name and description:
            base = f"遗物{name}（{rarity}，{pool}）：{description}"
            if flavor and "细节将在未来揭晓" not in flavor:
                base += f"。{flavor}"
            docs.append(base)

    # 药水
    for potion in potions:
        name = potion.get("name", "")
        description = clean_text(potion.get("description", ""))
        rarity = potion.get("rarity", "")

        if name and description:
            docs.append(f"药水{name}（{rarity}）：{description}")

    # 怪物
    for monster in monsters:
        name = monster.get("name", "")
        monster_type = monster.get("type", "")
        min_hp = monster.get("min_hp")
        max_hp = monster.get("max_hp")
        moves = monster.get("moves", [])

        if name and min_hp:
            move_names = [clean_text(m.get("name", "")) for m in moves if m.get("name") and m.get("name") != "无"]
            base = f"怪物{name}（{monster_type}）：血量{min_hp}"
            if max_hp and max_hp != min_hp:
                base += f"-{max_hp}"
            if move_names:
                base += f"，技能：{'、'.join(move_names)}"
            docs.append(base)

    return docs


def main():
    characters = fetch_characters()
    cards = fetch_cards()
    relics = fetch_relics()
    potions = fetch_potions()
    monsters = fetch_monsters()

    docs = build_knowledge(characters, cards, relics, potions, monsters)
    print(f"\n共生成 {len(docs)} 条知识")

    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump({"docs": docs}, f, ensure_ascii=False, indent=2)

    print(f"知识库已更新：{KNOWLEDGE_FILE}")


if __name__ == "__main__":
    main()