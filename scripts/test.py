import requests
import json

BASE_URL = "https://spire-codex.com/api"
LANG = "zhs"

endpoints = ["characters", "cards", "relics", "potions", "monsters"]

for endpoint in endpoints:
    response = requests.get(f"{BASE_URL}/{endpoint}", params={"lang": LANG})
    data = response.json()
    print(f"\n{'='*40}")
    print(f"[{endpoint}] 第一条数据：")
    print(json.dumps(data[0], ensure_ascii=False, indent=2))