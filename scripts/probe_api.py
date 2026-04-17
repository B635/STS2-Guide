"""Probe Spire Codex API to discover all available fields per entity type.

For each endpoint, fetches a handful of samples and prints the union of fields
seen with their value types. Used to design a structured knowledge schema.
"""
import json
import requests
import time

BASE_URL = "https://spire-codex.com/api"
LANG = "zhs"
ENDPOINTS = ["characters", "cards", "relics", "potions", "monsters"]
SAMPLES_PER_ENDPOINT = 3


def type_name(v):
    if v is None:
        return "null"
    if isinstance(v, list):
        if not v:
            return "list[?]"
        inner = type_name(v[0])
        return f"list[{inner}]"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def probe(endpoint: str):
    print(f"\n{'='*70}")
    print(f"Endpoint: /{endpoint}")
    print("=" * 70)
    resp = requests.get(f"{BASE_URL}/{endpoint}", params={"lang": LANG})
    data = resp.json()
    print(f"Total items: {len(data)}")

    # Union of keys seen across all samples
    all_fields = {}  # field_name -> set of types
    for item in data:
        for k, v in item.items():
            all_fields.setdefault(k, set()).add(type_name(v))

    print(f"\nField inventory (union across all {len(data)} items):")
    for k in sorted(all_fields.keys()):
        types = " | ".join(sorted(all_fields[k]))
        print(f"  {k:<25} :: {types}")

    # Show a concrete sample for reference
    print(f"\nSample item (first one):")
    sample = data[0]
    print(json.dumps(sample, ensure_ascii=False, indent=2)[:1500])

    time.sleep(1.1)


if __name__ == "__main__":
    for ep in ENDPOINTS:
        probe(ep)
