import copy
import json
import tempfile
import unittest
from pathlib import Path

from storage.assembly_facts import (
    DEFAULT_EXTRACT_PATH,
    DEFAULT_FACTS_PATH,
    STATUS_CONVERTED,
    STATUS_STALE,
    STATUS_UNREVIEWED,
    catalog_keys as committed_catalog_keys,
    evaluate_facts,
    load_extract,
    load_facts,
    refresh_facts,
)
from storage.release_database import ReleaseDatabaseError, build_release_database
from storage.relational import RelationalRepository


ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "data" / "knowledge.json"
COMMUNITY = ROOT / "data" / "community_scores.json"


def catalog_keys():
    return set(committed_catalog_keys(KNOWLEDGE))


class CommittedAssemblyFactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extract = load_extract()
        cls.facts = load_facts()
        cls.by_key = {entry["entity_key"]: entry for entry in cls.facts["entities"]}

    def test_extract_is_for_the_supported_game_assembly(self):
        manifest = json.loads(
            (ROOT / "packaging" / "compatibility.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            self.extract["assembly_sha256"], manifest["game"]["sts2_dll_sha256"]
        )
        self.assertEqual(self.facts["assembly_sha256"], self.extract["assembly_sha256"])

    def test_every_catalog_card_and_relic_has_a_fact_with_a_model_class(self):
        self.assertEqual(set(self.by_key), catalog_keys())
        missing_class = [k for k, e in self.by_key.items() if not e.get("class")]
        self.assertEqual(missing_class, [])

    def test_committed_facts_are_current_for_the_committed_extract(self):
        refreshed = refresh_facts(self.extract, catalog_keys(), self.facts)
        self.assertEqual(refreshed, self.facts)
        stale = [e["entity_key"] for e in self.facts["entities"] if e["status"] == STATUS_STALE]
        self.assertEqual(stale, [])

    def test_relic_triage_follows_hooks_not_descriptions(self):
        # War Paint only upgrades skills when picked up.
        self.assertEqual(
            self.by_key["relics:WAR_PAINT"]["triage"]["category"], "irrelevant"
        )
        # Cloak Clasp grants block from hand size at turn end: a baseline
        # defense source, not a relic that makes block cards better.
        self.assertEqual(
            self.by_key["relics:CLOAK_CLASP"]["triage"]["category"], "baseline"
        )
        # Reward-modifying relics always require a human review.
        self.assertEqual(
            self.by_key["relics:FROZEN_EGG"]["triage"]["category"], "needs_review"
        )
        triggers = [
            e for e in self.facts["entities"]
            if e.get("triage", {}).get("category") == "trigger"
        ]
        self.assertTrue(all(
            e["triage"]["hook_groups"]["trigger"] for e in triggers
        ))


    def test_irrelevant_relics_cannot_act_inside_a_fight(self):
        from storage.assembly_facts import COMBAT_COMMAND_PATTERN, RUN_HOOKS, extract_index

        index = extract_index(self.extract)
        for key, entry in self.by_key.items():
            if not key.startswith("relics:") or entry.get("category") != "irrelevant":
                continue
            if "(structural)" not in str(entry.get("reviewed_by")):
                # Reviewed from the IL: the evidence explains why it cannot
                # act inside a fight; only the structural rule is checked here.
                self.assertEqual(entry.get("mechanics"), [])
                self.assertTrue(entry.get("evidence"))
                continue
            with self.subTest(relic=key):
                self.assertEqual(entry.get("mechanics"), [])
                self.assertTrue(set(entry["hooks"]) <= RUN_HOOKS)
                if "AfterRoomEntered" in entry["hooks"]:
                    self.assertFalse(any(
                        COMBAT_COMMAND_PATTERN.search(call) for call in index[key]["calls"]
                    ))


class AssemblyFactValidityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.extract = load_extract(DEFAULT_EXTRACT_PATH)
        self.facts = copy.deepcopy(load_facts(DEFAULT_FACTS_PATH))

    def _mark(self, key, **fields):
        for entry in self.facts["entities"]:
            if entry["entity_key"] == key:
                entry.update(fields)
                return
        raise AssertionError(key)

    def _build(self):
        extract_path = self.root / "extract.json"
        facts_path = self.root / "facts.json"
        extract_path.write_text(json.dumps(self.extract), encoding="utf-8")
        facts_path.write_text(json.dumps(self.facts), encoding="utf-8")
        target = self.root / "release.db"
        build_release_database(
            target, KNOWLEDGE, COMMUNITY, extract_path, facts_path
        )
        return RelationalRepository(str(target))

    def test_only_reviewed_facts_with_matching_fingerprint_are_valid(self):
        self._mark("relics:WAR_PAINT", status=STATUS_CONVERTED, category="irrelevant")
        self._mark(
            "relics:CLOAK_CLASP",
            status=STATUS_CONVERTED,
            category="baseline",
            fingerprint="0" * 64,
        )
        self._mark("relics:ANCHOR", status=STATUS_UNREVIEWED)
        repository = self._build()

        self.assertEqual(
            repository.find_assembly_fact("relics", "WAR_PAINT")["category"],
            "irrelevant",
        )
        self.assertIsNone(repository.find_assembly_fact("relics", "CLOAK_CLASP"))
        self.assertIsNone(repository.find_assembly_fact("relics", "ANCHOR"))

        relics = [e for e in self.facts["entities"] if e["entity_key"].startswith("relics:")]
        coverage = repository.assembly_fact_coverage()["relics"]
        self.assertEqual(coverage["total"], len(relics))
        # A reviewed fact whose fingerprint no longer matches is stale in the
        # database report exactly as in the refresh report.
        self.assertEqual(coverage[STATUS_STALE], 1)
        self.assertEqual(
            coverage[STATUS_UNREVIEWED],
            sum(e["status"] == STATUS_UNREVIEWED for e in relics),
        )
        self.assertEqual(
            coverage[STATUS_CONVERTED],
            sum(e["status"] == STATUS_CONVERTED for e in relics) - 1,
        )
        with repository.connect() as connection:
            categories = dict(connection.execute(
                "SELECT entity_key, category FROM assembly_facts "
                "WHERE entity_key IN ('relics:CLOAK_CLASP', 'relics:ANCHOR')"
            ).fetchall())
        # Only valid reviewed rows expose a category; triage stays separate.
        self.assertEqual(categories, {"relics:CLOAK_CLASP": None, "relics:ANCHOR": None})

    def test_changed_class_code_makes_only_that_review_stale(self):
        self._mark("relics:WAR_PAINT", status=STATUS_CONVERTED, category="irrelevant")
        self._mark("relics:ANCHOR", status=STATUS_CONVERTED, category="baseline")
        for entry in self.extract["types"]:
            if entry["class"] == "WarPaint":
                entry["fingerprint"] = "f" * 64

        refreshed = refresh_facts(self.extract, catalog_keys(), self.facts)
        evaluated = evaluate_facts(refreshed, self.extract)

        war_paint = evaluated["relics:WAR_PAINT"]
        self.assertEqual(war_paint["status"], STATUS_STALE)
        self.assertEqual(war_paint["category"], "irrelevant")
        self.assertFalse(war_paint["valid"])
        self.assertTrue(evaluated["relics:ANCHOR"]["valid"])
        self.assertEqual(
            war_paint["hook_tokens"], {"AfterObtained": "06002376"}
        )

    def test_release_metadata_pins_extract_and_facts(self):
        repository = self._build()
        with repository.connect() as connection:
            metadata = dict(connection.execute(
                "SELECT key, value FROM schema_metadata"
            ).fetchall())
        self.assertRegex(metadata["assembly_extract_sha256"], "^[0-9a-f]{64}$")
        self.assertRegex(metadata["assembly_facts_sha256"], "^[0-9a-f]{64}$")

    def test_facts_for_another_extract_are_rejected(self):
        self.facts["assembly_sha256"] = "1" * 64
        with self.assertRaises(ReleaseDatabaseError):
            self._build()


if __name__ == "__main__":
    unittest.main()
