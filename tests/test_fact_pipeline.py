"""Offline unit tests for fact_pipeline (zero API calls, zero sts2.dll required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fact_pipeline.cli import sanitize_facts_for_eval
from fact_pipeline.compare import compare_facts
from fact_pipeline.evaluate import compute_eval_metrics, create_stratified_split
from fact_pipeline.extractor import (
    AssemblyProviderError,
    CachedAssemblyProvider,
    FixtureAssemblyProvider,
    MAX_DUMP_CHARS,
)
from fact_pipeline.llm import DEFAULT_MODEL, LLMResponse, LLMUsage, ScriptedLLMClient, make_tool_response
from fact_pipeline.pipeline import AuditOutcome, EntityPipelineResult, PipelineConfig, process_entity, run_batch
from fact_pipeline.validator import validate_fact


FP_CARD = "fp_card_1111"
FP_POWER = "fp_power_2222"
FP_RELIC = "fp_relic_3333"


def _sample_inputs():
    extract_by_key = {
        "cards:ABRASIVE": {
            "class": "Abrasive",
            "kind": "card",
            "namespace": "MegaCrit.Sts2.Core.Models.Cards",
            "base": "MegaCrit.Sts2.Core.Models.CardModel",
            "overrides": ["OnPlay", "OnUpgrade", "get_CanonicalKeywords", "get_CanonicalVars"],
            "calls": ["PowerCmd::Apply<DexterityPower>", "PowerCmd::Apply<ThornsPower>"],
            "canonical_vars": [
                {"type": "PowerVar`1<ThornsPower>", "name": None, "value": 4},
                {"type": "PowerVar`1<DexterityPower>", "name": None, "value": 1},
            ],
            "canonical_tags": [],
            "card_ctor": {"cost": 3, "type": "CardType.Power", "args": [3, 3, 4, 1, 1]},
            "upgrade_deltas": [{"var": "ThornsPower", "delta": 2}],
            "fingerprint": FP_CARD,
        },
        "powers:ACCURACY": {
            "class": "AccuracyPower",
            "kind": "power",
            "namespace": "MegaCrit.Sts2.Core.Models.Powers",
            "base": "MegaCrit.Sts2.Core.Models.PowerModel",
            "overrides": ["ModifyDamageAdditive"],
            "calls": [],
            "canonical_vars": [],
            "fingerprint": FP_POWER,
        },
        "relics:GOLD_PEARL": {
            "class": "GoldPearl",
            "kind": "relic",
            "namespace": "MegaCrit.Sts2.Core.Models.Relics",
            "base": "MegaCrit.Sts2.Core.Models.RelicModel",
            "overrides": ["AfterObtained", "get_CanonicalVars"],
            "calls": ["PlayerCmd::GainGold"],
            "canonical_vars": [{"type": "GoldVar", "name": None, "value": 150}],
            "fingerprint": FP_RELIC,
        },
    }
    facts_by_key = {
        "cards:ABRASIVE": {"entity_key": "cards:ABRASIVE", "class": "Abrasive", "status": "unreviewed"},
        "powers:ACCURACY": {"entity_key": "powers:ACCURACY", "class": "AccuracyPower", "status": "unreviewed"},
        "relics:GOLD_PEARL": {
            "entity_key": "relics:GOLD_PEARL",
            "class": "GoldPearl",
            "status": "unreviewed",
            "triage": {"category": "irrelevant"},
        },
    }
    catalog_by_type = {
        "cards": {
            "ABRASIVE": {
                "id": "ABRASIVE",
                "name": "磨砺",
                "cost": 3,
                "type_key": "Power",
                "description": "获得1点敏捷和4层荆棘。",
            }
        },
        "powers": {"ACCURACY": {"id": "ACCURACY", "name": "精准"}},
        "relics": {"GOLD_PEARL": {"id": "GOLD_PEARL", "name": "金珍珠"}},
    }
    assembly = FixtureAssemblyProvider(
        dumps={
            "Abrasive": "type Abrasive\nmethod .ctor() ldc.i4.3\nmethod OnPlay Apply Dexterity 1, Thorns 4 (+2)",
            "AccuracyPower": "type AccuracyPower\nmethod ModifyDamageAdditive tag 5 +Amount",
            "GoldPearl": "type GoldPearl\nmethod AfterObtained GainGold 150",
            "ThornsPower": "type ThornsPower\nmethod AfterDamageReceived",
        },
        enums={"CardTag": "MegaCrit.Sts2.Core.Entities.Cards.CardTag 5 CardTag.Shiv"},
    )
    return extract_by_key, facts_by_key, catalog_by_type, assembly


def _valid_card_fact(thorns: int = 4) -> dict:
    return {
        "entity_key": "cards:ABRASIVE",
        "fingerprint": FP_CARD,
        "status": "converted",
        "card_type": "Power",
        "cost": {"base": 3, "upgraded": 3},
        "tags": [],
        "keywords": ["Sly"],
        "effects": [
            {"kind": "apply_power", "power": "Dexterity", "amount": 1, "upgrade": 0, "target": "self"},
            {"kind": "apply_power", "power": "Thorns", "amount": thorns, "upgrade": 2, "target": "self"},
        ],
        "unknowable": [],
        "evidence": ["ctor cost=3 Power; OnPlay Apply Dexterity 1, Thorns 4 (+2)"],
    }


def _valid_power_fact() -> dict:
    return {
        "entity_key": "powers:ACCURACY",
        "fingerprint": FP_POWER,
        "status": "converted",
        "category": "trigger",
        "duration": {"kind": "combat", "stack_type": "Counter"},
        "mechanics": [
            {"modifier": "damage", "add_per_stack": 1, "filter": {"tag": "Shiv"}}
        ],
        "unknowable": [],
        "evidence": ["ModifyDamageAdditive: tag 5 (Shiv) +Amount"],
    }


def _valid_relic_fact() -> dict:
    return {
        "entity_key": "relics:GOLD_PEARL",
        "fingerprint": FP_RELIC,
        "status": "converted",
        "category": "irrelevant",
        "mechanics": [],
        "obtain": [{"kind": "gain_gold", "amount": 150}],
        "between_fights": [],
        "unknowable": [],
        "evidence": ["AfterObtained: PlayerCmd.GainGold(150)"],
    }


class ValidatorTests(unittest.TestCase):
    def test_valid_card_power_and_relic_pass(self):
        ext, facts, cat, _ = _sample_inputs()
        self.assertTrue(validate_fact(_valid_card_fact(), ext, facts, catalog_cards=cat["cards"]).ok)
        self.assertTrue(validate_fact(_valid_power_fact(), ext, facts).ok)
        self.assertTrue(validate_fact(_valid_relic_fact(), ext, facts).ok)

    def test_rejects_bad_fingerprint_missing_core_fields_and_unknown_vocab(self):
        ext, facts, _, _ = _sample_inputs()
        bad = _valid_card_fact()
        bad["fingerprint"] = "wrong"
        bad.pop("keywords")
        bad["effects"].append({"kind": "invented_effect", "amount": 5})
        res = validate_fact(bad, ext, facts, expected_entity_key="cards:ABRASIVE")
        self.assertFalse(res.ok)
        joined = "\n".join(res.problems)
        self.assertIn("fingerprint", joined)
        self.assertIn("keywords", joined)
        self.assertIn("invented_effect", joined)

    def test_supports_v0113_fool_vocabulary_and_flags_codex_conflict(self):
        ext, facts, cat, _ = _sample_inputs()
        fool_card = _valid_card_fact()
        fool_card["cost"] = {"base": 2, "upgraded": 2}  # conflicts with Codex cost 3
        fool_card["effects"] = [
            {"kind": "arrow_slots", "amount": 1, "upgrade": 0},
            {"kind": "dislodge_arrow", "count": "all", "owner_only": True},
        ]
        res = validate_fact(fool_card, ext, facts, catalog_cards=cat["cards"])
        self.assertTrue(res.ok, res.problems)
        self.assertTrue(any("cost 2 vs Codex 3" in c for c in res.codex_conflicts))

    def test_hardened_validator_catches_common_schema_mistakes(self):
        ext, facts, _, _ = _sample_inputs()
        bad_card = _valid_card_fact()
        bad_card["effects"] = [
            {"kind": "damage", "amount": 6, "target": "enemy"},
            {"kind": "block", "amount": 5, "target": "self"},
            {"kind": "generate", "count": 1, "pile": "draw_pile"},
            {"kind": "energy", "amount": -1, "when": "card_drawn"},
        ]
        res = validate_fact(bad_card, ext, facts)
        self.assertFalse(res.ok)
        text = "\n".join(res.problems)
        self.assertIn("damage target", text)
        self.assertIn("omit 'target'", text)
        self.assertIn("add_card", text)
        self.assertIn("belong in 'mechanics'", text)

        bad_relic = _valid_relic_fact()
        bad_relic["category"] = "trigger"
        bad_relic["mechanics"] = [{"when": "card_played", "count": 3, "effect": {"kind": "damage", "amount": 5, "target": "all"}}]
        bad_relic["obtain"] = [{"kind": "offer_rewards", "amount": 5}]
        res_r = validate_fact(bad_relic, ext, facts)
        self.assertFalse(res_r.ok)
        text_r = "\n".join(res_r.problems)
        self.assertIn("every", text_r)
        self.assertIn("rewards", text_r)


class ExtractorTests(unittest.TestCase):
    def test_fixture_provider_truncation_and_errors(self):
        huge = "x" * (MAX_DUMP_CHARS + 200)
        provider = FixtureAssemblyProvider(dumps={"Big": huge}, enums={"E": "E 1 A"})
        dumped = provider.dump_class("Big")
        self.assertIn("[TRUNCATED:", dumped)
        self.assertEqual(provider.read_enum("E"), "E 1 A")
        with self.assertRaises(AssemblyProviderError):
            provider.dump_class("Missing")

    def test_cached_provider_reads_from_disk_cache_without_dll(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            dump_file = cache / "abc123" / "dump" / "Abrasive.txt"
            dump_file.parent.mkdir(parents=True)
            dump_file.write_text("cached IL content", encoding="utf-8")
            provider = CachedAssemblyProvider(dll_path=None, extractor_path=None, cache_dir=cache, assembly_tag="abc123")
            self.assertEqual(provider.dump_class("Abrasive"), "cached IL content")


class PipelineStateMachineTests(unittest.TestCase):
    def test_happy_path_with_tool_call_and_strict_auditor_isolation(self):
        ext, facts, cat, assembly = _sample_inputs()
        llm = ScriptedLLMClient(
            [
                make_tool_response("dump_class", {"class_name": "ThornsPower"}, call_id="d1"),
                make_tool_response("submit_fact", {"fact": _valid_card_fact()}, call_id="d2"),
                make_tool_response(
                    "submit_audit",
                    {"verdict": "agree", "issues": [], "notes": "Checked ctor, OnPlay, OnUpgrade."},
                    call_id="a1",
                ),
            ]
        )
        res, trace = process_entity(
            "cards:ABRASIVE",
            extract_by_key=ext,
            facts_by_key=facts,
            catalog_by_type=cat,
            assembly=assembly,
            llm=llm,
            config=PipelineConfig(drafter_model=DEFAULT_MODEL, auditor_model=DEFAULT_MODEL),
        )
        self.assertEqual(res.outcome, "auto_approved")
        self.assertEqual(res.reason, "auditor_agreed")
        self.assertEqual(res.drafter_tool_calls, 1)
        self.assertEqual(res.auditor_tool_calls, 0)
        self.assertEqual(res.fact["audit"]["verdict"], "agree")
        # Auditor starts fresh: only 2 messages (system + user), no Drafter dump_class history leaked.
        auditor_call = llm.calls[2]
        self.assertEqual(len(auditor_call["messages"]), 2)
        self.assertNotIn("ThornsPower\nmethod AfterDamageReceived", auditor_call["messages"][1]["content"])
        self.assertGreaterEqual(len(trace), 3)

    def test_stringified_fact_coercion_and_schema_repair(self):
        ext, facts, cat, assembly = _sample_inputs()
        broken = _valid_card_fact()
        broken["effects"] = [{"kind": "bogus_kind", "amount": 4}]
        llm = ScriptedLLMClient(
            [
                make_tool_response("submit_fact", {"fact": broken}, call_id="d1"),
                make_tool_response("submit_fact", {"fact": json.dumps(_valid_card_fact())}, call_id="d2"),
                make_tool_response("submit_audit", {"verdict": "agree", "issues": [], "notes": "Matches IL."}, call_id="a1"),
            ]
        )
        res, _ = process_entity(
            "cards:ABRASIVE",
            extract_by_key=ext,
            facts_by_key=facts,
            catalog_by_type=cat,
            assembly=assembly,
            llm=llm,
        )
        self.assertEqual(res.outcome, "auto_approved")
        self.assertEqual(res.schema_repairs, 1)

    def test_schema_repair_exhaustion_routes_to_needs_human(self):
        ext, facts, cat, assembly = _sample_inputs()
        broken = _valid_card_fact()
        broken["fingerprint"] = "bad_fp"
        llm = ScriptedLLMClient(
            [
                make_tool_response("submit_fact", {"fact": broken}, call_id="d1"),
                make_tool_response("submit_fact", {"fact": broken}, call_id="d2"),
                make_tool_response("submit_fact", {"fact": broken}, call_id="d3"),
            ]
        )
        res, _ = process_entity(
            "cards:ABRASIVE",
            extract_by_key=ext,
            facts_by_key=facts,
            catalog_by_type=cat,
            assembly=assembly,
            llm=llm,
            config=PipelineConfig(max_schema_repairs=2),
        )
        self.assertEqual(res.outcome, "needs_human")
        self.assertEqual(res.reason, "schema_validation_failed")
        self.assertIsNone(res.fact)

    def test_auditor_disagree_triggers_one_revision_and_preserves_initial_draft(self):
        ext, facts, cat, assembly = _sample_inputs()
        buggy = _valid_card_fact(thorns=99)
        fixed = _valid_card_fact(thorns=4)
        llm = ScriptedLLMClient(
            [
                make_tool_response("submit_fact", {"fact": buggy}, call_id="d1"),
                make_tool_response(
                    "submit_audit",
                    {"verdict": "disagree", "issues": ["Thorns amount is 4 in CanonicalVars, not 99."], "notes": "Wrong Thorns."},
                    call_id="a1",
                ),
                make_tool_response("submit_fact", {"fact": fixed}, call_id="r1"),
                make_tool_response("submit_audit", {"verdict": "agree", "issues": [], "notes": "Fixed to 4."}, call_id="a2"),
            ]
        )
        res, _ = process_entity(
            "cards:ABRASIVE",
            extract_by_key=ext,
            facts_by_key=facts,
            catalog_by_type=cat,
            assembly=assembly,
            llm=llm,
        )
        self.assertEqual(res.outcome, "auto_approved")
        self.assertEqual(res.reason, "revised_auditor_agreed")
        self.assertTrue(res.revised)
        self.assertEqual(res.initial_draft["effects"][1]["amount"], 99)
        self.assertEqual(res.fact["effects"][1]["amount"], 4)
        self.assertEqual(res.initial_audit.verdict, "disagree")
        self.assertEqual(res.final_audit.verdict, "agree")

    def test_persistent_disagreement_stops_after_one_revision(self):
        ext, facts, cat, assembly = _sample_inputs()
        buggy = _valid_card_fact(thorns=99)
        llm = ScriptedLLMClient(
            [
                make_tool_response("submit_fact", {"fact": buggy}, call_id="d1"),
                make_tool_response("submit_audit", {"verdict": "disagree", "issues": ["Wrong Thorns."], "notes": "Bad."}, call_id="a1"),
                make_tool_response("submit_fact", {"fact": buggy}, call_id="r1"),
                make_tool_response("submit_audit", {"verdict": "disagree", "issues": ["Still wrong."], "notes": "Still bad."}, call_id="a2"),
            ]
        )
        res, _ = process_entity(
            "cards:ABRASIVE",
            extract_by_key=ext,
            facts_by_key=facts,
            catalog_by_type=cat,
            assembly=assembly,
            llm=llm,
        )
        self.assertEqual(res.outcome, "needs_human")
        self.assertEqual(res.reason, "auditor_disagreed_after_revision")
        self.assertTrue(res.revised)
        self.assertEqual(len(llm.calls), 4)

    def test_needs_new_vocabulary_escalates_immediately(self):
        ext, facts, cat, assembly = _sample_inputs()
        llm = ScriptedLLMClient(
            [
                make_tool_response(
                    "submit_fact",
                    {
                        "fact": _valid_power_fact(),
                        "needs_new_vocabulary": True,
                        "vocabulary_gap_note": "Needs new modifier 'custom_foo'",
                    },
                    call_id="d1",
                )
            ]
        )
        res, _ = process_entity(
            "powers:ACCURACY",
            extract_by_key=ext,
            facts_by_key=facts,
            catalog_by_type=cat,
            assembly=assembly,
            llm=llm,
        )
        self.assertEqual(res.outcome, "needs_human")
        self.assertEqual(res.reason, "needs_new_vocabulary")
        self.assertTrue(res.needs_new_vocabulary)
        self.assertEqual(len(llm.calls), 1)  # Did not waste Auditor tokens

    def test_skip_audit_ablation_and_batch_resume(self):
        ext, facts, cat, assembly = _sample_inputs()
        llm = ScriptedLLMClient(
            [
                make_tool_response("submit_fact", {"fact": _valid_relic_fact()}, call_id="d1"),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            first = run_batch(
                ["relics:GOLD_PEARL"],
                extract_by_key=ext,
                facts_by_key=facts,
                catalog_by_type=cat,
                assembly=assembly,
                llm=llm,
                out_dir=out_dir,
                config=PipelineConfig(skip_audit=True),
            )
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].outcome, "auto_approved")
            self.assertEqual(first[0].reason, "audit_skipped")
            self.assertTrue((out_dir / "auto_approved.json").is_file())
            self.assertTrue((out_dir / "review-queue.md").is_file())
            self.assertTrue((out_dir / "summary.json").is_file())

            # Second run with empty LLM script must resume from saved result without calling LLM.
            empty_llm = ScriptedLLMClient([])
            second = run_batch(
                ["relics:GOLD_PEARL"],
                extract_by_key=ext,
                facts_by_key=facts,
                catalog_by_type=cat,
                assembly=assembly,
                llm=empty_llm,
                out_dir=out_dir,
                config=PipelineConfig(skip_audit=True),
            )
            self.assertEqual(len(empty_llm.calls), 0)
            self.assertEqual(second[0].entity_key, "relics:GOLD_PEARL")
            self.assertEqual(second[0].outcome, "auto_approved")

    def test_lookup_fact_blocks_target_self_leakage(self):
        ext, facts, cat, assembly = _sample_inputs()
        # Even if facts_by_key had a reviewed copy, target self-lookup must be refused.
        facts["cards:ABRASIVE"] = _valid_card_fact()
        llm = ScriptedLLMClient(
            [
                make_tool_response("lookup_fact", {"entity_key": "cards:ABRASIVE"}, call_id="d1"),
                make_tool_response("submit_fact", {"fact": _valid_card_fact()}, call_id="d2"),
                make_tool_response("submit_audit", {"verdict": "agree", "issues": [], "notes": "OK."}, call_id="a1"),
            ]
        )
        res, _ = process_entity(
            "cards:ABRASIVE",
            extract_by_key=ext,
            facts_by_key=facts,
            catalog_by_type=cat,
            assembly=assembly,
            llm=llm,
        )
        self.assertEqual(res.outcome, "auto_approved")
        tool_msg = next(m for m in llm.calls[1]["messages"] if m.get("role") == "tool")
        self.assertIn("Cannot look up the target entity itself", tool_msg["content"])

    def test_sanitize_facts_for_eval_strips_held_out_answers(self):
        facts = {
            "cards:ABRASIVE": _valid_card_fact(),
            "relics:GOLD_PEARL": _valid_relic_fact(),
        }
        sanitized = sanitize_facts_for_eval(facts, ["cards:ABRASIVE"])
        self.assertEqual(sanitized["cards:ABRASIVE"]["status"], "unreviewed")
        self.assertNotIn("effects", sanitized["cards:ABRASIVE"])
        self.assertIn("obtain", sanitized["relics:GOLD_PEARL"])


class SemanticComparatorAndEvalTests(unittest.TestCase):
    def test_comparator_treats_harmless_variants_as_equal_and_catches_real_bugs(self):
        gold = _valid_card_fact(thorns=4)
        equivalent = {
            **gold,
            "effects": [
                {"kind": "apply_power", "power": "DexterityPower", "amount": 1.0, "target": "self"},
                {"kind": "apply_power", "power": "Thorns", "amount": 4, "upgrade": 2, "target": "self", "note": "extra"},
            ],
            "evidence": ["completely different evidence wording"],
        }
        cmp_ok = compare_facts("cards:ABRASIVE", equivalent, gold)
        self.assertTrue(cmp_ok.strict_match, cmp_ok.field_diffs)
        self.assertTrue(cmp_ok.core_match)

        buggy = _valid_card_fact(thorns=6)
        cmp_bad = compare_facts("cards:ABRASIVE", buggy, gold)
        self.assertFalse(cmp_bad.strict_match)
        self.assertFalse(cmp_bad.core_match)
        self.assertIn("effects", cmp_bad.field_diffs)

    def test_comparator_normalizes_exhaust_self_and_pool_prose(self):
        gold = _valid_card_fact()
        gold["keywords"] = ["Exhaust"]
        gold["effects"] = [{"kind": "add_card", "count": 1, "destination": "hand", "pool": "owner power pool"}]
        cand = _valid_card_fact()
        cand["keywords"] = []
        cand["effects"] = [
            {"kind": "add_card", "count": 1, "destination": "hand", "pool": "character Power cards"},
            {"kind": "exhaust_self"},
        ]
        res = compare_facts("cards:ABRASIVE", cand, gold)
        self.assertFalse(res.strict_match)
        self.assertTrue(res.core_match, res.field_diffs)

    def test_compute_eval_metrics_tracks_precision_auditor_recall_and_revision_net_gain(self):
        gold_map = {
            "cards:ABRASIVE": _valid_card_fact(thorns=4),
            "powers:ACCURACY": _valid_power_fact(),
            "relics:GOLD_PEARL": _valid_relic_fact(),
        }
        # 1) Buggy draft caught by auditor and fixed in revision -> auto_approved + core_ok
        r1 = EntityPipelineResult(
            entity_key="cards:ABRASIVE",
            entity_type="cards",
            class_name="Abrasive",
            fingerprint=FP_CARD,
            outcome="auto_approved",
            reason="revised_auditor_agreed",
            fact=_valid_card_fact(thorns=4),
            initial_draft=_valid_card_fact(thorns=9),
            initial_audit=AuditOutcome(verdict="disagree", issues=["wrong thorns"]),
            final_audit=AuditOutcome(verdict="agree"),
            revised=True,
            usage=LLMUsage(prompt_tokens=1000, completion_tokens=200, prompt_cache_hit_tokens=500, cost_usd=0.001),
        )
        # 2) Good draft wrongly flagged by auditor (false alarm) and sent to human
        r2 = EntityPipelineResult(
            entity_key="powers:ACCURACY",
            entity_type="powers",
            class_name="AccuracyPower",
            fingerprint=FP_POWER,
            outcome="needs_human",
            reason="auditor_disagreed_after_revision",
            fact=_valid_power_fact(),
            initial_draft=_valid_power_fact(),
            initial_audit=AuditOutcome(verdict="disagree", issues=["nitpick"]),
            final_audit=AuditOutcome(verdict="disagree"),
            revised=True,
            usage=LLMUsage(prompt_tokens=800, completion_tokens=150, prompt_cache_hit_tokens=400, cost_usd=0.0008),
        )
        # 3) Buggy draft missed by auditor -> auto_approved false positive
        bad_relic = _valid_relic_fact()
        bad_relic["obtain"] = [{"kind": "gain_gold", "amount": 999}]
        r3 = EntityPipelineResult(
            entity_key="relics:GOLD_PEARL",
            entity_type="relics",
            class_name="GoldPearl",
            fingerprint=FP_RELIC,
            outcome="auto_approved",
            reason="auditor_agreed",
            fact=bad_relic,
            initial_draft=bad_relic,
            initial_audit=AuditOutcome(verdict="agree"),
            final_audit=AuditOutcome(verdict="agree"),
            revised=False,
            usage=LLMUsage(prompt_tokens=600, completion_tokens=100, prompt_cache_hit_tokens=300, cost_usd=0.0005),
        )
        metrics = compute_eval_metrics([r1, r2, r3], gold_map)
        self.assertEqual(metrics["auto_approved"], 2)
        self.assertEqual(metrics["auto_approved_core_ok"], 1)
        self.assertAlmostEqual(metrics["auto_approved_core_precision"], 0.5)
        self.assertEqual(metrics["auditor"]["buggy_initial"], 2)
        self.assertEqual(metrics["auditor"]["caught_bugs"], 1)
        self.assertAlmostEqual(metrics["auditor"]["bug_recall"], 0.5)
        self.assertEqual(metrics["auditor"]["false_alarms"], 1)
        self.assertAlmostEqual(metrics["auditor"]["false_alarm_rate"], 1.0)
        self.assertEqual(metrics["revision"]["fixed_bug"], 1)
        self.assertEqual(metrics["revision"]["kept_good"], 1)


if __name__ == "__main__":
    unittest.main()
