from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from advisor.deck_fitness import (
    DECK_FITNESS_V4_POLICY_VERSION,
    PARAMETERS_KEY,
    DeckFitnessPolicy,
    DeckFitnessV4Policy,
)
from advisor.deck_mechanics import CardProfile, Situation, card_profile, relic_profile
from advisor.golden_set import card_reward_request, load_golden_set
from advisor.policies import create_card_reward_policy
from advisor.turn_micro_sim import compare_candidates_crn
from storage.relational import RelationalRepository
from storage.release_database import build_release_database

ROOT = Path(__file__).resolve().parents[1]


class TurnMicroSimUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        db = Path(cls.tmp.name) / "release.db"
        build_release_database(
            db, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json"
        )
        cls.repo = RelationalRepository(str(db))
        cls.params = cls.repo.mechanic_constant(PARAMETERS_KEY)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def card(self, cid: str, upgrades: int = 0) -> CardProfile:
        prof, _ = card_profile(self.repo, cid, upgrades)
        self.assertIsNotNone(prof, cid)
        return prof

    def relic(self, rid: str):
        prof, _ = relic_profile(self.repo, rid)
        self.assertIsNotNone(prof, rid)
        return prof

    def test_crn_coupling_is_deterministic_and_zero_variance_for_identical_option(self):
        strike = self.card("STRIKE_IRONCLAD")
        defend = self.card("DEFEND_IRONCLAD")
        deck = [strike] * 5 + [defend] * 4
        sit = Situation(6.0, 0.0, 12.0, {"MONSTER": 1.0}, 1.0, 0, enemy_hp=65.0)
        res1 = compare_candidates_crn(
            "det-case-1", deck, {"skip": None, "skip_copy": None, "bash": self.card("BASH")},
            (), sit, self.params, trajectories=24,
        )
        res2 = compare_candidates_crn(
            "det-case-1", deck, {"skip": None, "skip_copy": None, "bash": self.card("BASH")},
            (), sit, self.params, trajectories=24,
        )
        self.assertEqual(res1["skip"].losses, res2["skip"].losses)
        self.assertEqual(res1["bash"].losses, res2["bash"].losses)
        self.assertEqual(res1["skip"].losses, res1["skip_copy"].losses)
        self.assertGreaterEqual(res1["skip"].cvar25_hp_loss, res1["skip"].mean_hp_loss - 1e-9)

    def test_retain_block_card_holds_for_spike_turn_and_reduces_cvar25(self):
        strike = self.card("STRIKE_IRONCLAD")
        deck = [strike] * 8
        block_no_retain = CardProfile(
            card_id="SYNTH_BLK", cost=1.0, x_cost=False, card_type="Skill",
            tags=frozenset(), upgraded=False, exhausts_self=False, ethereal=False,
            sly=False, retain=False, innate=False, returns=None,
            effects=({"kind": "block", "amount": 16.0},), powers=(), generates=(),
        )
        block_retain = CardProfile(
            card_id="SYNTH_BLK_RETAIN", cost=1.0, x_cost=False, card_type="Skill",
            tags=frozenset(), upgraded=False, exhausts_self=False, ethereal=False,
            sly=False, retain=True, innate=False, returns=None,
            effects=({"kind": "block", "amount": 16.0},), powers=(), generates=(),
        )
        sit = Situation(6.0, 0.0, 11.0, {"MONSTER": 1.0}, 1.0, 0, enemy_hp=120.0)
        # Alternating 0-damage setup turns and 22-damage spike turns: Retain holds the
        # block card on 0-attack turns and deploys it on spike turns!
        templates = ((0.0, 22.0, 0.0, 22.0, 0.0, 22.0),)
        res = compare_candidates_crn(
            "retain-spike-test", deck,
            {"skip": None, "plain": block_no_retain, "retain": block_retain},
            (), sit, self.params, trajectories=24, attack_templates=templates,
        )
        self.assertGreater(res["retain"].retain_saves, 0.0)
        self.assertLess(res["retain"].mean_hp_loss, res["plain"].mean_hp_loss)
        self.assertLess(res["retain"].cvar25_hp_loss, res["plain"].cvar25_hp_loss)

    def test_high_cost_clumping_increases_clump_rate_vs_zero_cost_card(self):
        bludgeon = self.card("BLUDGEON")
        bash = self.card("BASH")
        defend = self.card("DEFEND_IRONCLAD")
        anger = self.card("ANGER")
        heavy_deck = [bludgeon, bash, bash] + [defend] * 5
        sit = Situation(6.0, 0.0, 12.0, {"MONSTER": 1.0}, 1.0, 0, enemy_hp=110.0)
        res = compare_candidates_crn(
            "clump-test", heavy_deck,
            {"skip": None, "extra_heavy": bludgeon, "zero_cost": anger},
            (), sit, self.params, trajectories=24,
        )
        self.assertGreater(res["extra_heavy"].clump_rate, res["zero_cost"].clump_rate)

    def test_unplayable_status_card_clogs_hand_and_worsens_hp_loss(self):
        strike = self.card("STRIKE_IRONCLAD")
        defend = self.card("DEFEND_IRONCLAD")
        wound = self.card("WOUND")
        deck = [strike] * 4 + [defend] * 4
        sit = Situation(6.0, 0.0, 12.0, {"MONSTER": 1.0}, 1.0, 0, enemy_hp=70.0)
        res = compare_candidates_crn(
            "wound-clog-test", deck, {"skip": None, "wound": wound},
            (), sit, self.params, trajectories=24,
        )
        self.assertGreaterEqual(res["wound"].mean_hp_loss, res["skip"].mean_hp_loss)

    def test_dynamic_cost_rule_and_x_cost_execute_cleanly(self):
        kingly = self.card("KINGLY_KICK")
        whirlwind = self.card("WHIRLWIND")
        strike = self.card("STRIKE_IRONCLAD")
        defend = self.card("DEFEND_IRONCLAD")
        deck = [strike] * 4 + [defend] * 4
        sit = Situation(6.0, 0.0, 10.0, {"MONSTER": 1.0}, 1.0, 0, enemy_hp=65.0)
        res = compare_candidates_crn(
            "dynamic-cost-x-test", deck,
            {"skip": None, "kingly": kingly, "whirlwind": whirlwind},
            (), sit, self.params, trajectories=24,
        )
        # Both Kingly Kick (cost 4 -> 3 on draw #1) and Whirlwind kill faster than skipping.
        self.assertLess(res["kingly"].mean_turns, res["skip"].mean_turns)
        self.assertLess(res["whirlwind"].mean_turns, res["skip"].mean_turns)

    def test_deck_fitness_v4_policy_registration_factors_and_latency(self):
        v3 = create_card_reward_policy("card_reward:deck_fitness_3", self.repo)
        v4 = create_card_reward_policy(DECK_FITNESS_V4_POLICY_VERSION, self.repo)
        self.assertIsInstance(v3, DeckFitnessPolicy)
        self.assertIsInstance(v4, DeckFitnessV4Policy)
        v4.warm_up()

        situations = load_golden_set()[:12]
        triggered = 0
        times_ms = []
        for sit in situations:
            req = card_reward_request(sit["scoring_view"])
            t0 = time.process_time()
            rec = v4.recommend(req)
            times_ms.append((time.process_time() - t0) * 1000.0)
            self.assertEqual(rec.policy_version, DECK_FITNESS_V4_POLICY_VERSION)
            stage2 = rec.payload.get("stage2") or {}
            if stage2.get("triggered"):
                triggered += 1
            if rec.recommended_candidate_id is not None:
                self.assertNotIn(rec.recommended_candidate_id, sit.get("obvious_errors") or ())
        self.assertGreater(triggered, 0)
        times_ms.sort()
        p95 = times_ms[min(len(times_ms) - 1, int(len(times_ms) * 0.95))]
        self.assertLess(p95, 300.0, f"v4 warm P95 latency too high: {p95:.1f}ms")


if __name__ == "__main__":
    unittest.main()
