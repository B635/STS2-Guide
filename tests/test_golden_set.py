import copy
import json
import tempfile
import unittest
from pathlib import Path

from advisor.decision_core import CARD_REWARD, Recommendation
from advisor.golden_set import (
    CARD_REWARD_VERSIONS,
    SCORING_FIELDS,
    GoldenSetError,
    evaluate_golden_set,
    load_golden_set,
    scoring_view,
)
from advisor.policies import CardRewardPolicy
from storage.relational import RelationalRepository
from storage.release_database import build_release_database


ROOT = Path(__file__).resolve().parents[1]
STATE_EVENT = ROOT / "protocol" / "state-event.example.json"


class FixedChoicePolicy:
    """A stand-in policy version that ranks one given candidate first."""

    decision_type = CARD_REWARD

    def __init__(self, repository, choice):
        self.inner = CardRewardPolicy(repository)
        self.choice = choice

    def recommend(self, request):
        base = self.inner.recommend(request)
        return Recommendation(
            decision_id=base.decision_id,
            decision_type=base.decision_type,
            payload=base.payload,
            world_sequence=base.world_sequence,
            policy_version="fixed-choice",
            candidates=base.candidates,
            recommended_candidate_id=self.choice,
            status="skip" if self.choice == "skip" else "recommend",
            confidence="low",
        )


class GoldenSetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        release = Path(cls.temporary.name) / "release.db"
        build_release_database(
            release, ROOT / "data" / "knowledge.json", ROOT / "data" / "community_scores.json"
        )
        cls.repository = RelationalRepository(str(release))
        event = json.loads(STATE_EVENT.read_text(encoding="utf-8"))
        cls.view = scoring_view(event)
        cls.ids = [c["candidate_id"] for c in cls.view["candidates"]]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def situation(self, **fields):
        base = {
            "id": "example",
            "character": self.view["state"]["character"],
            "source": {"kind": "live_snapshot"},
            "best": [self.ids[0]],
            "obvious_errors": [],
            "label_status": "draft",
            "rationale": "test",
            "scoring_view": copy.deepcopy(self.view),
        }
        base.update(fields)
        return base

    def _load(self, situations):
        path = Path(self.temporary.name) / "golden.json"
        path.write_text(
            json.dumps({"schema_version": 1, "situations": situations}), encoding="utf-8"
        )
        return load_golden_set(path)

    def test_committed_golden_set_runs_every_version_through_the_registry(self):
        situations = load_golden_set()
        for version, factory in CARD_REWARD_VERSIONS.items():
            report = evaluate_golden_set(situations, factory(self.repository))
            self.assertEqual(report["all"]["situations"], len(situations))
            for row in report["rows"]:
                # ``None`` means the version declined to choose (uncertain).
                self.assertTrue(row["first"] is None or row["first"] in row["scores"])

    def test_scoring_view_keeps_only_scoring_fields(self):
        event = json.loads(STATE_EVENT.read_text(encoding="utf-8"))
        view = scoring_view(event)
        self.assertEqual(set(view), set(SCORING_FIELDS))
        self.assertNotIn("release_fingerprint", view)
        event.pop("candidates")
        with self.assertRaises(GoldenSetError):
            scoring_view(event)

    def test_hits_obvious_errors_and_confirmed_scope_follow_the_labels(self):
        best, error = self.ids[0], self.ids[1]
        situations = self._load([
            self.situation(id="confirmed", best=[best], obvious_errors=[error],
                           label_status="confirmed"),
            self.situation(id="draft", best=[best], obvious_errors=[error]),
        ])

        hit = evaluate_golden_set(situations, FixedChoicePolicy(self.repository, best))
        wrong = evaluate_golden_set(situations, FixedChoicePolicy(self.repository, error))

        self.assertEqual(hit["all"]["hits"], 2)
        self.assertEqual(hit["confirmed"]["situations"], 1)
        self.assertEqual(hit["confirmed"]["hits"], 1)
        self.assertEqual(wrong["all"]["obvious_errors_first"], 2)
        self.assertEqual(wrong["confirmed"]["obvious_errors_first"], 1)

    def test_known_limits_are_reported_as_warnings(self):
        view = copy.deepcopy(self.view)
        view["map_context"] = None
        situations = self._load([
            self.situation(source={"kind": "reconstructed"}, scoring_view=view),
        ])
        report = evaluate_golden_set(situations, CardRewardPolicy(self.repository))
        self.assertEqual(report["rows"][0]["warnings"], ["reconstructed", "map_context_missing"])

    def test_labels_must_name_the_situations_own_candidates(self):
        with self.assertRaises(GoldenSetError):
            self._load([self.situation(best=["9:NOT_OFFERED"])])
        with self.assertRaises(GoldenSetError):
            self._load([self.situation(best=[self.ids[0]], obvious_errors=[self.ids[0]])])

    def test_situations_need_label_status_and_source_kind(self):
        with self.assertRaises(GoldenSetError):
            self._load([self.situation(label_status=None)])
        with self.assertRaises(GoldenSetError):
            self._load([self.situation(source={})])


if __name__ == "__main__":
    unittest.main()
