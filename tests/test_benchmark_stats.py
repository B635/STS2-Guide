"""Unit tests for review.benchmark_stats and stratified card-reward benchmark integrity."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from advisor.golden_set import card_reward_request
from review.benchmark_stats import (
    bootstrap_rate_ci,
    cohens_kappa,
    compare_policy_predictions,
    fleiss_kappa,
    mcnemar_test,
    multi_rater_kappa,
    paired_bootstrap_ci,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
STRATIFIED_BENCH_PATH = ROOT_DIR / "data" / "card_reward_stratified_bench.json"
GOLDEN_SET_PATH = ROOT_DIR / "data" / "golden_set" / "card_reward.json"
COACH_SPLIT_PATH = ROOT_DIR / "coach" / "eval" / "split.json"


class BenchmarkStatsTests(unittest.TestCase):
    def test_bootstrap_rate_ci_is_deterministic_and_bounded(self):
        data = [1] * 60 + [0] * 40
        ci1 = bootstrap_rate_ci(data, n_resamples=2000, seed=123)
        ci2 = bootstrap_rate_ci(data, n_resamples=2000, seed=123)
        self.assertEqual(ci1, ci2)
        self.assertAlmostEqual(ci1.mean, 0.6, places=6)
        self.assertLess(ci1.ci_low, ci1.mean)
        self.assertGreater(ci1.ci_high, ci1.mean)
        # Analytic binomial SE = sqrt(0.6 * 0.4 / 100) = 0.04899
        self.assertAlmostEqual(ci1.se, math.sqrt(0.24 / 100), delta=0.008)

        all_one = bootstrap_rate_ci([True] * 20, n_resamples=500, seed=7)
        self.assertEqual(all_one.mean, 1.0)
        self.assertEqual(all_one.ci_low, 1.0)
        self.assertEqual(all_one.ci_high, 1.0)

        with self.assertRaises(ValueError):
            bootstrap_rate_ci([])

    def test_paired_and_stratified_bootstrap_detects_gain_and_null(self):
        # Null case: identical policies
        same = [True, False] * 40
        res_null = paired_bootstrap_ci(same, same, n_resamples=1000, seed=42)
        self.assertEqual(res_null.delta, 0.0)
        self.assertEqual(res_null.p_value_two_sided, 1.0)
        self.assertFalse(res_null.significant)

        # Paired improvement: candidate strictly improves 12 items with 0 regressions
        base = [True] * 40 + [False] * 60
        cand = [True] * 52 + [False] * 48
        strata = ["act1"] * 50 + ["act2"] * 50
        res_gain = paired_bootstrap_ci(base, cand, n_resamples=3000, seed=42)
        res_strat = paired_bootstrap_ci(base, cand, strata=strata, n_resamples=3000, seed=42)
        self.assertAlmostEqual(res_gain.delta, 0.12, places=6)
        self.assertTrue(res_gain.significant)
        self.assertGreater(res_gain.ci_low, 0.0)
        self.assertLess(res_gain.p_value_two_sided, 0.01)
        self.assertTrue(res_strat.stratified)
        self.assertTrue(res_strat.significant)
        self.assertAlmostEqual(res_strat.delta, 0.12, places=6)

    def test_mcnemar_exact_and_chi2_match_analytical_values(self):
        # 1) +1 / -1 net flat (like engine-a2-comparison)
        base_flat = [True, True, False, False]
        cand_flat = [True, False, True, False]
        r_flat = mcnemar_test(base_flat, cand_flat)
        self.assertEqual((r_flat.both_correct, r_flat.baseline_only, r_flat.candidate_only, r_flat.both_wrong), (1, 1, 1, 1))
        self.assertEqual(r_flat.net_gain, 0)
        self.assertAlmostEqual(r_flat.exact_p_value, 1.0, places=9)
        self.assertFalse(r_flat.significant_05)

        # 2) b=0, c=6 -> exact p = 2 * 0.5^6 = 0.03125 < 0.05
        r_sig = mcnemar_test([False] * 6 + [True] * 20, [True] * 26)
        self.assertAlmostEqual(r_sig.exact_p_value, 2.0 / 64.0, places=9)
        self.assertTrue(r_sig.significant_05)

        # 3) b=0, c=5 -> exact p = 2 * 0.5^5 = 0.0625 >= 0.05
        r_ns = mcnemar_test([False] * 5 + [True] * 20, [True] * 25)
        self.assertAlmostEqual(r_ns.exact_p_value, 2.0 / 32.0, places=9)
        self.assertFalse(r_ns.significant_05)

        # 4) b=2, c=12 -> chi2_cc = (|2-12|-1)^2 / 14 = 81/14
        base_big = [True] * 2 + [False] * 12 + [True] * 30
        cand_big = [False] * 2 + [True] * 12 + [True] * 30
        r_big = mcnemar_test(base_big, cand_big)
        expected_chi2 = 81.0 / 14.0
        self.assertAlmostEqual(r_big.chi2_cc, expected_chi2, places=9)
        self.assertAlmostEqual(r_big.chi2_p_value, math.erfc(math.sqrt(expected_chi2 / 2.0)), places=9)
        self.assertTrue(r_big.significant_05)

        # 5) Zero discordant pairs
        r_zero = mcnemar_test([True, False, True], [True, False, True])
        self.assertEqual(r_zero.discordant, 0)
        self.assertEqual(r_zero.exact_p_value, 1.0)
        self.assertEqual(r_zero.chi2_cc, 0.0)

    def test_cohens_kappa_and_multi_rater_fleiss_kappa(self):
        # Analytical case: po=0.8, pe=0.5 -> kappa=0.6
        a = ["pick"] * 5 + ["skip"] * 5
        b = ["pick"] * 4 + ["skip"] + ["skip"] * 4 + ["pick"]
        res = cohens_kappa(a, b, n_bootstrap=500, seed=42)
        self.assertAlmostEqual(res.observed_agreement, 0.8, places=9)
        self.assertAlmostEqual(res.expected_agreement, 0.5, places=9)
        self.assertAlmostEqual(res.kappa, 0.6, places=9)
        self.assertEqual(res.interpretation, "moderate")
        self.assertLessEqual(res.ci_low, res.kappa)
        self.assertGreaterEqual(res.ci_high, res.kappa)

        # Perfect agreement
        perf = cohens_kappa(["A", "B", "C", "A"], ["A", "B", "C", "A"], n_bootstrap=200, seed=1)
        self.assertAlmostEqual(perf.kappa, 1.0, places=9)
        self.assertEqual(perf.interpretation, "almost_perfect")

        # Multi-rater Fleiss' kappa & pairwise matrix
        raters = {
            "human": ["A", "B", "A", "C", "B", "A", "C", "B"],
            "policy_v1": ["A", "B", "A", "C", "A", "A", "C", "B"],
            "policy_v2": ["A", "B", "A", "C", "B", "A", "C", "B"],
        }
        mk = multi_rater_kappa(raters, n_bootstrap=300, seed=42)
        self.assertEqual(mk.n_raters, 3)
        self.assertEqual(len(mk.pairwise), 3)
        self.assertAlmostEqual(mk.pairwise["human__vs__policy_v2"].kappa, 1.0, places=9)
        self.assertGreater(mk.fleiss_kappa, 0.8)
        self.assertAlmostEqual(mk.fleiss_kappa, fleiss_kappa(list(raters.values())), places=9)

    def test_compare_policy_predictions_returns_complete_bundle(self):
        gold = ["0:A", "skip", "1:B", "skip"] * 15
        base = ["0:A", "0:X", "1:B", "2:Y"] * 15
        cand = ["0:A", "skip", "1:B", "2:Y"] * 15
        strata = (["act1", "act1", "act2", "act2"]) * 15
        bundle = compare_policy_predictions(base, cand, gold, strata=strata, n_resamples=1000, seed=7)
        self.assertEqual(bundle["n"], 60)
        self.assertEqual(bundle["mcnemar"]["candidate_only"], 15)
        self.assertEqual(bundle["mcnemar"]["baseline_only"], 0)
        self.assertTrue(bundle["mcnemar"]["significant_05"])
        self.assertTrue(bundle["paired_delta"]["significant"])
        self.assertIn("act1", bundle["by_stratum"])
        self.assertIn("act2", bundle["by_stratum"])
        self.assertIn("gold__vs__candidate", bundle["kappa"]["pairwise"])


class StratifiedBenchmarkIntegrityTests(unittest.TestCase):
    def test_stratified_benchmark_meets_size_coverage_and_zero_test_leakage(self):
        self.assertTrue(STRATIFIED_BENCH_PATH.is_file(), f"Missing {STRATIFIED_BENCH_PATH}")
        bench = json.loads(STRATIFIED_BENCH_PATH.read_text(encoding="utf-8"))
        split = json.loads(COACH_SPLIT_PATH.read_text(encoding="utf-8"))
        test_runs = set(split["test"])

        situations = bench["situations"]
        self.assertGreaterEqual(len(situations), 240)

        seen_ids = set()
        chars = set()
        acts = set()
        decision_types = set()
        margin_buckets = set()
        for sit in situations:
            self.assertNotIn(sit["id"], seen_ids)
            seen_ids.add(sit["id"])

            run_id = str(sit["source"]["run"])
            self.assertNotIn(run_id, test_runs, f"Test run {run_id} leaked into stratified benchmark")

            chars.add(sit["character"])
            acts.add(sit["act"])
            decision_types.add(sit["decision_type"])
            margin_buckets.add(sit["margin_bucket"])

            view = sit["scoring_view"]
            self.assertIn("deck", view["state"])
            self.assertIn("relics", view["state"])
            self.assertGreaterEqual(len(view["candidates"]), 2)
            req = card_reward_request(view)
            self.assertEqual(req.decision_type, "card_reward")

        self.assertEqual(chars, {"IRONCLAD", "SILENT", "DEFECT", "REGENT", "NECROBINDER"})
        self.assertEqual(acts, {1, 2, 3})
        self.assertEqual(decision_types, {"pick", "skip"})
        self.assertEqual(margin_buckets, {"high_margin", "close_call"})

        # Original 79-case manually labeled golden set remains intact.
        golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(golden["situations"]), 79)


if __name__ == "__main__":
    unittest.main()
