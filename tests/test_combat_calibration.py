from __future__ import annotations

import unittest
from pathlib import Path

from advisor.combat_calibration import (
    CALIBRATION_DIMENSIONS,
    DEFAULT_CALIBRATION_PATH,
    calibrate_from_records,
    capability_feature_vector,
    fit_monotone_mean_model,
    fit_monotone_quantile_model,
    group_kfold_indices,
    load_calibration_model,
    pinball_loss,
    satisfaction_gain,
    spearman_rho,
)


class CombatCalibrationMathTests(unittest.TestCase):
    def test_group_kfold_keeps_every_run_in_a_single_fold(self):
        groups = [f"run-{i // 7}" for i in range(70)]
        splits = group_kfold_indices(groups, n_splits=5)
        self.assertEqual(len(splits), 5)
        seen_val = set()
        for train_idx, val_idx in splits:
            train_groups = {groups[i] for i in train_idx}
            val_groups = {groups[i] for i in val_idx}
            self.assertFalse(train_groups & val_groups)
            self.assertFalse(seen_val & set(val_idx))
            seen_val.update(val_idx)
        self.assertEqual(len(seen_val), len(groups))

    def test_monotone_fit_enforces_non_negative_weights_on_adversarial_data(self):
        # Adversarial data: higher feature value correlates with HIGHER hp_lost.
        # Physical monotonicity constraint must clamp every HP-saved weight >= 0.
        z = [[float(i % 5) * 0.3 for _ in CALIBRATION_DIMENSIONS] for i in range(30)]
        a = [30.0 + float(i % 4) * 10.0 for i in range(30)]
        y = [5.0 + float(i % 5) * 4.0 for i in range(30)]

        _, gm, wm = fit_monotone_mean_model(z, a, y)
        _, gq, wq = fit_monotone_quantile_model(z, a, y, tau=0.75)
        self.assertGreaterEqual(gm, 0.0)
        self.assertGreaterEqual(gq, 0.0)
        self.assertTrue(all(w >= 0.0 for w in wm), wm)
        self.assertTrue(all(w >= 0.0 for w in wq), wq)

    def test_pinball_loss_and_spearman_math(self):
        # y=[10, 0], pred=[6, 4], tau=0.75: r=[+4, -4] -> 0.75*4=3.0 and (-0.25)*(-4)=1.0 -> avg 2.0
        self.assertAlmostEqual(pinball_loss([10.0, 0.0], [6.0, 4.0], 0.75), 2.0, places=9)
        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=9)
        self.assertAlmostEqual(spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=9)
        # Satisfaction gain is strictly increasing across [0, 2].
        vals = [satisfaction_gain(x / 10.0) for x in range(21)]
        for prev, nxt in zip(vals, vals[1:]):
            self.assertLess(prev, nxt)

    def test_synthetic_end_to_end_calibration_recovers_signal(self):
        records = []
        for run_idx in range(20):
            for f_idx in range(6):
                fl = 0.2 + 0.35 * ((run_idx + f_idx) % 5)
                df = 0.2 + 0.30 * ((run_idx * 2 + f_idx) % 5)
                atk = 30.0 + 10.0 * (f_idx % 4)
                caps = {
                    "frontload": 10.0 * fl, "scaling": 5.0, "aoe": 4.0,
                    "defense": 10.0 * df, "cycle": 1.0, "energy": 1.0,
                }
                needs = {
                    "frontload": 10.0, "scaling": 5.0, "aoe": 4.0,
                    "defense": 10.0, "cycle": 1.0, "energy": 1.0,
                }
                feats = capability_feature_vector(caps, needs)
                hp_lost = max(
                    0.0, 35.0 + 0.35 * atk - 10.0 * feats["frontload"] - 8.0 * feats["defense"]
                )
                records.append({
                    "run_id": f"r{run_idx}",
                    "act": 1,
                    "fight_kind": "MONSTER",
                    "attack_load": atk,
                    "hp_lost": hp_lost,
                    "features": feats,
                })
        model = calibrate_from_records(records, n_splits=5)
        self.assertGreater(model.cv_metrics["overall"]["oof_spearman_mean"], 0.80)
        self.assertLess(
            model.cv_metrics["overall"]["oof_mae"],
            model.cv_metrics["overall"]["naive_stratum_mae"],
        )
        # With light regularization on clean synthetic data, OOF Spearman approaches 1.0.
        tight = calibrate_from_records(records, n_splits=5, l2_mean=0.002, l2_q75=0.002, l2_attack=0.002)
        self.assertGreater(tight.cv_metrics["overall"]["oof_spearman_mean"], 0.98)


class CommittedCombatCalibrationTests(unittest.TestCase):
    def test_committed_artifact_is_leak_free_and_monotone_across_all_strata(self):
        self.assertTrue(DEFAULT_CALIBRATION_PATH.exists(), DEFAULT_CALIBRATION_PATH)
        model = load_calibration_model(DEFAULT_CALIBRATION_PATH)
        self.assertIsNotNone(model)

        ds = model.dataset_summary
        self.assertEqual(ds.get("test_split_leakage"), 0)
        self.assertEqual(ds.get("excluded_test_runs"), 15)
        self.assertGreaterEqual(int(ds.get("valid_fights") or 0), 500)

        ov = model.cv_metrics["overall"]
        self.assertLessEqual(ov["oof_mae"], ov["naive_stratum_mae"])
        self.assertGreater(ov["oof_spearman_mean"], 0.45)
        self.assertGreaterEqual(ov["oof_q75_coverage"], 0.68)
        self.assertLessEqual(ov["oof_q75_coverage"], 0.82)

        # Every stratum must have non-negative weights and monotonically non-increasing
        # HP-loss predictions when ANY capability improves.
        needs = {
            "frontload": 15.0, "scaling": 8.0, "aoe": 6.0,
            "defense": 10.0, "cycle": 1.5, "energy": 1.0,
        }
        base_caps = {
            "frontload": 8.0, "scaling": 3.0, "aoe": 2.0,
            "defense": 5.0, "cycle": 0.5, "energy": 0.6,
        }
        for act in (1, 2, 3):
            for kind in ("MONSTER", "ELITE", "BOSS"):
                sw = model.weights_for(act, kind)
                self.assertGreaterEqual(sw.mean_attack_coef, 0.0)
                self.assertGreaterEqual(sw.q75_attack_coef, 0.0)
                for dim in CALIBRATION_DIMENSIONS:
                    self.assertGreaterEqual(sw.mean_weights[dim], 0.0)
                    self.assertGreaterEqual(sw.q75_weights[dim], 0.0)
                base_m, base_q = model.predict_hp_loss(
                    base_caps, needs, act=act, fight_kind=kind, attack_load=60.0
                )
                self.assertGreaterEqual(base_q, base_m - 1e-9)
                for cap_key in ("frontload", "scaling", "aoe", "defense", "cycle", "energy"):
                    improved = dict(base_caps)
                    improved[cap_key] = base_caps[cap_key] * 2.0
                    imp_m, imp_q = model.predict_hp_loss(
                        improved, needs, act=act, fight_kind=kind, attack_load=60.0
                    )
                    self.assertLessEqual(imp_m, base_m + 1e-9, (act, kind, cap_key))
                    self.assertLessEqual(imp_q, base_q + 1e-9, (act, kind, cap_key))
                    self.assertGreaterEqual(imp_q, imp_m - 1e-9, (act, kind, cap_key))

        # Marginal HP saved over a fight mix is non-negative when capabilities dominate.
        stronger = {k: v * 1.6 for k, v in base_caps.items()}
        saved = model.marginal_hp_saved(
            base_caps, stronger, needs,
            act=1, fight_mix={"MONSTER": 3.0, "ELITE": 1.0, "BOSS": 1.0}, attack_load=60.0,
        )
        self.assertGreaterEqual(saved["mean_hp_saved"], 0.0)
        self.assertGreaterEqual(saved["q75_hp_saved"], 0.0)
        self.assertGreaterEqual(saved["blended_hp_saved"], 0.0)


if __name__ == "__main__":
    unittest.main()
