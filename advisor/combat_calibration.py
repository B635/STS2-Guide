"""Monotone-constrained real-combat HP-loss calibration model.

Runtime inference (``load_calibration_model`` and ``CombatCalibrationModel``)
uses only the Python standard library so frozen builds stay lightweight.
Offline fitting helpers lazily import ``numpy`` and ``scipy`` and are invoked
by ``scripts/calibrate_combat_model.py`` and unit tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_DIMENSIONS: Tuple[str, ...] = (
    "frontload",
    "scaling",
    "aoe",
    "defense",
    "consistency",
    "energy",
)
# Map deck_mechanics / deck_fitness capability keys to calibration dimensions.
CAPABILITY_TO_CALIBRATION_KEY: Dict[str, str] = {
    "frontload": "frontload",
    "scaling": "scaling",
    "aoe": "aoe",
    "defense": "defense",
    "consistency": "cycle",
    "energy": "energy",
}
DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "combat_calibration.json"
)


def satisfaction_gain(
    ratio: float,
    cap: float = 2.0,
    surplus_slope: float = 0.25,
) -> float:
    """Strictly non-decreasing concave satisfaction transform matching DeckFitness.

    Maps a supply/demand ratio ``s >= 0`` onto ``[0, 1.5 + surplus_slope * ln(cap)]``.
    Because the transform is monotonically non-decreasing in ``s``, pairing it
    with non-negative HP-saved weights ``w_d >= 0`` guarantees that improving
    any deck capability never predicts higher HP loss.
    """

    s = max(0.0, min(float(cap), float(ratio)))
    if s <= 1.0:
        val = s - 0.5 * (1.0 - s) ** 2
    else:
        val = 1.0 + float(surplus_slope) * math.log(s)
    return val + 0.5


def capability_feature_vector(
    capabilities: Mapping[str, float],
    needs: Mapping[str, float],
    *,
    cap: float = 2.0,
    surplus_slope: float = 0.25,
) -> Dict[str, float]:
    """Compute the 6-dimensional monotone satisfaction vector for one deck/fight."""

    features: Dict[str, float] = {}
    for dim in CALIBRATION_DIMENSIONS:
        src_key = CAPABILITY_TO_CALIBRATION_KEY[dim]
        need_val = float(needs.get(src_key, needs.get(dim, 0.0)) or 0.0)
        cap_val = float(capabilities.get(src_key, capabilities.get(dim, 0.0)) or 0.0)
        ratio = (cap_val / need_val) if need_val > 0.0 else float(cap)
        features[dim] = satisfaction_gain(ratio, cap=cap, surplus_slope=surplus_slope)
    return features


def pinball_loss(y_true: Sequence[float], y_pred: Sequence[float], tau: float) -> float:
    """Mean pinball (check) loss at quantile ``tau`` in (0, 1)."""

    if not y_true:
        return 0.0
    total = 0.0
    for yt, yp in zip(y_true, y_pred):
        diff = float(yt) - float(yp)
        total += (tau * diff) if diff >= 0.0 else ((tau - 1.0) * diff)
    return total / len(y_true)


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if not y_true:
        return 0.0
    return sum(abs(float(yt) - float(yp)) for yt, yp in zip(y_true, y_pred)) / len(y_true)


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pure-Python Spearman rank correlation with average tie-breaking."""

    n = len(xs)
    if n < 3 or len(ys) != n:
        return 0.0
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return num / (dx * dy)


def group_kfold_indices(
    groups: Sequence[str],
    n_splits: int = 5,
) -> List[Tuple[List[int], List[int]]]:
    """Deterministic balanced GroupKFold split by group identifier (``run_id``).

    Every occurrence of a group lands in the exact same validation fold so no
    run can ever span both train and validation splits.
    """

    by_group: Dict[str, List[int]] = {}
    for idx, g in enumerate(groups):
        by_group.setdefault(str(g), []).append(idx)
    if len(by_group) < 2:
        raise ValueError("GroupKFold requires at least 2 distinct groups.")
    k = min(int(n_splits), len(by_group))
    # Greedy largest-group-first assignment for balanced fold sizes.
    ordered_groups = sorted(by_group.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    fold_bins: List[List[int]] = [[] for _ in range(k)]
    for _, idxs in ordered_groups:
        smallest = min(range(k), key=lambda f: (len(fold_bins[f]), f))
        fold_bins[smallest].extend(idxs)
    all_set = set(range(len(groups)))
    splits: List[Tuple[List[int], List[int]]] = []
    for f in range(k):
        val = sorted(fold_bins[f])
        train = sorted(all_set - set(val))
        splits.append((train, val))
    return splits


@dataclass(frozen=True)
class StratumWeights:
    """Monotone HP-loss parameters for one ``(act, fight_kind)`` stratum.

    Predicted HP loss is:
        ``max(0, intercept + attack_coef * attack_load - sum(w_d * g(s_d)))``
    with ``attack_coef >= 0`` and every ``w_d >= 0`` (enforcing physical
    non-positive marginal HP loss as capabilities increase).
    """

    key: str
    act: Optional[int]
    fight_kind: Optional[str]
    sample_count: int
    mean_intercept: float
    mean_attack_coef: float
    mean_weights: Mapping[str, float]
    q75_intercept: float
    q75_attack_coef: float
    q75_weights: Mapping[str, float]

    def predict(
        self,
        features: Mapping[str, float],
        attack_load: float,
    ) -> Tuple[float, float]:
        load = max(0.0, float(attack_load))
        mean_raw = (
            self.mean_intercept
            + self.mean_attack_coef * load
            - sum(
                max(0.0, float(self.mean_weights.get(d, 0.0))) * float(features.get(d, 0.0))
                for d in CALIBRATION_DIMENSIONS
            )
        )
        q75_raw = (
            self.q75_intercept
            + self.q75_attack_coef * load
            - sum(
                max(0.0, float(self.q75_weights.get(d, 0.0))) * float(features.get(d, 0.0))
                for d in CALIBRATION_DIMENSIONS
            )
        )
        mean_pred = max(0.0, mean_raw)
        # Coherent upper tail: Q0.75 is never below expected HP loss; taking max
        # of two non-increasing functions preserves monotonicity in every dim.
        q75_pred = max(mean_pred, max(0.0, q75_raw))
        return mean_pred, q75_pred

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "act": self.act,
            "fight_kind": self.fight_kind,
            "sample_count": int(self.sample_count),
            "mean": {
                "intercept": round(float(self.mean_intercept), 6),
                "attack_load_coef": round(max(0.0, float(self.mean_attack_coef)), 6),
                "hp_saved_weights": {
                    d: round(max(0.0, float(self.mean_weights.get(d, 0.0))), 6)
                    for d in CALIBRATION_DIMENSIONS
                },
            },
            "q75": {
                "intercept": round(float(self.q75_intercept), 6),
                "attack_load_coef": round(max(0.0, float(self.q75_attack_coef)), 6),
                "hp_saved_weights": {
                    d: round(max(0.0, float(self.q75_weights.get(d, 0.0))), 6)
                    for d in CALIBRATION_DIMENSIONS
                },
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StratumWeights":
        mean_block = payload["mean"]
        q75_block = payload["q75"]
        return cls(
            key=str(payload["key"]),
            act=int(payload["act"]) if payload.get("act") is not None else None,
            fight_kind=str(payload["fight_kind"]).upper() if payload.get("fight_kind") else None,
            sample_count=int(payload.get("sample_count") or 0),
            mean_intercept=float(mean_block["intercept"]),
            mean_attack_coef=max(0.0, float(mean_block["attack_load_coef"])),
            mean_weights={
                d: max(0.0, float(mean_block["hp_saved_weights"].get(d, 0.0)))
                for d in CALIBRATION_DIMENSIONS
            },
            q75_intercept=float(q75_block["intercept"]),
            q75_attack_coef=max(0.0, float(q75_block["attack_load_coef"])),
            q75_weights={
                d: max(0.0, float(q75_block["hp_saved_weights"].get(d, 0.0)))
                for d in CALIBRATION_DIMENSIONS
            },
        )


@dataclass(frozen=True)
class CombatCalibrationModel:
    """Loaded monotone combat HP-loss calibration parameters and CV metadata."""

    schema_version: int
    cap: float
    surplus_slope: float
    global_weights: StratumWeights
    by_kind: Mapping[str, StratumWeights]
    by_stratum: Mapping[str, StratumWeights]
    cv_metrics: Mapping[str, Any]
    dataset_summary: Mapping[str, Any]

    def weights_for(self, act: int, fight_kind: str) -> StratumWeights:
        kind = str(fight_kind or "MONSTER").upper()
        key = f"act{int(act)}:{kind}"
        if key in self.by_stratum:
            return self.by_stratum[key]
        if kind in self.by_kind:
            return self.by_kind[kind]
        return self.global_weights

    def predict_hp_loss(
        self,
        capabilities: Mapping[str, float],
        needs: Mapping[str, float],
        *,
        act: int,
        fight_kind: str,
        attack_load: float,
    ) -> Tuple[float, float]:
        features = capability_feature_vector(
            capabilities, needs, cap=self.cap, surplus_slope=self.surplus_slope
        )
        return self.weights_for(act, fight_kind).predict(features, attack_load)

    def predict_mix_hp_loss(
        self,
        capabilities: Mapping[str, float],
        needs: Mapping[str, float],
        *,
        act: int,
        fight_mix: Mapping[str, float],
        attack_load: float,
    ) -> Tuple[float, float]:
        features = capability_feature_vector(
            capabilities, needs, cap=self.cap, surplus_slope=self.surplus_slope
        )
        total_w = sum(max(0.0, float(v)) for v in fight_mix.values())
        if total_w <= 0.0:
            return self.weights_for(act, "MONSTER").predict(features, attack_load)
        mean_sum = 0.0
        q75_sum = 0.0
        for kind, count in fight_mix.items():
            w = max(0.0, float(count))
            if w <= 0.0:
                continue
            m, q = self.weights_for(act, kind).predict(features, attack_load)
            mean_sum += w * m
            q75_sum += w * q
        return mean_sum / total_w, q75_sum / total_w

    def marginal_hp_saved(
        self,
        baseline_caps: Mapping[str, float],
        candidate_caps: Mapping[str, float],
        needs: Mapping[str, float],
        *,
        act: int,
        fight_mix: Mapping[str, float],
        attack_load: float,
        tail_weight: float = 0.35,
    ) -> Dict[str, Any]:
        """Expected & Q75 HP saved per fight by adding candidate vs baseline."""

        base_m, base_q = self.predict_mix_hp_loss(
            baseline_caps, needs, act=act, fight_mix=fight_mix, attack_load=attack_load
        )
        cand_m, cand_q = self.predict_mix_hp_loss(
            candidate_caps, needs, act=act, fight_mix=fight_mix, attack_load=attack_load
        )
        mean_saved = base_m - cand_m
        q75_saved = base_q - cand_q
        blended = (1.0 - tail_weight) * mean_saved + tail_weight * q75_saved

        base_feat = capability_feature_vector(
            baseline_caps, needs, cap=self.cap, surplus_slope=self.surplus_slope
        )
        cand_feat = capability_feature_vector(
            candidate_caps, needs, cap=self.cap, surplus_slope=self.surplus_slope
        )
        total_w = sum(max(0.0, float(v)) for v in fight_mix.values()) or 1.0
        dim_saved: Dict[str, float] = {d: 0.0 for d in CALIBRATION_DIMENSIONS}
        for kind, count in (fight_mix.items() if fight_mix else [("MONSTER", 1.0)]):
            w = max(0.0, float(count)) / total_w
            if w <= 0.0:
                continue
            st = self.weights_for(act, kind)
            for d in CALIBRATION_DIMENSIONS:
                eff_w = (1.0 - tail_weight) * st.mean_weights[d] + tail_weight * st.q75_weights[d]
                dim_saved[d] += w * eff_w * (cand_feat[d] - base_feat[d])
        return {
            "baseline_mean_hp_loss": base_m,
            "candidate_mean_hp_loss": cand_m,
            "baseline_q75_hp_loss": base_q,
            "candidate_q75_hp_loss": cand_q,
            "mean_hp_saved": mean_saved,
            "q75_hp_saved": q75_saved,
            "blended_hp_saved": blended,
            "dimension_hp_saved": dim_saved,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "dimensions": list(CALIBRATION_DIMENSIONS),
            "transform": {
                "kind": "deck_fitness_concave_satisfaction",
                "cap": float(self.cap),
                "surplus_slope": float(self.surplus_slope),
            },
            "dataset": dict(self.dataset_summary),
            "cv_metrics": dict(self.cv_metrics),
            "global": self.global_weights.to_dict(),
            "by_kind": {k: v.to_dict() for k, v in sorted(self.by_kind.items())},
            "by_stratum": {k: v.to_dict() for k, v in sorted(self.by_stratum.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CombatCalibrationModel":
        tx = payload.get("transform") or {}
        return cls(
            schema_version=int(payload.get("schema_version") or CALIBRATION_SCHEMA_VERSION),
            cap=float(tx.get("cap", 2.0)),
            surplus_slope=float(tx.get("surplus_slope", 0.25)),
            global_weights=StratumWeights.from_dict(payload["global"]),
            by_kind={
                k.upper(): StratumWeights.from_dict(v)
                for k, v in (payload.get("by_kind") or {}).items()
            },
            by_stratum={
                k: StratumWeights.from_dict(v)
                for k, v in (payload.get("by_stratum") or {}).items()
            },
            cv_metrics=dict(payload.get("cv_metrics") or {}),
            dataset_summary=dict(payload.get("dataset") or {}),
        )


_MODEL_CACHE: Dict[Path, CombatCalibrationModel] = {}


def load_calibration_model(
    path: Optional[Path] = None,
) -> Optional[CombatCalibrationModel]:
    """Load and cache the calibrated HP-loss model, or return ``None`` if absent."""

    resolved = Path(path or DEFAULT_CALIBRATION_PATH).resolve()
    if resolved in _MODEL_CACHE:
        return _MODEL_CACHE[resolved]
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    model = CombatCalibrationModel.from_dict(payload)
    _MODEL_CACHE[resolved] = model
    return model


# ---------------------------------------------------------------------------
# Offline monotone-constrained fitting (lazy numpy/scipy imports)
# ---------------------------------------------------------------------------

def _fit_centered_monotone_quantile(
    z_matrix: Sequence[Sequence[float]],
    attack_loads: Sequence[float],
    y_values: Sequence[float],
    *,
    tau: float = 0.5,
    l2_weights: float = 0.05,
    l2_attack: float = 0.20,
    prior_weights: Optional[Sequence[float]] = None,
    prior_attack_coef: Optional[float] = None,
    prior_strength: float = 0.0,
    group_ids: Optional[Sequence[int]] = None,
) -> Tuple[Any, float, List[float]]:
    """Centered monotone-constrained check-loss regression (global convex optimum).

    Solves ``y ~ b_g + gamma * (a / 60) - z @ w`` with ``gamma >= 0, w >= 0``
    in centered coordinates so regularization shrinks smoothly toward stratum
    quantiles without intercept drift or Simpson's-paradox sign distortion.
    Returns ``(uncentered_intercept_or_map, attack_coef_per_unit, weights)``.
    """

    import numpy as np
    from scipy.optimize import minimize

    z = np.asarray(z_matrix, dtype=float)
    a_scaled = np.asarray(attack_loads, dtype=float) / 60.0
    y = np.asarray(y_values, dtype=float)
    n, p = z.shape
    zm = np.mean(z, axis=0)
    am = float(np.mean(a_scaled))
    zc = z - zm
    ac = a_scaled - am

    if group_ids is None:
        g_idx = np.zeros(n, dtype=int)
        m = 1
    else:
        g_idx = np.asarray(group_ids, dtype=int)
        m = int(np.max(g_idx)) + 1

    w0 = np.zeros(p, dtype=float) if prior_weights is None else np.asarray(prior_weights, dtype=float)
    g0 = 0.0 if prior_attack_coef is None else float(prior_attack_coef) * 60.0
    delta = 0.25

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        bs = theta[:m]
        gamma = theta[m]
        w = theta[m + 1:]
        pred = bs[g_idx] + gamma * ac - zc @ w
        r = y - pred
        abs_r = np.abs(r)
        quad = abs_r <= delta
        huber_abs = np.where(quad, 0.5 * (r * r) / delta + 0.5 * delta, abs_r)
        loss_vec = 0.5 * huber_abs + (tau - 0.5) * r
        d_huber = np.where(quad, r / delta, np.sign(r))
        d_loss_dr = 0.5 * d_huber + (tau - 0.5)

        loss = float(np.mean(loss_vec)) + float(l2_weights) * float(np.dot(w, w)) + float(l2_attack) * gamma * gamma
        grad_bs = np.array([float(np.sum(-d_loss_dr[g_idx == j])) / n for j in range(m)], dtype=float)
        grad_g = float(np.mean(-d_loss_dr * ac)) + 2.0 * float(l2_attack) * gamma
        grad_w = (zc.T @ d_loss_dr) / n + 2.0 * float(l2_weights) * w

        if prior_weights is not None and prior_strength > 0.0:
            diff_w = w - w0
            loss += (float(prior_strength) / n) * float(np.dot(diff_w, diff_w))
            grad_w = grad_w + (2.0 * float(prior_strength) / n) * diff_w
        if prior_attack_coef is not None and prior_strength > 0.0:
            eff_g = float(prior_strength) / n
            loss += eff_g * (gamma - g0) ** 2
            grad_g += 2.0 * eff_g * (gamma - g0)

        return loss, np.concatenate([grad_bs, [grad_g], grad_w])

    init = np.zeros(m + 1 + p, dtype=float)
    for j in range(m):
        mask = g_idx == j
        init[j] = float(np.quantile(y[mask], tau)) if np.any(mask) else float(np.quantile(y, tau))
    init[m] = max(0.0, g0)
    init[m + 1:] = np.maximum(0.0, w0)
    bounds = [(None, None)] * m + [(0.0, None)] + [(0.0, None)] * p
    res = minimize(objective, init, jac=True, method="L-BFGS-B", bounds=bounds)
    bs = res.x[:m]
    gamma_scaled = max(0.0, float(res.x[m]))
    w = np.maximum(0.0, res.x[m + 1:])
    gamma_unit = gamma_scaled / 60.0
    uncentered = [float(bs[j] - gamma_scaled * am + float(np.dot(zm, w))) for j in range(m)]
    if group_ids is None:
        return uncentered[0], gamma_unit, [float(v) for v in w]
    return uncentered, gamma_unit, [float(v) for v in w]


def fit_monotone_mean_model(
    z_matrix: Sequence[Sequence[float]],
    attack_loads: Sequence[float],
    y_values: Sequence[float],
    *,
    l2_penalty: float = 0.05,
    l2_attack: float = 0.20,
    prior_weights: Optional[Sequence[float]] = None,
    prior_attack_coef: Optional[float] = None,
    prior_strength: float = 0.0,
) -> Tuple[float, float, List[float]]:
    """Fit robust MAE-aligned monotone HP-loss model with ``gamma >= 0, w >= 0``."""

    b, g, w = _fit_centered_monotone_quantile(
        z_matrix, attack_loads, y_values,
        tau=0.5, l2_weights=l2_penalty, l2_attack=l2_attack,
        prior_weights=prior_weights, prior_attack_coef=prior_attack_coef,
        prior_strength=prior_strength,
    )
    return float(b), g, w


def fit_monotone_quantile_model(
    z_matrix: Sequence[Sequence[float]],
    attack_loads: Sequence[float],
    y_values: Sequence[float],
    *,
    tau: float = 0.75,
    l2_penalty: float = 0.05,
    l2_attack: float = 0.20,
    prior_weights: Optional[Sequence[float]] = None,
    prior_attack_coef: Optional[float] = None,
    prior_strength: float = 0.0,
) -> Tuple[float, float, List[float]]:
    """Fit monotone quantile regression ``Q_tau(y)`` with ``gamma >= 0, w >= 0``."""

    b, g, w = _fit_centered_monotone_quantile(
        z_matrix, attack_loads, y_values,
        tau=tau, l2_weights=l2_penalty, l2_attack=l2_attack,
        prior_weights=prior_weights, prior_attack_coef=prior_attack_coef,
        prior_strength=prior_strength,
    )
    return float(b), g, w


def _fit_hierarchical(
    records: Sequence[Mapping[str, Any]],
    *,
    l2_mean: float = 0.05,
    l2_q75: float = 0.05,
    l2_attack: float = 0.20,
    prior_strength_kind: float = 15.0,
    prior_strength_stratum: float = 25.0,
) -> Tuple[StratumWeights, Dict[str, StratumWeights], Dict[str, StratumWeights]]:
    """Fit global (stratum-fixed-effect) -> fight_kind -> (act, fight_kind)
    hierarchical monotone models."""

    def extract(subset: Sequence[Mapping[str, Any]]) -> Tuple[List[List[float]], List[float], List[float]]:
        z = [[float(r["features"][d]) for d in CALIBRATION_DIMENSIONS] for r in subset]
        a = [float(r["attack_load"]) for r in subset]
        y = [float(r["hp_lost"]) for r in subset]
        return z, a, y

    z_all, a_all, y_all = extract(records)
    # Use stratum fixed effects when fitting global slopes to avoid Simpson's
    # paradox (late-act bosses having both higher defense and higher damage).
    strata_keys = sorted({(int(r["act"]), str(r["fight_kind"]).upper()) for r in records})
    s_map = {k: idx for idx, k in enumerate(strata_keys)}
    g_ids = [s_map[(int(r["act"]), str(r["fight_kind"]).upper())] for r in records]

    gm_bs, gm_g, gm_w = _fit_centered_monotone_quantile(
        z_all, a_all, y_all, tau=0.5, l2_weights=l2_mean, l2_attack=l2_attack, group_ids=g_ids
    )
    gq_bs, gq_g, gq_w = _fit_centered_monotone_quantile(
        z_all, a_all, y_all, tau=0.75, l2_weights=l2_q75, l2_attack=l2_attack, group_ids=g_ids
    )
    gm_b = sum(gm_bs) / max(1, len(gm_bs))
    gq_b = sum(gq_bs) / max(1, len(gq_bs))
    global_sw = StratumWeights(
        key="global",
        act=None,
        fight_kind=None,
        sample_count=len(records),
        mean_intercept=gm_b,
        mean_attack_coef=gm_g,
        mean_weights=dict(zip(CALIBRATION_DIMENSIONS, gm_w)),
        q75_intercept=gq_b,
        q75_attack_coef=gq_g,
        q75_weights=dict(zip(CALIBRATION_DIMENSIONS, gq_w)),
    )

    by_kind: Dict[str, StratumWeights] = {}
    for kind in ("MONSTER", "ELITE", "BOSS"):
        sub = [r for r in records if str(r["fight_kind"]).upper() == kind]
        if not sub:
            continue
        zk, ak, yk = extract(sub)
        km_b, km_g, km_w = fit_monotone_mean_model(
            zk, ak, yk, l2_penalty=l2_mean * 0.5, l2_attack=l2_attack,
            prior_weights=gm_w, prior_attack_coef=gm_g, prior_strength=prior_strength_kind,
        )
        kq_b, kq_g, kq_w = fit_monotone_quantile_model(
            zk, ak, yk, tau=0.75, l2_penalty=l2_q75 * 0.5, l2_attack=l2_attack,
            prior_weights=gq_w, prior_attack_coef=gq_g, prior_strength=prior_strength_kind,
        )
        by_kind[kind] = StratumWeights(
            key=kind,
            act=None,
            fight_kind=kind,
            sample_count=len(sub),
            mean_intercept=km_b,
            mean_attack_coef=km_g,
            mean_weights=dict(zip(CALIBRATION_DIMENSIONS, km_w)),
            q75_intercept=kq_b,
            q75_attack_coef=kq_g,
            q75_weights=dict(zip(CALIBRATION_DIMENSIONS, kq_w)),
        )

    by_stratum: Dict[str, StratumWeights] = {}
    for act in (1, 2, 3):
        for kind in ("MONSTER", "ELITE", "BOSS"):
            sub = [
                r for r in records
                if int(r["act"]) == act and str(r["fight_kind"]).upper() == kind
            ]
            if not sub:
                continue
            parent = by_kind.get(kind, global_sw)
            pm_w = [parent.mean_weights[d] for d in CALIBRATION_DIMENSIONS]
            pq_w = [parent.q75_weights[d] for d in CALIBRATION_DIMENSIONS]
            zs, as_, ys = extract(sub)
            sm_b, sm_g, sm_w = fit_monotone_mean_model(
                zs, as_, ys, l2_penalty=l2_mean * 0.25, l2_attack=l2_attack * 0.25,
                prior_weights=pm_w, prior_attack_coef=parent.mean_attack_coef,
                prior_strength=prior_strength_stratum,
            )
            sq_b, sq_g, sq_w = fit_monotone_quantile_model(
                zs, as_, ys, tau=0.75, l2_penalty=l2_q75 * 0.25, l2_attack=l2_attack * 0.25,
                prior_weights=pq_w, prior_attack_coef=parent.q75_attack_coef,
                prior_strength=prior_strength_stratum,
            )
            key = f"act{act}:{kind}"
            by_stratum[key] = StratumWeights(
                key=key,
                act=act,
                fight_kind=kind,
                sample_count=len(sub),
                mean_intercept=sm_b,
                mean_attack_coef=sm_g,
                mean_weights=dict(zip(CALIBRATION_DIMENSIONS, sm_w)),
                q75_intercept=sq_b,
                q75_attack_coef=sq_g,
                q75_weights=dict(zip(CALIBRATION_DIMENSIONS, sq_w)),
            )
    return global_sw, by_kind, by_stratum


def calibrate_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 5,
    cap: float = 2.0,
    surplus_slope: float = 0.25,
    dataset_summary: Optional[Mapping[str, Any]] = None,
    l2_mean: float = 0.05,
    l2_q75: float = 0.05,
    l2_attack: float = 0.20,
    prior_strength_kind: float = 15.0,
    prior_strength_stratum: float = 25.0,
) -> CombatCalibrationModel:
    """Run GroupKFold(by ``run_id``) OOF evaluation and fit final full model."""

    import numpy as np

    if not records:
        raise ValueError("Cannot calibrate combat model on an empty dataset.")
    groups = [str(r["run_id"]) for r in records]
    splits = group_kfold_indices(groups, n_splits=n_splits)

    oof_mean = [0.0] * len(records)
    oof_q75 = [0.0] * len(records)
    naive_mean = [0.0] * len(records)
    naive_q75 = [0.0] * len(records)

    for train_idx, val_idx in splits:
        train_rows = [records[i] for i in train_idx]
        g_sw, k_sw, s_sw = _fit_hierarchical(
            train_rows,
            l2_mean=l2_mean,
            l2_q75=l2_q75,
            l2_attack=l2_attack,
            prior_strength_kind=prior_strength_kind,
            prior_strength_stratum=prior_strength_stratum,
        )
        fold_model = CombatCalibrationModel(
            schema_version=CALIBRATION_SCHEMA_VERSION,
            cap=cap,
            surplus_slope=surplus_slope,
            global_weights=g_sw,
            by_kind=k_sw,
            by_stratum=s_sw,
            cv_metrics={},
            dataset_summary={},
        )
        # Naive baseline: stratum median / stratum Q0.75 (falling back to global).
        train_y = [float(r["hp_lost"]) for r in train_rows]
        g_med = float(np.median(train_y))
        g_q75 = float(np.quantile(train_y, 0.75))
        stratum_ys: Dict[str, List[float]] = {}
        for r in train_rows:
            sk = f"act{int(r['act'])}:{str(r['fight_kind']).upper()}"
            stratum_ys.setdefault(sk, []).append(float(r["hp_lost"]))

        for i in val_idx:
            row = records[i]
            sw = fold_model.weights_for(int(row["act"]), str(row["fight_kind"]))
            mp, qp = sw.predict(row["features"], float(row["attack_load"]))
            oof_mean[i] = mp
            oof_q75[i] = qp
            sk = f"act{int(row['act'])}:{str(row['fight_kind']).upper()}"
            sy = stratum_ys.get(sk)
            naive_mean[i] = float(np.median(sy)) if sy else g_med
            naive_q75[i] = float(np.quantile(sy, 0.75)) if sy else g_q75

    y_all = [float(r["hp_lost"]) for r in records]

    def metrics_for(indices: Sequence[int]) -> Dict[str, Any]:
        ys = [y_all[i] for i in indices]
        ms = [oof_mean[i] for i in indices]
        qs = [oof_q75[i] for i in indices]
        nm = [naive_mean[i] for i in indices]
        nq = [naive_q75[i] for i in indices]
        cov = sum(1 for yt, qp in zip(ys, qs) if yt <= qp + 1e-9) / max(1, len(ys))
        return {
            "n": len(indices),
            "oof_mae": round(mean_absolute_error(ys, ms), 4),
            "naive_stratum_mae": round(mean_absolute_error(ys, nm), 4),
            "oof_pinball_q75": round(pinball_loss(ys, qs, 0.75), 4),
            "naive_stratum_pinball_q75": round(pinball_loss(ys, nq, 0.75), 4),
            "oof_spearman_mean": round(spearman_rho(ms, ys), 4),
            "oof_spearman_q75": round(spearman_rho(qs, ys), 4),
            "oof_q75_coverage": round(cov, 4),
        }

    by_kind_metrics = {}
    for kind in ("MONSTER", "ELITE", "BOSS"):
        idxs = [i for i, r in enumerate(records) if str(r["fight_kind"]).upper() == kind]
        if idxs:
            by_kind_metrics[kind] = metrics_for(idxs)
    by_act_metrics = {}
    for act in (1, 2, 3):
        idxs = [i for i, r in enumerate(records) if int(r["act"]) == act]
        if idxs:
            by_act_metrics[f"act{act}"] = metrics_for(idxs)

    cv_metrics = {
        "cv_scheme": f"GroupKFold(n_splits={len(splits)}, group=run_id)",
        "distinct_runs": len(set(groups)),
        "overall": metrics_for(list(range(len(records)))),
        "by_kind": by_kind_metrics,
        "by_act": by_act_metrics,
    }

    g_sw, k_sw, s_sw = _fit_hierarchical(
        records,
        l2_mean=l2_mean,
        l2_q75=l2_q75,
        l2_attack=l2_attack,
        prior_strength_kind=prior_strength_kind,
        prior_strength_stratum=prior_strength_stratum,
    )
    return CombatCalibrationModel(
        schema_version=CALIBRATION_SCHEMA_VERSION,
        cap=cap,
        surplus_slope=surplus_slope,
        global_weights=g_sw,
        by_kind=k_sw,
        by_stratum=s_sw,
        cv_metrics=cv_metrics,
        dataset_summary=dict(dataset_summary or {"fights": len(records), "runs": len(set(groups))}),
    )
