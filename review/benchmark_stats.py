"""Statistical significance and inter-rater agreement toolkit for STS2 benchmarks.

Zero external scientific dependencies (pure Python 3.9+ stdlib ``math`` + ``random``).
Provides:
- Paired & stratified Bootstrap 95% confidence intervals for rates and paired deltas;
- Exact binomial and continuity-corrected McNemar's test for paired binary comparisons;
- Pairwise Cohen's kappa (with bootstrap 95% CI) and multi-rater Fleiss' kappa.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BootstrapCI:
    n: int
    mean: float
    ci_low: float
    ci_high: float
    se: float
    alpha: float
    n_resamples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean": round(self.mean, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "se": round(self.se, 6),
            "alpha": self.alpha,
            "n_resamples": self.n_resamples,
        }


@dataclass(frozen=True)
class PairedBootstrapResult:
    n: int
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    se: float
    p_value_two_sided: float
    significant: bool
    alpha: float
    n_resamples: int
    stratified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "baseline_mean": round(self.baseline_mean, 6),
            "candidate_mean": round(self.candidate_mean, 6),
            "delta": round(self.delta, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "se": round(self.se, 6),
            "p_value_two_sided": round(self.p_value_two_sided, 6),
            "significant": self.significant,
            "alpha": self.alpha,
            "n_resamples": self.n_resamples,
            "stratified": self.stratified,
        }


@dataclass(frozen=True)
class McNemarResult:
    n: int
    both_correct: int
    baseline_only: int  # b: regressions (baseline=1, candidate=0)
    candidate_only: int  # c: improvements (baseline=0, candidate=1)
    both_wrong: int
    discordant: int
    net_gain: int
    exact_p_value: float
    chi2_cc: float
    chi2_p_value: float
    significant_05: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "both_correct": self.both_correct,
            "baseline_only": self.baseline_only,
            "candidate_only": self.candidate_only,
            "both_wrong": self.both_wrong,
            "discordant": self.discordant,
            "net_gain": self.net_gain,
            "exact_p_value": round(self.exact_p_value, 6),
            "chi2_cc": round(self.chi2_cc, 6),
            "chi2_p_value": round(self.chi2_p_value, 6),
            "significant_05": self.significant_05,
        }


@dataclass(frozen=True)
class KappaResult:
    n: int
    kappa: float
    observed_agreement: float
    expected_agreement: float
    ci_low: float
    ci_high: float
    se: float
    interpretation: str
    categories: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "kappa": round(self.kappa, 6),
            "observed_agreement": round(self.observed_agreement, 6),
            "expected_agreement": round(self.expected_agreement, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "se": round(self.se, 6),
            "interpretation": self.interpretation,
            "categories": list(self.categories),
        }


@dataclass(frozen=True)
class MultiRaterKappaResult:
    n_items: int
    n_raters: int
    rater_names: Tuple[str, ...]
    fleiss_kappa: float
    mean_pairwise_kappa: float
    pairwise: Dict[str, KappaResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_items": self.n_items,
            "n_raters": self.n_raters,
            "rater_names": list(self.rater_names),
            "fleiss_kappa": round(self.fleiss_kappa, 6),
            "mean_pairwise_kappa": round(self.mean_pairwise_kappa, 6),
            "pairwise": {pair: res.to_dict() for pair, res in self.pairwise.items()},
        }


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("Cannot compute percentile of empty sequence")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _sample_std(vals: Sequence[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return math.sqrt(var)


def _interpret_kappa(k: float) -> str:
    val = round(k, 12)
    if val < 0.0:
        return "poor"
    if val <= 0.20:
        return "slight"
    if val <= 0.40:
        return "fair"
    if val <= 0.60:
        return "moderate"
    if val <= 0.80:
        return "substantial"
    return "almost_perfect"


def bootstrap_rate_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Compute non-parametric bootstrap percentile CI for a mean / binary accuracy rate."""
    nums = [float(v) for v in values]
    n = len(nums)
    if n == 0:
        raise ValueError("values must not be empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be >= 100, got {n_resamples}")

    point = sum(nums) / n
    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += nums[rng.randrange(n)]
        boot_means.append(total / n)
    boot_means.sort()
    ci_low = _percentile(boot_means, alpha / 2.0)
    ci_high = _percentile(boot_means, 1.0 - alpha / 2.0)
    return BootstrapCI(
        n=n,
        mean=point,
        ci_low=ci_low,
        ci_high=ci_high,
        se=_sample_std(boot_means),
        alpha=alpha,
        n_resamples=n_resamples,
    )


def paired_bootstrap_ci(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    strata: Optional[Sequence[Hashable]] = None,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Compute paired (optionally stratified) bootstrap 95% CI for delta = candidate - baseline."""
    base = [float(x) for x in baseline]
    cand = [float(x) for x in candidate]
    n = len(base)
    if n == 0 or len(cand) != n:
        raise ValueError("baseline and candidate must be non-empty sequences of equal length")
    if strata is not None and len(strata) != n:
        raise ValueError("strata length must match baseline/candidate length")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be >= 100, got {n_resamples}")

    diffs = [c - b for b, c in zip(base, cand)]
    base_mean = sum(base) / n
    cand_mean = sum(cand) / n
    obs_delta = cand_mean - base_mean

    rng = random.Random(seed)
    boot_deltas: List[float] = []

    if strata is None:
        for _ in range(n_resamples):
            total = 0.0
            for _ in range(n):
                total += diffs[rng.randrange(n)]
            boot_deltas.append(total / n)
    else:
        groups: Dict[Hashable, List[float]] = collections.defaultdict(list)
        for s, d in zip(strata, diffs):
            groups[s].append(d)
        grouped_lists = [groups[k] for k in sorted(groups.keys(), key=str)]
        for _ in range(n_resamples):
            total = 0.0
            for g in grouped_lists:
                m = len(g)
                for _ in range(m):
                    total += g[rng.randrange(m)]
            boot_deltas.append(total / n)

    boot_deltas.sort()
    ci_low = _percentile(boot_deltas, alpha / 2.0)
    ci_high = _percentile(boot_deltas, 1.0 - alpha / 2.0)

    le_zero = sum(1 for d in boot_deltas if d <= 0.0)
    ge_zero = sum(1 for d in boot_deltas if d >= 0.0)
    if obs_delta == 0.0 and all(d == 0.0 for d in diffs):
        p_val = 1.0
    else:
        p_val = min(1.0, 2.0 * min(le_zero, ge_zero) / n_resamples)

    return PairedBootstrapResult(
        n=n,
        baseline_mean=base_mean,
        candidate_mean=cand_mean,
        delta=obs_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        se=_sample_std(boot_deltas),
        p_value_two_sided=p_val,
        significant=(ci_low > 0.0 or ci_high < 0.0),
        alpha=alpha,
        n_resamples=n_resamples,
        stratified=strata is not None,
    )


def _exact_binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided p-value for Binom(n, 0.5) using log-probabilities for overflow safety."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    log_half_n = -n * math.log(2.0)
    log_fact_n = math.lgamma(n + 1)
    tail = 0.0
    for i in range(lo + 1):
        log_p = log_fact_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + log_half_n
        tail += math.exp(log_p)
    return min(1.0, 2.0 * tail)


def mcnemar_test(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
) -> McNemarResult:
    """Compute exact two-sided binomial and continuity-corrected chi-square McNemar's test."""
    n = len(baseline)
    if n == 0 or len(candidate) != n:
        raise ValueError("baseline and candidate must be non-empty sequences of equal length")

    n11 = 0
    n10 = 0  # baseline correct, candidate wrong
    n01 = 0  # baseline wrong, candidate correct
    n00 = 0
    for b_raw, c_raw in zip(baseline, candidate):
        b = bool(b_raw)
        c = bool(c_raw)
        if b and c:
            n11 += 1
        elif b and not c:
            n10 += 1
        elif not b and c:
            n01 += 1
        else:
            n00 += 1

    discordant = n10 + n01
    exact_p = _exact_binomial_two_sided(min(n10, n01), discordant)
    if discordant == 0:
        chi2 = 0.0
        chi2_p = 1.0
    else:
        diff = abs(n10 - n01)
        chi2 = (max(0.0, diff - 1.0) ** 2) / discordant
        chi2_p = math.erfc(math.sqrt(chi2 / 2.0))

    return McNemarResult(
        n=n,
        both_correct=n11,
        baseline_only=n10,
        candidate_only=n01,
        both_wrong=n00,
        discordant=discordant,
        net_gain=n01 - n10,
        exact_p_value=exact_p,
        chi2_cc=chi2,
        chi2_p_value=chi2_p,
        significant_05=exact_p < 0.05,
    )


def _cohens_kappa_point(
    rater_a: Sequence[Hashable],
    rater_b: Sequence[Hashable],
) -> Tuple[float, float, float, Tuple[str, ...]]:
    n = len(rater_a)
    if n == 0 or len(rater_b) != n:
        raise ValueError("rater_a and rater_b must be non-empty sequences of equal length")
    counts_a: collections.Counter = collections.Counter(rater_a)
    counts_b: collections.Counter = collections.Counter(rater_b)
    cats = sorted(set(counts_a.keys()) | set(counts_b.keys()), key=str)

    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    pe = sum((counts_a[c] / n) * (counts_b[c] / n) for c in cats)
    if abs(1.0 - pe) < 1e-12:
        kappa = 1.0 if abs(po - 1.0) < 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)
    return kappa, po, pe, tuple(str(c) for c in cats)


def cohens_kappa(
    rater_a: Sequence[Hashable],
    rater_b: Sequence[Hashable],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> KappaResult:
    """Compute pairwise Cohen's kappa with non-parametric bootstrap confidence interval."""
    n = len(rater_a)
    kappa, po, pe, cats = _cohens_kappa_point(rater_a, rater_b)
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap must be >= 100, got {n_bootstrap}")

    rng = random.Random(seed)
    boot_kappas: List[float] = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        sa = [rater_a[i] for i in idxs]
        sb = [rater_b[i] for i in idxs]
        k_val, _, _, _ = _cohens_kappa_point(sa, sb)
        boot_kappas.append(k_val)
    boot_kappas.sort()
    ci_low = _percentile(boot_kappas, alpha / 2.0)
    ci_high = _percentile(boot_kappas, 1.0 - alpha / 2.0)

    return KappaResult(
        n=n,
        kappa=kappa,
        observed_agreement=po,
        expected_agreement=pe,
        ci_low=ci_low,
        ci_high=ci_high,
        se=_sample_std(boot_kappas),
        interpretation=_interpret_kappa(kappa),
        categories=cats,
    )


def fleiss_kappa(raters: Sequence[Sequence[Hashable]]) -> float:
    """Compute Fleiss' kappa across m >= 2 raters scoring the same N items."""
    m = len(raters)
    if m < 2:
        raise ValueError("fleiss_kappa requires at least 2 raters")
    n = len(raters[0])
    if n == 0 or any(len(r) != n for r in raters):
        raise ValueError("All raters must score the same non-zero number of items")

    cats = sorted({label for r in raters for label in r}, key=str)
    cat_index = {c: i for i, c in enumerate(cats)}
    k_cats = len(cats)

    cat_totals = [0] * k_cats
    p_i_sum = 0.0
    for item_idx in range(n):
        row_counts = [0] * k_cats
        for r in raters:
            ci = cat_index[r[item_idx]]
            row_counts[ci] += 1
            cat_totals[ci] += 1
        p_i = (sum(c * c for c in row_counts) - m) / (m * (m - 1))
        p_i_sum += p_i

    p_bar = p_i_sum / n
    p_e = sum((ct / (n * m)) ** 2 for ct in cat_totals)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_bar - 1.0) < 1e-12 else 0.0
    return (p_bar - p_e) / (1.0 - p_e)


def multi_rater_kappa(
    raters: Mapping[str, Sequence[Hashable]],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> MultiRaterKappaResult:
    """Compute Fleiss' kappa and all pairwise Cohen's kappa scores across multiple raters/policies."""
    names = tuple(raters.keys())
    if len(names) < 2:
        raise ValueError("multi_rater_kappa requires at least 2 named raters")
    n_items = len(raters[names[0]])
    if n_items == 0 or any(len(raters[name]) != n_items for name in names):
        raise ValueError("All raters must have the same non-zero length")

    pairwise: Dict[str, KappaResult] = {}
    pair_idx = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            pairwise[f"{a_name}__vs__{b_name}"] = cohens_kappa(
                raters[a_name],
                raters[b_name],
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                seed=seed + pair_idx,
            )
            pair_idx += 1

    mean_k = sum(r.kappa for r in pairwise.values()) / len(pairwise)
    fk = fleiss_kappa([raters[name] for name in names])
    return MultiRaterKappaResult(
        n_items=n_items,
        n_raters=len(names),
        rater_names=names,
        fleiss_kappa=fk,
        mean_pairwise_kappa=mean_k,
        pairwise=pairwise,
    )


def compare_policy_predictions(
    baseline_preds: Sequence[Hashable],
    candidate_preds: Sequence[Hashable],
    gold_labels: Sequence[Hashable],
    *,
    strata: Optional[Sequence[Hashable]] = None,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full paired statistical comparison between two policies on the same benchmark items."""
    n = len(gold_labels)
    if n == 0 or len(baseline_preds) != n or len(candidate_preds) != n:
        raise ValueError("baseline_preds, candidate_preds, and gold_labels must share non-zero length")

    base_hits = [p == g for p, g in zip(baseline_preds, gold_labels)]
    cand_hits = [p == g for p, g in zip(candidate_preds, gold_labels)]

    base_ci = bootstrap_rate_ci(base_hits, n_resamples=n_resamples, seed=seed)
    cand_ci = bootstrap_rate_ci(cand_hits, n_resamples=n_resamples, seed=seed + 1)
    paired_ci = paired_bootstrap_ci(
        base_hits,
        cand_hits,
        strata=strata,
        n_resamples=n_resamples,
        seed=seed + 2,
    )
    mcnemar = mcnemar_test(base_hits, cand_hits)
    agreement = multi_rater_kappa(
        {
            "gold": list(gold_labels),
            "baseline": list(baseline_preds),
            "candidate": list(candidate_preds),
        },
        n_bootstrap=min(2000, n_resamples),
        seed=seed + 3,
    )

    by_stratum: Dict[str, Dict[str, Any]] = {}
    if strata is not None:
        buckets: Dict[str, List[int]] = collections.defaultdict(list)
        for idx, s in enumerate(strata):
            buckets[str(s)].append(idx)
        for s_name, idxs in sorted(buckets.items()):
            bh = [base_hits[i] for i in idxs]
            ch = [cand_hits[i] for i in idxs]
            m_s = mcnemar_test(bh, ch)
            by_stratum[s_name] = {
                "n": len(idxs),
                "baseline_rate": round(sum(bh) / len(idxs), 4),
                "candidate_rate": round(sum(ch) / len(idxs), 4),
                "delta": round((sum(ch) - sum(bh)) / len(idxs), 4),
                "mcnemar_exact_p": round(m_s.exact_p_value, 6),
            }

    return {
        "n": n,
        "baseline": base_ci.to_dict(),
        "candidate": cand_ci.to_dict(),
        "paired_delta": paired_ci.to_dict(),
        "mcnemar": mcnemar.to_dict(),
        "kappa": agreement.to_dict(),
        "by_stratum": by_stratum,
    }


def _cmd_summarize_bench(bench_path: Path, n_resamples: int, seed: int) -> Dict[str, Any]:
    data = json.loads(bench_path.read_text(encoding="utf-8"))
    situations = data.get("situations") or []
    hits = [bool(s.get("baseline_reference", {}).get("agreed")) for s in situations]
    player_actions = [str(s.get("decision_type", "pick")) for s in situations]
    engine_actions = [
        "skip" if s.get("baseline_reference", {}).get("engine_top") == "skip" else "pick"
        for s in situations
    ]
    rate_ci = bootstrap_rate_ci(hits, n_resamples=n_resamples, seed=seed)
    pick_vs_skip_kappa = cohens_kappa(player_actions, engine_actions, n_bootstrap=min(2000, n_resamples), seed=seed + 1)
    exact_pick_kappa = cohens_kappa(
        [str(s.get("player_choice")) for s in situations],
        [str(s.get("baseline_reference", {}).get("engine_top")) for s in situations],
        n_bootstrap=min(2000, n_resamples),
        seed=seed + 2,
    )
    return {
        "benchmark": str(bench_path),
        "n": len(situations),
        "agreement_rate_ci": rate_ci.to_dict(),
        "exact_choice_kappa": exact_pick_kappa.to_dict(),
        "pick_vs_skip_kappa": pick_vs_skip_kappa.to_dict(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Statistical significance tools for STS2 benchmarks.")
    parser.add_argument("benchmark", type=Path, help="Path to stratified card-reward benchmark JSON.")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = _cmd_summarize_bench(args.benchmark, n_resamples=args.resamples, seed=args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
