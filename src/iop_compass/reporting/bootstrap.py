"""Patient-level bootstrap confidence intervals.

Resampling is over *patients*, never images, so all five views of a patient are
kept together in every replicate.  Because every reported metric is a pure
function of summed per-image statistics, a replicate is just a re-summation and
1,000+ replicates are cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..segmentation.metrics import ImageEval, aggregate


@dataclass
class Interval:
    point: float
    low: float
    high: float
    n_replicates: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "n_replicates": self.n_replicates,
        }


def group_by_patient(evals: list[ImageEval]) -> dict[str, list[ImageEval]]:
    groups: dict[str, list[ImageEval]] = {}
    for ev in evals:
        groups.setdefault(ev.patient_id, []).append(ev)
    return groups


def bootstrap_metrics(
    evals: list[ImageEval],
    metrics: list[str],
    n_replicates: int = 1000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> dict[str, Interval]:
    """Percentile bootstrap CIs for the named pooled metrics."""
    groups = group_by_patient(evals)
    patient_ids = sorted(groups)
    if not patient_ids:
        return {}

    point = aggregate(evals)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}

    n = len(patient_ids)
    for _ in range(n_replicates):
        picks = rng.integers(0, n, size=n)
        sample: list[ImageEval] = []
        for idx in picks:
            sample.extend(groups[patient_ids[idx]])
        agg = aggregate(sample)
        for m in metrics:
            draws[m].append(agg.get(m, float("nan")))

    out: dict[str, Interval] = {}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for m in metrics:
        arr = np.asarray(draws[m], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[m] = Interval(point.get(m, float("nan")), float("nan"), float("nan"), 0)
            continue
        out[m] = Interval(
            point=float(point.get(m, float("nan"))),
            low=float(np.percentile(arr, lo_q)),
            high=float(np.percentile(arr, hi_q)),
            n_replicates=int(arr.size),
        )
    return out


def paired_bootstrap(
    evals_a: list[ImageEval],
    evals_b: list[ImageEval],
    metrics: list[str],
    n_replicates: int = 1000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> dict[str, Interval]:
    """Paired CI for ``metric(b) - metric(a)`` over the shared patients.

    The same resampled patient set is used for both methods, so the interval
    reflects the paired difference and not two independent samples.
    """
    groups_a = group_by_patient(evals_a)
    groups_b = group_by_patient(evals_b)
    shared = sorted(set(groups_a) & set(groups_b))
    if not shared:
        return {}

    base_a = aggregate([e for p in shared for e in groups_a[p]])
    base_b = aggregate([e for p in shared for e in groups_b[p]])
    point = {m: base_b.get(m, float("nan")) - base_a.get(m, float("nan")) for m in metrics}

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    n = len(shared)
    for _ in range(n_replicates):
        picks = rng.integers(0, n, size=n)
        sample_a: list[ImageEval] = []
        sample_b: list[ImageEval] = []
        for idx in picks:
            pid = shared[idx]
            sample_a.extend(groups_a[pid])
            sample_b.extend(groups_b[pid])
        agg_a = aggregate(sample_a)
        agg_b = aggregate(sample_b)
        for m in metrics:
            draws[m].append(agg_b.get(m, float("nan")) - agg_a.get(m, float("nan")))

    out: dict[str, Interval] = {}
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for m in metrics:
        arr = np.asarray(draws[m], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            out[m] = Interval(point[m], float("nan"), float("nan"), 0)
            continue
        out[m] = Interval(
            point=float(point[m]),
            low=float(np.percentile(arr, lo_q)),
            high=float(np.percentile(arr, hi_q)),
            n_replicates=int(arr.size),
        )
    return out


def is_inconclusive(interval: Interval) -> bool:
    """True when the paired interval spans zero, i.e. no superiority claim."""
    if np.isnan(interval.low) or np.isnan(interval.high):
        return True
    return interval.low <= 0.0 <= interval.high
