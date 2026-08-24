"""View-classification metrics.

Primary outcome: exact five-view patient-set accuracy after one-to-one
constrained assignment.  Secondary: image-level macro F1 and accuracy, per-view
recall, expected calibration error, and the fraction of patients whose five views
are all independently correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constrained_assignment import constrained_assign
from .dataset import VIEW_CLASSES


@dataclass
class ClassificationEval:
    n_images: int
    n_patients: int
    accuracy: float
    macro_f1: float
    per_view_recall: dict[str, float]
    per_view_precision: dict[str, float]
    ece: float
    mean_confidence: float
    patient_all_correct_independent: float
    patient_set_exact_constrained: float
    n_patients_all_correct_independent: int
    n_patients_exact_constrained: int
    accuracy_constrained: float
    macro_f1_constrained: float
    confusion: list[list[int]] = field(default_factory=list)
    confusion_constrained: list[list[int]] = field(default_factory=list)
    n_incomplete_patients: int = 0

    def to_dict(self) -> dict:
        payload = self.__dict__.copy()
        return payload


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, k: int) -> np.ndarray:
    matrix = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(labels, preds):
        matrix[int(t), int(p)] += 1
    return matrix


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    scores = []
    for k in range(matrix.shape[0]):
        tp = matrix[k, k]
        fp = matrix[:, k].sum() - tp
        fn = matrix[k, :].sum() - tp
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def per_class_recall(matrix: np.ndarray) -> dict[str, float]:
    out = {}
    for k, name in enumerate(VIEW_CLASSES):
        support = matrix[k, :].sum()
        out[name] = float(matrix[k, k] / support) if support else float("nan")
    return out


def per_class_precision(matrix: np.ndarray) -> dict[str, float]:
    out = {}
    for k, name in enumerate(VIEW_CLASSES):
        predicted = matrix[:, k].sum()
        out[name] = float(matrix[k, k] / predicted) if predicted else float("nan")
    return out


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, float]:
    """Equal-width-bin ECE on the top-1 confidence, plus the mean confidence."""
    if probs.size == 0:
        return float("nan"), float("nan")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.sum() / n * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece), float(confidence.mean())


def evaluate_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[str],
    views_required: int = len(VIEW_CLASSES),
) -> ClassificationEval:
    k = len(VIEW_CLASSES)
    preds = probs.argmax(axis=1)
    matrix = confusion_matrix(labels, preds, k)

    by_patient: dict[str, list[int]] = {}
    for idx, pid in enumerate(patient_ids):
        by_patient.setdefault(pid, []).append(idx)

    all_correct_independent = 0
    exact_constrained = 0
    incomplete = 0
    constrained_preds = preds.copy()
    for pid, indices in by_patient.items():
        idx = np.asarray(sorted(indices))
        if len(idx) != views_required:
            incomplete += 1
            continue
        if bool((preds[idx] == labels[idx]).all()):
            all_correct_independent += 1
        assigned = constrained_assign(probs[idx])
        constrained_preds[idx] = assigned
        if bool((assigned == labels[idx]).all()):
            exact_constrained += 1

    matrix_constrained = confusion_matrix(labels, constrained_preds, k)
    ece, mean_conf = expected_calibration_error(probs, labels)
    n_patients = len(by_patient)
    complete_patients = max(n_patients - incomplete, 1)

    return ClassificationEval(
        n_images=int(len(labels)),
        n_patients=n_patients,
        accuracy=float((preds == labels).mean()) if len(labels) else float("nan"),
        macro_f1=macro_f1_from_confusion(matrix),
        per_view_recall=per_class_recall(matrix),
        per_view_precision=per_class_precision(matrix),
        ece=ece,
        mean_confidence=mean_conf,
        patient_all_correct_independent=all_correct_independent / complete_patients,
        patient_set_exact_constrained=exact_constrained / complete_patients,
        n_patients_all_correct_independent=all_correct_independent,
        n_patients_exact_constrained=exact_constrained,
        accuracy_constrained=(
            float((constrained_preds == labels).mean()) if len(labels) else float("nan")
        ),
        macro_f1_constrained=macro_f1_from_confusion(matrix_constrained),
        confusion=matrix.tolist(),
        confusion_constrained=matrix_constrained.tolist(),
        n_incomplete_patients=incomplete,
    )


def bootstrap_classification(
    probs: np.ndarray,
    labels: np.ndarray,
    patient_ids: list[str],
    metrics: tuple[str, ...] = (
        "accuracy",
        "macro_f1",
        "patient_set_exact_constrained",
        "patient_all_correct_independent",
        "accuracy_constrained",
        "macro_f1_constrained",
    ),
    n_replicates: int = 1000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Patient-level bootstrap CIs for the classification metrics."""
    by_patient: dict[str, list[int]] = {}
    for idx, pid in enumerate(patient_ids):
        by_patient.setdefault(pid, []).append(idx)
    keys = sorted(by_patient)
    if not keys:
        return {}

    point = evaluate_predictions(probs, labels, patient_ids).to_dict()
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    n = len(keys)
    for _ in range(n_replicates):
        picks = rng.integers(0, n, size=n)
        idx: list[int] = []
        pids: list[str] = []
        for rep, choice in enumerate(picks):
            pid = keys[choice]
            for i in by_patient[pid]:
                idx.append(i)
                # rename so repeated patients stay separate groups
                pids.append(f"{pid}#{rep}")
        arr = np.asarray(idx)
        sample = evaluate_predictions(probs[arr], labels[arr], pids).to_dict()
        for m in metrics:
            draws[m].append(float(sample[m]))

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict[str, float]] = {}
    for m in metrics:
        values = np.asarray(draws[m], dtype=float)
        values = values[~np.isnan(values)]
        out[m] = {
            "point": float(point[m]),
            "ci_low": float(np.percentile(values, lo_q)) if values.size else float("nan"),
            "ci_high": float(np.percentile(values, hi_q)) if values.size else float("nan"),
            "n_replicates": int(values.size),
        }
    return out
