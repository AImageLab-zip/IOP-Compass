"""Patient-set constrained view assignment."""

from __future__ import annotations

import numpy as np
import pytest

from iop_compass.classification.constrained_assignment import (
    constrained_assign,
    patient_set_exact,
)
from iop_compass.classification.dataset import (
    FLIP_INDEX,
    FLIP_REMAP,
    VIEW_CLASSES,
    label_to_index,
)
from iop_compass.classification.evaluate import (
    evaluate_predictions,
    expected_calibration_error,
)


def one_hot(indices: list[int], n: int = 5, peak: float = 0.9) -> np.ndarray:
    probs = np.full((len(indices), n), (1 - peak) / (n - 1))
    for row, index in enumerate(indices):
        probs[row, index] = peak
    return probs


def test_confident_correct_predictions_are_unchanged():
    probs = one_hot([0, 1, 2, 3, 4])
    assert list(constrained_assign(probs)) == [0, 1, 2, 3, 4]


def test_duplicate_argmax_is_resolved_one_to_one():
    # two images both prefer class 0; nothing prefers class 1
    probs = np.array(
        [
            [0.90, 0.05, 0.02, 0.02, 0.01],
            [0.50, 0.45, 0.02, 0.02, 0.01],
            [0.02, 0.02, 0.90, 0.03, 0.03],
            [0.02, 0.02, 0.03, 0.90, 0.03],
            [0.02, 0.02, 0.03, 0.03, 0.90],
        ]
    )
    assert list(probs.argmax(axis=1)) == [0, 0, 2, 3, 4]
    assigned = list(constrained_assign(probs))
    assert sorted(assigned) == [0, 1, 2, 3, 4]
    # the less confident of the two class-0 images gives way
    assert assigned[0] == 0 and assigned[1] == 1


def test_assignment_maximises_total_log_probability():
    rng = np.random.default_rng(0)
    for _ in range(20):
        probs = rng.random((5, 5))
        probs /= probs.sum(axis=1, keepdims=True)
        assigned = constrained_assign(probs)
        score = np.log(probs[np.arange(5), assigned]).sum()
        import itertools

        best = max(
            np.log(probs[np.arange(5), list(perm)]).sum()
            for perm in itertools.permutations(range(5))
        )
        assert score == pytest.approx(best)


def test_zero_probability_does_not_break_the_solver():
    probs = np.zeros((5, 5))
    probs[np.arange(5), np.arange(5)] = 1.0
    assert sorted(constrained_assign(probs)) == [0, 1, 2, 3, 4]


def test_patient_set_exact_helper():
    labels = np.array([0, 1, 2, 3, 4])
    assert patient_set_exact(labels, labels)
    assert not patient_set_exact(np.array([1, 0, 2, 3, 4]), labels)


def test_constrained_assignment_can_only_help_a_full_patient_set():
    # three patients: one perfect, one with a duplicate argmax, one badly wrong
    probs = np.vstack(
        [
            one_hot([0, 1, 2, 3, 4]),
            np.array(
                [
                    [0.90, 0.05, 0.02, 0.02, 0.01],
                    [0.50, 0.45, 0.02, 0.02, 0.01],
                    [0.02, 0.02, 0.90, 0.03, 0.03],
                    [0.02, 0.02, 0.03, 0.90, 0.03],
                    [0.02, 0.02, 0.03, 0.03, 0.90],
                ]
            ),
            one_hot([4, 3, 2, 1, 0]),
        ]
    )
    labels = np.array([0, 1, 2, 3, 4] * 3)
    patients = ["p1"] * 5 + ["p2"] * 5 + ["p3"] * 5
    metrics = evaluate_predictions(probs, labels, patients)
    assert metrics.n_patients_all_correct_independent == 1
    assert metrics.n_patients_exact_constrained == 2
    assert metrics.accuracy_constrained >= metrics.accuracy


def test_incomplete_patients_are_reported_not_silently_scored():
    probs = one_hot([0, 1, 2])
    labels = np.array([0, 1, 2])
    metrics = evaluate_predictions(probs, labels, ["p1"] * 3)
    assert metrics.n_incomplete_patients == 1
    assert metrics.n_patients_exact_constrained == 0


def test_horizontal_flip_remap_is_an_involution():
    for view in VIEW_CLASSES:
        assert FLIP_REMAP[FLIP_REMAP[view]] == view
    assert FLIP_REMAP["left_buccal"] == "right_buccal"
    assert FLIP_REMAP["frontal"] == "frontal"
    for index, view in enumerate(VIEW_CLASSES):
        assert FLIP_INDEX[index] == label_to_index(FLIP_REMAP[view])


def test_ece_is_zero_for_a_perfectly_calibrated_case():
    probs = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]] * 4)
    labels = np.zeros(4, dtype=int)
    ece, confidence = expected_calibration_error(probs, labels, n_bins=5)
    assert ece == pytest.approx(0.0)
    assert confidence == pytest.approx(1.0)


def test_ece_is_one_for_confidently_wrong_predictions():
    probs = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]] * 4)
    labels = np.ones(4, dtype=int)
    ece, _ = expected_calibration_error(probs, labels, n_bins=5)
    assert ece == pytest.approx(1.0)
