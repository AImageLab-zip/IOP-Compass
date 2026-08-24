"""Patient-set constrained view assignment.

A standardised acquisition contains exactly one image per clinical view, so the
five per-image posteriors of a patient can be resolved jointly instead of
independently.  Maximising the total log-probability under a one-to-one
image/view constraint is a linear assignment problem, solved here with the
Hungarian algorithm.

This guarantees exactly one frontal, one left buccal, one right buccal, one
maxillary occlusal and one mandibular occlusal label per patient.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = 1e-12


def constrained_assign(probs: np.ndarray) -> np.ndarray:
    """Assign views to images under a one-to-one constraint.

    Parameters
    ----------
    probs
        ``(n_images, n_views)`` class-probability matrix for one patient.

    Returns
    -------
    ``(n_images,)`` array of assigned view indices.  When the matrix is not
    square the assignment covers ``min(n_images, n_views)`` rows and the
    remaining images fall back to their independent argmax.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"expected a 2-D probability matrix, got {probs.shape}")
    n_images, n_views = probs.shape
    assigned = probs.argmax(axis=1)
    if n_images == 0 or n_views == 0:
        return assigned

    cost = -np.log(np.clip(probs, EPS, 1.0))
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        assigned[r] = c
    return assigned


def patient_set_exact(
    assigned: np.ndarray, true_labels: np.ndarray
) -> bool:
    """True when every image of the patient received its correct view label."""
    return bool(np.array_equal(np.asarray(assigned), np.asarray(true_labels)))


def assign_all(
    probs_by_patient: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {pid: constrained_assign(p) for pid, p in probs_by_patient.items()}
