"""Display names, canonical orderings and number formatters for the reporting code.

One definition of how a view, a variant or a metric is named and rendered, shared by
:mod:`iop_compass.reporting.figures` and :mod:`iop_compass.reporting.aggregate` so a
label never differs between a figure and the aggregate.
"""

from __future__ import annotations

import math

ROI_ORDER = ["R0_full", "R1_geometric", "R2_learned", "R3_sam3"]
ROI_SHORT = {
    "R0_full": "R0 full image",
    "R1_geometric": "R1 geometric prior",
    "R2_learned": "R2 learned box",
    "R3_sam3": "R3 SAM 3 concept",
}
VIEW_SHORT = {
    "frontal": "Frontal",
    "left_buccal": "Left buccal",
    "right_buccal": "Right buccal",
    "upper_occlusal": "Maxillary occl.",
    "lower_occlusal": "Mandibular occl.",
}
CLASSIFIER_ORDER = ["C0", "C1", "C2"]
CLASSIFIER_SHORT = {
    "C0": "pretrained, no augmentation",
    "C1": "pretrained, augmented",
    "C2": "from scratch, augmented",
}


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "n/a"
    return f"{100 * number:.{digits}f}"


def integer(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{int(round(float(value)))}"
    except (TypeError, ValueError):
        return str(value)
