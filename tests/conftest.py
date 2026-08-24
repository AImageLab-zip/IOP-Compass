"""Shared fixtures: a tiny synthetic dataset in the real on-disk layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# The vendored SAM 3 package lives here, exactly as scripts/run_segmentation.py adds it.
sys.path.insert(0, str(REPO / "third_party"))

VIEW_TOKENS = ["Center", "Left", "Right", "Upper", "Down"]
VIEW_FDI = {
    "Center": [11, 12, 21, 22, 31, 32, 41, 42],
    "Left": [22, 23, 24, 32, 33, 34],
    "Right": [12, 13, 14, 42, 43, 44],
    "Upper": [11, 12, 13, 21, 22, 23],
    "Down": [31, 32, 33, 41, 42, 43],
}


def _square(cx: int, cy: int, half: int) -> list[list[int]]:
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


def make_patient(root: Path, index: int, width: int = 160, height: int = 120) -> None:
    directory = root / f"Patient_{index}"
    directory.mkdir(parents=True, exist_ok=True)
    for token in VIEW_TOKENS:
        image = np.full((height, width, 3), 40, dtype=np.uint8)
        teeth = []
        for slot, fdi in enumerate(VIEW_FDI[token]):
            cx = 20 + 18 * slot
            cy = 40 + 10 * (slot % 3)
            half = 6
            contour = _square(cx, cy, half)
            xs = [p[0] for p in contour]
            ys = [p[1] for p in contour]
            cv2.rectangle(image, (cx - half, cy - half), (cx + half, cy + half), (200, 200, 200), -1)
            teeth.append(
                {
                    "appearance_idx": slot + 1,
                    "FDI_NUM": str(fdi),
                    "class_name": "tooth",
                    "color": "#777777",
                    "pixel_area": float((2 * half) * (2 * half)),
                    "x_min": min(xs),
                    "y_min": min(ys),
                    "x_max": max(xs),
                    "y_max": max(ys),
                    "centroid_x": float(cx),
                    "centroid_y": float(cy),
                    "arch": None,
                    "contour_count": 1,
                    "contour": contour,
                    "contours": [contour],
                }
            )
        cv2.imwrite(str(directory / f"IOP_{token}_RawImage_{index}.png"), image)
        (directory / f"IOP_{token}_{index}.json").write_text(
            json.dumps(
                {
                    "patient_id": f"internal_{index}",
                    "case_name": f"{token}_intraoral_1_patient_{index}",
                    "annotation_count": len(teeth),
                    "teeth": teeth,
                }
            )
        )


@pytest.fixture
def synthetic_dataset(tmp_path: Path):
    """Return ``(config_path, DatasetConfig)`` for a 10-patient synthetic cohort."""
    root = tmp_path / "IOP_Dataset"
    root.mkdir()
    for index in range(1, 11):
        make_patient(root, index)

    config = {
        "dataset": {
            "name": "synthetic",
            "root": str(root),
            "derived_root": str(tmp_path / "derived"),
            "patient_dir_regex": r"^Patient_(?P<pid>\d+)$",
            "image_pattern": "IOP_{view_token}_RawImage_{pid}.png",
            "annotation_pattern": "IOP_{view_token}_{pid}.json",
            "view_map": {
                "Center": "frontal",
                "Left": "left_buccal",
                "Right": "right_buccal",
                "Upper": "upper_occlusal",
                "Down": "lower_occlusal",
            },
            "views_required": [
                "frontal",
                "left_buccal",
                "right_buccal",
                "upper_occlusal",
                "lower_occlusal",
            ],
            "fdi_labels": [
                11, 12, 13, 14, 15, 16, 17, 18,
                21, 22, 23, 24, 25, 26, 27, 28,
                31, 32, 33, 34, 35, 36, 37, 38,
                41, 42, 43, 44, 45, 46, 47, 48,
            ],
        },
        "resolution": {
            "inference_long_side": 160,
            "eval_long_side": 160,
            "classifier_image_size": 32,
        },
        "audit": {
            "drop_zero_area_instances": True,
            "min_instance_area_warn": 20,
            "missing_file_tolerance": 0.0,
            "view_quadrant_check": True,
            "view_quadrant_min_agreement": 0.6,
        },
        "splits": {"seed": 42, "n_train": 6, "n_val": 2, "n_test": 2, "require_exact": True},
    }
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(yaml.safe_dump(config))

    from iop_compass.data.adapter import load_dataset_config

    return config_path, load_dataset_config(config_path)
