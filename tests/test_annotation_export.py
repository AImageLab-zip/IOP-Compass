"""Round-trip of the annotation record the correction interface writes.

The interface is external to this repository, so what must be pinned is the
*contract*: the JSON schema it writes, and the guarantee that a dataset rebuilt from
that JSON reproduces the same instances, FDI codes and geometry. If the contract
drifts, these tests fail rather than the dataset silently changing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from iop_compass.data.adapter import discover_patients, parse_annotation
from iop_compass.data.rasterize import (
    instance_masks,
    load_instance_raster,
    load_instance_sidecar,
    rasterize_record,
)
from iop_compass.segmentation.infer import Prediction, load_prediction, save_prediction

# The keys the interface writes per tooth, as consumed by the adapter.
REQUIRED_TOOTH_KEYS = {
    "appearance_idx",
    "FDI_NUM",
    "class_name",
    "pixel_area",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "centroid_x",
    "centroid_y",
    "contour",
    "contours",
}


def export_record(record) -> dict:
    """Re-serialise an ImageRecord into the interface's JSON schema."""
    teeth = []
    for inst in record.instances:
        teeth.append(
            {
                "appearance_idx": inst.appearance_idx,
                "FDI_NUM": str(inst.fdi) if inst.fdi is not None else "",
                "class_name": "tooth",
                "color": "#777777",
                "pixel_area": inst.pixel_area,
                "x_min": inst.bbox[0],
                "y_min": inst.bbox[1],
                "x_max": inst.bbox[2],
                "y_max": inst.bbox[3],
                "centroid_x": inst.centroid[0],
                "centroid_y": inst.centroid[1],
                "arch": inst.arch,
                "contour_count": len(inst.contours),
                "contour": [list(p) for p in (inst.contours[0] if inst.contours else [])],
                "contours": [[list(p) for p in c] for c in inst.contours],
            }
        )
    return {
        "patient_id": record.source_patient_id,
        "case_name": record.case_name,
        "annotation_count": len(teeth),
        "teeth": teeth,
    }


def test_released_annotations_carry_the_expected_keys(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    path = patients[0].images["frontal"].annotation_path
    payload = json.loads(path.read_text())
    assert {"patient_id", "case_name", "annotation_count", "teeth"} <= set(payload)
    assert payload["annotation_count"] == len(payload["teeth"])
    for tooth in payload["teeth"]:
        assert REQUIRED_TOOTH_KEYS <= set(tooth), REQUIRED_TOOTH_KEYS - set(tooth)


def test_export_import_round_trip_preserves_instances(synthetic_dataset, tmp_path):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    original = patients[0].images["frontal"]

    exported = tmp_path / "roundtrip.json"
    exported.write_text(json.dumps(export_record(original)))
    source_id, case_name, reimported = parse_annotation(exported)

    assert source_id == original.source_patient_id
    assert case_name == original.case_name
    assert len(reimported) == len(original.instances)
    for before, after in zip(original.instances, reimported):
        assert after.fdi == before.fdi
        assert after.appearance_idx == before.appearance_idx
        assert after.bbox == before.bbox
        assert after.contours == before.contours
        assert after.pixel_area == pytest.approx(before.pixel_area)


def test_raster_and_json_agree(synthetic_dataset, tmp_path):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    record = patients[0].images["frontal"]
    result = rasterize_record(record, tmp_path / "gt", eval_long_side=None)

    raster = load_instance_raster(result.instance_png)
    sidecar = load_instance_sidecar(result.sidecar_json)
    masks, fdis = instance_masks(raster, sidecar)

    # every instance in the JSON exists in the raster with the recorded area
    assert len(masks) == sidecar["n_instances"]
    for mask, row in zip(masks, sidecar["instances"]):
        assert int(mask.sum()) == int(row["raster_area"])
    # and the FDI codes agree with the source annotation
    assert fdis == [i.fdi for i in record.instances]
    # no instance id in the raster is missing from the JSON
    ids_in_raster = set(np.unique(raster).tolist()) - {0}
    ids_in_json = {int(r["instance_id"]) for r in sidecar["instances"]}
    assert ids_in_raster <= ids_in_json


def test_prediction_record_round_trip(tmp_path):
    masks = [np.zeros((32, 48), dtype=bool) for _ in range(3)]
    masks[0][2:10, 2:10] = True
    masks[1][2:10, 20:30] = True
    masks[2][15:25, 5:20] = True
    prediction = Prediction(
        image_id="P0001_frontal",
        patient_id="P0001",
        view_label="frontal",
        variant="no-roi+sat+post",
        width=48,
        height=32,
        instances=[
            {"instance_id": i + 1, "fdi": fdi, "score": 0.9 - 0.1 * i, "bbox": [0, 0, 1, 1], "area": int(m.sum())}
            for i, (fdi, m) in enumerate(zip([11, 12, 13], masks))
        ],
        roi={"strategy": "R3_sam3", "box": [0, 0, 48, 32], "box_eval": [0, 0, 48, 32], "fallback": False},
        runtime={"total_seconds": 1.5},
        meta={"eval_long_side": 48},
    )
    save_prediction(prediction, masks, tmp_path)
    payload, raster = load_prediction(tmp_path / "P0001_frontal_pred.json")

    assert payload["image_id"] == "P0001_frontal"
    assert payload["variant"] == "no-roi+sat+post"
    assert raster.shape == (32, 48)
    assert len(payload["instances"]) == 3
    # areas recorded in the JSON match the raster exactly, and instances are disjoint
    total = 0
    for row in payload["instances"]:
        area = int((raster == int(row["instance_id"])).sum())
        assert area == int(row["raster_area"])
        total += area
    assert total == int((raster > 0).sum())
    # FDI codes and scores survive
    assert [r["fdi"] for r in payload["instances"]] == [11, 12, 13]
    assert payload["instances"][0]["score"] == pytest.approx(0.9)


def test_overlapping_predictions_are_resolved_not_double_counted(tmp_path):
    big = np.zeros((32, 48), dtype=bool)
    big[0:20, 0:30] = True
    small = np.zeros((32, 48), dtype=bool)
    small[5:12, 5:12] = True
    prediction = Prediction(
        image_id="x",
        patient_id="p",
        view_label="frontal",
        variant="no-roi+sat+no-post",
        width=48,
        height=32,
        instances=[
            {"instance_id": 1, "fdi": 11, "score": 1.0, "bbox": [0, 0, 1, 1], "area": int(big.sum())},
            {"instance_id": 2, "fdi": 12, "score": 1.0, "bbox": [0, 0, 1, 1], "area": int(small.sum())},
        ],
    )
    save_prediction(prediction, [big, small], tmp_path)
    payload, raster = load_prediction(tmp_path / "x_pred.json")
    # the smaller instance is painted last, so it survives inside the larger one
    assert int((raster == 2).sum()) == int(small.sum())
    assert int((raster == 1).sum()) == int(big.sum()) - int(small.sum())
    assert payload["instances"][1]["raster_area"] == int(small.sum())
