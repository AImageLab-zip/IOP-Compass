"""Manifest, adapter, rasterisation and audit behaviour."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from iop_compass.data.adapter import (
    discover_patients,
    parse_annotation,
    parse_fdi,
    png_dimensions,
)
from iop_compass.data.manifest import (
    MANIFEST_COLUMNS,
    build_rows,
    manifest_hash,
    read_manifest,
    write_manifest,
)
from iop_compass.data.rasterize import (
    instance_masks,
    load_instance_raster,
    load_instance_sidecar,
    rasterize_record,
)
from iop_compass.data.splits import make_splits
from iop_compass.data.validation import audit


def test_discovery_finds_every_patient_and_view(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    assert len(patients) == 10
    assert all(p.is_complete(cfg.views_required) for p in patients)
    assert patients[0].patient_id == "P0001"
    assert set(patients[0].images) == set(cfg.views_required)
    assert patients[0].images["frontal"].view_token == "Center"


def test_manifest_columns_and_content(synthetic_dataset, tmp_path):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    assignment = make_splits(patients, cfg)
    rows = build_rows(patients, cfg, checksums=True, splits=assignment.mapping)
    assert len(rows) == 50

    path = tmp_path / "manifest.csv"
    digest = write_manifest(rows, path)
    assert digest == manifest_hash(path)

    loaded = read_manifest(path)
    assert list(loaded[0].keys()) == MANIFEST_COLUMNS
    assert all(row["annotation_status"] == "annotated" for row in loaded)
    # paths are stored relative to the dataset root
    assert not loaded[0]["image_path"].startswith("/")
    assert len({row["checksum"] for row in loaded}) > 1


def test_relative_paths_resolve(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    rows = build_rows(patients, cfg, checksums=False)
    for row in rows[:5]:
        assert (cfg.root / row.image_path).exists()
        assert (cfg.root / row.metadata_path).exists()


def test_png_dimensions_matches_decode(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    path = patients[0].images["frontal"].image_path
    width, height = png_dimensions(path)
    decoded = cv2.imread(str(path))
    assert (height, width) == decoded.shape[:2]


def test_rasterisation_reproduces_declared_areas(synthetic_dataset, tmp_path):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    record = patients[0].images["frontal"]
    result = rasterize_record(record, tmp_path / "gt", eval_long_side=None)
    assert result.n_instances == len(record.instances)

    raster = load_instance_raster(result.instance_png)
    sidecar = load_instance_sidecar(result.sidecar_json)
    assert raster.dtype == np.uint16
    assert sidecar["width"] == record.width
    for row in sidecar["instances"]:
        # cv2.fillPoly includes the boundary pixels, so the rasterised area is the
        # polygon area plus roughly half the perimeter.  What must hold exactly is
        # that the recorded area equals the pixels actually painted.
        assert row["polygon_area"] == row["raster_area"]
        declared = row["declared_area"]
        perimeter_slack = 4 * np.sqrt(max(declared, 1.0)) + 4
        assert declared <= row["polygon_area"] <= declared + perimeter_slack

    masks, fdis = instance_masks(raster, sidecar)
    assert len(masks) == result.n_instances
    assert all(f is not None for f in fdis)
    # instances are disjoint in the raster
    total = sum(int(m.sum()) for m in masks)
    assert total == int((raster > 0).sum())


def test_rasterisation_at_reduced_resolution(synthetic_dataset, tmp_path):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    record = patients[0].images["frontal"]
    result = rasterize_record(record, tmp_path / "gt_small", eval_long_side=80)
    sidecar = load_instance_sidecar(result.sidecar_json)
    assert sidecar["width"] == 80
    assert sidecar["native_width"] == record.width
    assert sidecar["n_instances"] == len(record.instances)


def test_degenerate_instances_are_dropped(synthetic_dataset, tmp_path):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    record = patients[0].images["frontal"]
    from iop_compass.data.adapter import ToothInstance

    record.instances.append(
        ToothInstance(
            appearance_idx=99,
            fdi=48,
            pixel_area=0.0,
            bbox=(0, 0, 0, 0),
            centroid=(0.0, 0.0),
            contours=[],
            raw_fdi="48",
        )
    )
    result = rasterize_record(record, tmp_path / "gt_drop", eval_long_side=None)
    assert result.n_dropped == 1


def test_parse_fdi_rejects_nonsense():
    assert parse_fdi("11") == 11
    assert parse_fdi(48) == 48
    assert parse_fdi("") is None
    assert parse_fdi(None) is None
    assert parse_fdi("tooth") is None
    assert parse_fdi("19") is None  # position 9 does not exist
    assert parse_fdi("10") is None


def test_labelme_schema_is_accepted(tmp_path):
    payload = {
        "shapes": [
            {"label": "11", "points": [[0, 0], [10, 0], [10, 10], [0, 10]]},
            {"label": "21", "points": [[20, 0], [30, 0], [30, 10], [20, 10]]},
        ]
    }
    path = tmp_path / "labelme.json"
    path.write_text(json.dumps(payload))
    _, _, instances = parse_annotation(path)
    assert [i.fdi for i in instances] == [11, 21]
    assert instances[0].pixel_area == pytest.approx(100.0)


def test_audit_reports_a_clean_synthetic_cohort(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    report = audit(patients, cfg)
    assert report.stats["n_complete_patients"] == 10
    assert report.stats["n_images"] == 50
    assert not report.fatal, [f.code for f in report.fatal]


def test_audit_detects_an_unsupported_label(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    patients[0].images["frontal"].instances[0].fdi = 55  # primary dentition
    report = audit(patients, cfg)
    codes = {f.code for f in report.fatal}
    assert "uninterpretable_fdi" in codes


def test_audit_detects_duplicate_fdi(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    instances = patients[0].images["frontal"].instances
    instances[1].fdi = instances[0].fdi
    report = audit(patients, cfg)
    codes = {f.code for f in report.findings}
    assert "duplicate_fdi_in_image" in codes


def test_audit_detects_cross_split_duplicate_content(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    assignment = make_splits(patients, cfg)
    mapping = assignment.mapping
    a = assignment.train[0]
    b = assignment.test[0]
    checksums = {}
    for patient in patients:
        for view in cfg.views_required:
            rec = patient.images[view]
            digest = "same" if patient.patient_id in (a, b) and view == "frontal" else rec.image_id
            checksums[rec.image_id] = digest
    report = audit(patients, cfg, checksums=checksums, split_map=mapping)
    codes = {f.code for f in report.fatal}
    assert "duplicate_content_across_splits" in codes


def test_audit_flags_a_stale_case_name_view(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg)
    rec = patients[0].images["left_buccal"]
    rec.case_name = rec.case_name.replace("Left", "Right")
    report = audit(patients, cfg)
    codes = {f.code for f in report.findings}
    assert "stale_case_name_view" in codes
