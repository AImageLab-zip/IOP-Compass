"""Excluded files/patients must be named, with their reason, in one report."""

from __future__ import annotations

import dataclasses

from iop_compass.data.adapter import discover_patients
from iop_compass.data.splits import make_splits, write_splits
from iop_compass.reporting.exclusions import build_exclusion_report


def test_report_names_excluded_patient_and_file(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)

    victim = patients[0]
    image_id = victim.images["frontal"].image_id
    cfg.derived_root.mkdir(parents=True, exist_ok=True)
    (cfg.derived_root / "undecodable_images.json").write_text(
        '{"%s": "decode_failed:test"}' % image_id
    )

    # One fewer eligible patient now, so the split sizes must shrink to match.
    reduced = dataclasses.replace(
        cfg, splits={**cfg.splits, "n_train": 5, "n_val": 2, "n_test": 2}
    )
    assignment = make_splits(patients, reduced)
    write_splits(assignment, reduced.derived_root)

    report = build_exclusion_report(reduced)

    assert reduced.name in report
    assert "1 excluded" in report
    assert victim.patient_id in report
    assert "undecodable_images" in report
    assert image_id in report
    assert "decode_failed:test" in report


def test_report_summarizes_zero_exclusions(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    assignment = make_splits(patients, cfg)
    write_splits(assignment, cfg.derived_root)

    report = build_exclusion_report(cfg)

    assert "0 excluded" in report
    assert "10 patient directories found" in report
    assert "10 eligible" in report
