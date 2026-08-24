"""Aggregate the scattered per-image/per-patient QC artifacts into one report.

``eligibility()`` (see ``iop_compass.data.splits``) already excludes patients for
reasons drawn from three separate JSON files
(``undecodable_images.json``/``unusable_annotations.json``/``unusable_images.json``)
plus a few checks of its own, and ``splits.json`` records the resulting
``{patient_id: reason}`` map.  This module turns that plus the three source
files into a single human-readable Markdown report naming every excluded
patient and, where available, the specific file and check responsible.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data.adapter import DatasetConfig, PatientRecord, discover_patients
from ..data.splits import (
    load_undecodable,
    load_unusable_annotations,
    load_unusable_images,
)

_SOURCE_LABELS = {
    "undecodable": "undecodable_images.json (full image decode failed)",
    "unusable_annotation": "unusable_annotations.json (annotation has no polygon geometry)",
    "unusable_image": "unusable_images.json (audit-flagged: overshoot, view/quadrant conflict, or duplicate content)",
}


def _image_owner_map(patients: list[PatientRecord]) -> dict[str, tuple[str, str]]:
    """``image_id -> (patient_id, view_label)`` for every discovered image."""
    owners: dict[str, tuple[str, str]] = {}
    for patient in patients:
        for view, rec in patient.images.items():
            owners[rec.image_id] = (patient.patient_id, view)
    return owners


def build_exclusion_report(cfg: DatasetConfig) -> str:
    patients = discover_patients(cfg, load_annotations=False, read_dimensions=False)
    owners = _image_owner_map(patients)

    splits_path = cfg.derived_root / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} not found; run scripts/create_splits.py first"
        )
    assignment = json.loads(splits_path.read_text())
    excluded: dict[str, str] = assignment["excluded"]

    file_sources = {
        "undecodable": load_undecodable(cfg),
        "unusable_annotation": load_unusable_annotations(cfg),
        "unusable_image": load_unusable_images(cfg),
    }

    by_category: dict[str, int] = {}
    for reason in excluded.values():
        category = reason.split(":")[0]
        by_category[category] = by_category.get(category, 0) + 1

    lines: list[str] = []
    lines.append(f"# Excluded files report — {cfg.name}")
    lines.append("")
    lines.append(
        f"{len(patients)} patient directories found, "
        f"{assignment['n_eligible']} eligible, "
        f"{len(excluded)} excluded."
    )
    lines.append("")
    lines.append("## Summary by reason")
    lines.append("")
    lines.append("| Reason | Patients |")
    lines.append("| --- | --- |")
    for category, count in sorted(by_category.items()):
        lines.append(f"| {category} | {count} |")
    lines.append("")

    lines.append("## Excluded patients")
    lines.append("")
    lines.append("| Patient | Reason |")
    lines.append("| --- | --- |")
    for pid, reason in sorted(excluded.items()):
        lines.append(f"| {pid} | {reason} |")
    lines.append("")

    lines.append("## Excluded files, by source check")
    lines.append("")
    for source, label in _SOURCE_LABELS.items():
        entries = file_sources[source]
        lines.append(f"### {label}")
        lines.append("")
        if not entries:
            lines.append("None.")
            lines.append("")
            continue
        lines.append("| Image | Patient | View | Reason |")
        lines.append("| --- | --- | --- | --- |")
        for image_id, reason in sorted(entries.items()):
            pid, view = owners.get(image_id, ("?", "?"))
            lines.append(f"| {image_id} | {pid} | {view} | {reason} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_exclusion_report(cfg: DatasetConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "excluded_files_report.md"
    out_path.write_text(build_exclusion_report(cfg))
    return out_path
