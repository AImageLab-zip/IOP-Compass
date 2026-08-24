"""Canonical dataset manifest: one row per image.

The manifest is the single source of truth for every downstream stage.  It is
plain CSV so it can be inspected, diffed and shipped with the paper.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .adapter import DatasetConfig, PatientRecord, sha256_file

MANIFEST_COLUMNS = [
    "patient_id",
    "patient_key",
    "image_id",
    "image_path",
    "view_label",
    "view_token",
    "mask_path",
    "metadata_path",
    "width",
    "height",
    "number_of_instances",
    "available_FDI_labels",
    "annotation_status",
    "reviewer_id",
    "split",
    "checksum",
]


@dataclass
class ManifestRow:
    patient_id: str
    patient_key: str
    image_id: str
    image_path: str
    view_label: str
    view_token: str
    mask_path: str
    metadata_path: str
    width: str
    height: str
    number_of_instances: str
    available_FDI_labels: str
    annotation_status: str
    reviewer_id: str
    split: str
    checksum: str


def _status(patient: PatientRecord, view: str, cfg: DatasetConfig) -> str:
    rec = patient.images.get(view)
    if rec is None:
        return "missing_view"
    if rec.image_path is None and rec.annotation_path is None:
        return "missing_view"
    if rec.image_path is None:
        return "missing_image"
    if rec.annotation_path is None:
        return "missing_annotation"
    if rec.width is None or rec.height is None:
        return "unreadable_image"
    if not rec.instances:
        return "empty_annotation"
    if not patient.is_complete(cfg.views_required):
        return "incomplete_patient"
    return "annotated"


def build_rows(
    patients: list[PatientRecord],
    cfg: DatasetConfig,
    checksums: bool = True,
    splits: dict[str, str] | None = None,
) -> list[ManifestRow]:
    splits = splits or {}
    rows: list[ManifestRow] = []
    for patient in patients:
        for view in cfg.views_required:
            rec = patient.images.get(view)
            if rec is None:
                continue
            fdis = sorted({i.fdi for i in rec.instances if i.fdi is not None})
            checksum = ""
            if checksums and rec.image_path is not None:
                checksum = sha256_file(rec.image_path)
            rows.append(
                ManifestRow(
                    patient_id=patient.patient_id,
                    patient_key=patient.patient_key,
                    image_id=rec.image_id,
                    image_path=cfg.rel(rec.image_path) if rec.image_path else "",
                    view_label=rec.view_label,
                    view_token=rec.view_token,
                    mask_path=f"{rec.image_id}_inst.png",
                    metadata_path=(
                        cfg.rel(rec.annotation_path) if rec.annotation_path else ""
                    ),
                    width="" if rec.width is None else str(rec.width),
                    height="" if rec.height is None else str(rec.height),
                    number_of_instances=str(len(rec.instances)),
                    available_FDI_labels=";".join(str(f) for f in fdis),
                    annotation_status=_status(patient, view, cfg),
                    reviewer_id="",
                    split=splits.get(patient.patient_id, "unassigned"),
                    checksum=checksum,
                )
            )
    return rows


def write_manifest(rows: list[ManifestRow], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return manifest_hash(path)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def manifest_hash(path: Path) -> str:
    """Content hash of the manifest file, used to bind results to the data."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows_by_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [r for r in rows if r["split"] == split]


def patients_in_split(rows: list[dict[str, str]], split: str) -> list[str]:
    return sorted({r["patient_id"] for r in rows if r["split"] == split})


assert MANIFEST_COLUMNS == [f.name for f in fields(ManifestRow)]
