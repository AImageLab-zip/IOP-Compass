"""Dataset discovery and annotation parsing.

The source tree is never renamed or modified; everything is driven by the
patterns in the dataset config so a differently laid out export can be adopted
by editing YAML only.

Expected layout (configurable)::

    <root>/Patient_<pid>/IOP_<ViewToken>_RawImage_<pid>.png
    <root>/Patient_<pid>/IOP_<ViewToken>_<pid>.json
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetConfig:
    """Typed view over the ``dataset`` / ``audit`` / ``splits`` config blocks."""

    name: str
    root: Path
    derived_root: Path
    patient_dir_regex: str
    image_pattern: str
    annotation_pattern: str
    view_map: dict[str, str]
    views_required: tuple[str, ...]
    fdi_labels: tuple[int, ...]
    resolution: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    splits: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    @property
    def inference_long_side(self) -> int:
        return int(self.resolution.get("inference_long_side", 2048))

    @property
    def eval_long_side(self) -> int:
        return int(self.resolution.get("eval_long_side", 1024))

    @property
    def classifier_image_size(self) -> int:
        return int(self.resolution.get("classifier_image_size", 256))

    @property
    def view_tokens(self) -> tuple[str, ...]:
        return tuple(self.view_map.keys())

    @property
    def token_for_view(self) -> dict[str, str]:
        return {v: k for k, v in self.view_map.items()}

    @property
    def fdi_index(self) -> dict[int, int]:
        """FDI code -> contiguous 1-based class index (0 is background)."""
        return {fdi: i + 1 for i, fdi in enumerate(self.fdi_labels)}

    def rel(self, path: Path) -> str:
        """Path relative to the dataset root, for manifest storage."""
        try:
            return str(Path(path).resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    path = Path(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    ds = raw["dataset"]
    return DatasetConfig(
        name=ds["name"],
        root=Path(ds["root"]),
        derived_root=Path(ds["derived_root"]),
        patient_dir_regex=ds["patient_dir_regex"],
        image_pattern=ds["image_pattern"],
        annotation_pattern=ds["annotation_pattern"],
        view_map=dict(ds["view_map"]),
        views_required=tuple(ds["views_required"]),
        fdi_labels=tuple(int(x) for x in ds["fdi_labels"]),
        resolution=dict(raw.get("resolution", {})),
        audit=dict(raw.get("audit", {})),
        splits=dict(raw.get("splits", {})),
        config_path=path,
    )


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #


@dataclass
class ToothInstance:
    """One annotated tooth instance."""

    appearance_idx: int
    fdi: int | None
    pixel_area: float
    bbox: tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    centroid: tuple[float, float]
    contours: list[list[tuple[int, int]]]
    raw_fdi: str | None = None

    @property
    def arch(self) -> str:
        """Arch derived from the FDI code.

        The source field ``arch`` is ``null`` for a non-trivial number of
        instances, so it is never trusted.
        """
        if self.fdi is None:
            return "unknown"
        return "upper" if self.fdi // 10 in (1, 2) else "lower"

    @property
    def quadrant(self) -> int | None:
        return None if self.fdi is None else self.fdi // 10

    @property
    def is_degenerate(self) -> bool:
        return self.pixel_area <= 0 or not any(len(c) >= 3 for c in self.contours)


@dataclass
class ImageRecord:
    """One image plus its annotation file."""

    patient_key: str  # directory-derived, e.g. "Patient_1"
    patient_id: str  # canonical anonymous id, e.g. "P0001"
    source_patient_id: str | None  # internal id inside the JSON, never released
    view_token: str  # e.g. "Center"
    view_label: str  # e.g. "frontal"
    image_path: Path | None
    annotation_path: Path | None
    case_name: str | None = None
    width: int | None = None
    height: int | None = None
    instances: list[ToothInstance] = field(default_factory=list)

    @property
    def image_id(self) -> str:
        return f"{self.patient_id}_{self.view_label}"

    @property
    def is_complete(self) -> bool:
        return self.image_path is not None and self.annotation_path is not None


@dataclass
class PatientRecord:
    patient_key: str
    patient_id: str
    images: dict[str, ImageRecord]  # view_label -> record

    def is_complete(self, views_required: tuple[str, ...]) -> bool:
        return all(
            v in self.images and self.images[v].is_complete for v in views_required
        )

    @property
    def source_patient_id(self) -> str | None:
        for rec in self.images.values():
            if rec.source_patient_id:
                return rec.source_patient_id
        return None


# --------------------------------------------------------------------------- #
# low level helpers
# --------------------------------------------------------------------------- #


def png_dimensions(path: Path) -> tuple[int, int]:
    """Read (width, height) from the PNG IHDR without decoding the image."""
    with open(path, "rb") as fh:
        header = fh.read(33)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _as_pairs(points: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not points:
        return out
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((int(round(float(p[0]))), int(round(float(p[1])))))
    return out


def parse_fdi(value: Any) -> int | None:
    """Parse an FDI code, returning ``None`` when it is not interpretable."""
    if value is None:
        return None
    try:
        fdi = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if fdi // 10 in (1, 2, 3, 4, 5, 6, 7, 8) and 1 <= fdi % 10 <= 8:
        return fdi
    return None


def parse_annotation(path: Path) -> tuple[str | None, str | None, list[ToothInstance]]:
    """Parse one annotation JSON.

    Returns ``(source_patient_id, case_name, instances)``.  Two schemas are
    accepted: the project's ``teeth`` schema and a LabelMe-style ``shapes``
    list, so externally produced files can be imported without conversion.
    """
    with open(path) as fh:
        payload = json.load(fh)

    source_pid = payload.get("patient_id")
    case_name = payload.get("case_name")
    instances: list[ToothInstance] = []

    if isinstance(payload.get("teeth"), list):
        rows = payload["teeth"]
        for i, row in enumerate(rows, start=1):
            contours = [_as_pairs(c) for c in (row.get("contours") or [])]
            if not contours:
                single = _as_pairs(row.get("contour"))
                contours = [single] if single else []
            raw_fdi = row.get("FDI_NUM", row.get("fdi", row.get("label")))
            bbox = (
                int(row.get("x_min", 0)),
                int(row.get("y_min", 0)),
                int(row.get("x_max", 0)),
                int(row.get("y_max", 0)),
            )
            instances.append(
                ToothInstance(
                    appearance_idx=int(row.get("appearance_idx", i)),
                    fdi=parse_fdi(raw_fdi),
                    pixel_area=float(row.get("pixel_area", 0.0) or 0.0),
                    bbox=bbox,
                    centroid=(
                        float(row.get("centroid_x", 0.0) or 0.0),
                        float(row.get("centroid_y", 0.0) or 0.0),
                    ),
                    contours=contours,
                    raw_fdi=None if raw_fdi is None else str(raw_fdi),
                )
            )
    elif isinstance(payload.get("shapes"), list):
        for i, shape in enumerate(payload["shapes"], start=1):
            pts = _as_pairs(shape.get("points"))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            raw_fdi = shape.get("label")
            instances.append(
                ToothInstance(
                    appearance_idx=i,
                    fdi=parse_fdi(raw_fdi),
                    pixel_area=float(_polygon_area(pts)),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
                    contours=[pts],
                    raw_fdi=None if raw_fdi is None else str(raw_fdi),
                )
            )
    else:
        raise ValueError(f"unrecognised annotation schema: {path}")

    return source_pid, case_name, instances


def _polygon_area(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


def _fill(pattern: str, view_token: str, pid: str) -> str:
    return pattern.format(view_token=view_token, pid=pid)


def discover_patients(
    cfg: DatasetConfig, load_annotations: bool = True, read_dimensions: bool = True
) -> list[PatientRecord]:
    """Walk the dataset root and build one :class:`PatientRecord` per directory.

    Patient directories are sorted numerically when the regex exposes a numeric
    ``pid`` group, so the canonical anonymous ids are stable across runs.
    """
    rx = re.compile(cfg.patient_dir_regex)
    found: list[tuple[Any, str, Path]] = []
    for child in sorted(cfg.root.iterdir()):
        if not child.is_dir():
            continue
        m = rx.match(child.name)
        if not m:
            continue
        pid = m.groupdict().get("pid", child.name)
        sort_key: Any = int(pid) if str(pid).isdigit() else str(pid)
        found.append((sort_key, str(pid), child))

    found.sort(key=lambda t: (isinstance(t[0], str), t[0]))

    patients: list[PatientRecord] = []
    width = max(4, len(str(len(found))))
    for ordinal, (_, pid, directory) in enumerate(found, start=1):
        patient_id = f"P{ordinal:0{width}d}"
        images: dict[str, ImageRecord] = {}
        for token, view_label in cfg.view_map.items():
            img = directory / _fill(cfg.image_pattern, token, pid)
            ann = directory / _fill(cfg.annotation_pattern, token, pid)
            rec = ImageRecord(
                patient_key=directory.name,
                patient_id=patient_id,
                source_patient_id=None,
                view_token=token,
                view_label=view_label,
                image_path=img if img.exists() else None,
                annotation_path=ann if ann.exists() else None,
            )
            if rec.image_path is not None and read_dimensions:
                try:
                    rec.width, rec.height = png_dimensions(rec.image_path)
                except Exception:  # unreadable / truncated file
                    rec.width = rec.height = None
            if rec.annotation_path is not None and load_annotations:
                try:
                    src, case, inst = parse_annotation(rec.annotation_path)
                    rec.source_patient_id, rec.case_name, rec.instances = src, case, inst
                except Exception:
                    rec.case_name = None
                    rec.instances = []
            images[view_label] = rec
        patients.append(
            PatientRecord(patient_key=directory.name, patient_id=patient_id, images=images)
        )
    return patients


def iter_images(patients: list[PatientRecord]) -> Iterator[ImageRecord]:
    for patient in patients:
        for view in sorted(patient.images):
            yield patient.images[view]
