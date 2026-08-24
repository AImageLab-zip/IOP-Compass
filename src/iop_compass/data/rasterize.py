"""Rasterise polygon annotations into instance-ID label maps.

The released annotations store exact integer polygons: on a 200-file sample the
shoelace area of ``contour`` equals the stored ``pixel_area`` to three decimals
and every point lies inside the image, so ``cv2.fillPoly`` reproduces the
original raster losslessly.

Output per image:

* ``<image_id>_inst.png`` - uint16 single-channel PNG, 0 = background,
  otherwise the instance id (1-based, in annotation order).
* ``<image_id>_inst.json`` - instance id -> FDI code plus per-instance geometry
  and the overlap bookkeeping.

Instances are painted in *descending* polygon area so that a small tooth
overlapping a large one is never erased.  Overlapping pixels are counted and
reported rather than silently resolved.

Two conventions are worth stating because they affect the absolute metric values
(identically for every method, so comparisons are unaffected):

* ``cv2.fillPoly`` includes the boundary pixels, so the rasterised area exceeds
  the polygon's mathematical area by roughly half the perimeter.  This is the same
  convention COCO and LabelMe use.
* Polygons are scaled to the evaluation resolution and filled there rather than
  filled at native resolution and downsampled; the two differ only within a
  sub-pixel boundary, and the former is about a hundred times cheaper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .adapter import ImageRecord, ToothInstance


@dataclass
class RasterResult:
    instance_png: Path
    sidecar_json: Path
    n_instances: int
    n_dropped: int
    overlap_pixels: int
    overlapping_pairs: int
    lost_instances: list[int]


def instance_polygons(inst: ToothInstance) -> list[np.ndarray]:
    polys = []
    for contour in inst.contours:
        if len(contour) >= 3:
            polys.append(np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2))
    return polys


def rasterize_record(
    record: ImageRecord,
    out_dir: Path,
    drop_zero_area: bool = True,
    write: bool = True,
    eval_long_side: int | None = None,
) -> RasterResult:
    """Rasterise one image's annotation.

    When ``eval_long_side`` is given the polygons are scaled to that resolution
    and filled there.  Storing rasters at the evaluation resolution keeps the
    reference data at a few hundred megabytes instead of a hundred gigabytes, and
    makes reference and prediction directly comparable.
    """
    if record.width is None or record.height is None:
        raise ValueError(f"missing image dimensions for {record.image_id}")
    native_h, native_w = int(record.height), int(record.width)
    if eval_long_side:
        from .imaging import target_size

        w, h = target_size(native_w, native_h, eval_long_side)
    else:
        w, h = native_w, native_h
    scale_x = w / native_w
    scale_y = h / native_h

    kept: list[tuple[int, ToothInstance]] = []
    dropped = 0
    for idx, inst in enumerate(record.instances, start=1):
        if drop_zero_area and inst.is_degenerate:
            dropped += 1
            continue
        if not instance_polygons(inst):
            dropped += 1
            continue
        kept.append((idx, inst))

    label = np.zeros((h, w), dtype=np.uint16)
    coverage = np.zeros((h, w), dtype=np.uint8)

    # Paint large instances first so small ones stay visible.
    order = sorted(kept, key=lambda t: -t[1].pixel_area)
    per_instance: list[dict] = []
    buffer = np.zeros((h, w), dtype=np.uint8)
    for inst_id, inst in order:
        buffer[:] = 0
        # Polygons are scaled to the target resolution and filled there.  Filling
        # at native resolution and downsampling afterwards is equivalent to
        # within a sub-pixel boundary but costs ~13 s per image (a 24-megapixel
        # buffer per instance), which is 100x the cost of this path.
        polys = instance_polygons(inst)
        if scale_x != 1.0 or scale_y != 1.0:
            polys = [
                np.rint(p.astype(np.float64) * np.array([scale_x, scale_y])).astype(np.int32)
                for p in polys
            ]
        cv2.fillPoly(buffer, polys, 1)
        piece = buffer.copy()
        piece_bool = piece.astype(bool)
        label[piece_bool] = inst_id
        np.clip(coverage + piece, 0, 255, out=coverage)
        per_instance.append(
            {
                "instance_id": int(inst_id),
                "appearance_idx": int(inst.appearance_idx),
                "fdi": inst.fdi,
                "raw_fdi": inst.raw_fdi,
                "polygon_area": float(int(piece_bool.sum())),
                "declared_area": float(inst.pixel_area),
                "bbox": [
                    int(round(inst.bbox[0] * scale_x)),
                    int(round(inst.bbox[1] * scale_y)),
                    int(round(inst.bbox[2] * scale_x)),
                    int(round(inst.bbox[3] * scale_y)),
                ],
                "bbox_native": [int(v) for v in inst.bbox],
                "centroid": [
                    float(inst.centroid[0] * scale_x),
                    float(inst.centroid[1] * scale_y),
                ],
                "n_contours": len(inst.contours),
            }
        )

    overlap_pixels = int((coverage > 1).sum())
    present = set(np.unique(label).tolist()) - {0}
    lost = sorted(int(i) for i, _ in kept if i not in present)

    # A pair counts as overlapping when the later-painted instance took pixels
    # from an earlier one.
    overlapping_pairs = 0
    if overlap_pixels:
        ids = [i for i, _ in order]
        for pos, inst_id in enumerate(ids):
            if pos == 0:
                continue
            # cheap proxy: does this instance's final footprint differ from its
            # painted footprint?
            painted = next(r for r in per_instance if r["instance_id"] == inst_id)
            final_area = int((label == inst_id).sum())
            if final_area < painted["polygon_area"]:
                overlapping_pairs += 1

    for row in per_instance:
        row["raster_area"] = float(int((label == row["instance_id"]).sum()))
    per_instance.sort(key=lambda r: r["instance_id"])

    sidecar = {
        "image_id": record.image_id,
        "patient_id": record.patient_id,
        "view_label": record.view_label,
        "width": w,
        "height": h,
        "native_width": native_w,
        "native_height": native_h,
        "n_instances": len(per_instance),
        "n_dropped_degenerate": dropped,
        "overlap_pixels": overlap_pixels,
        "overlapping_pairs": overlapping_pairs,
        "fully_occluded_instances": lost,
        "instances": per_instance,
    }

    png_path = out_dir / f"{record.image_id}_inst.png"
    json_path = out_dir / f"{record.image_id}_inst.json"
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(png_path), label):
            raise IOError(f"failed to write {png_path}")
        with json_path.open("w") as fh:
            json.dump(sidecar, fh, indent=1)

    return RasterResult(
        instance_png=png_path,
        sidecar_json=json_path,
        n_instances=len(per_instance),
        n_dropped=dropped,
        overlap_pixels=overlap_pixels,
        overlapping_pairs=overlapping_pairs,
        lost_instances=lost,
    )


def load_instance_raster(png_path: Path) -> np.ndarray:
    arr = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"cannot read {png_path}")
    if arr.ndim != 2:
        raise ValueError(f"expected single-channel instance map, got {arr.shape}")
    return arr.astype(np.uint16)


def load_instance_sidecar(json_path: Path) -> dict:
    with open(json_path) as fh:
        return json.load(fh)


def instance_masks(label: np.ndarray, sidecar: dict) -> tuple[list[np.ndarray], list[int | None]]:
    """Split an instance-ID map into boolean masks plus their FDI codes."""
    masks: list[np.ndarray] = []
    fdis: list[int | None] = []
    for row in sidecar["instances"]:
        mask = label == int(row["instance_id"])
        if not mask.any():
            continue
        masks.append(mask)
        fdis.append(row["fdi"])
    return masks, fdis
