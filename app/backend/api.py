"""HTTP surface of the viewer.

Endpoints
    ``GET  /api/health``                       readiness, device, loaded models
    ``GET  /api/config``                       views, segmenters, FDI vocabulary
    ``POST /api/cases``                        upload N photographs, get a case
    ``GET  /api/cases/<case>/images/<image>``  the original photograph
    ``POST /api/cases/<case>/classify``        predict the view of every image
    ``POST /api/cases/<case>/segment``         segment one image
    ``POST /api/cases/<case>/save``            write the corrected annotations

Errors are JSON with an ``error`` key and an accurate status code; the stack trace
goes to the log, never to the client.
"""

from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request, send_file

from cases import CaseImage, patient_id_from_filename, sanitize
from contours import polygons_area, polygons_bbox, polygons_centroid, polygons_to_mask
from iop_compass.classification.dataset import VIEW_CLASSES
from pipeline import FDI_CODES, SEGMENTER_CELLS, SEGMENTER_LABELS

log = logging.getLogger(__name__)

api = Blueprint("api", __name__, url_prefix="/api")

VIEW_LABELS = {
    "frontal": "Frontal",
    "left_buccal": "Left buccal",
    "right_buccal": "Right buccal",
    "upper_occlusal": "Upper occlusal",
    "lower_occlusal": "Lower occlusal",
}


class ApiError(Exception):
    """An error whose message is safe to return to the client."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


@api.errorhandler(ApiError)
def _handle_api_error(exc: ApiError):
    return jsonify({"error": exc.message}), exc.status


@api.errorhandler(Exception)
def _handle_unexpected(exc: Exception):
    log.error("unhandled error on %s\n%s", request.path, traceback.format_exc())
    return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


def _pipeline():
    return current_app.extensions["iopc_pipeline"]


def _store():
    return current_app.extensions["iopc_cases"]


def _settings():
    return current_app.extensions["iopc_settings"]


def _case_or_404(case_id: str):
    case = _store().get(case_id)
    if case is None:
        raise ApiError(f"unknown or expired case {case_id!r}", 404)
    return case


def _image_or_404(case, image_id: str) -> CaseImage:
    image = case.image(image_id)
    if image is None:
        raise ApiError(f"unknown image {image_id!r} in case {case.case_id!r}", 404)
    return image


# --------------------------------------------------------------------- status
@api.get("/health")
def health():
    pipeline = _pipeline()
    return jsonify(
        {
            "status": "ok",
            "device": pipeline.device,
            "device_name": pipeline.device_name(),
            "segmenters": list(pipeline.segmenters),
        }
    )


@api.get("/config")
def config():
    pipeline = _pipeline()
    return jsonify(
        {
            "views": [
                {"value": v, "label": VIEW_LABELS[v]} for v in VIEW_CLASSES
            ],
            "segmenters": [
                {
                    "value": s,
                    "label": SEGMENTER_LABELS[s],
                    "cell": SEGMENTER_CELLS[s],
                }
                for s in pipeline.segmenters
            ],
            "default_segmenter": "sat" if "sat" in pipeline.segmenters else
                                 (pipeline.segmenters[0] if pipeline.segmenters else None),
            "fdi_codes": list(FDI_CODES),
            "max_upload_bytes": _settings().max_upload_bytes,
            "device_name": pipeline.device_name(),
        }
    )


# ---------------------------------------------------------------------- cases
@api.post("/cases")
def create_case():
    """Accept the photographs of one patient and register them as a case."""
    files = request.files.getlist("images")
    if not files:
        raise ApiError("no images uploaded; expected one or more 'images' parts")

    settings = _settings()
    patient_id = sanitize(
        request.form.get("patient_id", "").strip()
        or patient_id_from_filename(files[0].filename),
        "unknown_patient",
    )
    case = _store().create(patient_id)

    for index, item in enumerate(files):
        if item.mimetype not in settings.accepted_mimetypes:
            raise ApiError(
                f"{item.filename!r} has unsupported type {item.mimetype!r}; "
                f"accepted: {', '.join(settings.accepted_mimetypes)}",
                415,
            )
        raw = item.read()
        if len(raw) > settings.max_upload_bytes:
            raise ApiError(
                f"{item.filename!r} is {len(raw) / 1e6:.1f} MB, over the "
                f"{settings.max_upload_bytes / 1e6:.0f} MB limit",
                413,
            )
        decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ApiError(f"could not decode {item.filename!r} as an image")

        image_id = f"img{index}"
        height, width = decoded.shape[:2]
        case.images[image_id] = CaseImage(
            image_id=image_id,
            filename=item.filename or f"{image_id}.png",
            stem=sanitize(Path(item.filename or image_id).stem, image_id),
            width=width,
            height=height,
            bgr=decoded,
            original_bytes=raw,
            content_type=item.mimetype,
        )

    log.info("case %s: %d image(s) for %s", case.case_id, len(case.images), patient_id)
    return jsonify(
        {
            "case_id": case.case_id,
            "patient_id": case.patient_id,
            "images": [
                {
                    "image_id": image.image_id,
                    "filename": image.filename,
                    "width": image.width,
                    "height": image.height,
                    "url": f"/api/cases/{case.case_id}/images/{image.image_id}",
                }
                for image in case.images.values()
            ],
        }
    )


@api.get("/cases/<case_id>/images/<image_id>")
def get_image(case_id: str, image_id: str):
    image = _image_or_404(_case_or_404(case_id), image_id)
    from io import BytesIO

    return send_file(
        BytesIO(image.original_bytes),
        mimetype=image.content_type,
        download_name=image.filename,
    )


# --------------------------------------------------------------------- stages
@api.post("/cases/<case_id>/classify")
def classify(case_id: str):
    case = _case_or_404(case_id)
    predictions = _pipeline().classify(
        {image_id: image.bgr for image_id, image in case.images.items()}
    )
    for prediction in predictions:
        image = case.images[prediction.image_id]
        image.view = prediction.view
        image.view_confidence = prediction.confidence

    return jsonify(
        {
            "constrained": bool(predictions and predictions[0].constrained),
            "predictions": [
                {
                    "image_id": p.image_id,
                    "view": p.view,
                    "view_label": VIEW_LABELS[p.view],
                    "confidence": round(p.confidence, 4),
                    "probabilities": {
                        k: round(v, 4) for k, v in p.probabilities.items()
                    },
                }
                for p in predictions
            ],
        }
    )


@api.post("/cases/<case_id>/segment")
def segment(case_id: str):
    case = _case_or_404(case_id)
    payload = request.get_json(silent=True) or {}
    image = _image_or_404(case, payload.get("image_id", ""))

    view = payload.get("view") or image.view
    if view is None:
        raise ApiError("image has no view yet; run classify first or pass 'view'")
    if view not in VIEW_CLASSES:
        raise ApiError(f"unknown view {view!r}; expected one of {list(VIEW_CLASSES)}")

    segmenter = payload.get("segmenter", "sat")
    if segmenter not in _pipeline().segmenters:
        raise ApiError(
            f"segmenter {segmenter!r} is not enabled; "
            f"available: {list(_pipeline().segmenters)}"
        )

    image.view = view
    result = _pipeline().segment(
        image.bgr,
        view=view,
        segmenter=segmenter,
        postprocess=bool(payload.get("postprocess", True)),
    )
    log.info(
        "case %s image %s: %s -> %d instance(s) in %.2fs",
        case.case_id,
        image.image_id,
        result.cell,
        len(result.instances),
        sum(result.timings.values()),
    )
    return jsonify({"image_id": image.image_id, "view": view, **result.to_dict()})


# ----------------------------------------------------------------------- save
@api.post("/cases/<case_id>/save")
def save(case_id: str):
    """Write the corrected case in the schema ``iop_compass.data.adapter`` reads.

    One directory per patient, one set of files per image: the original, a label
    mask, the annotation JSON, and a per-instance overlay-free record.  Contours
    arrive in the uploaded image's coordinates, so the mask written here is at
    sensor resolution.
    """
    case = _case_or_404(case_id)
    payload = request.get_json(silent=True) or {}
    images_payload = payload.get("images")
    if not isinstance(images_payload, list) or not images_payload:
        raise ApiError("expected a non-empty 'images' list")

    # A patient id corrected after the upload wins: the id decides the output
    # directory, and a case filed under 'unknown_patient' is hard to find again.
    supplied = (payload.get("patient_id") or "").strip()
    if supplied:
        case.patient_id = sanitize(supplied, case.patient_id)

    out_dir = Path(_settings().output_dir) / case.patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []

    for entry in images_payload:
        image = _image_or_404(case, entry.get("image_id", ""))
        view = entry.get("view") or image.view
        instances = entry.get("instances") or []

        teeth, label_mask = _build_annotations(instances, image)
        stem = image.stem

        (out_dir / f"{stem}_original.png").write_bytes(image.original_bytes)
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), label_mask)

        record = {
            "patient_id": case.patient_id,
            "case_name": stem,
            "view": view,
            "image_width": image.width,
            "image_height": image.height,
            "annotation_count": len(teeth),
            "saved_utc": datetime.now(timezone.utc).isoformat(),
            "teeth": teeth,
        }
        (out_dir / f"{stem}_annotations.json").write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )
        written.append(
            {
                "image_id": image.image_id,
                "annotation_count": len(teeth),
                "directory": str(out_dir),
            }
        )

    log.info("case %s saved to %s", case.case_id, out_dir)
    return jsonify({"status": "saved", "patient_id": case.patient_id, "images": written})


def _build_annotations(instances: list[dict], image: CaseImage):
    """Annotation records plus the uint8 label mask, both at the uploaded resolution."""
    label_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    teeth: list[dict] = []

    for index, item in enumerate(instances):
        contours = [c for c in (item.get("contours") or []) if len(c) >= 6]
        if not contours:
            continue
        fdi = item.get("fdi")
        if fdi is not None and int(fdi) not in FDI_CODES:
            raise ApiError(
                f"FDI code {fdi} is not in the permanent-dentition vocabulary"
            )

        mask = polygons_to_mask(contours, image.width, image.height)
        # Label 0 is background, so instances are numbered from 1.  An instance with
        # no FDI still occupies the mask; its identity lives in the JSON record.
        label_mask[mask] = (index + 1) % 256

        x0, y0, x1, y1 = polygons_bbox(contours)
        cx, cy = polygons_centroid(contours)
        teeth.append(
            {
                "appearance_idx": index + 1,
                "FDI_NUM": str(fdi) if fdi is not None else "",
                "class_name": "tooth",
                "pixel_area": int(mask.sum()),
                "polygon_area": round(polygons_area(contours), 1),
                "x_min": round(x0, 1),
                "y_min": round(y0, 1),
                "x_max": round(x1, 1),
                "y_max": round(y1, 1),
                "centroid_x": round(cx, 1),
                "centroid_y": round(cy, 1),
                "arch": _arch(fdi),
                "contour_count": len(contours),
                "contour": contours[0],
                "contours": contours,
            }
        )
    return teeth, label_mask


def _arch(fdi) -> str:
    """Upper or lower arch from an FDI code, validated against the vocabulary."""
    try:
        code = int(fdi)
    except (TypeError, ValueError):
        return ""
    if code not in FDI_CODES:
        return ""
    return "upper" if code // 10 in (1, 2) else "lower"
