"""Viewer configuration.

Every path and threshold is read from the environment so the same code runs on a
workstation with downloaded weights and on the cluster against a run directory.
Nothing is discovered implicitly: :func:`Settings.missing` reports exactly which
files are absent, and the server refuses to start until they are there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Environment variable -> what it relocates, printed verbatim when a file is missing.
WEIGHT_ENV = {
    "classifier": "IOPC_CLASSIFIER_WEIGHTS",
    "maskrcnn": "IOPC_MASKRCNN_WEIGHTS",
    "sat_weights": "SAT_WEIGHT_DIR",
    "sat_code": "SAT_CODE_DIR",
}

# Files SegmentAnyTooth needs inside its weight directory.
SAT_WEIGHT_FILES = (
    "segmentanytooth_vit_tiny.pt",
    "segmentanytooth_yolo11_front.pt",
    "segmentanytooth_yolo11_upper.pt",
    "segmentanytooth_yolo11_lower.pt",
    "segmentanytooth_yolo11_right.pt",
)


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser() if value else default


@dataclass
class Settings:
    """Resolved viewer settings."""

    classifier_weights: Path
    maskrcnn_weights: Path
    sat_weight_dir: Path
    sat_code_dir: Path
    postprocessing_config: Path
    output_dir: Path

    device: str = "cuda"
    #: Long side the segmenters see; matches ``inference_long_side`` in the dataset
    #: configs, so the viewer runs the models at the resolution they were measured at.
    inference_long_side: int = 2048
    #: Square input size of the view classifier, matching ``classifier_image_size``.
    classifier_image_size: int = 256
    #: Largest accepted upload, per file.
    max_upload_bytes: int = 40 * 1024 * 1024
    accepted_mimetypes: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/tiff",
        "image/bmp",
    )
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])

    @classmethod
    def from_env(cls) -> "Settings":
        weights = _path("IOPC_WEIGHTS_DIR", REPO / "weights")
        origins = os.environ.get("IOPC_CORS_ORIGINS", "http://localhost:5173")
        return cls(
            classifier_weights=_path(
                "IOPC_CLASSIFIER_WEIGHTS", weights / "view_classifier.pt"
            ),
            maskrcnn_weights=_path(
                "IOPC_MASKRCNN_WEIGHTS", weights / "maskrcnn.pt"
            ),
            sat_weight_dir=_path("SAT_WEIGHT_DIR", REPO / "third_party" / "sat_weights"),
            sat_code_dir=_path(
                "SAT_CODE_DIR", REPO / "third_party" / "segmentanytooth_src"
            ),
            postprocessing_config=_path(
                "IOPC_POSTPROCESSING_CONFIG",
                REPO / "configs" / "segmentation" / "postprocessing.yaml",
            ),
            output_dir=_path("IOPC_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs"),
            device=os.environ.get("IOPC_DEVICE", "cuda"),
            inference_long_side=int(os.environ.get("IOPC_INFERENCE_LONG_SIDE", 2048)),
            classifier_image_size=int(os.environ.get("IOPC_CLASSIFIER_IMAGE_SIZE", 256)),
            max_upload_bytes=int(
                os.environ.get("IOPC_MAX_UPLOAD_BYTES", 40 * 1024 * 1024)
            ),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        )

    def missing(self, segmenters: tuple[str, ...]) -> list[str]:
        """Human-readable problems, each naming the env var that fixes it.

        Only the segmenters actually enabled are checked, so a deployment that
        serves SegmentAnyTooth alone does not need the Mask R-CNN checkpoint.
        """
        problems: list[str] = []

        if not self.classifier_weights.is_file():
            problems.append(
                f"view classifier checkpoint not found: {self.classifier_weights} "
                f"(set {WEIGHT_ENV['classifier']}, or rename/symlink the release "
                "checkpoint to this name -- it is distributed under its training "
                "run's name, not this one)"
            )

        if "sat" in segmenters:
            if not self.sat_code_dir.is_dir():
                problems.append(
                    f"SegmentAnyTooth source not found: {self.sat_code_dir} "
                    f"(set {WEIGHT_ENV['sat_code']}; the code is MIT and can be "
                    "cloned from github.com/thangngoc89/SegmentAnyTooth -- only the "
                    "weights need the licence agreement)"
                )
            for name in SAT_WEIGHT_FILES:
                if not (self.sat_weight_dir / name).is_file():
                    problems.append(
                        f"SegmentAnyTooth weight not found: {self.sat_weight_dir / name} "
                        f"(set {WEIGHT_ENV['sat_weights']})"
                    )

        if "mask-rcnn" in segmenters and not self.maskrcnn_weights.is_file():
            problems.append(
                f"Mask R-CNN checkpoint not found: {self.maskrcnn_weights} "
                f"(set {WEIGHT_ENV['maskrcnn']}, or rename/symlink the release "
                "checkpoint to this name -- it is distributed under its training "
                "run's name, not this one)"
            )

        if not self.postprocessing_config.is_file():
            problems.append(
                f"post-processing config not found: {self.postprocessing_config} "
                "(set IOPC_POSTPROCESSING_CONFIG)"
            )
        return problems
