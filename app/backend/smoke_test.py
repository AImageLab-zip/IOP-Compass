#!/usr/bin/env python
"""End-to-end check that the viewer serves the benchmarked configuration.

    python app/backend/smoke_test.py --config configs/dataset_final_1000.yaml

Uploads one held-out patient's five photographs through the real HTTP API, runs
classification and both segmenters, and asserts:

* every predicted view matches the manifest's view label;
* every returned FDI code is in the permanent-dentition vocabulary;
* the instance count is within a tolerance of the reference annotation count;
* saving produces a mask at the *uploaded* resolution, not at the inference one.

It is a wiring check, not a metric: the benchmark numbers come from
``scripts/evaluate_segmentation.py`` over the whole split. What this catches is a
viewer that silently runs a different pipeline than the one that was measured.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from config import Settings  # noqa: E402
from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from pipeline import FDI_CODES, SEGMENTERS  # noqa: E402
from server import create_app  # noqa: E402

#: Instance counts differ from the reference by real detector errors, so the check
#: is a sanity bound rather than an equality.
COUNT_TOLERANCE = 8


def pick_patient(cfg, split: str) -> list[dict]:
    """The first complete patient of ``split``, with ``image_path`` made absolute.

    Manifest paths are relative to the dataset root, which is what binds a manifest
    to a cohort rather than to one machine's directory layout.
    """
    manifest = list(csv.DictReader((cfg.derived_root / "manifest.csv").open()))
    for row in manifest:
        row["image_path"] = str(cfg.root / row["image_path"])
    splits = json.loads((cfg.derived_root / "splits.json").read_text())
    for patient in splits.get(split, []):
        rows = [r for r in manifest if r["patient_id"] == patient]
        if len(rows) == 5 and all(Path(r["image_path"]).exists() for r in rows):
            return rows
    raise SystemExit(f"no complete {split} patient with readable images")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset_final_1000.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--segmenters", default=",".join(SEGMENTERS), help="which segmenters to exercise"
    )
    args = parser.parse_args()

    cfg = load_dataset_config(REPO / args.config)
    rows = pick_patient(cfg, args.split)
    patient = rows[0]["patient_id"]
    print(f"patient {patient}: {len(rows)} images")

    segmenters = tuple(s.strip() for s in args.segmenters.split(",") if s.strip())
    settings = Settings.from_env()
    problems = settings.missing(segmenters)
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    app = create_app(settings, segmenters)
    client = app.test_client()
    failures: list[str] = []

    payload = [
        ("images", (Path(r["image_path"]).open("rb"), Path(r["image_path"]).name))
        for r in rows
    ]
    response = client.post(
        "/api/cases",
        data={"patient_id": patient, "images": [p[1] for p in payload]},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200, response.get_json()
    case = response.get_json()
    case_id = case["case_id"]
    by_name = {Path(r["image_path"]).name: r for r in rows}
    print(f"case {case_id}: uploaded {len(case['images'])} image(s)")

    # ------------------------------------------------------------ classification
    response = client.post(f"/api/cases/{case_id}/classify")
    assert response.status_code == 200, response.get_json()
    classified = response.get_json()
    print(f"constrained assignment: {classified['constrained']}")
    views: dict[str, str] = {}
    for prediction in classified["predictions"]:
        image = next(
            i for i in case["images"] if i["image_id"] == prediction["image_id"]
        )
        expected = by_name[image["filename"]]["view_label"]
        views[prediction["image_id"]] = prediction["view"]
        mark = "ok " if prediction["view"] == expected else "BAD"
        if prediction["view"] != expected:
            failures.append(
                f"{image['filename']}: predicted {prediction['view']}, expected {expected}"
            )
        print(
            f"  {mark} {image['filename']:<38} {prediction['view']:<15} "
            f"p={prediction['confidence']:.4f}"
        )

    # -------------------------------------------------------------- segmentation
    for segmenter in segmenters:
        print(f"\nsegmenter {segmenter}")
        for image in case["images"]:
            expected_n = int(by_name[image["filename"]]["number_of_instances"])
            response = client.post(
                f"/api/cases/{case_id}/segment",
                json={
                    "image_id": image["image_id"],
                    "segmenter": segmenter,
                    "postprocess": True,
                },
            )
            assert response.status_code == 200, response.get_json()
            result = response.get_json()

            if (result["width"], result["height"]) != (image["width"], image["height"]):
                failures.append(
                    f"{image['filename']}: {segmenter} returned "
                    f"{result['width']}x{result['height']}, uploaded "
                    f"{image['width']}x{image['height']}"
                )
            bad = [
                i["fdi"]
                for i in result["instances"]
                if i["fdi"] is not None and i["fdi"] not in FDI_CODES
            ]
            if bad:
                failures.append(f"{image['filename']}: invalid FDI codes {bad}")

            n = len(result["instances"])
            if abs(n - expected_n) > COUNT_TOLERANCE:
                failures.append(
                    f"{image['filename']}: {segmenter} found {n} instances, "
                    f"reference has {expected_n}"
                )
            print(
                f"  {result['cell']:<24} {views[image['image_id']]:<15} "
                f"{n:>3} instances (ref {expected_n:>3}) "
                f"{sum(result['timings'].values()):.2f}s"
            )

    # ---------------------------------------------------------------------- save
    response = client.post(
        f"/api/cases/{case_id}/segment",
        json={"image_id": case["images"][0]["image_id"], "segmenter": segmenters[0]},
    )
    instances = response.get_json()["instances"]
    response = client.post(
        f"/api/cases/{case_id}/save",
        json={
            "images": [
                {
                    "image_id": case["images"][0]["image_id"],
                    "view": views[case["images"][0]["image_id"]],
                    "instances": instances,
                }
            ]
        },
    )
    assert response.status_code == 200, response.get_json()
    saved = response.get_json()
    first = case["images"][0]
    mask_path = (
        Path(saved["images"][0]["directory"]) / f"{Path(first['filename']).stem}_mask.png"
    )
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        failures.append(f"no mask written at {mask_path}")
    elif mask.shape[:2] != (first["height"], first["width"]):
        failures.append(
            f"saved mask is {mask.shape[1]}x{mask.shape[0]}, uploaded image is "
            f"{first['width']}x{first['height']}"
        )
    else:
        print(f"\nsaved mask {mask.shape[1]}x{mask.shape[0]} at uploaded resolution")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
