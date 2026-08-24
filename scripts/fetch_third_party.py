#!/usr/bin/env python
"""Link the third-party code and weights this benchmark depends on.

    python scripts/fetch_third_party.py
    python scripts/fetch_third_party.py --record runs/final_1000/third_party.json

Nothing is downloaded and no licence is accepted. The script only links assets that
are already present on this machine and records their path, size and SHA-256 so a
result can be tied to the exact weights that produced it:

  * SegmentAnyTooth source (MIT) and its per-view weights, which are covered by a
    separate non-commercial agreement shipped with them;
  * the SAM 3 package and checkpoint.

Override any location with the matching environment variable. If an asset is
missing the script reports it as a blocked dependency and exits non-zero, so a
missing licence never turns into a silently different experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import sha256_file  # noqa: E402

TARGET = REPO / "third_party"

ASSETS = {
    "segmentanytooth_src": {
        "env": "SAT_CODE_DIR",
        "default": "/work/yazelelew_phd/Tooth_Segmentation/SegmentAnyTooth_Dependance",
        "kind": "dir",
        "licence": "MIT (code)",
        "note": "provides the `segmentanytooth` and `sam` modules",
    },
    "sat_weights": {
        "env": "SAT_WEIGHT_DIR",
        "default": str(REPO / "inputs" / "SegmentAnyTooth weights"),
        "kind": "dir",
        "licence": "separate non-commercial agreement (see the PDF shipped with the weights)",
        "note": "segmentanytooth_vit_tiny.pt + per-view YOLO11 detectors; not redistributed",
    },
    "sam3": {
        "env": "SAM3_CODE_DIR",
        "default": "/work/yazelelew_phd/Tooth_Segmentation_App/backend/sam3",
        "kind": "dir",
        "licence": "Meta SAM 3 licence",
        "note": "importable `sam3` package",
    },
    "sam3.pt": {
        "env": "SAM3_CHECKPOINT",
        "default": str(REPO / "inputs" / "sam3.pt"),
        "kind": "file",
        "licence": "Meta SAM 3 licence",
        "note": "local checkpoint; keeps inference offline",
    },
    "torch_home": {
        "env": "TORCH_HOME",
        "default": str(TARGET / "torch_home"),
        "kind": "dir",
        "licence": "torchvision (BSD)",
        "note": "ImageNet / COCO backbones pre-downloaded so compute nodes need no network",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None, help="write the inventory to this JSON path")
    ap.add_argument("--hash-weights", action="store_true", help="checksum every weight file")
    args = ap.parse_args()

    TARGET.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, dict] = {}
    missing: list[str] = []

    for name, spec in ASSETS.items():
        source = Path(os.environ.get(spec["env"], spec["default"]))
        link = TARGET / name
        entry = {
            "source": str(source),
            "env_var": spec["env"],
            "licence": spec["licence"],
            "note": spec["note"],
            "present": source.exists(),
        }
        if not source.exists():
            missing.append(f"{name} (expected at {source}, override with ${spec['env']})")
        else:
            if link.is_symlink() or link.exists():
                if link.is_symlink() and Path(os.readlink(link)) != source:
                    link.unlink()
                    link.symlink_to(source)
            else:
                link.symlink_to(source)
            if spec["kind"] == "file":
                entry["bytes"] = source.stat().st_size
                entry["sha256"] = sha256_file(source)
            elif args.hash_weights:
                entry["files"] = {
                    p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                    for p in sorted(source.glob("*.pt"))
                }
        inventory[name] = entry
        state = "ok" if entry["present"] else "MISSING"
        print(f"[third_party] {name:22s} {state:8s} {source}")

    if args.record:
        path = Path(args.record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(inventory, indent=1))
        print(f"[third_party] inventory -> {path}")

    if missing:
        print("\n[third_party] BLOCKED DEPENDENCIES:", file=sys.stderr)
        for item in missing:
            print(f"[third_party]   - {item}", file=sys.stderr)
        print(
            "[third_party] obtain them yourself and accept their licences; this "
            "script never downloads or accepts anything on your behalf.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
