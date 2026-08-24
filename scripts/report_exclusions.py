#!/usr/bin/env python
"""Write a human-readable report of every excluded patient/file and why.

    python scripts/report_exclusions.py --config configs/dataset_final_1000.yaml

Reads ``<derived_root>/splits.json`` (must already exist — run
``scripts/create_splits.py`` first) plus the three per-image QC files
(``undecodable_images.json``, ``unusable_annotations.json``,
``unusable_images.json``) and writes
``results/<name>/audit/excluded_files_report.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.reporting.exclusions import write_exclusion_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output directory (default results/<name>/audit)")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    out_dir = Path(args.out) if args.out else REPO / "results" / cfg.name / "audit"
    out_path = write_exclusion_report(cfg, out_dir)
    print(f"[report_exclusions] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
