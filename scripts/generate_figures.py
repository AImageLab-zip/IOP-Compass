#!/usr/bin/env python
"""Generate the paper figures from the aggregate results.

    python scripts/generate_figures.py --run final_1000 \
        --config configs/dataset_final_1000.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.reporting.aggregate import build_aggregate  # noqa: E402
from iop_compass.reporting.figures import generate_all  # noqa: E402
from iop_compass.reporting.qualitative import five_view_showcase  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--five-view-patient", default="P0001")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    agg = build_aggregate(REPO, cfg, args.run)
    out_dir = REPO / "paper" / "generated" / "figures"
    produced = generate_all(agg, out_dir, args.split)
    for path in produced:
        print(f"[figures] wrote {path.relative_to(REPO)}")

    five_view_panels = five_view_showcase(cfg, out_dir, args.five_view_patient)
    for path in five_view_panels:
        print(f"[figures] wrote {path.relative_to(REPO)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
