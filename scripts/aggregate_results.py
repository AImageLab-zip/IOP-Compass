#!/usr/bin/env python
"""Aggregate every result file for a run into one machine-readable document.

    python scripts/aggregate_results.py --run final_1000 \
        --config configs/dataset_final_1000.yaml

Writes ``results/<dataset>/aggregate.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.reporting.aggregate import build_aggregate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    agg = build_aggregate(REPO, cfg, args.run)
    out_dir = REPO / "results" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregate.json").write_text(json.dumps(agg.to_dict(), indent=1))
    print(f"[aggregate] wrote {out_dir / 'aggregate.json'}")
    print(
        f"[aggregate] classification splits={sorted(agg.classification)} "
        f"roi splits={sorted(agg.roi)} segmentation splits={sorted(agg.segmentation)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
