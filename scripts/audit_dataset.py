#!/usr/bin/env python
"""Audit the dataset and write the audit report + proposed repairs.

    python scripts/audit_dataset.py --config configs/dataset_final_1000.yaml
    python scripts/audit_dataset.py --config ... --splits <derived>/splits.json
    python scripts/audit_dataset.py --config ... --fail-on-fatal   # CI / job gate

Exit codes: 0 clean, 1 fatal findings (only with --fail-on-fatal), 2 usage error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import (  # noqa: E402
    discover_patients,
    load_dataset_config,
    sha256_file,
)
from iop_compass.data.splits import load_splits  # noqa: E402
from iop_compass.data.validation import audit, render_markdown  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--out", default=None, help="output directory (default results/<name>/audit)")
    ap.add_argument("--checksums", action="store_true", help="hash every image (slow)")
    ap.add_argument(
        "--checksums-from",
        default=None,
        help="reuse a previously written image_checksums.json instead of re-hashing",
    )
    ap.add_argument("--fail-on-fatal", action="store_true")
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    patients = discover_patients(cfg)

    checksums: dict[str, str] = {}
    if args.checksums_from and Path(args.checksums_from).exists():
        checksums = json.loads(Path(args.checksums_from).read_text())
        print(f"[audit] reused {len(checksums)} checksums from {args.checksums_from}")
    elif args.checksums:
        for patient in patients:
            for view in cfg.views_required:
                rec = patient.images.get(view)
                if rec is not None and rec.image_path is not None:
                    checksums[rec.image_id] = sha256_file(rec.image_path)

    split_map = load_splits(Path(args.splits)) if args.splits else {}
    report = audit(patients, cfg, checksums=checksums, split_map=split_map)

    out_dir = Path(args.out) if args.out else REPO / "results" / cfg.name / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_report.json").write_text(json.dumps(report.to_dict(), indent=1))
    (out_dir / "audit_report.md").write_text(render_markdown(report))
    with (out_dir / "proposed_repairs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scope", "target", "action", "reason"])
        writer.writeheader()
        for repair in report.repairs:
            writer.writerow(asdict(repair))
    if checksums:
        (out_dir / "image_checksums.json").write_text(json.dumps(checksums, indent=0))

    # Exclusion evidence for the split builder.  Written only when the audit had
    # the checksums it needs to see duplicate content, so a cheap audit run cannot
    # silently drop the duplicate-image evidence.
    exclusions_written = bool(report.unusable_images and checksums)
    if exclusions_written:
        target = cfg.derived_root / "unusable_images.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report.unusable_images, indent=1, sort_keys=True))
        print(
            f"[audit] {len(report.unusable_images)} unusable image(s) recorded in "
            f"{target}"
        )

    s = report.stats
    print(
        f"[audit] {s['n_patient_directories']} dirs, "
        f"{s['n_complete_patients']} complete patients, "
        f"{s['n_images']} images, {s['n_instances']} instances, "
        f"{s['n_distinct_fdi']} FDI codes",
        flush=True,
    )
    for finding in report.findings:
        print(f"[audit] {finding.severity:<7} {finding.code}: {len(finding.items)}")
    print(f"[audit] wrote {out_dir}")

    # A fatal finding whose images the audit itself recorded in unusable_images.json is
    # already excluded from every split, so it is evidence the exclusion machinery
    # worked, not a reason to stop the pipeline.  Only an unhandled fatal blocks.
    if report.fatal:
        # Without the exclusion file on disk the split builder cannot act on the
        # findings, so nothing counts as handled however well the audit recorded it.
        handled = report.fatal_handled if exclusions_written else []
        unhandled = report.fatal_unhandled if exclusions_written else report.fatal
        print(
            f"[audit] fatal, already excluded: "
            f"{sum(len(f.image_ids) for f in handled)} image(s) in "
            f"{len(handled)} finding(s)",
            file=sys.stderr,
        )
        print(
            f"[audit] fatal, unhandled: {len(unhandled)} finding(s)"
            + (f" ({', '.join(f.code for f in unhandled)})" if unhandled else ""),
            file=sys.stderr,
        )
        if unhandled and args.fail_on_fatal:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
