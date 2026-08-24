#!/usr/bin/env python
"""Collect the benchmark grid into one Markdown report and one JSON document.

    python scripts/report_grid.py --run final_1000 \
        --config configs/dataset_final_1000.yaml --split test

Reads the per-cell result files written by ``scripts/evaluate_segmentation.py`` and
emits:

* ``results/<run>/segmentation/<split>/grid.json`` -- machine-readable;
* ``results/<run>/benchmark_grid_<split>.md`` -- the table to read.

Two things the sixteen-row table on its own cannot tell you, and which this report
therefore leads with:

*Replicate spread versus sampling error.*  With one replicate there is no spread, so
the +/- shown is half the patient-level bootstrap 95% CI of that single measurement.
With several replicates it is the standard deviation over them.  The two answer
different questions and are labelled differently; they are never mixed in one column.

*Main effects.*  Sixteen cells make 120 pairwise comparisons, and on a hold-out of
this size most of them are inconclusive.  A marginal effect -- one ROI level against
another, pooled over both segmenters and both post settings -- uses four times the
data and is the comparison that can actually resolve.  Marginals are computed by
pooling the per-image statistics of the cells sharing a level and running the paired
patient-level bootstrap, so both arms are always resampled over the same patients.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.data.adapter import load_dataset_config  # noqa: E402
from iop_compass.reporting.bootstrap import (  # noqa: E402
    is_inconclusive,
    paired_bootstrap,
)
from iop_compass.segmentation import grid  # noqa: E402
from iop_compass.segmentation.metrics import ImageEval  # noqa: E402

# Columns of the main table, in order: (metric key, header, format).
TABLE_METRICS = [
    ("fdi_instance_f1", "FDI F1", "{:.4f}"),
    ("fdi_accuracy_matched", "FDI acc.", "{:.4f}"),
    ("n_false_positive", "FP", "{:.0f}"),
    ("n_false_negative", "FN", "{:.0f}"),
    ("n_fdi_errors", "FDI err.", "{:.0f}"),
    ("n_merges", "Merge", "{:.0f}"),
    ("n_splits", "Split", "{:.0f}"),
    ("complete_image_rate", "Compl.", "{:.3f}"),
    ("matched_dice", "mDice", "{:.4f}"),
    ("mean_runtime_s", "s/img", "{:.2f}"),
]

# Metrics the marginal comparisons are run on.  Kept short on purpose: every extra
# metric is another interval a reader has to discount for multiplicity.
EFFECT_METRICS = ["fdi_instance_f1", "fdi_accuracy_matched", "complete_image_rate"]

PRIMARY = "fdi_instance_f1"


def load_cell(results_dir: Path, cell: str, replicate: int):
    """``(payload, evals)`` for one cell and replicate, or ``(None, None)``."""
    name = grid.result_name(cell, replicate)
    js = results_dir / f"{name}.json"
    csv_path = results_dir / f"{name}_per_image.csv"
    if not js.exists() or not csv_path.exists():
        return None, None
    payload = json.loads(js.read_text())
    with csv_path.open() as fh:
        evals = [ImageEval.from_row(row) for row in csv.DictReader(fh)]
    return payload, evals


def discover_replicates(results_dir: Path) -> list[int]:
    found = set()
    for path in results_dir.glob("*.json"):
        parsed = grid.parse_result_name(path.stem)
        if parsed is not None:
            found.add(parsed[1])
    return sorted(found)


def summarise_cell(payloads: list[dict], evals_by_rep: list[list[ImageEval]]) -> dict:
    """Per-metric point estimate plus a spread whose meaning depends on replicate count."""
    out: dict[str, dict] = {}
    n = len(payloads)
    for key, _, _ in TABLE_METRICS:
        values = [float(p["overall"][key]) for p in payloads if key in p["overall"]]
        if not values:
            continue
        entry = {"mean": statistics.fmean(values), "n_replicates": n}
        if n > 1:
            entry["std"] = statistics.pstdev(values)
            entry["spread"] = "std_over_replicates"
        else:
            ci = (payloads[0].get("bootstrap") or {}).get(key)
            if ci is not None:
                entry["ci_low"] = ci["ci_low"]
                entry["ci_high"] = ci["ci_high"]
                entry["half_ci"] = (ci["ci_high"] - ci["ci_low"]) / 2.0
                entry["spread"] = "half_bootstrap_ci"
            else:
                entry["spread"] = "none"
        out[key] = entry
    out["n_images"] = sum(len(e) for e in evals_by_rep) // max(n, 1)
    return out


def other_levels(cell: str, index: int) -> tuple:
    """The cell's levels on the axes *other* than ``index``."""
    parsed = grid.parse_cell(cell)
    return tuple(v for i, v in enumerate(parsed) if i != index)


def balanced_arms(
    cells: dict[tuple[str, int], list[ImageEval]],
    index: int,
    level_a: str,
    level_b: str,
    restrict=None,
):
    """Pool two levels of one axis over exactly the same background cells.

    A marginal effect is only an effect of *this* axis if both arms average over the
    same combinations of the other axes.  If one level is missing a cell the other
    has, pooling everything available would fold that other axis into the difference
    -- e.g. comparing a ROI level that only exists with post-processing against one
    that exists both ways would report part of the post-processing effect as a ROI
    effect.  So the background combinations are intersected first, and how many
    survived is reported alongside the interval.
    """
    def keep(cell, level):
        return grid.parse_cell(cell)[index] == level and (
            restrict is None or restrict(cell)
        )

    backgrounds: dict[str, set] = {}
    for level in (level_a, level_b):
        backgrounds[level] = {
            other_levels(cell, index)
            for (cell, _rep) in cells
            if keep(cell, level)
        }
    shared = backgrounds[level_a] & backgrounds[level_b]

    arms: dict[str, list[ImageEval]] = {level_a: [], level_b: []}
    for (cell, _rep), evals in sorted(cells.items()):
        for level in (level_a, level_b):
            if keep(cell, level) and other_levels(cell, index) in shared:
                arms[level].extend(evals)
    dropped = sorted(
        (backgrounds[level_a] | backgrounds[level_b]) - shared
    )
    return arms[level_a], arms[level_b], len(shared), dropped


def axis_effects(
    cells: dict[tuple[str, int], list[ImageEval]],
    axis: str,
    levels: tuple[str, ...],
    index: int,
    bootstrap: int,
    seed: int,
    restrict=None,
) -> list[dict]:
    """Every level of one axis against the first level, marginalising the others."""
    reference = levels[0]
    rows = []
    for level in levels[1:]:
        arm_a, arm_b, n_shared, dropped = balanced_arms(
            cells, index, reference, level, restrict
        )
        if not arm_a or not arm_b:
            continue
        deltas = paired_bootstrap(arm_a, arm_b, EFFECT_METRICS, bootstrap, seed)
        if not deltas:
            continue
        rows.append(
            {
                "axis": axis,
                "level": level,
                "reference": reference,
                "n_background_cells": n_shared,
                "dropped_backgrounds": ["+".join(d) for d in dropped],
                "deltas": {
                    k: {**v.to_dict(), "inconclusive": is_inconclusive(v)}
                    for k, v in deltas.items()
                },
            }
        )
    return rows


def fmt_point(entry: dict, fmt: str) -> str:
    """Render one table cell, spread included, never mixing the two spread kinds."""
    value = fmt.format(entry["mean"])
    if entry.get("spread") == "std_over_replicates":
        return f"{value} ± {fmt.format(entry['std'])}"
    if entry.get("spread") == "half_bootstrap_ci":
        return f"{value} ± {fmt.format(entry['half_ci'])}"
    return value


def fmt_delta(interval: dict) -> str:
    verdict = "inconclusive" if interval["inconclusive"] else "conclusive"
    return (
        f"{interval['point']:+.4f} [{interval['ci_low']:+.4f}, "
        f"{interval['ci_high']:+.4f}] {verdict}"
    )


def build_markdown(doc: dict) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"# Segmentation benchmark grid — {doc['split']} split")
    a("")
    a(f"Run `{doc['run']}` · {doc['n_cells_present']}/16 cells present · "
      f"replicates {doc['replicates']} · "
      f"{doc['n_images']} images / {doc['n_patients']} patients per replicate")
    a("")
    a(f"Primary outcome: FDI-aware instance F1 at mask IoU ≥ 0.50. "
      f"Spread shown is **{doc['spread_meaning']}**.")
    a("")

    if doc["missing_cells"]:
        a("> **Incomplete.** These cells have no result file, so every marginal below "
          "is computed without them:")
        a(">")
        for cell in doc["missing_cells"]:
            a(f"> - `{cell}`")
        a("")

    a("## Cells")
    a("")
    header = ["ROI", "Segmenter", "Post"] + [h for _, h, _ in TABLE_METRICS]
    a("| " + " | ".join(header) + " |")
    a("|" + "|".join(["---"] * len(header)) + "|")
    for row in doc["cells"]:
        roi, seg, post = grid.parse_cell(row["cell"])
        cells = [grid.LABELS[roi], grid.LABELS[seg], grid.LABELS[post]]
        for key, _, fmt in TABLE_METRICS:
            entry = row["metrics"].get(key)
            cells.append(fmt_point(entry, fmt) if entry else "—")
        a("| " + " | ".join(cells) + " |")
    a("")
    a("`Compl.` is the fraction of images with no error of any kind. FP/FN/FDI err./"
      "Merge/Split are absolute counts over the split, not rates.")
    a("")

    a("## Main effects")
    a("")
    a("Each row pools the cells sharing a level and compares them against the "
      "reference level over the same patients, so it uses every image in the split "
      "rather than one sixteenth of it. Differences are `level − reference` on the "
      "primary outcome; an interval spanning zero is reported as inconclusive and "
      "supports no claim.")
    a("")
    a("| Axis | Level | vs reference | Paired over | Δ FDI F1 [95% CI] | Δ FDI acc. | Δ Compl. |")
    a("|---|---|---|---|---|---|---|")
    for row in doc["main_effects"]:
        d = row["deltas"]
        a(
            f"| {row['axis']} | {grid.LABELS.get(row['level'], row['level'])} "
            f"| {grid.LABELS.get(row['reference'], row['reference'])} "
            f"| {row['n_background_cells']} "
            f"cell{'s' if row['n_background_cells'] != 1 else ''} "
            f"| {fmt_delta(d[PRIMARY])} "
            f"| {d['fdi_accuracy_matched']['point']:+.4f} "
            f"| {d['complete_image_rate']['point']:+.4f} |"
        )
    a("")
    a("*Paired over* is how many combinations of the other two axes both levels have "
      "in common; only those are pooled, so the difference is an effect of this axis "
      "and not of a missing cell elsewhere.")
    dropped = {
        d for row in doc["main_effects"] for d in row.get("dropped_backgrounds", [])
    }
    if dropped:
        a("")
        a("Background combinations excluded from at least one comparison because a "
          "cell was missing: " + ", ".join(f"`{d}`" for d in sorted(dropped)) + ".")
    a("")

    a("## Interactions")
    a("")
    a("The same marginals computed inside one segmenter at a time. An axis whose "
      "effect changes sign or magnitude between the two blocks does not have a single "
      "answer, and reporting only its pooled main effect would hide that.")
    a("")
    for seg, rows in doc["interactions"].items():
        a(f"### Within {grid.LABELS[seg]}")
        a("")
        if not rows:
            a("Not enough cells of this segmenter to compare anything within it.")
            a("")
            continue
        a("| Axis | Level | vs reference | Paired over | Δ FDI F1 [95% CI] |")
        a("|---|---|---|---|---|")
        for row in rows:
            a(
                f"| {row['axis']} | {grid.LABELS.get(row['level'], row['level'])} "
                f"| {grid.LABELS.get(row['reference'], row['reference'])} "
                f"| {row['n_background_cells']} "
            f"cell{'s' if row['n_background_cells'] != 1 else ''} "
                f"| {fmt_delta(row['deltas'][PRIMARY])} |"
            )
        a("")

    a("## Provenance")
    a("")
    for key, value in doc["provenance"].items():
        a(f"- **{key}**: `{value}`")
    a("")
    a("Limits worth carrying into any reading of the table: with a single replicate "
      "every number is conditional on one hold-out partition; both segmenters are "
      "used off the shelf and neither was tuned here; and the post-processing axis is "
      "one bundled module, whose individual rules are ablated separately by "
      "`scripts/ablate_postprocessing.py`.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--replicates",
        default=None,
        help="comma-separated replicate indices (default: every one found)",
    )
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    run = args.run or cfg.name
    results_dir = REPO / "results" / run / "segmentation" / args.split
    if not results_dir.exists():
        print(f"[report] no results at {results_dir}", file=sys.stderr)
        return 1

    replicates = (
        [int(v) for v in args.replicates.split(",") if v.strip()]
        if args.replicates
        else discover_replicates(results_dir)
    )
    if not replicates:
        print(
            f"[report] no grid result files in {results_dir}; run "
            f"evaluate_segmentation.py first",
            file=sys.stderr,
        )
        return 2

    evals: dict[tuple[str, int], list[ImageEval]] = {}
    payloads: dict[str, list[dict]] = {}
    per_rep: dict[str, list[list[ImageEval]]] = {}
    for cell in grid.all_cells():
        for rep in replicates:
            payload, cell_evals = load_cell(results_dir, cell, rep)
            if payload is None:
                continue
            evals[(cell, rep)] = cell_evals
            payloads.setdefault(cell, []).append(payload)
            per_rep.setdefault(cell, []).append(cell_evals)

    present = sorted(payloads)
    missing = [c for c in grid.all_cells() if c not in payloads]
    if not present:
        print("[report] no cells found", file=sys.stderr)
        return 3

    cell_rows = [
        {
            "cell": cell,
            "label": grid.label(cell),
            "metrics": summarise_cell(payloads[cell], per_rep[cell]),
        }
        for cell in grid.all_cells()
        if cell in payloads
    ]

    main_effects = (
        axis_effects(evals, "ROI", grid.ROI_LEVELS, 0, args.bootstrap, args.bootstrap_seed)
        + axis_effects(
            evals, "Segmenter", grid.SEG_LEVELS, 1, args.bootstrap, args.bootstrap_seed
        )
        + axis_effects(
            evals, "Post-processing", grid.POST_LEVELS, 2, args.bootstrap, args.bootstrap_seed
        )
    )

    interactions: dict[str, list[dict]] = {}
    for seg in grid.SEG_LEVELS:
        def within(cell, seg=seg):
            return grid.parse_cell(cell)[1] == seg

        interactions[seg] = axis_effects(
            evals, "ROI", grid.ROI_LEVELS, 0, args.bootstrap, args.bootstrap_seed, within
        ) + axis_effects(
            evals,
            "Post-processing",
            grid.POST_LEVELS,
            2,
            args.bootstrap,
            args.bootstrap_seed,
            within,
        )

    any_payload = payloads[present[0]][0]
    n_images = any_payload["n_images"]
    n_patients = int(any_payload["overall"].get("n_patients", 0))
    frozen = REPO / "runs" / run / "frozen_config.sha256"

    doc = {
        "run": run,
        "split": args.split,
        "replicates": replicates,
        "n_cells_present": len(present),
        "missing_cells": missing,
        "n_images": n_images,
        "n_patients": n_patients,
        "spread_meaning": (
            "the standard deviation over replicates"
            if len(replicates) > 1
            else "half the patient-level bootstrap 95% CI of the single replicate"
        ),
        "cells": cell_rows,
        "main_effects": main_effects,
        "interactions": interactions,
        "provenance": {
            "split_hash": any_payload.get("split_hash"),
            "manifest_hash": any_payload.get("dataset_manifest_hash"),
            "frozen_config_sha256": (
                frozen.read_text().strip() if frozen.exists() else "not frozen"
            ),
            "eval_long_side": any_payload.get("eval_long_side"),
            "iou_threshold": any_payload.get("iou_threshold"),
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
        },
    }

    (results_dir / "grid.json").write_text(json.dumps(doc, indent=1))
    md_path = REPO / "results" / run / f"benchmark_grid_{args.split}.md"
    md_path.write_text(build_markdown(doc))
    print(f"[report] wrote {results_dir / 'grid.json'}")
    print(f"[report] wrote {md_path}")
    if missing:
        print(f"[report] WARNING: {len(missing)} of 16 cells missing: {missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
