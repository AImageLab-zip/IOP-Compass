"""Dataset audit.

Produces a machine-readable report plus a proposed-repair table.  Nothing is
ever repaired implicitly: :func:`audit` only *describes* the dataset and lists
what a repair would do.

Hard gates (``fatal`` findings) that must block training:

* patient leakage between splits,
* duplicate image content across splits,
* labels that cannot be interpreted,
* missing files beyond the configured tolerance,
* split sizes that do not match the requested configuration.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from .adapter import DatasetConfig, PatientRecord

# Expected FDI quadrants per canonical view.  Used only as a *cross-check* of the
# filename-derived view label against the annotation content; it never rewrites a
# label.
VIEW_QUADRANTS: dict[str, tuple[int, ...]] = {
    "frontal": (1, 2, 3, 4),
    "left_buccal": (2, 3),
    "right_buccal": (1, 4),
    "upper_occlusal": (1, 2),
    "lower_occlusal": (3, 4),
}


@dataclass
class Finding:
    severity: str  # "fatal" | "warning" | "info"
    code: str
    message: str
    items: list[str] = field(default_factory=list)
    # Image ids the finding concerns, when it is per-image.  `items` is
    # human-readable and carries the measured value, so it cannot be matched against
    # `AuditReport.unusable_images`; this can.  Empty means "not attributable to
    # specific images", which the gate treats as unhandled.
    image_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "count": len(self.items),
            "items": self.items[:200],
            "items_truncated": len(self.items) > 200,
        }


@dataclass
class Repair:
    scope: str  # "instance" | "image" | "patient"
    target: str
    action: str
    reason: str


@dataclass
class AuditReport:
    dataset: str
    stats: dict[str, Any]
    findings: list[Finding]
    repairs: list[Repair]
    # image ids whose defect makes them unusable, written to
    # derived/unusable_images.json for the split builder
    unusable_images: dict[str, str] = field(default_factory=dict)

    @property
    def fatal(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "fatal"]

    @property
    def fatal_handled(self) -> list[Finding]:
        """Fatal findings whose every image is already excluded from the splits.

        The audit both flags a defect and records the affected images in
        ``unusable_images``, which the split builder consults, so such a finding is
        evidence that the exclusion machinery worked -- not a reason to stop the
        pipeline.  Only a fatal finding that is *not* covered there means the data
        would reach training unhandled.
        """
        return [
            f
            for f in self.fatal
            if f.image_ids and all(i in self.unusable_images for i in f.image_ids)
        ]

    @property
    def fatal_unhandled(self) -> list[Finding]:
        handled = {id(f) for f in self.fatal_handled}
        return [f for f in self.fatal if id(f) not in handled]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "stats": self.stats,
            "findings": [f.to_dict() for f in self.findings],
            "n_repairs": len(self.repairs),
        }


def audit(
    patients: list[PatientRecord],
    cfg: DatasetConfig,
    checksums: dict[str, str] | None = None,
    split_map: dict[str, str] | None = None,
) -> AuditReport:
    findings: list[Finding] = []
    repairs: list[Repair] = []
    checksums = checksums or {}
    split_map = split_map or {}

    allowed_fdi = set(cfg.fdi_labels)
    min_area_warn = float(cfg.audit.get("min_instance_area_warn", 0) or 0)
    tolerance = float(cfg.audit.get("missing_file_tolerance", 0.0) or 0.0)
    overshoot_tol = int(cfg.audit.get("contour_overshoot_tolerance_px", 8) or 0)
    conflict_thr = float(cfg.audit.get("view_quadrant_conflict_threshold", 0.40))
    # image ids whose defect makes the image unusable; written out so the split
    # builder can exclude their patients without re-deriving the evidence
    unusable_images: dict[str, str] = {}

    # ---------------------------------------------------------------- counters
    n_patients = len(patients)
    images_per_patient: collections.Counter[int] = collections.Counter()
    view_counts: collections.Counter[str] = collections.Counter()
    resolutions: collections.Counter[tuple[int, int]] = collections.Counter()
    fdi_counts: collections.Counter[int] = collections.Counter()
    instances_per_view: dict[str, list[int]] = collections.defaultdict(list)

    empty_dirs: list[str] = []
    missing_images: list[str] = []
    missing_annotations: list[str] = []
    unreadable_images: list[str] = []
    empty_annotations: list[str] = []
    unparsable_annotations: list[str] = []
    duplicate_fdi: list[str] = []
    invalid_fdi: list[str] = []
    invalid_fdi_ids: list[str] = []
    degenerate_instances: list[str] = []
    tiny_instances: list[str] = []
    out_of_bounds: list[str] = []
    out_of_bounds_severe: list[str] = []
    out_of_bounds_severe_ids: list[str] = []
    view_mismatch: list[str] = []
    quadrant_conflict: list[str] = []
    incomplete_patients: list[str] = []
    n_expected_files = 0
    n_missing_files = 0
    n_instances_total = 0

    for patient in patients:
        present = 0
        for view in cfg.views_required:
            rec = patient.images.get(view)
            n_expected_files += 2  # one image + one annotation
            if rec is None:
                n_missing_files += 2
                missing_images.append(f"{patient.patient_key}/{view}")
                missing_annotations.append(f"{patient.patient_key}/{view}")
                continue
            if rec.image_path is None:
                n_missing_files += 1
                missing_images.append(f"{patient.patient_key}/{view}")
            else:
                present += 1
                view_counts[view] += 1
                if rec.width is None or rec.height is None:
                    unreadable_images.append(f"{patient.patient_key}/{view}")
                else:
                    resolutions[(rec.width, rec.height)] += 1
            if rec.annotation_path is None:
                n_missing_files += 1
                missing_annotations.append(f"{patient.patient_key}/{view}")
                continue

            if not rec.instances:
                # distinguishes "file unreadable" from "genuinely zero teeth"
                empty_annotations.append(f"{patient.patient_key}/{view}")

            instances_per_view[view].append(len(rec.instances))
            n_instances_total += len(rec.instances)

            seen_fdi: set[int] = set()
            for inst in rec.instances:
                # An instance whose FDI code is blank or outside the vocabulary cannot be
                # scored: the primary metric needs the code to decide a true positive.
                # Excluding the image here -- rather than only reporting it -- is what
                # lets the exclusion machinery handle the finding, exactly as the severe
                # contour overshoot below does.
                if inst.fdi is None:
                    invalid_fdi.append(
                        f"{patient.patient_key}/{view}#{inst.appearance_idx}"
                        f"={inst.raw_fdi}"
                    )
                    invalid_fdi_ids.append(rec.image_id)
                    unusable_images.setdefault(rec.image_id, "uninterpretable_fdi")
                else:
                    fdi_counts[inst.fdi] += 1
                    if inst.fdi not in allowed_fdi:
                        invalid_fdi.append(
                            f"{patient.patient_key}/{view}#{inst.appearance_idx}"
                            f"={inst.fdi}"
                        )
                        invalid_fdi_ids.append(rec.image_id)
                        unusable_images.setdefault(
                            rec.image_id, f"fdi_outside_vocabulary_{inst.fdi}"
                        )
                    if inst.fdi in seen_fdi:
                        duplicate_fdi.append(
                            f"{patient.patient_key}/{view}#{inst.appearance_idx}"
                            f"=FDI{inst.fdi}"
                        )
                    seen_fdi.add(inst.fdi)

                if inst.is_degenerate:
                    target = f"{patient.patient_key}/{view}#{inst.appearance_idx}"
                    degenerate_instances.append(target)
                    repairs.append(
                        Repair(
                            scope="instance",
                            target=target,
                            action="drop_instance",
                            reason="empty contour and zero pixel area",
                        )
                    )
                    continue

                if min_area_warn and inst.pixel_area < min_area_warn:
                    tiny_instances.append(
                        f"{patient.patient_key}/{view}#{inst.appearance_idx}"
                        f"={inst.pixel_area:.0f}px"
                    )

                if rec.width and rec.height:
                    overshoot = 0
                    for contour in inst.contours:
                        for x, y in contour:
                            overshoot = max(
                                overshoot,
                                x - (rec.width - 1),
                                y - (rec.height - 1),
                                -x,
                                -y,
                            )
                    if overshoot > 0:
                        target = (
                            f"{patient.patient_key}/{view}#{inst.appearance_idx}"
                            f" overshoot={overshoot}px"
                        )
                        if overshoot > overshoot_tol:
                            out_of_bounds_severe.append(target)
                            out_of_bounds_severe_ids.append(rec.image_id)
                            unusable_images[rec.image_id] = (
                                f"contour_overshoot_{overshoot}px"
                            )
                        else:
                            out_of_bounds.append(target)

            # stale `case_name` view token vs the authoritative filename token
            if rec.case_name:
                token = rec.case_name.split("_")[0]
                if token != rec.view_token:
                    view_mismatch.append(
                        f"{patient.patient_key}/{view}: filename={rec.view_token} "
                        f"case_name={token}"
                    )

            if cfg.audit.get("view_quadrant_check", True) and rec.instances:
                agreement = _quadrant_agreement(rec.view_label, rec.instances)
                threshold = float(cfg.audit.get("view_quadrant_min_agreement", 0.6))
                if agreement is not None and agreement < threshold:
                    quadrant_conflict.append(
                        f"{patient.patient_key}/{view}: expected-quadrant share "
                        f"{agreement:.2f} < {threshold:.2f}"
                    )
                    if agreement < conflict_thr:
                        unusable_images[rec.image_id] = (
                            f"view_quadrant_conflict_{agreement:.2f}"
                        )
                        repairs.append(
                            Repair(
                                scope="image",
                                target=f"{patient.patient_key}/{view}",
                                action="review_view_label",
                                reason=(
                                    "annotation quadrants indicate the opposite "
                                    f"side (share {agreement:.2f})"
                                ),
                            )
                        )

        images_per_patient[present] += 1
        if present == 0:
            empty_dirs.append(patient.patient_key)
        if not patient.is_complete(cfg.views_required):
            incomplete_patients.append(patient.patient_key)
            repairs.append(
                Repair(
                    scope="patient",
                    target=patient.patient_key,
                    action="exclude_from_experiments",
                    reason="incomplete five-view set",
                )
            )

    # -------------------------------------------------------------- duplicates
    by_checksum: dict[str, list[str]] = collections.defaultdict(list)
    for image_id, digest in checksums.items():
        if digest:
            by_checksum[digest].append(image_id)
    duplicate_content = {k: v for k, v in by_checksum.items() if len(v) > 1}

    cross_split_dupes: list[str] = []
    for digest, image_ids in duplicate_content.items():
        splits = {split_map.get(i.split("_")[0], "unassigned") for i in image_ids}
        splits.discard("excluded")
        splits.discard("unassigned")
        if len(splits) > 1:
            cross_split_dupes.append(f"{digest[:12]}: {sorted(image_ids)}")

    # ---------------------------------------------------------------- findings
    def add(
        severity: str,
        code: str,
        message: str,
        items: list[str],
        image_ids: list[str] | None = None,
    ) -> None:
        if items:
            findings.append(Finding(severity, code, message, items, image_ids or []))

    add("info", "empty_patient_directory", "patient directory contains no files", empty_dirs)
    add("warning", "missing_image", "expected image file absent", missing_images)
    add(
        "warning",
        "missing_annotation",
        "expected annotation file absent",
        missing_annotations,
    )
    add("fatal", "unreadable_image", "image header could not be parsed", unreadable_images)
    add(
        "fatal",
        "unparsable_annotation",
        "annotation file could not be parsed",
        unparsable_annotations,
    )
    add("warning", "empty_annotation", "annotation contains zero instances", empty_annotations)
    add(
        "fatal",
        "uninterpretable_fdi",
        "FDI label is missing or outside the configured label set",
        invalid_fdi,
        invalid_fdi_ids,
    )
    add(
        "warning",
        "duplicate_fdi_in_image",
        "the same FDI code appears on more than one instance in an image",
        duplicate_fdi,
    )
    add(
        "warning",
        "degenerate_instance",
        "instance has an empty contour and zero area",
        degenerate_instances,
    )
    add(
        "info",
        "tiny_instance",
        f"instance smaller than {min_area_warn:.0f} px",
        tiny_instances,
    )
    add(
        "warning",
        "contour_out_of_bounds",
        f"contour point lies up to {overshoot_tol} px outside the image (rounding; "
        "clipped at rasterisation)",
        out_of_bounds,
    )
    add(
        "fatal",
        "contour_out_of_bounds_severe",
        f"contour exceeds the image by more than {overshoot_tol} px, so the "
        "annotation was made against a differently sized image",
        out_of_bounds_severe,
        out_of_bounds_severe_ids,
    )
    add(
        "warning",
        "stale_case_name_view",
        "case_name view token disagrees with the filename view token; the "
        "filename is authoritative (verified against FDI quadrant composition)",
        view_mismatch,
    )
    add(
        "warning",
        "view_quadrant_conflict",
        "annotation quadrant composition does not support the filename view "
        "label; needs human review",
        quadrant_conflict,
    )
    add(
        "info",
        "incomplete_patient",
        "patient does not have the complete required view set",
        incomplete_patients,
    )
    add(
        "warning",
        "duplicate_image_content",
        "identical image checksum on more than one image",
        [f"{k[:12]}: {sorted(v)}" for k, v in duplicate_content.items()],
    )
    for digest, image_ids in duplicate_content.items():
        # keep the lowest patient id of a duplicate group; identical pixels cannot
        # be two different patients, and keeping one copy also removes any risk of
        # the same content landing in two splits
        for image_id in sorted(image_ids)[1:]:
            unusable_images.setdefault(image_id, f"duplicate_of_{sorted(image_ids)[0]}")
    add(
        "fatal",
        "duplicate_content_across_splits",
        "identical image content appears in more than one split",
        cross_split_dupes,
    )

    if n_expected_files:
        missing_frac = n_missing_files / n_expected_files
        if missing_frac > tolerance:
            findings.append(
                Finding(
                    "fatal",
                    "missing_files_over_tolerance",
                    f"{n_missing_files}/{n_expected_files} expected files missing "
                    f"({missing_frac:.4f} > tolerance {tolerance:.4f})",
                    [],
                )
            )

    if split_map:
        leak = _leakage(split_map)
        if leak:
            findings.append(
                Finding("fatal", "patient_leakage", "patient assigned to >1 split", leak)
            )

    # ------------------------------------------------------------------- stats
    stats: dict[str, Any] = {
        "n_patient_directories": n_patients,
        "n_complete_patients": sum(
            1 for p in patients if p.is_complete(cfg.views_required)
        ),
        "n_incomplete_patients": len(incomplete_patients),
        "n_empty_directories": len(empty_dirs),
        "n_images": sum(view_counts.values()),
        "n_annotations": sum(len(v) for v in instances_per_view.values()),
        "n_instances": n_instances_total,
        "images_per_patient": dict(sorted(images_per_patient.items())),
        "images_per_view": dict(sorted(view_counts.items())),
        "instances_per_view": {
            view: {
                "n_images": len(values),
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "mean": round(sum(values) / len(values), 3) if values else 0.0,
            }
            for view, values in sorted(instances_per_view.items())
        },
        "fdi_distribution": {str(k): v for k, v in sorted(fdi_counts.items())},
        "n_distinct_fdi": len(fdi_counts),
        "n_distinct_resolutions": len(resolutions),
        "resolution_top20": [
            {"width": w, "height": h, "count": c}
            for (w, h), c in resolutions.most_common(20)
        ],
        "resolution_extremes": _resolution_extremes(resolutions),
        "n_expected_files": n_expected_files,
        "n_missing_files": n_missing_files,
        "split_sizes": collections.Counter(split_map.values()) if split_map else {},
    }

    report = AuditReport(
        dataset=cfg.name, stats=stats, findings=findings, repairs=repairs
    )
    report.unusable_images = unusable_images
    return report


def _quadrant_agreement(view_label: str, instances) -> float | None:
    expected = VIEW_QUADRANTS.get(view_label)
    if not expected:
        return None
    quads = [i.quadrant for i in instances if i.quadrant is not None]
    if not quads:
        return None
    hits = sum(1 for q in quads if q in expected)
    return hits / len(quads)


def _leakage(split_map: dict[str, str]) -> list[str]:
    # split_map is patient -> single split, so leakage can only be introduced by
    # a caller merging maps; check defensively.
    seen: dict[str, set[str]] = collections.defaultdict(set)
    for pid, split in split_map.items():
        seen[pid].add(split)
    return sorted(pid for pid, splits in seen.items() if len(splits) > 1)


def _resolution_extremes(resolutions: collections.Counter) -> dict[str, Any]:
    if not resolutions:
        return {}
    keys = list(resolutions)
    areas = {k: k[0] * k[1] for k in keys}
    smallest = min(keys, key=lambda k: areas[k])
    largest = max(keys, key=lambda k: areas[k])
    return {
        "smallest": {"width": smallest[0], "height": smallest[1]},
        "largest": {"width": largest[0], "height": largest[1]},
    }


def render_markdown(report: AuditReport) -> str:
    lines = [f"# Dataset audit - `{report.dataset}`", ""]
    s = report.stats
    lines += [
        "## Cohort",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| patient directories | {s['n_patient_directories']} |",
        f"| complete patients | {s['n_complete_patients']} |",
        f"| incomplete patients | {s['n_incomplete_patients']} |",
        f"| empty directories | {s['n_empty_directories']} |",
        f"| images | {s['n_images']} |",
        f"| annotation files | {s['n_annotations']} |",
        f"| tooth instances | {s['n_instances']} |",
        f"| distinct FDI codes | {s['n_distinct_fdi']} |",
        f"| distinct resolutions | {s['n_distinct_resolutions']} |",
        "",
        "## Images per view",
        "",
        "| view | images | min | max | mean instances |",
        "| --- | --- | --- | --- | --- |",
    ]
    for view, stat in s["instances_per_view"].items():
        lines.append(
            f"| {view} | {stat['n_images']} | {stat['min']} | {stat['max']} | "
            f"{stat['mean']} |"
        )
    lines += ["", "## Findings", ""]
    if not report.findings:
        lines.append("No findings.")
    else:
        lines += ["| severity | code | count | message |", "| --- | --- | --- | --- |"]
        for f in sorted(report.findings, key=lambda f: ("fatal", "warning", "info").index(f.severity)):
            lines.append(f"| {f.severity} | `{f.code}` | {len(f.items)} | {f.message} |")
        lines += ["", "### Detail", ""]
        for f in report.findings:
            lines.append(f"#### `{f.code}` ({len(f.items)})")
            lines.append("")
            for item in f.items[:50]:
                lines.append(f"- {item}")
            if len(f.items) > 50:
                lines.append(f"- ... {len(f.items) - 50} more")
            lines.append("")
    lines += [
        "## Proposed repairs",
        "",
        f"{len(report.repairs)} proposed operations; see `proposed_repairs.csv`. "
        "Repairs are applied only with an explicit `--apply-repairs` flag.",
        "",
    ]
    return "\n".join(lines)
