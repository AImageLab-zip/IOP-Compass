"""Deterministic patient-level splits.

All five images of a patient always stay in the same split.  Sizes come from the
config; nothing about the cohort is hard-coded here, so the same code produces
the 495-patient and the 1,000-patient splits.

Seeds select a *rotating hold-out block*: the eligible patients are shuffled once
with the configuration seed, and seed index ``k`` (1-based) takes the ``k``-th block
of ``n_val + n_test`` patients from the end of that order as its validation and test
splits.  Consecutive seeds therefore never reuse a validation or test patient, which
is what makes a mean over seeds a statement about the cohort rather than about one
partition.  Seed index 1 is the canonical partition, so its ``split_hash`` and every
artefact derived from it stay valid.

The rotation only fits if ``n_seeds * (n_val + n_test) <= len(eligible)``;
:func:`check_seed_capacity` says so up front instead of letting the last seed
silently overlap the first.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from .adapter import DatasetConfig, PatientRecord


class SplitError(RuntimeError):
    pass


@dataclass
class SplitAssignment:
    seed: int
    train: list[str]
    val: list[str]
    test: list[str]
    eligible: list[str]
    excluded: dict[str, str]
    # Which rotating hold-out block this assignment used.  Deliberately absent from
    # `hash()`: the hash identifies the partition, and seed index 1 is the canonical
    # partition.
    seed_index: int = 1

    @property
    def mapping(self) -> dict[str, str]:
        out = {}
        for name in ("train", "val", "test"):
            for pid in getattr(self, name):
                out[pid] = name
        for pid in self.excluded:
            out.setdefault(pid, "excluded")
        return out

    def hash(self) -> str:
        payload = json.dumps(
            {
                "seed": self.seed,
                "train": self.train,
                "val": self.val,
                "test": self.test,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "seed_index": self.seed_index,
            "split_hash": self.hash(),
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_eligible": len(self.eligible),
            "n_excluded": len(self.excluded),
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "excluded": self.excluded,
        }


def load_undecodable(cfg: DatasetConfig) -> dict[str, str]:
    """Image ids that failed a full decode, as recorded by verify_decodable.py."""
    path = cfg.derived_root / "undecodable_images.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_unusable_annotations(cfg: DatasetConfig) -> dict[str, str]:
    """Image ids whose annotation has no polygon geometry.

    Recorded by ``scripts/build_manifest.py --reduce-geometry``.  Such a file
    carries FDI codes and bounding boxes but no contour, so no reference mask can
    be built from it and the image cannot take part in a segmentation benchmark.
    """
    path = cfg.derived_root / "unusable_annotations.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_unusable_images(cfg: DatasetConfig) -> dict[str, str]:
    """Image ids the audit judged unusable.

    Written by ``scripts/audit_dataset.py``: contours that exceed the image by more
    than the configured tolerance, buccal views whose annotation quadrants indicate
    the opposite side, and all but the first member of a duplicate-content group.
    """
    path = cfg.derived_root / "unusable_images.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def eligibility(
    patients: list[PatientRecord], cfg: DatasetConfig
) -> tuple[list[str], dict[str, str]]:
    """Split patients into eligible ids and ``{patient_id: reason}`` exclusions."""
    eligible: list[str] = []
    excluded: dict[str, str] = {}
    undecodable = load_undecodable(cfg)
    unusable = load_unusable_annotations(cfg)
    flagged = load_unusable_images(cfg)
    for patient in patients:
        broken = [
            v
            for v in cfg.views_required
            if v in patient.images and patient.images[v].image_id in undecodable
        ]
        geometryless = [
            v
            for v in cfg.views_required
            if v in patient.images and patient.images[v].image_id in unusable
        ]
        audit_flagged = sorted(
            {
                f"{v}({flagged[patient.images[v].image_id].split(':')[0]})"
                for v in cfg.views_required
                if v in patient.images and patient.images[v].image_id in flagged
            }
        )
        missing_images = [
            v for v in cfg.views_required
            if v not in patient.images or patient.images[v].image_path is None
        ]
        missing_ann = [
            v for v in cfg.views_required
            if v not in patient.images or patient.images[v].annotation_path is None
        ]
        unreadable = [
            v for v in cfg.views_required
            if v in patient.images
            and patient.images[v].image_path is not None
            and patient.images[v].width is None
        ]
        if len(missing_images) == len(cfg.views_required):
            excluded[patient.patient_id] = "empty_directory"
        elif broken:
            excluded[patient.patient_id] = "undecodable_images:" + ",".join(broken)
        elif geometryless:
            excluded[patient.patient_id] = "annotations_without_geometry:" + ",".join(
                geometryless
            )
        elif audit_flagged:
            excluded[patient.patient_id] = "audit_flagged:" + ",".join(audit_flagged)
        elif missing_images:
            excluded[patient.patient_id] = "missing_images:" + ",".join(missing_images)
        elif missing_ann:
            excluded[patient.patient_id] = "missing_annotations:" + ",".join(missing_ann)
        elif unreadable:
            excluded[patient.patient_id] = "unreadable_images:" + ",".join(unreadable)
        else:
            eligible.append(patient.patient_id)
    return sorted(eligible), excluded


def check_seed_capacity(n_eligible: int, cfg: DatasetConfig, n_seeds: int) -> None:
    """Raise unless ``n_seeds`` rotating hold-out blocks fit without overlapping.

    Called before any seed is materialised, so an infeasible campaign fails at the
    split step rather than after a day of GPU time.
    """
    spec = cfg.splits
    block = int(spec["n_val"]) + int(spec["n_test"])
    needed = n_seeds * block
    if needed > n_eligible:
        raise SplitError(
            f"{n_seeds} seeds need {needed} patients for disjoint val+test blocks "
            f"({block} per seed) but only {n_eligible} are eligible; reduce the seed "
            f"count or shrink n_val/n_test"
        )


def make_splits(
    patients: list[PatientRecord],
    cfg: DatasetConfig,
    seed_index: int = 1,
) -> SplitAssignment:
    spec = cfg.splits
    seed = int(spec.get("seed", 42))
    n_train = int(spec["n_train"])
    n_val = int(spec["n_val"])
    n_test = int(spec["n_test"])
    require_exact = bool(spec.get("require_exact", True))
    if seed_index < 1:
        raise SplitError(f"seed_index must be at least 1, got {seed_index}")

    eligible, excluded = eligibility(patients, cfg)
    wanted = n_train + n_val + n_test
    if require_exact and len(eligible) != wanted:
        # The likeliest failure on a new cohort: the nominal size is round but some
        # patients are excluded by the audit.  Say what to change instead of only
        # what is wrong, since the hold-out sizes are usually the ones to keep.
        raise SplitError(
            f"config asks for {n_train}/{n_val}/{n_test} = {wanted} patients but "
            f"{len(eligible)} are eligible (excluded: {len(excluded)}); either set "
            f"n_train: {len(eligible) - n_val - n_test} to absorb the shortfall while "
            f"keeping n_val/n_test, or set require_exact: false"
        )
    if len(eligible) < wanted:
        raise SplitError(
            f"only {len(eligible)} eligible patients for a requested {wanted}"
        )

    shuffled = list(eligible)
    random.Random(seed).shuffle(shuffled)

    # Rotating hold-out: block `seed_index` counted back from the end of the shuffled
    # order.  Seed index 1 is the tail block, which is the canonical partition.
    check_seed_capacity(len(shuffled), cfg, seed_index)
    block = n_val + n_test
    hi = len(shuffled) - (seed_index - 1) * block
    lo = hi - block
    holdout = shuffled[lo:hi]
    val = sorted(holdout[:n_val])
    test = sorted(holdout[n_val:])
    # Everything outside the hold-out is available for training; the config decides
    # how much of it is used, so a cohort larger than the requested sizes is trimmed
    # rather than silently inflating the training set.
    train = sorted((shuffled[:lo] + shuffled[hi:])[:n_train])

    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise SplitError(f"patient leakage across splits: {sorted(overlap)}")

    return SplitAssignment(
        seed=seed,
        train=train,
        val=val,
        test=test,
        eligible=eligible,
        excluded=excluded,
        seed_index=seed_index,
    )


def holdout_block(eligible: list[str], cfg: DatasetConfig, seed_index: int) -> list[str]:
    """The val+test patients a seed index claims, without building a full assignment.

    Used by the disjointness tests and by ``create_splits.py`` to report the rotation
    before writing anything.
    """
    spec = cfg.splits
    shuffled = list(eligible)
    random.Random(int(spec.get("seed", 42))).shuffle(shuffled)
    block = int(spec["n_val"]) + int(spec["n_test"])
    hi = len(shuffled) - (seed_index - 1) * block
    return shuffled[hi - block : hi]


def split_paths(out_dir: Path, seed_index: int | None) -> tuple[Path, Path]:
    """Where a seed's split lives.

    ``seed_index=None`` gives the unsuffixed ``splits.json``, which is the canonical
    seed-1 partition; any other index gives ``splits_seed<k>.json``.
    """
    stem = "splits" if seed_index is None else f"splits_seed{seed_index}"
    return out_dir / f"{stem}.json", out_dir / f"{stem}.csv"


def write_splits(
    assignment: SplitAssignment, out_dir: Path, seed_index: int | None = None
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = split_paths(out_dir, seed_index)
    with json_path.open("w") as fh:
        json.dump(assignment.to_dict(), fh, indent=1)
    with csv_path.open("w") as fh:
        fh.write("patient_id,split\n")
        for pid, split in sorted(assignment.mapping.items()):
            fh.write(f"{pid},{split}\n")
    return json_path, csv_path


def apply_seed_split(
    rows: list[dict], derived_root: Path, seed_index: int | None
) -> list[dict]:
    """Rewrite each manifest row's ``split`` from that seed's split file.

    The manifest carries the split column of whichever partition it was built
    against, so a run using a rotated hold-out has to re-derive membership or it
    would evaluate the wrong patients.  Doing it here keeps one manifest for the
    whole campaign.  ``seed_index=None`` leaves the rows untouched.

    Patients absent from the split file are marked ``excluded``, which every caller
    already filters out.
    """
    if seed_index is None:
        return rows
    path, _ = split_paths(derived_root, seed_index)
    if not path.exists():
        raise SplitError(
            f"missing {path}; run scripts/create_splits.py --seeds {seed_index}"
        )
    mapping = load_splits(path)
    for row in rows:
        row["split"] = mapping.get(row["patient_id"], "excluded")
    return rows


def split_hash_for(derived_root: Path, seed_index: int | None) -> str:
    """``split_hash`` recorded for a seed, for provenance in every result file."""
    path, _ = split_paths(derived_root, seed_index)
    if not path.exists() and seed_index is not None:
        raise SplitError(f"missing {path}; run scripts/create_splits.py first")
    return json.loads(path.read_text())["split_hash"]


def load_splits(path: Path) -> dict[str, str]:
    with open(path) as fh:
        payload = json.load(fh)
    mapping: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for pid in payload.get(name, []):
            mapping[pid] = name
    for pid in payload.get("excluded", {}):
        mapping.setdefault(pid, "excluded")
    return mapping
