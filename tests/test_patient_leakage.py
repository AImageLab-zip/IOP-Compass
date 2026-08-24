"""Patient-level isolation must be structurally impossible to violate."""

from __future__ import annotations

import pytest

from iop_compass.data.adapter import discover_patients
from iop_compass.data.splits import (
    SplitError,
    check_seed_capacity,
    eligibility,
    holdout_block,
    make_splits,
    write_splits,
)


def test_no_patient_in_two_splits(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    assignment = make_splits(patients, cfg)

    train, val, test = set(assignment.train), set(assignment.val), set(assignment.test)
    assert not train & val
    assert not train & test
    assert not val & test
    assert len(train) + len(val) + len(test) == len(assignment.eligible)


def test_all_five_images_share_a_split(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    assignment = make_splits(patients, cfg)
    mapping = assignment.mapping

    from iop_compass.data.manifest import build_rows

    rows = build_rows(patients, cfg, checksums=False, splits=mapping)
    by_patient: dict[str, set[str]] = {}
    for row in rows:
        by_patient.setdefault(row.patient_id, set()).add(row.split)
    for patient_id, splits in by_patient.items():
        assert len(splits) == 1, f"{patient_id} spread over {splits}"


def test_split_is_deterministic(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    first = make_splits(patients, cfg)
    second = make_splits(patients, cfg)
    assert first.hash() == second.hash()
    assert first.train == second.train


def test_seed_changes_the_split(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    baseline = make_splits(patients, cfg)
    import dataclasses

    other = dataclasses.replace(cfg, splits={**cfg.splits, "seed": 7})
    assert make_splits(patients, other).hash() != baseline.hash()


def test_requested_sizes_are_enforced(synthetic_dataset):
    _, cfg = synthetic_dataset
    import dataclasses

    patients = discover_patients(cfg, load_annotations=False)
    too_big = dataclasses.replace(
        cfg, splits={**cfg.splits, "n_train": 20, "require_exact": True}
    )
    with pytest.raises(SplitError):
        make_splits(patients, too_big)


def test_undecodable_image_excludes_its_patient(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    eligible_before, _ = eligibility(patients, cfg)

    cfg.derived_root.mkdir(parents=True, exist_ok=True)
    victim = patients[0]
    image_id = victim.images["frontal"].image_id
    (cfg.derived_root / "undecodable_images.json").write_text(
        '{"%s": "decode_failed:test"}' % image_id
    )

    eligible_after, excluded = eligibility(patients, cfg)
    assert victim.patient_id not in eligible_after
    assert victim.patient_id in excluded
    assert "undecodable" in excluded[victim.patient_id]
    assert len(eligible_after) == len(eligible_before) - 1


def test_split_files_round_trip(synthetic_dataset, tmp_path):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    assignment = make_splits(patients, cfg)
    json_path, _ = write_splits(assignment, tmp_path / "splits")

    from iop_compass.data.splits import load_splits

    mapping = load_splits(json_path)
    for pid in assignment.train:
        assert mapping[pid] == "train"
    for pid in assignment.test:
        assert mapping[pid] == "test"


def test_replicate_one_reproduces_the_single_partition(synthetic_dataset):
    """The rotation must not move replicate 1.

    Every checkpoint and stored prediction of a run was produced against replicate 1,
    so if its hash moved they would all silently become invalid.
    """
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    assert make_splits(patients, cfg, seed_index=1).hash() == make_splits(
        patients, cfg
    ).hash()


def test_replicates_never_reuse_a_holdout_patient(synthetic_dataset):
    """A patient is a validation or test patient for at most one replicate."""
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    eligible, _ = eligibility(patients, cfg)
    block = cfg.splits["n_val"] + cfg.splits["n_test"]
    n_replicates = len(eligible) // block

    holdouts = []
    for k in range(1, n_replicates + 1):
        assignment = make_splits(patients, cfg, seed_index=k)
        holdouts.append(set(assignment.val) | set(assignment.test))
        # sizes stay as configured for every replicate
        assert len(assignment.val) == cfg.splits["n_val"]
        assert len(assignment.test) == cfg.splits["n_test"]
        assert len(assignment.train) == cfg.splits["n_train"]

    for i in range(len(holdouts)):
        for j in range(i + 1, len(holdouts)):
            assert not holdouts[i] & holdouts[j], f"replicates {i} and {j} overlap"


def test_replicates_produce_different_partitions(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    a = make_splits(patients, cfg, seed_index=1)
    b = make_splits(patients, cfg, seed_index=2)
    assert a.hash() != b.hash()


def test_holdout_block_agrees_with_the_assignment(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    eligible, _ = eligibility(patients, cfg)
    for k in (1, 2):
        assignment = make_splits(patients, cfg, seed_index=k)
        assert set(holdout_block(eligible, cfg, k)) == set(assignment.val) | set(
            assignment.test
        )


def test_too_many_replicates_is_refused_before_any_work(synthetic_dataset):
    """An infeasible rotation must fail at the split step, not after a day of GPU."""
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    eligible, _ = eligibility(patients, cfg)
    block = cfg.splits["n_val"] + cfg.splits["n_test"]
    too_many = len(eligible) // block + 1

    with pytest.raises(SplitError):
        check_seed_capacity(len(eligible), cfg, too_many)
    with pytest.raises(SplitError):
        make_splits(patients, cfg, seed_index=too_many)


def test_replicate_zero_is_refused(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    with pytest.raises(SplitError):
        make_splits(patients, cfg, seed_index=0)


def test_all_five_images_share_a_split_for_every_replicate(synthetic_dataset):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    from iop_compass.data.manifest import build_rows

    for k in (1, 2):
        assignment = make_splits(patients, cfg, seed_index=k)
        rows = build_rows(patients, cfg, checksums=False, splits=assignment.mapping)
        by_patient: dict[str, set[str]] = {}
        for row in rows:
            by_patient.setdefault(row.patient_id, set()).add(row.split)
        for patient_id, splits in by_patient.items():
            assert len(splits) == 1, f"{patient_id} spread over {splits} at replicate {k}"


def test_per_replicate_split_files_do_not_collide(synthetic_dataset, tmp_path):
    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    out = tmp_path / "derived"
    paths = set()
    for k in (1, 2):
        assignment = make_splits(patients, cfg, seed_index=k)
        json_path, csv_path = write_splits(assignment, out, seed_index=k)
        paths.add(json_path)
        assert json_path.exists() and csv_path.exists()
    assert len(paths) == 2

    # the unsuffixed file stays the canonical replicate-1 partition
    canonical, _ = write_splits(make_splits(patients, cfg, seed_index=1), out)
    assert canonical.name == "splits.json"


def test_apply_seed_split_rewrites_manifest_membership(synthetic_dataset, tmp_path):
    """Otherwise a rotated replicate would evaluate the partition of another."""
    from iop_compass.data.splits import apply_seed_split

    _, cfg = synthetic_dataset
    patients = discover_patients(cfg, load_annotations=False)
    out = tmp_path / "derived"
    for k in (1, 2):
        write_splits(make_splits(patients, cfg, seed_index=k), out, seed_index=k)

    assignment_2 = make_splits(patients, cfg, seed_index=2)
    # rows carrying replicate 1's membership
    rows = [
        {"patient_id": pid, "split": split}
        for pid, split in make_splits(patients, cfg, seed_index=1).mapping.items()
    ]
    rewritten = apply_seed_split(rows, out, 2)
    got = {r["patient_id"]: r["split"] for r in rewritten}
    for pid in assignment_2.test:
        assert got[pid] == "test"
    for pid in assignment_2.val:
        assert got[pid] == "val"
