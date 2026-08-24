"""A frozen ROI artefact may only be used on the split it was fitted on.

Both ROI artefacts are fitted on *train* patients.  A replicate rotates the hold-out,
so reusing another replicate's artefact would fit on patients that are validation or
test patients here -- leakage into a reported number, with nothing in the output to
show it happened.  These tests lock the guard that makes that impossible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iop_compass.roi.factory import (  # noqa: E402
    RoiProvenanceError,
    _check_split,
    geometric_prior_path,
    learned_roi_checkpoint,
)
from iop_compass.roi.geometric_crop import (  # noqa: E402
    GeometricPriorConfig,
    ViewPrior,
    load_priors,
    save_priors,
)


def test_artefact_paths_are_per_run_and_per_replicate():
    class Cfg:
        name = "somerun"

    cfg = Cfg()
    priors = {geometric_prior_path(cfg, r) for r in (1, 2, 3)}
    ckpts = {learned_roi_checkpoint(cfg, r) for r in (1, 2, 3)}
    assert len(priors) == 3, "replicates must not share a geometric prior"
    assert len(ckpts) == 3, "replicates must not share a learned ROI checkpoint"
    for path in priors | ckpts:
        assert "somerun" in str(path), "artefacts must live under their own run"


def test_mismatched_split_is_refused():
    with pytest.raises(RoiProvenanceError):
        _check_split("geometric prior", "aaaa", "bbbb", Path("p"))


def test_matching_split_is_accepted():
    _check_split("geometric prior", "aaaa", "aaaa", Path("p"))


def test_unknown_provenance_does_not_block():
    """Artefacts predating the provenance block stay usable, just unchecked."""
    _check_split("geometric prior", None, "aaaa", Path("p"))
    _check_split("geometric prior", "aaaa", None, Path("p"))


def test_prior_round_trips_with_its_provenance(tmp_path):
    priors = {"frontal": ViewPrior(x0=0.0, y0=0.1, x1=1.0, y1=0.9, n_images=284)}
    cfg = GeometricPriorConfig()
    path = tmp_path / "geometric_prior_r1.json"
    save_priors(
        priors,
        cfg,
        path,
        provenance={"split_hash": "abc123", "replicate": 1, "n_patients": 284},
    )
    loaded, _, provenance = load_priors(path)
    assert loaded["frontal"].n_images == 284
    assert provenance["split_hash"] == "abc123"
    assert provenance["replicate"] == 1


def test_prior_written_without_provenance_is_still_readable(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "config": {},
                "priors": {
                    "frontal": {
                        "x0": 0.0,
                        "y0": 0.0,
                        "x1": 1.0,
                        "y1": 1.0,
                        "n_images": 1,
                    }
                },
            }
        )
    )
    _, _, provenance = load_priors(path)
    assert provenance == {}
