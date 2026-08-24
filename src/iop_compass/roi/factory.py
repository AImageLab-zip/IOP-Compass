"""Build ROI strategies from the frozen configuration.

One factory is used by the segmentation driver and by the ROI evaluation script so
both see exactly the same frozen priors, checkpoints and prompts.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import FullImageRoi
from .geometric_crop import GeometricRoi, load_priors
from .learned_roi import LearnedRoi, LearnedRoiConfig
from .sam3_roi import Sam3Roi

REPO = Path(__file__).resolve().parents[3]
ALL_STRATEGIES = ("R0_full", "R1_geometric", "R2_learned", "R3_sam3")


class RoiProvenanceError(RuntimeError):
    """A frozen ROI artefact was not fitted on the split it is about to be used on."""


def geometric_prior_path(cfg, replicate: int) -> Path:
    return REPO / "runs" / cfg.name / "roi" / f"geometric_prior_r{replicate}.json"


def learned_roi_checkpoint(cfg, replicate: int) -> Path:
    return REPO / "runs" / cfg.name / "roi_detector" / f"r{replicate}" / "best.pt"


def _check_split(artefact: str, recorded, expected, path) -> None:
    """Refuse an artefact fitted on a different split.

    Both ROI artefacts are fitted on *train* patients.  A replicate rotates the
    hold-out, so an artefact from another replicate was fitted on patients that are
    validation or test patients here -- using it would leak them into the result, with
    nothing in the output to show it happened.
    """
    if expected is None or recorded is None:
        return
    if recorded != expected:
        raise RoiProvenanceError(
            f"{artefact} at {path} was fitted on split {recorded[:12]} but this run "
            f"evaluates split {expected[:12]}; refit it for this replicate "
            f"(its training patients are validation or test patients here)"
        )


def build_strategies(
    cfg,
    names: list[str],
    roi_config: str | Path = "configs/roi/learned_roi.yaml",
    device: str = "cuda",
    replicate: int = 0,
    split_hash: str | None = None,
) -> dict[str, object]:
    payload = yaml.safe_load(Path(roi_config).read_text()) or {}
    wanted = set(names)
    strategies: dict[str, object] = {}

    if "R0_full" in wanted:
        strategies["R0_full"] = FullImageRoi()

    if "R1_geometric" in wanted:
        prior_path = geometric_prior_path(cfg, replicate)
        if not prior_path.exists():
            raise FileNotFoundError(
                f"missing frozen geometric prior {prior_path}; run "
                f"scripts/fit_roi_priors.py --config <cfg> --replicate {replicate}"
            )
        priors, _, provenance = load_priors(prior_path)
        _check_split(
            "geometric prior", provenance.get("split_hash"), split_hash, prior_path
        )
        strategies["R1_geometric"] = GeometricRoi(priors)

    if "R2_learned" in wanted:
        ckpt = learned_roi_checkpoint(cfg, replicate)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"missing learned ROI checkpoint {ckpt}; run "
                f"scripts/train_roi_detector.py --config <cfg> --replicate {replicate}"
            )
        import torch

        recorded = torch.load(str(ckpt), map_location="cpu", weights_only=False).get(
            "split_hash"
        )
        _check_split("learned ROI detector", recorded, split_hash, ckpt)
        strategies["R2_learned"] = LearnedRoi(
            ckpt, LearnedRoiConfig.from_dict(payload.get("learned_roi")), device=device
        )

    if "R3_sam3" in wanted:
        strategies["R3_sam3"] = Sam3Roi(
            params=payload.get("sam3_roi"), device=device, allow_fallback=True
        )

    # preserve the canonical order
    return {name: strategies[name] for name in ALL_STRATEGIES if name in strategies}
