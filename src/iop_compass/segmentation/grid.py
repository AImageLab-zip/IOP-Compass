"""The segmentation benchmark grid: ROI x segmenter x post-processing.

Three axes, fully crossed, sixteen cells:

============  ================================================================
axis          levels
============  ================================================================
ROI           ``no-roi`` | ``geometric`` | ``fast-r`` | ``sam3``
segmenter     ``sat`` | ``mask-rcnn``
post          ``no-post`` | ``post``
============  ================================================================

A cell id is the three levels joined by ``+``, e.g. ``geometric+sat+post``.  This
module is the single source of truth for the set: the submitter, the inference
driver, the evaluator and the report all enumerate cells from here rather than from
hard-coded strings.

The ROI is applied the same way in every cell that has one: crop to the ROI box,
run the segmenter on the crop, paste the masks back.  Neither segmenter is
retrained per ROI level, so the axis measures the ROI source and nothing else.

``post`` is a pure function of the ``no-post`` masks plus the frozen
post-processing configuration, so only the eight ``no-post`` cells need a forward
pass; the ``post`` half is derived by ``scripts/repostprocess.py``.
"""

from __future__ import annotations

SEP = "+"

ROI_LEVELS = ("no-roi", "geometric", "fast-r", "sam3")
SEG_LEVELS = ("sat", "mask-rcnn")
POST_LEVELS = ("no-post", "post")

LABELS = {
    "no-roi": "No ROI (full image)",
    "geometric": "Geometric prior",
    "fast-r": "FAST-R (Faster R-CNN box)",
    "sam3": "SAM 3 concept prompts",
    "sat": "SegmentAnyTooth",
    "mask-rcnn": "Mask R-CNN R50-FPN (in-dataset)",
    "no-post": "off",
    "post": "on",
}

# Level -> the roi.factory strategy key recorded in the `strategy` column of every
# ROI result.
ROI_STRATEGY = {
    "no-roi": "R0_full",
    "geometric": "R1_geometric",
    "fast-r": "R2_learned",
    "sam3": "R3_sam3",
}
STRATEGY_ROI = {v: k for k, v in ROI_STRATEGY.items()}

# Only the SAT path applies the upstream photometric ROI guidance; it is a
# SegmentAnyTooth preprocessing trick, not a property of the ROI.
ROI_GUIDANCE = {"sat": True, "mask-rcnn": False}


class GridError(ValueError):
    pass


def cell_id(roi: str, seg: str, post: str | bool) -> str:
    """Build a cell id, validating every level."""
    post = _post_level(post)
    if roi not in ROI_LEVELS:
        raise GridError(f"unknown ROI level {roi!r}; expected one of {ROI_LEVELS}")
    if seg not in SEG_LEVELS:
        raise GridError(f"unknown segmenter {seg!r}; expected one of {SEG_LEVELS}")
    return SEP.join((roi, seg, post))


def parse_cell(cell: str) -> tuple[str, str, str]:
    """Split a cell id into ``(roi, seg, post)``, validating every level."""
    parts = cell.split(SEP)
    if len(parts) != 3:
        raise GridError(
            f"malformed cell id {cell!r}; expected roi{SEP}segmenter{SEP}post"
        )
    roi, seg, post = parts
    if roi not in ROI_LEVELS:
        raise GridError(f"unknown ROI level {roi!r} in {cell!r}")
    if seg not in SEG_LEVELS:
        raise GridError(f"unknown segmenter {seg!r} in {cell!r}")
    if post not in POST_LEVELS:
        raise GridError(f"unknown post level {post!r} in {cell!r}")
    return roi, seg, post


def all_cells() -> list[str]:
    """All sixteen cells in canonical order: ROI outer, then segmenter, then post."""
    return [
        SEP.join((roi, seg, post))
        for roi in ROI_LEVELS
        for seg in SEG_LEVELS
        for post in POST_LEVELS
    ]


def raw_cells() -> list[str]:
    """The eight cells that need a forward pass; the rest are derived from these."""
    return [c for c in all_cells() if parse_cell(c)[2] == "no-post"]


def post_pairs() -> list[tuple[str, str]]:
    """``(no-post cell, post cell)`` for every ROI x segmenter combination."""
    return [
        (SEP.join((roi, seg, "no-post")), SEP.join((roi, seg, "post")))
        for roi in ROI_LEVELS
        for seg in SEG_LEVELS
    ]


def needs_trained_checkpoint(seg: str) -> bool:
    """Whether a segmenter is trained in-dataset and so has a per-replicate weight set.

    SegmentAnyTooth is frozen upstream and has no checkpoint of ours; Mask R-CNN is
    trained per replicate.  Both still vary between replicates, because a replicate
    also rotates the split.
    """
    if seg not in SEG_LEVELS:
        raise GridError(f"unknown segmenter {seg!r}")
    return seg == "mask-rcnn"


def prediction_dir_name(cell: str, replicate: int) -> str:
    """Directory under ``runs/<run>/predictions/<split>/`` for a cell and replicate.

    Every cell carries the replicate suffix, including the deterministic
    SegmentAnyTooth ones: a replicate rotates the validation/test hold-out, so the
    same cell evaluated under two replicates is two different measurements and must
    not share a directory.
    """
    parse_cell(cell)
    if replicate is None or int(replicate) < 1:
        raise GridError(f"cell {cell!r} needs a positive replicate index")
    return f"{cell}{SEP}r{int(replicate)}"


def result_name(cell: str, replicate: int) -> str:
    """Stem of the per-cell metrics JSON written by the evaluator."""
    return prediction_dir_name(cell, replicate)


def parse_result_name(name: str) -> tuple[str, int] | None:
    """Split ``<cell>+r<k>`` back into ``(cell, replicate)``, or ``None`` if ``name``
    is not a grid result name."""
    marker = f"{SEP}r"
    idx = name.rfind(marker)
    if idx < 0:
        return None
    cell, suffix = name[:idx], name[idx + len(marker) :]
    if not suffix.isdigit():
        return None
    try:
        parse_cell(cell)
    except GridError:
        return None
    return cell, int(suffix)


def roi_strategy(cell_or_level: str) -> str:
    """The roi.factory key for a cell id or a bare ROI level."""
    level = (
        cell_or_level
        if cell_or_level in ROI_LEVELS
        else parse_cell(cell_or_level)[0]
    )
    return ROI_STRATEGY[level]


def strategies_for(cells: list[str]) -> list[str]:
    """The roi.factory keys needed by a set of cells, in canonical ROI order."""
    wanted = {parse_cell(c)[0] for c in cells}
    return [ROI_STRATEGY[level] for level in ROI_LEVELS if level in wanted]


def label(cell: str) -> str:
    """Human-readable cell label for the report."""
    roi, seg, post = parse_cell(cell)
    return f"{LABELS[roi]} + {LABELS[seg]}, post {LABELS[post]}"


def _post_level(post: str | bool) -> str:
    if isinstance(post, bool):
        return "post" if post else "no-post"
    if post not in POST_LEVELS:
        raise GridError(f"unknown post level {post!r}; expected one of {POST_LEVELS}")
    return post
