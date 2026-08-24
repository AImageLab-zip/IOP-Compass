"""The audit gate must distinguish handled from unhandled fatal findings.

A fatal finding whose images the audit recorded in ``unusable_images`` is already
excluded from every split, so it is evidence the exclusion machinery worked.  The
gate used to fail on any fatal finding at all, which stopped the whole pipeline on
data defects it had itself already handled.
"""

from __future__ import annotations

from iop_compass.data.validation import AuditReport, Finding


def _report(findings: list[Finding], unusable: dict[str, str]) -> AuditReport:
    report = AuditReport(dataset="t", stats={}, findings=findings, repairs=[])
    report.unusable_images = unusable
    return report


def test_fatal_finding_covered_by_exclusions_is_handled():
    finding = Finding(
        "fatal",
        "contour_out_of_bounds_severe",
        "msg",
        ["Patient_105/right_buccal#5 overshoot=161px"],
        ["P0105_right_buccal"],
    )
    report = _report([finding], {"P0105_right_buccal": "contour_overshoot_161px"})
    assert report.fatal == [finding]
    assert report.fatal_handled == [finding]
    assert report.fatal_unhandled == []


def test_partially_covered_fatal_finding_is_unhandled():
    """One uncovered image is enough: that image would reach training."""
    finding = Finding(
        "fatal", "contour_out_of_bounds_severe", "msg", ["a", "b"],
        ["P0105_right_buccal", "P0999_frontal"],
    )
    report = _report([finding], {"P0105_right_buccal": "contour_overshoot_161px"})
    assert report.fatal_handled == []
    assert report.fatal_unhandled == [finding]


def test_fatal_finding_without_image_ids_is_unhandled():
    """Not attributable to specific images, so the gate cannot clear it."""
    finding = Finding("fatal", "unreadable_image", "msg", ["something"])
    report = _report([finding], {"P0105_right_buccal": "x"})
    assert report.fatal_unhandled == [finding]


def test_warnings_are_never_fatal():
    report = _report([Finding("warning", "duplicate_fdi_in_image", "msg", ["a"], ["A"])], {})
    assert report.fatal == []
    assert report.fatal_unhandled == []


def test_identical_findings_are_counted_separately():
    """`fatal_unhandled` must not collapse two findings that compare equal."""
    covered = Finding("fatal", "c", "m", ["a"], ["P1_frontal"])
    uncovered = Finding("fatal", "c", "m", ["a"], ["P2_frontal"])
    report = _report([covered, uncovered], {"P1_frontal": "x"})
    assert report.fatal_handled == [covered]
    assert len(report.fatal_unhandled) == 1
    assert report.fatal_unhandled[0].image_ids == ["P2_frontal"]
