"""FDI label handling, view vocabularies and the SegmentAnyTooth class mapping.

Five different view vocabularies exist in this project (source filename token,
canonical label, SegmentAnyTooth view, classifier class index, FDI quadrant), and a
mismatch between any two of them would silently corrupt every result, so each
mapping is pinned by a test.
"""

from __future__ import annotations

import pytest

from iop_compass.classification.dataset import VIEW_CLASSES, label_to_index
from iop_compass.data.adapter import ToothInstance, load_dataset_config, parse_fdi
from iop_compass.data.validation import VIEW_QUADRANTS
from iop_compass.segmentation.mask_rcnn import mirror_fdi
from iop_compass.segmentation.segmentanytooth_adapter import LEFT_CLASSES, VIEW_TO_SAT

REAL_CONFIGS = ("configs/dataset_final_1000.yaml",)


def test_canonical_view_labels_agree_across_modules():
    assert set(VIEW_CLASSES) == set(VIEW_TO_SAT)
    assert set(VIEW_CLASSES) == set(VIEW_QUADRANTS)
    for view in VIEW_CLASSES:
        assert 0 <= label_to_index(view) < len(VIEW_CLASSES)


def test_sat_view_names_are_the_upstream_ones():
    assert VIEW_TO_SAT == {
        "frontal": "front",
        "left_buccal": "left",
        "right_buccal": "right",
        "upper_occlusal": "upper",
        "lower_occlusal": "lower",
    }


def test_sat_left_class_list_decodes_to_valid_fdi():
    assert len(LEFT_CLASSES) == 24
    for name in LEFT_CLASSES:
        fdi = parse_fdi(name[-2:])
        assert fdi is not None, name
    # the left detector is the right model, so it can only emit quadrants 1-4
    quadrants = {parse_fdi(name[-2:]) // 10 for name in LEFT_CLASSES}
    assert quadrants <= {1, 2, 3, 4}


@pytest.mark.parametrize("config_path", REAL_CONFIGS)
def test_real_configs_declare_the_permanent_dentition(config_path):
    cfg = load_dataset_config(config_path)
    assert len(cfg.fdi_labels) == 32
    expected = tuple(
        quadrant * 10 + position
        for quadrant in (1, 2, 3, 4)
        for position in range(1, 9)
    )
    assert cfg.fdi_labels == expected
    assert len(cfg.fdi_index) == 32
    assert min(cfg.fdi_index.values()) == 1  # 0 stays free for background


@pytest.mark.parametrize("config_path", REAL_CONFIGS)
def test_real_configs_agree_on_the_view_vocabulary(config_path):
    cfg = load_dataset_config(config_path)
    assert set(cfg.view_map.values()) == set(VIEW_CLASSES)
    assert set(cfg.views_required) == set(VIEW_CLASSES)
    assert set(cfg.token_for_view) == set(VIEW_CLASSES)


def test_quadrant_expectations_match_clinical_anatomy():
    # the maxillary occlusal view shows only maxillary quadrants, and so on
    assert VIEW_QUADRANTS["upper_occlusal"] == (1, 2)
    assert VIEW_QUADRANTS["lower_occlusal"] == (3, 4)
    assert VIEW_QUADRANTS["left_buccal"] == (2, 3)
    assert VIEW_QUADRANTS["right_buccal"] == (1, 4)
    assert VIEW_QUADRANTS["frontal"] == (1, 2, 3, 4)


def test_arch_is_derived_from_fdi_not_from_the_source_field():
    upper = ToothInstance(1, 17, 10.0, (0, 0, 1, 1), (0.0, 0.0), [])
    lower = ToothInstance(2, 47, 10.0, (0, 0, 1, 1), (0.0, 0.0), [])
    unknown = ToothInstance(3, None, 10.0, (0, 0, 1, 1), (0.0, 0.0), [])
    assert upper.arch == "upper"
    assert lower.arch == "lower"
    assert unknown.arch == "unknown"
    assert upper.quadrant == 1 and lower.quadrant == 4


def test_mirror_fdi_swaps_sides_and_is_an_involution():
    assert mirror_fdi(11) == 21
    assert mirror_fdi(21) == 11
    assert mirror_fdi(46) == 36
    assert mirror_fdi(36) == 46
    for quadrant in (1, 2, 3, 4):
        for position in range(1, 9):
            fdi = quadrant * 10 + position
            assert mirror_fdi(mirror_fdi(fdi)) == fdi
            assert mirror_fdi(fdi) % 10 == position


def test_fdi_class_index_round_trip():
    cfg = load_dataset_config(REAL_CONFIGS[0])
    index_to_fdi = {i: fdi for fdi, i in cfg.fdi_index.items()}
    for fdi, index in cfg.fdi_index.items():
        assert index_to_fdi[index] == fdi
