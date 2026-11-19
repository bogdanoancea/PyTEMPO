"""Resolving a filter mapping to the service's own value identifiers."""

from __future__ import annotations

import pytest

from tempo_ins._get import resolve_filters
from tempo_ins.errors import TempoInputError

METADATA = {
    "dimensionsMap": [
        {
            "label": "Judete",
            "options": [
                {"nomItemId": 1, "label": "Cluj"},
                {"nomItemId": 2, "label": "Bihor"},
                {"nomItemId": 3, "label": "Bucuresti"},
            ],
        },
        {
            "label": "Ani",
            "options": [
                {"nomItemId": 10, "label": "Anul 2019"},
                {"nomItemId": 11, "label": "Anul 2020"},
                {"nomItemId": 12, "label": "Anul 2021"},
            ],
        },
    ],
    "details": {"matMaxDim": 2, "matUMSpec": 0},
}


def test_no_filters_selects_everything():
    assert resolve_filters(None, METADATA, "X") == [
        ["1", "2", "3"],
        ["10", "11", "12"],
    ]
    assert resolve_filters({}, METADATA, "X") == [
        ["1", "2", "3"],
        ["10", "11", "12"],
    ]


def test_labels_are_matched_exactly_ignoring_case_and_space():
    selection = resolve_filters({"Judete": ["cluj", "  Bihor "]}, METADATA, "X")

    assert selection[0] == ["1", "2"]
    assert selection[1] == ["10", "11", "12"]


def test_a_year_matches_its_label_as_a_whole_token():
    selection = resolve_filters({"Ani": range(2019, 2021)}, METADATA, "X")
    assert selection[1] == ["10", "11"]


def test_service_identifiers_can_be_used_directly():
    assert resolve_filters({"Ani": "12"}, METADATA, "X")[1] == ["12"]


def test_dimensions_can_be_named_partially_or_by_position():
    assert resolve_filters({"jude": "Cluj"}, METADATA, "X")[0] == ["1"]
    assert resolve_filters({2: 2021}, METADATA, "X")[1] == ["12"]
    assert resolve_filters({"2": 2021}, METADATA, "X")[1] == ["12"]


def test_a_none_filter_means_the_whole_dimension():
    assert resolve_filters({"Judete": None}, METADATA, "X")[0] == ["1", "2", "3"]


def test_an_unmatched_value_is_an_error_that_lists_what_is_available():
    with pytest.raises(TempoInputError, match="Available values") as caught:
        resolve_filters({"Judete": "Atlantis"}, METADATA, "X")

    # The message names the offending value, not just the dimension.
    assert "Atlantis" in str(caught.value)


def test_an_unknown_dimension_is_an_error_that_lists_the_real_ones():
    with pytest.raises(TempoInputError, match="Available dimensions"):
        resolve_filters({"Nowhere": "x"}, METADATA, "X")


def test_filters_must_be_a_mapping_with_usable_keys():
    with pytest.raises(TempoInputError, match="must be a mapping"):
        resolve_filters(["Cluj"], METADATA, "X")

    with pytest.raises(TempoInputError, match="selects no values"):
        resolve_filters({"Judete": []}, METADATA, "X")

    with pytest.raises(TempoInputError, match="must each name a dimension"):
        resolve_filters({"": "Cluj"}, METADATA, "X")


def test_partial_matching_does_not_mask_a_wrong_token():
    # "202" must not quietly match "Anul 2020"; whole tokens only.
    with pytest.raises(TempoInputError):
        resolve_filters({"Ani": "202"}, METADATA, "X")


def test_an_ambiguous_dimension_name_asks_for_a_position():
    ambiguous = {
        "dimensionsMap": [
            {"label": "Ani", "options": [{"nomItemId": 1, "label": "a"}]},
            {"label": "ani", "options": [{"nomItemId": 2, "label": "b"}]},
        ],
        "details": {"matMaxDim": 2, "matUMSpec": 0},
    }

    with pytest.raises(TempoInputError, match="use the position instead"):
        resolve_filters({"Ani": "a"}, ambiguous, "X")
