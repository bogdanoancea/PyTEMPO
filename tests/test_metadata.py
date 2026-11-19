"""Reading dimensions, labels, dates and the query payload from metadata."""

from __future__ import annotations

import warnings
from datetime import date

import pytest

from tempo_ins._api import (
    ancestor,
    dimension_names,
    metadata_date,
    metadata_update_string,
    option_ids,
    option_labels,
    payload,
    strip_html_one,
    validate_metadata,
)
from tempo_ins.errors import TempoAPIError, TempoDateWarning, reset_warnings

METADATA = {
    "dimensionsMap": [
        {
            "label": "Judete",
            "options": [
                {"nomItemId": 1, "label": "Cluj"},
                {"nomItemId": 2, "label": "Bihor"},
            ],
        },
        {
            "label": "Ani",
            "options": [{"nomItemId": 10, "label": "Anul 2020"}],
        },
    ],
    "details": {"matMaxDim": 2, "matUMSpec": 0},
    "ultimaActualizare": "05-11-2025",
    "ancestors": [
        {"name": "ROOT"},
        {"name": 'AGRICULTURE <a href="https://insse.ro/x">Press releases</a>'},
    ],
}


def test_option_identifiers_and_labels_are_read_from_the_metadata():
    assert option_ids(METADATA, "AAA") == [["1", "2"], ["10"]]
    assert option_labels(METADATA, "AAA") == [["Cluj", "Bihor"], ["Anul 2020"]]
    assert dimension_names(METADATA, "AAA") == ["Judete", "Ani"]


def test_dimension_names_fall_back_to_positions_when_absent():
    bare = {
        **METADATA,
        "dimensionsMap": [
            {"options": METADATA["dimensionsMap"][0]["options"]},
            METADATA["dimensionsMap"][1],
        ],
    }

    assert dimension_names(bare, "AAA") == ["dimension_1", "Ani"]


def test_labels_are_read_from_any_of_the_plausible_field_names():
    variants = {
        "dimensionsMap": [
            {
                "name": "Named dimension",
                "options": [
                    {"nomItemId": 1, "nomItemName": "From nomItemName"},
                    {"nomItemId": 2, "text": "From text"},
                ],
            }
        ],
        "details": {"matMaxDim": 1, "matUMSpec": 0},
    }

    assert dimension_names(variants, "V") == ["Named dimension"]
    assert option_labels(variants, "V")[0] == [
        "From nomItemName",
        "From text",
    ]


def test_the_query_payload_is_built_from_the_selected_identifiers():
    built = payload("AAA101A", "en", METADATA, option_ids(METADATA, "AAA101A"))

    assert built["encQuery"] == "1, 2:10"
    assert built["matCode"] == "AAA101A"
    assert built["language"] == "en"
    assert built["matMaxDim"] == 2


def test_ancestors_are_returned_as_plain_text():
    assert strip_html_one(ancestor(METADATA, 2)) == "AGRICULTURE"
    assert ancestor(METADATA, 9) is None


def test_incomplete_metadata_is_rejected_with_a_classed_error():
    with pytest.raises(TempoAPIError):
        validate_metadata({"details": {}}, "AAA")

    with pytest.raises(TempoAPIError, match="incomplete query metadata"):
        payload("AAA", "ro", {"details": {}}, [["1"]])


def test_an_invalid_option_identifier_is_rejected():
    broken = {
        "dimensionsMap": [{"label": "D", "options": [{"label": "no id"}]}],
        "details": {"matMaxDim": 1, "matUMSpec": 0},
    }

    with pytest.raises(TempoAPIError, match="invalid option identifier"):
        option_ids(broken, "AAA")


def test_a_dimension_with_no_options_is_rejected():
    broken = {
        "dimensionsMap": [{"label": "D", "options": []}],
        "details": {"matMaxDim": 1, "matUMSpec": 0},
    }

    with pytest.raises(TempoAPIError, match="no selectable options in dimension 1"):
        option_ids(broken, "AAA")


@pytest.mark.parametrize(
    "value", ["05-11-2025", "05.11.2025", "05/11/2025", "2025-11-05", "2025/11/05"]
)
def test_the_update_date_is_parsed_in_every_format_the_service_might_use(value):
    assert metadata_date({"ultimaActualizare": value}) == date(2025, 11, 5)


def test_an_unparseable_update_date_warns_instead_of_failing_silently():
    reset_warnings()

    with pytest.warns(TempoDateWarning):
        result = metadata_date({"ultimaActualizare": "November 5th, 2025"}, code="AAA")

    assert result is None

    # Warned once, not once per matrix.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metadata_date({"ultimaActualizare": "November 5th, 2025"})
    assert caught == []


def test_a_missing_update_date_is_none_without_a_warning():
    reset_warnings()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert metadata_date({}) is None
    assert caught == []

    assert metadata_update_string({}) is None
