"""End-to-end tests against the fake service in conftest.py."""

from __future__ import annotations

import os
import warnings

import pandas as pd
import pytest

import tempo_ins as tempo
from tempo_ins import _api, _http
from tempo_ins._cache import cache_get, cache_key
from tempo_ins._hierarchy import _context_children, _context_root
from tempo_ins.errors import (
    TempoAPIError,
    TempoDateWarning,
    TempoHTTPError,
    TempoOfflineError,
    reset_warnings,
)
from tempo_ins.options import option_context


def test_toc_lists_matrices_and_strips_html_from_names(service):
    toc = tempo.toc()

    assert isinstance(toc, pd.DataFrame)
    assert toc["code"].tolist() == ["SML101A", "BIG101A", "TAG101A"]
    assert toc["name"].iloc[2] == "Tagged table"
    assert toc["comment"].isna().tolist() == [True, False, True]
    assert toc["comment"].iloc[1] == "a comment"


def test_toc_caches_the_listing(service):
    tempo.toc()
    tempo.toc()

    cached = cache_get(cache_key("matrices", "ro", _api.base_url()))
    assert cached["code"].iloc[0] == "SML101A"
    assert service.toc_calls == 1


def test_toc_full_description_returns_plain_text_classification(service):
    described = tempo.toc(full_description=True)

    assert described["Statistical_domain"].unique().tolist() == ["AGRICULTURE"]
    assert described["Statistical_sub_domain"].unique().tolist() == ["Sub-domain"]
    assert described["Survey_name"].unique().tolist() == ["Survey name"]
    assert described["Press_releases"].unique().tolist() == [
        "https://insse.ro/press?x=1"
    ]
    assert pd.api.types.is_datetime64_any_dtype(described["Last_update"])
    assert described["Last_update"].unique().tolist() == [pd.Timestamp("2025-11-05")]


def test_search_filters_the_listing(service):
    assert tempo.search("big")["code"].tolist() == ["BIG101A"]
    assert len(tempo.search("fake")) == 2
    assert len(tempo.search("nothing here")) == 0


def test_meta_reports_structure_without_downloading(service):
    info = tempo.meta("BIG101A")

    assert info.code == "BIG101A"
    assert info.dimensions["name"].tolist() == ["Judete", "Ani"]
    assert info.dimensions["values"].tolist() == [7, 5]
    assert info.cells == 35
    assert info.requests == 1
    assert info.domain == "AGRICULTURE"
    assert info.last_update.isoformat() == "2025-11-05"
    assert info.values["Judete"]["label"].iloc[0] == "County 1"
    assert info[0]["label"].iloc[0] == "County 1"
    assert len(info) == 2
    assert service.pivot_calls == 0

    rendered = repr(info)
    assert "BIG101A" in rendered
    assert "35 cells" in rendered
    assert "AGRICULTURE > Sub-domain > Survey name" in rendered


def test_get_returns_a_tidy_data_frame(service):
    data = tempo.get("SML101A")

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 6
    assert set(data.columns) == {"Judete", "Ani", "unit", "value"}
    assert pd.api.types.is_float_dtype(data["value"])
    assert pd.api.types.is_integer_dtype(data["Ani"])
    assert data.attrs["tempo_code"] == "SML101A"
    assert data.attrs["tempo_requests"] == 1
    assert data.attrs["tempo_language"] == "ro"
    assert data.attrs["tempo_service_update"] == "05-11-2025"


def test_get_sends_the_filter_to_the_service(service):
    data = tempo.get(
        "BIG101A",
        filters={"Judete": ["County 1", "County 3"], "Ani": range(2020, 2022)},
    )

    assert len(data) == 4
    assert set(data["Judete"]) == {"County 1", "County 3"}
    assert set(data["Ani"]) == {2020, 2021}


def test_a_bad_filter_is_rejected_before_any_download(service):
    with pytest.raises(tempo.TempoInputError):
        tempo.get("SML101A", filters={"Judete": "Atlantis"})

    assert service.pivot_calls == 0


def test_get_untidied_keeps_the_service_layout(service):
    raw = tempo.get("SML101A", tidy=False)

    # The service's headers arrive ", "-separated, so compare stripped names,
    # which is what tidy() does.
    names = [name.strip() for name in raw.columns]
    assert "Valoare" in names
    assert any(name.startswith("UM:") for name in names)
    # Untidied means untouched: the service's ", " padding is still on both
    # the header and the values, its own ":" marker is still there, and
    # nothing was parsed into a number.
    value_column = next(name for name in raw.columns if name.strip() == "Valoare")
    assert not pd.api.types.is_numeric_dtype(raw[value_column])
    assert raw[value_column].iloc[0] == " :"
    assert raw[" Ani"].iloc[0].startswith(" ")


def test_a_chunked_download_reassembles_every_row_exactly_once(service):
    # 7 x 5 = 35 cells with a 6-cell budget: several requests, each returning
    # its own header and its own footer line.
    with option_context(max_cells=6):
        data = tempo.get("BIG101A")

    assert data.attrs["tempo_requests"] > 1
    assert len(data) == 35
    assert len(data.drop_duplicates(subset=["Judete", "Ani"])) == 35
    # No footer line survived into the data.
    assert not data["Judete"].isna().any()
    assert (data["Judete"] != "").all()


def test_inconsistent_headers_across_chunks_are_an_error(service):
    with option_context(max_cells=1):
        with pytest.raises(TempoAPIError, match="inconsistent CSV headers"):
            tempo.get("BADHDR")


def test_bulk_writes_a_csv_a_manifest_and_reports_what_it_did(service, download_dir):
    files = tempo.bulk("SML101A", download_dir)

    assert list(files) == ["SML101A"]
    assert os.path.exists(files["SML101A"])
    assert files.downloaded["SML101A"] is True
    assert files.reason["SML101A"] == "missing"
    assert os.path.exists(os.path.join(download_dir, ".tempo-manifest.json"))

    frame = tempo.tidy(files["SML101A"], code="SML101A")
    assert len(frame) == 6

    described = files.to_frame()
    assert described["code"].tolist() == ["SML101A"]
    assert described["requests"].tolist() == [1]


def test_bulk_returns_paths_for_skipped_files_too(service, download_dir):
    tempo.bulk("SML101A", download_dir)
    again = tempo.bulk("SML101A", download_dir)

    assert list(again) == ["SML101A"]
    assert os.path.exists(again["SML101A"])
    assert again.downloaded["SML101A"] is False
    assert again.reason["SML101A"] == "current"


def test_switching_language_redownloads_instead_of_reusing_the_other_one(
    service, download_dir
):
    tempo.bulk("SML101A", download_dir, language="ro")
    switched = tempo.bulk("SML101A", download_dir, language="en")

    assert switched.downloaded["SML101A"] is True
    assert switched.reason["SML101A"] == "language"


def test_overwrite_forces_a_download_of_a_current_file(service, download_dir):
    tempo.bulk("SML101A", download_dir)
    forced = tempo.bulk("SML101A", download_dir, overwrite=True)

    assert forced.downloaded["SML101A"] is True
    assert forced.reason["SML101A"] == "overwrite"


def test_bulk_handles_several_codes_at_once(service, download_dir):
    files = tempo.bulk(["SML101A", "TAG101A"], download_dir)

    assert sorted(files) == ["SML101A", "TAG101A"]
    assert all(os.path.exists(path) for path in files.values())
    assert len(files.to_frame()) == 2


def test_a_retryable_status_is_retried_and_then_succeeds(service):
    # The fake service fails this endpoint twice with HTTP 500.
    response = _http.request(_api.url("flaky/matrices"))

    assert response.status_code == 200
    assert service.counters["flaky"] == 3


def test_a_non_retryable_status_raises_a_http_error_with_its_code(service):
    with pytest.raises(TempoHTTPError) as caught:
        tempo.meta("NOSUCH")

    assert caught.value.status_code == 404
    assert "matrix/NOSUCH" in caught.value.url
    assert "HTTP 404" in str(caught.value)


def test_an_unreachable_service_raises_an_offline_error():
    with (
        option_context(
            base_url="http://127.0.0.1:1/tempo-ins", max_attempts=1, backoff=0
        ),
        pytest.raises(TempoOfflineError) as caught,
    ):
        tempo.toc(cache=False)

    assert caught.value.url is not None


def test_concurrent_fetching_returns_responses_in_request_order(service):
    urls = [_api.url(f"matrix/{code}") for code in ("SML101A", "BIG101A", "TAG101A")]

    responses = _http.request_many(urls)
    names = [
        _api.parse_json(response, "metadata")["matrixName"] for response in responses
    ]

    assert names == [
        "Fake matrix SML101A",
        "Fake matrix BIG101A",
        "Fake matrix TAG101A",
    ]


def test_a_single_connection_still_fetches_everything(service):
    urls = [_api.url(f"matrix/{code}") for code in ("SML101A", "BIG101A")]

    with option_context(max_connections=1):
        responses = _http.request_many(urls)

    assert [response.status_code for response in responses] == [200, 200]


def test_an_unparseable_service_date_warns_but_keeps_caching_working(
    service, download_dir
):
    reset_warnings()

    with pytest.warns(TempoDateWarning):
        tempo.bulk("ODDDATE", download_dir)

    # The raw update string still identifies the revision, so the second call
    # reuses the file rather than downloading for ever.
    again = tempo.bulk("ODDDATE", download_dir)
    assert again.downloaded["ODDDATE"] is False
    assert again.reason["ODDDATE"] == "current"


def test_hierarchy_returns_the_branch_structure_in_one_request(service):
    branches = tempo.hierarchy(matrices=False)

    assert isinstance(branches, pd.DataFrame)
    assert len(branches) == 3
    assert not branches["is_matrix"].any()
    # HTML chrome stripped, 1-based levels, root parent reported as None.
    assert branches["name"].iloc[0] == "AGRICULTURE"
    assert branches["level"].tolist() == [1, 2, 3]
    assert pd.isna(branches["parent_code"].iloc[0])
    assert branches["parent_code"].iloc[2] == "10"
    assert branches["path"].iloc[2] == "AGRICULTURE > Land use > Areas and yields"
    assert service.context_calls == 1


def test_hierarchy_expands_matrices_under_the_deepest_branches(service):
    tree = tempo.hierarchy()

    matrices = tree[tree["is_matrix"]]
    assert sorted(matrices["code"]) == ["BIG101A", "SML101A"]
    assert matrices["level"].unique().tolist() == [4]
    assert (
        tree.loc[tree["code"] == "SML101A", "path"].iloc[0]
        == "AGRICULTURE > Land use > Areas and yields > Small fake table"
    )
    assert tree.loc[tree["code"] == "SML101A", "parent_code"].iloc[0] == "1010"
    # Only the deepest branch is expanded, not every branch.
    assert len(tree) == 5


def test_an_unexpected_context_tree_shape_is_reported_not_guessed_at(service):
    with option_context(base_url=_api.base_url() + "/nonexistent"):
        with pytest.raises(TempoAPIError) as caught:
            tempo.hierarchy(cache=False)

    assert "unexpected context-tree format" in str(caught.value)
    assert "toc(full_description" in str(caught.value)


def test_the_context_parsers_handle_each_shape_in_isolation():
    root = [
        {
            "parentCode": "0",
            "level": 0,
            "context": {
                "name": 'DOMAIN <a href="x">Press releases</a>',
                "code": "1",
                "childrenUrl": "context",
            },
        },
        {
            "parentCode": "1",
            "level": 1,
            "context": {"name": "Sub", "code": "10", "childrenUrl": "context"},
        },
    ]

    parsed = _context_root(root)
    assert parsed["code"].tolist() == ["1", "10"]
    assert parsed["name"].tolist() == ["DOMAIN", "Sub"]
    assert parsed["level"].tolist() == [1, 2]
    assert pd.isna(parsed["parent_code"].iloc[0])

    node = {
        "context": {"name": "Survey", "code": "1010", "childrenUrl": "context"},
        "ancestors": [{"name": "home", "code": "", "childrenUrl": "context"}],
        "children": [{"name": "A table", "code": "POP105A", "childrenUrl": "matrix"}],
    }

    children = _context_children(node)
    assert children["code"].tolist() == ["POP105A"]
    assert children["children_url"].tolist() == ["matrix"]

    assert _context_root([]) is None
    assert _context_children({"children": []}) is None


def test_hierarchy_tolerates_unwrapped_root_nodes():
    # The service currently wraps each node, but the parser accepts a bare one.
    parsed = _context_root(
        [{"name": "DOMAIN", "code": "1", "childrenUrl": "context", "level": 0}]
    )

    assert parsed["code"].tolist() == ["1"]
    assert parsed["level"].tolist() == [1]


def test_the_services_padded_headers_are_handled(service):
    # Regression test for the header shape the live service sends: fields
    # separated by ", ", so every column but the first has a leading space.
    raw = tempo.get("SML101A", tidy=False)

    assert [name for name in raw.columns if name.startswith(" ")], (
        "the fake service should reproduce the live leading-space headers"
    )

    # tidy() has to see through that: no padded names, and the unit and value
    # columns still found.
    tidied = tempo.get("SML101A")

    assert [name for name in tidied.columns if name != name.strip()] == []
    assert "value" in tidied.columns
    assert "unit" in tidied.columns
    assert tidied["unit"].unique().tolist() == ["Numar"]
    assert not any(name.startswith("UM:") for name in tidied.columns)


def test_the_value_column_is_found_under_its_romanian_name_in_english(service):
    # The live service leaves the observation column named "Valoare" even when
    # language="en", so tidy() must accept the Romanian spelling either way.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tidied = tempo.get("SML101A", language="en")

    assert "value" in tidied.columns
    assert tidied["value"].notna().sum() == 5  # 6 cells, the first is ":"
    # In particular, no "no Valoare/Value column found" warning.
    assert [str(w.message) for w in caught] == []
