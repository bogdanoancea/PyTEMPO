"""Integration tests against the real TEMPO Online service.

These are skipped unless ``TEMPO_LIVE_TESTS=true``, so a failure here means
the service changed, not that the package is broken.  They are deliberately
frugal: the branch listing is fetched without expanding every survey into its
matrices, which would cost a few hundred more requests against a small public
service.

Run them with::

    TEMPO_LIVE_TESTS=true pytest -m live
"""

from __future__ import annotations

import os
import re

import pandas as pd
import pytest

import tempo_ins as tempo
from tempo_ins import _api, _http
from tempo_ins._hierarchy import _context_children

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("TEMPO_LIVE_TESTS") != "true",
        reason="set TEMPO_LIVE_TESTS=true to run tests against the live service",
    ),
]

CODE = "ACC101B"


def test_the_live_table_of_contents_has_the_expected_shape():
    toc = tempo.toc(language="en", cache=False)

    assert isinstance(toc, pd.DataFrame)
    assert {"name", "code"} <= set(toc.columns)
    assert len(toc) > 1000
    assert toc["code"].str.match(r"^[A-Za-z0-9]+$").all()
    # Codes must be unique, or the download cache keys collide.
    assert not toc["code"].duplicated().any()


def test_live_metadata_exposes_dimensions_and_a_parseable_update_date():
    info = tempo.meta(CODE, language="en", cache=False)

    assert len(info.dimensions) > 0
    assert (info.dimensions["values"] > 0).all()
    assert info.last_update is not None
    assert info.cells > 0


def test_a_filtered_live_download_returns_only_the_requested_values():
    info = tempo.meta(CODE, language="en", cache=False)
    dimension = info.dimensions["name"].iloc[0]
    wanted = info.value_frames[0]["label"].iloc[0]

    data = tempo.get(CODE, filters={dimension: wanted}, language="en")

    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert "value" in data.columns


def test_the_live_csv_layout_still_matches_what_tidy_expects():
    raw = tempo.get(CODE, language="en", tidy=False)

    # The service separates its header fields with ", ", so every column but
    # the first arrives with a leading space. tidy() strips them, so compare
    # the stripped names here too.
    names = [name.strip() for name in raw.columns]

    # If either of these fails, tidy() needs updating. Note that the service
    # leaves the observation column named "Valoare" even in English, which is
    # why tidy() accepts both spellings.
    assert any(name.lower() in {"valoare", "value"} for name in names)
    assert any(re.match(r"^(UM|MU)\s*:", name) for name in names)


def test_the_live_context_tree_has_the_shape_hierarchy_expects():
    branches = tempo.hierarchy(language="en", matrices=False, cache=False)

    assert isinstance(branches, pd.DataFrame)
    assert {
        "code",
        "name",
        "parent_code",
        "level",
        "is_matrix",
        "path",
    } <= set(branches.columns)
    assert len(branches) > 100
    assert not branches["is_matrix"].any()

    # Levels are 1-based and contiguous from the root.
    levels = branches["level"].dropna().astype(int)
    assert levels.min() == 1
    assert set(levels) == set(range(1, int(levels.max()) + 1))

    # Root nodes have no parent; every other parent must resolve to a node.
    roots = branches["level"] == 1
    assert branches.loc[roots, "parent_code"].isna().all()
    assert branches.loc[~roots, "parent_code"].isin(branches["code"]).all()

    # HTML chrome is stripped from the domain labels, which do contain anchors.
    assert not branches["name"].str.contains(r"<a |</a>", regex=True).any()
    assert (branches["name"].str.len() > 0).all()

    # Paths are built from the parent links, so depth must match level.
    depth = branches["path"].str.split(" > ").str.len()
    assert depth.tolist() == levels.tolist()


def test_live_survey_nodes_list_matrices_as_bare_children():
    branches = tempo.hierarchy(language="en", matrices=False, cache=False)
    deepest = branches[branches["level"] == branches["level"].max()]
    if not len(deepest):  # pragma: no cover - service would have to change
        pytest.skip("no deepest branch found")

    response = _http.request(
        _api.url(f"context/{deepest['code'].iloc[0]}", language="en")
    )
    children = _context_children(_api.parse_json(response, "context-node"))

    assert isinstance(children, pd.DataFrame)
    assert {"code", "name", "children_url"} <= set(children.columns)
    assert len(children) > 0
    # A survey's children are matrices, and their codes must be usable.
    assert (children["children_url"] == "matrix").any()
    matrices = children.loc[children["children_url"] == "matrix", "code"]
    assert matrices.str.match(r"^[A-Za-z0-9]+$").all()
