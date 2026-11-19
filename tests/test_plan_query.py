"""Query planning budgets on the product of selected values."""

from __future__ import annotations

import math

import pytest

from tempo_ins._api import chunk, each_selection, plan_query
from tempo_ins.errors import TempoAPIError, TempoInputError


def _ids(count: int):
    return [str(value) for value in range(1, count + 1)]


def test_a_small_matrix_is_fetched_in_a_single_request():
    plan = plan_query([_ids(5), _ids(4)], code="SML", max_cells=1000)

    assert len(plan.chunks) == 2
    assert plan.requests == 1
    assert plan.cells == 20
    assert plan.chunks[0] == [_ids(5)]


def test_the_budget_is_the_product_not_one_dimension():
    # Five dimensions of 200 options each: no dimension crosses the historical
    # 300-option threshold, yet the query asks for 3.2e11 cells.  This is the
    # case per-dimension chunking let through untouched.
    ids = [_ids(200) for _ in range(5)]

    plan = plan_query(ids, code="HUGE", max_cells=50_000, max_requests=math.inf)

    assert plan.requests > 1
    assert math.prod(plan.chunk_sizes) <= 50_000
    # Every option still appears exactly once per dimension.
    for index, dimension in enumerate(ids):
        flattened = [value for part in plan.chunks[index] for value in part]
        assert sorted(flattened) == sorted(dimension)


def test_a_single_huge_dimension_is_split_and_fully_covered():
    plan = plan_query([_ids(1000)], code="ONE", max_cells=100)

    assert math.prod(plan.chunk_sizes) == 100
    assert plan.requests == 10
    flattened = [int(value) for part in plan.chunks[0] for value in part]
    assert sorted(flattened) == list(range(1, 1001))


def test_the_per_dimension_cap_is_honoured_even_when_the_product_is_small():
    plan = plan_query([_ids(400)], code="CAP", max_cells=1e9, max_options=300)

    assert plan.chunk_sizes == [300]
    assert plan.requests == 2


def test_an_absurd_download_is_refused_rather_than_attempted():
    ids = [_ids(500) for _ in range(4)]

    with pytest.raises(TempoInputError, match="would need"):
        plan_query(ids, code="NOPE", max_cells=10, max_requests=100)


def test_the_refusal_carries_the_request_count():
    ids = [_ids(500) for _ in range(4)]

    with pytest.raises(TempoInputError) as caught:
        plan_query(ids, code="NOPE", max_cells=10, max_requests=100)

    assert caught.value.code == "NOPE"
    assert caught.value.requests > 100


def test_empty_dimensions_are_rejected():
    with pytest.raises(TempoAPIError):
        plan_query([[]], code="EMPTY")

    with pytest.raises(TempoAPIError):
        plan_query([], code="NONE")


def test_chunking_splits_into_equal_pieces_with_a_remainder():
    chunks = chunk(_ids(301), 100)

    assert len(chunks) == 4
    assert [len(part) for part in chunks] == [100, 100, 100, 1]
    assert chunk(["1", "2", "3"], 10) == [["1", "2", "3"]]


def test_the_cartesian_walk_visits_every_combination_once_in_order():
    chunks = chunk(_ids(301), 100)
    selections = list(each_selection([chunks, [["x"], ["y"]]]))

    assert len(selections) == 8
    assert selections[0] == (chunks[0], ["x"])
    assert selections[7] == (chunks[3], ["y"])
    assert len({tuple(map(tuple, pair)) for pair in selections}) == 8
