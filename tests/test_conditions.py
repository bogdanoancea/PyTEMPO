"""The error and warning classes, and what they carry."""

from __future__ import annotations

import warnings

import pytest

from tempo_ins import _http
from tempo_ins._api import validate_language
from tempo_ins.errors import (
    TempoAPIError,
    TempoError,
    TempoInputError,
    TempoTidyWarning,
    reset_warnings,
    warn_once,
)
from tempo_ins.options import option_context


def test_every_tempo_error_inherits_from_tempo_error():
    error = TempoAPIError("boom", code="X1")

    assert isinstance(error, TempoAPIError)
    assert isinstance(error, TempoError)
    assert isinstance(error, Exception)
    assert error.code == "X1"
    assert str(error) == "boom"


def test_fields_default_to_none_when_the_service_supplied_none():
    assert TempoAPIError("boom").code is None
    assert TempoAPIError("boom").url is None


def test_input_errors_are_catchable_by_class_not_by_message_text():
    try:
        validate_language("fr")
    except TempoInputError:
        handled = "input"
    except TempoError:  # pragma: no cover - would be the wrong branch
        handled = "other"

    assert handled == "input"


def test_a_plain_exception_handler_still_catches_tempo_errors():
    # Blind on purpose: the point is that a caller who knows nothing about this
    # package's classes still catches its errors.
    with pytest.raises(Exception):  # noqa: B017
        validate_language("fr")


def test_warnings_are_raised_once_per_process_and_reset_on_demand():
    reset_warnings()

    with pytest.warns(TempoTidyWarning, match="first time"):
        assert warn_once("demo", "first time", TempoTidyWarning) is True

    # The second call is silent, and says so by returning False.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert warn_once("demo", "second time", TempoTidyWarning) is False
    assert caught == []

    reset_warnings()
    with pytest.warns(TempoTidyWarning):
        assert warn_once("demo", "again", TempoTidyWarning) is True


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (429, True),
        (503, True),
        (500, True),
        (408, True),
        (425, True),
        (404, False),
        (400, False),
        (200, False),
        (None, False),
    ],
)
def test_retryable_statuses_are_classified_correctly(status, retryable):
    assert _http._status_retryable(status) is retryable


def test_the_user_agent_names_the_package():
    assert _http.user_agent().startswith("tempo-ins/")

    with option_context(user_agent="custom/1.0"):
        assert _http.user_agent() == "custom/1.0"


def test_retry_after_is_clamped_and_validated():
    class _Response:
        def __init__(self, value):
            self.headers = {"Retry-After": value} if value is not None else {}

    assert _http._retry_after(_Response("5")) == 5.0
    assert _http._retry_after(_Response("600")) == 60.0
    assert _http._retry_after(_Response("-1")) is None
    assert _http._retry_after(_Response("soon")) is None
    assert _http._retry_after(_Response(None)) is None
