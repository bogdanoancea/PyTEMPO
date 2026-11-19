"""Splitting the service's CSV responses into header and body."""

from __future__ import annotations

import pytest

from tempo_ins._api import csv_split, header_text, without_csv_header
from tempo_ins.errors import TempoAPIError


def test_the_header_is_separated_from_the_body():
    parts = csv_split(b"a,b\n1,2\n3,4\n")

    assert parts.header == b"a,b\n"
    assert parts.body == b"1,2\n3,4\n"


def test_a_header_only_response_yields_an_empty_body():
    assert csv_split(b"a,b\n").body == b""


def test_csv_chunks_retain_one_header():
    assert without_csv_header(b"a,b\n1,2\n") == b"1,2\n"


def test_a_response_with_no_header_line_is_an_error():
    with pytest.raises(TempoAPIError, match="without a header line"):
        csv_split(b"a,b")


def test_header_text_ignores_trailing_newlines():
    assert header_text(b"a,b\r\n") == "a,b"
    assert header_text(b"a,b") == "a,b"
