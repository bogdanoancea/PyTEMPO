"""HTML in service labels is stripped, and the URLs kept separately."""

from __future__ import annotations

import pytest

from tempo_ins._api import (
    extract_links_one,
    strip_html_one,
    toc_from_response,
)
from tempo_ins.errors import TempoAPIError


class _Response:
    """The parts of a requests.Response the parsers actually read."""

    def __init__(self, text: str) -> None:
        self.content = text.encode("utf-8")
        self.headers: dict = {}
        self.status_code = 200


def test_anchors_are_stripped_from_service_labels():
    # This is exactly what the service returns for a top-level domain.
    domain = (
        'AGRICULTURE <a href="https://insse.ro/cms/en/comunicate-de-presa-view'
        '?field_categorie_value_i18n%5B%5D=1" target="_blank">Press releases</a>'
    )

    assert strip_html_one(domain) == "AGRICULTURE"


def test_stripping_handles_several_anchors_entities_and_none():
    value = (
        'LABOUR FORCE <a href="https://insse.ro/a">One-Press releases</a>; '
        '<a href="https://insse.ro/b">Two-Press releases</a>'
    )

    assert strip_html_one(value) == "LABOUR FORCE"
    assert strip_html_one("Trade &amp; industry") == "Trade & industry"
    assert strip_html_one("  spaced   out  ") == "spaced out"
    assert strip_html_one(None) is None
    assert strip_html_one("plain text") == "plain text"
    # A label that is nothing but a link keeps its text, or it would vanish.
    assert (
        strip_html_one('<a href="https://insse.ro/x">Only a link</a>') == "Only a link"
    )


def test_press_release_urls_are_kept_rather_than_discarded():
    value = (
        'LABOUR FORCE <a href="https://insse.ro/a">One</a>; '
        '<a href="https://insse.ro/b">Two</a>'
    )

    assert extract_links_one(value) == "https://insse.ro/a https://insse.ro/b"
    assert extract_links_one("no links here") is None


def test_toc_parser_strips_html_and_keeps_comments():
    response = _Response(
        '[{"name":"First table <a href=\\"https://insse.ro/x\\">Press releases'
        '</a>","code":"AAA101A","comment":null,"extra":"ignored"},'
        '{"name":"Second table","code":"BBB101B","comment":"note"}]'
    )

    toc = toc_from_response(response)

    assert toc["code"].tolist() == ["AAA101A", "BBB101B"]
    assert toc["name"].tolist() == ["First table", "Second table"]
    # A comment the service omitted is missing; pandas 2 represents that as
    # None and pandas 3 as nan, so assert missingness rather than identity.
    assert toc["comment"].isna().tolist() == [True, False]
    assert toc["comment"].iloc[1] == "note"


def test_toc_parser_rejects_malformed_responses():
    response = _Response('[{"title":"Missing code"}]')

    with pytest.raises(TempoAPIError, match="unexpected table-of-contents format"):
        toc_from_response(response)


def test_toc_parser_rejects_non_json():
    with pytest.raises(TempoAPIError, match="invalid table-of-contents JSON"):
        toc_from_response(_Response("not json at all"))
