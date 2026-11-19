"""Listing and searching the TEMPO table of contents."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import pandas as pd

from ._api import (
    ancestor,
    base_url,
    extract_links,
    matrix_metadata_many,
    metadata_date,
    strip_html,
    toc_from_response,
    url,
    validate_flag,
    validate_language,
)
from ._cache import cache_key, cached
from ._http import request
from .errors import TempoInputError

__all__ = ["toc", "search"]


def toc(
    full_description: bool = False,
    language: str = "ro",
    cache: bool = True,
) -> pd.DataFrame:
    """List the tables available from TEMPO Online.

    Retrieves the table of contents for the Romanian National Institute of
    Statistics TEMPO Online database: one row per statistical matrix, with the
    matrix code needed by :func:`tempo_ins.bulk`, :func:`tempo_ins.get` and
    :func:`tempo_ins.meta`.

    Args:
        full_description: Also retrieve the statistical domain, sub-domain,
            survey name and last-update date of every table.  This needs one
            request per table, which the service answers for roughly 1500
            matrices, so the first call takes a while even though the requests
            are issued concurrently and cached afterwards.
        language: ``"ro"`` for Romanian (the default) or ``"en"`` for English.
        cache: Whether to reuse and populate the response cache described in
            :func:`tempo_ins.cache_dir`.

    Returns:
        A data frame with one row per matrix and the columns ``name``, ``code``
        and, when the service supplies any, ``comment``.  With
        ``full_description=True`` it additionally holds
        ``Statistical_domain``, ``Statistical_sub_domain`` and ``Survey_name``
        (the first three levels of the service's classification, as plain
        text), ``Press_releases`` (press-release URLs that the service embeds
        in the domain label, space separated) and ``Last_update`` (the
        publication date of the matrix, as a datetime).

    Text and HTML:
        The service embeds HTML anchors in some labels, most visibly in the
        top-level statistical domains, which arrive as
        ``AGRICULTURE <a href="https://insse.ro/...">Press releases</a>``.
        Those tags are stripped from every text column, and the URLs are kept
        separately in ``Press_releases``.  Table names are returned without
        Romanian diacritics, because that is how the service stores them.

    Raises:
        TempoOfflineError: The service could not be reached.
        TempoAPIError: The service returned an unexpected format.

    See Also:
        :func:`tempo_ins.search` to filter the result,
        :func:`tempo_ins.meta` for the full metadata of a single matrix,
        :func:`tempo_ins.hierarchy` for the classification tree.

    Example:
        >>> import tempo_ins as tempo
        >>> tables = tempo.toc(language="en")           # doctest: +SKIP
        >>> tables.head()                               # doctest: +SKIP
    """
    full_description = validate_flag(full_description, "full_description")
    language = validate_language(language)
    cache = validate_flag(cache, "cache")

    def fetch() -> pd.DataFrame:
        return toc_from_response(request(url("matrix/matrices", language=language)))

    if cache:
        frame = cached(cache_key("matrices", language, base_url()), fetch)
    else:
        frame = fetch()

    frame = frame.copy()

    if not full_description:
        return frame

    codes = frame["code"].tolist()
    metadata = matrix_metadata_many(
        codes,
        language=language,
        progress_label="Fetching table descriptions",
    )

    domain = [ancestor(entry, 2) for entry in metadata]

    frame["Statistical_domain"] = strip_html(domain)
    frame["Statistical_sub_domain"] = strip_html(
        [ancestor(entry, 3) for entry in metadata]
    )
    frame["Survey_name"] = strip_html([ancestor(entry, 4) for entry in metadata])
    frame["Press_releases"] = extract_links(domain)
    frame["Last_update"] = pd.to_datetime(
        pd.Series(
            [
                metadata_date(entry, code=code)
                for entry, code in zip(metadata, codes, strict=True)
            ],
            dtype="object",
        )
    )

    return frame


# Bound here so that `search(toc=...)` can still reach the function above
# despite the parameter of the same name.
_fetch_toc = toc


def search(
    pattern: str | Sequence[str],
    language: str = "ro",
    fields: Sequence[str] | None = None,
    ignore_case: bool = True,
    fixed: bool = False,
    toc: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Search the TEMPO table of contents.

    A convenience wrapper around :func:`tempo_ins.toc` for the common task of
    finding the matrix codes whose name matches a pattern.

    Args:
        pattern: A regular expression, or a literal string when ``fixed=True``.
            Several patterns are combined with a logical OR.
        language: ``"ro"`` for Romanian (the default) or ``"en"`` for English.
            The service stores table names without diacritics, so search for
            the unaccented spelling, as in ``"sanatate"``.
        fields: Columns to search.  Defaults to every text column of ``toc``.
        ignore_case: Whether to match case-insensitively.
        fixed: Whether to match ``pattern`` literally rather than as a regular
            expression.
        toc: An existing table of contents, as returned by
            :func:`tempo_ins.toc`.  Pass one to search a table you already
            hold, including a fully described one, instead of fetching it
            again.

    Returns:
        The matching rows, with the index reset.

    Example:
        >>> import tempo_ins as tempo
        >>> tempo.search("population", language="en")        # doctest: +SKIP
        >>> tables = tempo.toc(language="en")                # doctest: +SKIP
        >>> tempo.search(["unemployment", "employment"], toc=tables)
        ...                                                  # doctest: +SKIP
    """
    if isinstance(pattern, str):
        patterns: list[str] = [pattern]
    elif isinstance(pattern, Iterable):
        patterns = list(pattern)
    else:
        patterns = [pattern]  # type: ignore[list-item]

    if not patterns or any(
        value is None or not isinstance(value, str) for value in patterns
    ):
        raise TempoInputError(
            "`pattern` must be a string, or a sequence with at least one string."
        )

    ignore_case = validate_flag(ignore_case, "ignore_case")
    fixed = validate_flag(fixed, "fixed")

    if toc is None:
        toc = _fetch_toc(language=language)

    if not isinstance(toc, pd.DataFrame) or not len(toc):
        raise TempoInputError(
            "`toc` must be a data frame with at least one row, as returned by toc()."
        )

    if fields is None:
        selected = [
            name
            for name in toc.columns
            if pd.api.types.is_object_dtype(toc[name])
            or pd.api.types.is_string_dtype(toc[name])
        ]
    else:
        selected = [fields] if isinstance(fields, str) else list(fields)
        unknown = [name for name in selected if name not in toc.columns]
        if unknown:
            raise TempoInputError(
                "`fields` names columns that are not in `toc`: "
                + ", ".join(str(name) for name in unknown)
                + "."
            )

    if not selected:
        raise TempoInputError("`toc` holds no text columns to search.")

    haystack = toc[selected[0]].fillna("").astype(str)
    for name in selected[1:]:
        haystack = haystack + "\n" + toc[name].fillna("").astype(str)

    flags = re.IGNORECASE if ignore_case else 0
    matched = pd.Series(False, index=toc.index)
    for one in patterns:
        expression = re.escape(one) if fixed else one
        matched |= haystack.str.contains(expression, flags=flags, regex=True, na=False)

    return toc[matched].reset_index(drop=True)
