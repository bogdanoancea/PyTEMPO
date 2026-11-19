"""Describing a TEMPO matrix without downloading it."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from ._api import (
    ancestor,
    dimension_names,
    first_field,
    matrix_metadata,
    metadata_date,
    metadata_update_string,
    option_ids,
    option_labels,
    plan_query,
    strip_html_one,
    validate_code,
    validate_flag,
    validate_language,
)
from .errors import TempoError
from .options import get_option

__all__ = ["Meta", "meta"]


class Meta:
    """The structure of one TEMPO matrix, as returned by :func:`meta`.

    Attributes:
        code: The matrix code.
        name: The matrix name, or ``None``.
        language: The language the labels are in.
        last_update: Publication date, or ``None`` when the service reports a
            date this package cannot parse.
        service_update: The service's own update string, verbatim.
        domain: Statistical domain, as plain text.
        sub_domain: Statistical sub-domain, as plain text.
        survey: Survey name, as plain text.
        dimensions: A data frame with one row per dimension, giving its
            ``position``, ``name`` and number of ``values``.
        values: One data frame per dimension, keyed by dimension name, each
            holding the ``id`` and ``label`` of every selectable value.  A
            duplicated dimension name gets a ``" (2)"``-style suffix here; use
            :attr:`value_frames` to index by position instead.
        value_frames: The same data frames, in dimension order.
        cells: Number of cells a full download would request.
        requests: Number of HTTP requests a full download would need under the
            current query-size options, or ``None`` when that is above
            ``max_requests``.
        details: The service's query parameters (``matMaxDim``, ``matUMSpec``).
        raw: The unparsed service response.

    The unit of measure:
        The service exposes the unit of measure as a dimension of its own,
        which is why a matrix measured in persons reports a final dimension
        such as ``MU: Persons`` with a single value.  It is part of the query,
        so it appears here; :func:`tempo_ins.tidy` moves it out of the data
        into a ``unit`` column.

    Example:
        >>> import tempo_ins as tempo
        >>> info = tempo.meta("ACC101B", language="en")   # doctest: +SKIP
        >>> info.dimensions                               # doctest: +SKIP
        >>> info["Counties"].head()                       # doctest: +SKIP
    """

    __slots__ = (
        "code",
        "name",
        "language",
        "last_update",
        "service_update",
        "domain",
        "sub_domain",
        "survey",
        "dimensions",
        "values",
        "value_frames",
        "cells",
        "requests",
        "details",
        "raw",
    )

    def __init__(
        self,
        code: str,
        name: str | None,
        language: str,
        last_update: date | None,
        service_update: str | None,
        domain: str | None,
        sub_domain: str | None,
        survey: str | None,
        dimensions: pd.DataFrame,
        values: dict[str, pd.DataFrame],
        value_frames: list[pd.DataFrame],
        cells: int,
        requests: int | None,
        details: dict[str, Any],
        raw: dict[str, Any],
    ) -> None:
        self.code = code
        self.name = name
        self.language = language
        self.last_update = last_update
        self.service_update = service_update
        self.domain = domain
        self.sub_domain = sub_domain
        self.survey = survey
        self.dimensions = dimensions
        self.values = values
        self.value_frames = value_frames
        self.cells = cells
        self.requests = requests
        self.details = details
        self.raw = raw

    def __getitem__(self, key: int | str) -> pd.DataFrame:
        """The selectable values of one dimension, by name or 0-based position."""
        if isinstance(key, int):
            return self.value_frames[key]
        return self.values[key]

    def __len__(self) -> int:
        return len(self.value_frames)

    def to_dict(self) -> dict[str, Any]:
        """Every field except :attr:`raw`, as a plain dictionary."""
        return {name: getattr(self, name) for name in self.__slots__ if name != "raw"}

    def __repr__(self) -> str:
        lines = [f"<tempo_meta> {self.code}"]
        if self.name:
            lines[0] += f" — {self.name}"

        classification = [
            value for value in (self.domain, self.sub_domain, self.survey) if value
        ]
        if classification:
            lines.append("  " + " > ".join(classification))

        lines.append(f"  language:    {self.language}")

        if self.last_update is None:
            last = f'unknown (service reported "{self.service_update}")'
        else:
            last = self.last_update.isoformat()
        lines.append(f"  last update: {last}")

        if self.requests is None:
            budget = float(get_option("max_requests"))
            lines.append(
                f"  full table:  {self.cells:,} cells, over {budget:,.0f} requests"
            )
            lines.append(
                "               too large to fetch whole; select values with get()"
            )
        else:
            plural = "" if self.requests == 1 else "s"
            lines.append(
                f"  full table:  {self.cells:,} cells, {self.requests} request{plural}"
            )

        lines.append(f"  dimensions:  {len(self.dimensions)}")
        for position, row in enumerate(self.dimensions.itertuples(index=False)):
            examples = [
                label
                for label in self.value_frames[position]["label"].tolist()[:3]
                if label is not None
            ]
            plural = "" if row.values == 1 else "s"
            shown = ""
            if examples:
                more = ", ..." if row.values > len(examples) else ""
                shown = ": " + ", ".join(examples) + more
            lines.append(
                f"    {row.position}. {row.name} ({row.values} value{plural}){shown}"
            )

        return "\n".join(lines)

    __str__ = __repr__


def meta(code: str, language: str = "ro", cache: bool = True) -> Meta:
    """Describe a TEMPO matrix without downloading it.

    Returns the structure of a single matrix: its dimensions, the selectable
    values of each dimension, the publication date, the position in the
    statistical classification, and how large a full download would be.  This
    is the information needed to choose sensible ``filters`` for
    :func:`tempo_ins.get`, and it answers "how big is this table?" in one
    request instead of a download.

    Args:
        code: A single TEMPO matrix code, such as ``"ACC101B"``.  Use
            :func:`tempo_ins.toc` or :func:`tempo_ins.search` to find one.
        language: ``"ro"`` for Romanian (the default) or ``"en"`` for English.
            Dimension and value labels are returned in this language.
        cache: Whether to reuse and populate the response cache described in
            :func:`tempo_ins.cache_dir`.

    Returns:
        A :class:`Meta` object.

    Undocumented field names:
        TEMPO Online has no published API contract.  The dimension and value
        labels are read from the first of several plausible field names
        (``label``, ``name``, ``nomItemName``, ``text``), and dimensions fall
        back to ``dimension_1``, ``dimension_2``, ... if none of them is
        present.  The value identifiers, which the query payload actually
        needs, come from the documented-by-use ``nomItemId`` field and are
        validated.

    See Also:
        :func:`tempo_ins.get` to download a subset, :func:`tempo_ins.bulk` for
        whole tables, :func:`tempo_ins.toc` to list matrices.

    Example:
        >>> import tempo_ins as tempo
        >>> info = tempo.meta("ACC101B", language="en")   # doctest: +SKIP
        >>> info.cells                                    # doctest: +SKIP
    """
    code = validate_code(code)
    language = validate_language(language)
    cache = validate_flag(cache, "cache")

    metadata = matrix_metadata(code, language, cache=cache)

    names = dimension_names(metadata, code)
    ids = option_ids(metadata, code)
    labels = option_labels(metadata, code)

    value_frames = [
        pd.DataFrame({"id": ids[index], "label": labels[index]})
        for index in range(len(ids))
    ]

    keyed: dict[str, pd.DataFrame] = {}
    for index, name in enumerate(names):
        key = name if name is not None else f"dimension_{index + 1}"
        if key in keyed:
            suffix = 2
            while f"{key} ({suffix})" in keyed:
                suffix += 1
            key = f"{key} ({suffix})"
        keyed[key] = value_frames[index]

    sizes = [len(group) for group in ids]

    try:
        plan = plan_query(ids, code=code)
        requests_needed: int | None = plan.requests
    except TempoError:
        requests_needed = None

    cells = 1
    for size in sizes:
        cells *= size

    details = metadata.get("details") or {}

    return Meta(
        code=code,
        name=first_field(metadata, ("matrixName", "name", "nume", "label")),
        language=language,
        last_update=metadata_date(metadata, code=code),
        service_update=metadata_update_string(metadata),
        domain=strip_html_one(ancestor(metadata, 2)),
        sub_domain=strip_html_one(ancestor(metadata, 3)),
        survey=strip_html_one(ancestor(metadata, 4)),
        dimensions=pd.DataFrame(
            {
                "position": list(range(1, len(names) + 1)),
                "name": names,
                "values": sizes,
            }
        ),
        values=keyed,
        value_frames=value_frames,
        cells=cells,
        requests=requests_needed,
        details={
            "matMaxDim": details.get("matMaxDim"),
            "matUMSpec": details.get("matUMSpec"),
        },
        raw=metadata,
    )
