"""Downloading part of a TEMPO Online table into memory."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ._api import (
    dimension_names,
    fetch_matrix,
    matrix_metadata,
    metadata_update_string,
    option_ids,
    option_labels,
    plan_query,
    validate_code,
    validate_flag,
    validate_language,
)
from ._tidy import read_csv_text
from ._tidy import tidy as _tidy
from .errors import TempoInputError
from .options import progress_enabled

__all__ = ["get"]


def _normalise(value: Any) -> str:
    return str(value).strip().lower()


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _match_dimension(key: Any, names: Sequence[str | None]) -> int:
    """Resolve a filter key to a 0-based dimension index."""
    exact = [
        index for index, name in enumerate(names) if _normalise(name) == _normalise(key)
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise TempoInputError(
            f'`filters` name "{key}" matches {len(exact)} dimensions; use the '
            "position instead."
        )

    position = _as_int(key)
    if position is not None and 1 <= position <= len(names):
        return position - 1

    pattern = re.escape(str(key).strip())
    partial = [
        index
        for index, name in enumerate(names)
        if name is not None and re.search(pattern, name, re.IGNORECASE)
    ]
    if len(partial) == 1:
        return partial[0]

    available = "; ".join(f"{index + 1}. {name}" for index, name in enumerate(names))
    raise TempoInputError(
        f'`filters` names an unknown dimension "{key}". '
        f"Available dimensions: {available}."
    )


def _wanted_values(wanted: Any) -> list[str]:
    if isinstance(wanted, (str, bytes)):
        candidates: list[Any] = [wanted]
    elif isinstance(wanted, Iterable):
        candidates = list(wanted)
    else:
        candidates = [wanted]

    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in candidates
        if value is not None and not (isinstance(value, float) and pd.isna(value))
    ]


def _match_values(
    wanted: Any,
    ids: Sequence[str],
    labels: Sequence[str | None],
    dimension: str | None,
) -> list[str]:
    """Resolve a filter's values to the service's own value identifiers."""
    values = _wanted_values(wanted)
    if not values:
        raise TempoInputError(
            f'`filters` for dimension "{dimension}" selects no values.'
        )

    chosen: list[int] = []
    unmatched: list[str] = []

    for value in values:
        hits = [
            index
            for index, label in enumerate(labels)
            if _normalise(label) == _normalise(value)
        ]

        if not hits:
            # Whole-token match, so that 2020 finds "Anul 2020" but not "12020".
            token = re.compile(
                rf"(^|[^0-9A-Za-z]){re.escape(value.strip())}([^0-9A-Za-z]|$)",
                re.IGNORECASE,
            )
            hits = [
                index
                for index, label in enumerate(labels)
                if label is not None and token.search(label)
            ]

        if not hits:
            hits = [
                index
                for index, identifier in enumerate(ids)
                if identifier == value.strip()
            ]

        if hits:
            chosen.extend(hits)
        else:
            unmatched.append(value)

    if unmatched:
        available = [label for label in labels if label is not None]
        shown = available[:10]
        extra = (
            f" (first {len(shown)} of {len(available)})"
            if len(available) > len(shown)
            else ""
        )
        raise TempoInputError(
            f'No value of dimension "{dimension}" matches '
            + ", ".join(f'"{value}"' for value in unmatched)
            + f". Available values{extra}: "
            + "; ".join(shown)
            + "."
        )

    return [ids[index] for index in sorted(set(chosen))]


def resolve_filters(
    filters: Mapping[Any, Any] | None,
    metadata: dict[str, Any],
    code: str,
) -> list[list[str]]:
    """Turn a filter mapping into one list of value identifiers per dimension."""
    ids = option_ids(metadata, code)

    if not filters:
        return ids

    if not isinstance(filters, Mapping):
        raise TempoInputError(
            "`filters` must be a mapping, where each key is a dimension of "
            "the matrix and each value selects from that dimension, as in "
            '{"Judete": "Cluj", "Ani": range(2015, 2025)}.'
        )

    if any(key is None or str(key).strip() == "" for key in filters):
        raise TempoInputError(
            "`filters` keys must each name a dimension of the matrix."
        )

    names = dimension_names(metadata, code)
    labels = option_labels(metadata, code)
    selection = list(ids)

    for key, wanted in filters.items():
        position = _match_dimension(key, names)
        if wanted is None:
            continue

        selection[position] = _match_values(
            wanted, ids[position], labels[position], names[position]
        )

    return selection


def get(
    code: str,
    filters: Mapping[Any, Any] | None = None,
    language: str = "ro",
    tidy: bool = True,
    cache: bool = True,
    progress: bool | None = None,
    **tidy_options: Any,
) -> pd.DataFrame:
    """Download a TEMPO Online table, or part of one, into a data frame.

    Nothing is written to disk.  Unlike :func:`tempo_ins.bulk`, which always
    downloads a whole table to CSV, this sends the dimension selection to the
    service, so asking for one county and ten years costs one small request
    rather than a download of the entire table.

    Args:
        code: A single TEMPO matrix code, such as ``"ACC101B"``.
        filters: A mapping selecting values from one or more dimensions, or
            ``None`` for the complete table.  Keys are dimension names as
            reported by :func:`tempo_ins.meta`; a partial, case-insensitive
            name or a 1-based dimension position also works.  Dimensions you do
            not mention, or set to ``None``, are taken whole.  See *Matching*.
        language: ``"ro"`` for Romanian (the default) or ``"en"`` for English.
            Dimension names, value labels and the values you filter on are all
            in this language.
        tidy: Whether to pass the result through :func:`tempo_ins.tidy`: a
            numeric ``value`` column, the unit of measure as a column, parsed
            periods and no footer rows.  Pass ``False`` to get the service's
            own CSV layout, every column a string.
        cache: Passed to the metadata lookup.  The data itself is never cached.
        progress: Whether to report progress, or ``None`` to follow the
            ``progress`` option.
        **tidy_options: Further keyword arguments for :func:`tempo_ins.tidy`,
            such as ``periods=False`` or ``clean_names=True``.

    Returns:
        A data frame.  ``DataFrame.attrs`` records the provenance of the
        result under ``tempo_code``, ``tempo_language``, ``tempo_requests``,
        ``tempo_service_update`` and ``tempo_retrieved_at``.

    Matching:
        Each element of a filter is matched against the dimension's value
        labels, in three steps: an exact match ignoring case and surrounding
        space; then a whole-token match, so ``2020`` finds ``"Anul 2020"``;
        then an exact match against the service's own numeric value
        identifiers.  An element that matches nothing is an error listing the
        available values, rather than a silently smaller result.

    Size:
        A selection still has to fit the service's query limits, and a large
        one is split into several requests exactly as in
        :func:`tempo_ins.bulk`.  Call :func:`tempo_ins.meta` first to see how
        many cells and requests a selection implies.

    See Also:
        :func:`tempo_ins.meta` for the dimensions available to filter on,
        :func:`tempo_ins.bulk` for whole tables as CSV files,
        :func:`tempo_ins.tidy` for the cleaning rules.

    Example:
        >>> import tempo_ins as tempo
        >>> cluj = tempo.get(
        ...     "ACC101B",
        ...     filters={"Counties": "Cluj", "Years": range(2015, 2025)},
        ...     language="en",
        ... )                                              # doctest: +SKIP
    """
    code = validate_code(code)
    language = validate_language(language)
    tidy = validate_flag(tidy, "tidy")
    cache = validate_flag(cache, "cache")
    if progress is not None:
        progress = validate_flag(progress, "progress")

    metadata = matrix_metadata(code, language, cache=cache)
    selection = resolve_filters(filters, metadata, code)
    plan = plan_query(selection, code=code)

    show_progress = progress_enabled() if progress is None else progress
    label = f"Fetching {code}" if show_progress and plan.requests > 1 else None

    fetched = fetch_matrix(
        code=code,
        language=language,
        metadata=metadata,
        plan=plan,
        progress_label=label,
    )

    result = read_csv_text(fetched.text)

    if tidy:
        result = _tidy(result, code=code, **tidy_options)

    result.attrs["tempo_code"] = code
    result.attrs["tempo_language"] = language
    result.attrs["tempo_requests"] = fetched.requests
    result.attrs["tempo_service_update"] = metadata_update_string(metadata)
    result.attrs["tempo_retrieved_at"] = datetime.now(timezone.utc)
    return result
