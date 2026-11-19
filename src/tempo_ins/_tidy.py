"""Turn the service's CSV layout into analysis-ready data."""

from __future__ import annotations

import io
import os
import re
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from ._api import validate_flag
from .errors import TempoInputError, TempoTidyWarning, warn_once

__all__ = ["tidy", "read_csv_text", "read_csv_file"]

_ISSUES = "https://github.com/bogdanoancea/PyTEMPO/issues"

_READ_OPTIONS: dict[str, Any] = {
    "sep": ",",
    "dtype": str,
    "keep_default_na": False,
    "na_values": [],
}


def read_csv_text(text: str) -> pd.DataFrame:
    """Read CSV text exactly as the service sent it: every column a string."""
    return pd.read_csv(io.StringIO(text), **_READ_OPTIONS)


def read_csv_file(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a CSV file written by :func:`tempo_ins.bulk`."""
    return pd.read_csv(path, encoding="utf-8", **_READ_OPTIONS)


# Period labels ---------------------------------------------------------------
#
# The service labels time values in words: "Anul 2020", "luna ianuarie",
# "Trimestrul III".  Parsing is best-effort and deliberately matches on the
# month or ordinal itself rather than on an exact label format, because the
# label layout is not contractual.

_MONTHS: dict[str, int] = {
    "ianuarie": 1,
    "februarie": 2,
    "martie": 3,
    "aprilie": 4,
    "mai": 5,
    "iunie": 6,
    "iulie": 7,
    "august": 8,
    "septembrie": 9,
    "octombrie": 10,
    "noiembrie": 11,
    "decembrie": 12,
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_ROMAN: dict[str, int] = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
    "xi": 11,
    "xii": 12,
}

_YEAR_PREFIX_RE = re.compile(r"^(anul|ani|year|years)\s*")
_YEAR_ANYWHERE_RE = re.compile(r"(18|19|20|21)[0-9]{2}")
_YEAR_TOKEN_RE = re.compile(r"(^|[^0-9])(18|19|20|21)[0-9]{2}([^0-9]|$)")

_MONTH_RES: dict[str, re.Pattern[str]] = {
    name: re.compile(rf"(^|[^a-z]){re.escape(name)}([^a-z]|$)") for name in _MONTHS
}

_QUARTER_RE = re.compile(
    r"(?:^|[^a-z])(?:trimestr[a-z]*|trim\.?|quarters?|q)[^a-z0-9]*([ivx]+|[0-9]+)"
)
_SEMESTER_RE = re.compile(
    r"(?:^|[^a-z])(?:semestr[a-z]*|sem\.?|semesters?)[^a-z0-9]*([ivx]+|[0-9]+)"
)


def _normalise(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _as_int(text: str) -> int | None:
    """Parse an integer, tolerating a decimal point, or return ``None``."""
    try:
        return int(text)
    except (TypeError, ValueError):
        pass

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _map_unique(values: Iterable[Any], parse) -> list[Any]:
    """Apply ``parse`` to each distinct value once, then map back.

    Time columns have few distinct values even in very large tables, so parsing
    the unique set and mapping is what keeps tidying a 700,000-row download
    fast.
    """
    cache: dict[str, Any] = {}
    result = []
    for value in values:
        key = _normalise(value)
        if key not in cache:
            cache[key] = parse(key)
        result.append(cache[key])
    return result


def _parse_year_one(value: str) -> int | None:
    direct = _as_int(_YEAR_PREFIX_RE.sub("", value))
    if direct is not None:
        return direct

    match = _YEAR_ANYWHERE_RE.search(value)
    return int(match.group(0)) if match else None


def parse_year(values: Iterable[Any]) -> list[int | None]:
    """Read a year out of each label, as in ``"Anul 2020"`` or ``"2020"``."""
    return _map_unique(values, _parse_year_one)


def _parse_month_one(value: str) -> int | None:
    if not value:
        return None
    for name, pattern in _MONTH_RES.items():
        if pattern.search(value):
            return _MONTHS[name]
    return None


def parse_month(values: Iterable[Any]) -> list[int | None]:
    """Read a month number out of each label, in Romanian or English."""
    return _map_unique(values, _parse_month_one)


def _parse_ordinal_one(
    value: str,
    pattern: re.Pattern[str],
    maximum: int,
) -> int | None:
    match = pattern.search(value)
    if match is None:
        return None

    token = match.group(1)
    parsed = int(token) if token.isdigit() else _ROMAN.get(token)
    if parsed is None or parsed < 1 or parsed > maximum:
        return None

    return parsed


def parse_quarter(values: Iterable[Any]) -> list[int | None]:
    """Read a quarter number out of each label, Arabic or Roman."""
    return _map_unique(values, lambda value: _parse_ordinal_one(value, _QUARTER_RE, 4))


def parse_semester(values: Iterable[Any]) -> list[int | None]:
    """Read a semester number out of each label, Arabic or Roman."""
    return _map_unique(values, lambda value: _parse_ordinal_one(value, _SEMESTER_RE, 2))


_YEAR_NAME_RE = re.compile(r"^(ani|anul|anii|years?|year)$")
_MONTH_NAME_RE = re.compile(r"^(luni|luna|lunile|months?|month)$")
_PERIOD_NAME_RE = re.compile(r"^(perioade|perioada|periods?|period)$")


def time_role(name: Any, values: Sequence[Any]) -> str | None:
    """Classify a column as a time column, by name first and then by content.

    TEMPO uses two time layouts.  Some matrices carry a dedicated
    ``Ani``/``Years`` dimension, optionally alongside ``Luni``/``Months``.
    Others carry a single ``Perioade``/``Periods`` dimension whose labels mix
    frequencies, as in ``"Year 2010"``, ``"January 2010"``,
    ``"February 2010"``.  The second layout must not be treated as a year
    column: reducing every label to its year would make an annual total and a
    January value indistinguishable.

    Returns one of ``"year"``, ``"month"``, ``"quarter"``, ``"semester"``,
    ``"period"`` or ``None``.
    """
    normalised = _normalise(name)
    if _YEAR_NAME_RE.match(normalised):
        return "year"
    if _MONTH_NAME_RE.match(normalised):
        return "month"
    if "trimestr" in normalised or "quarter" in normalised:
        return "quarter"
    if "semestr" in normalised or "semester" in normalised:
        return "semester"
    if _PERIOD_NAME_RE.match(normalised):
        return "period"

    distinct = []
    seen = set()
    for value in values:
        text = "" if value is None else str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        distinct.append(text)

    if not distinct:
        return None

    months = parse_month(distinct)
    quarters = parse_quarter(distinct)
    years = parse_year(distinct)

    if len(distinct) <= 12 and all(value is not None for value in months):
        return "month"
    if len(distinct) <= 4 and all(value is not None for value in quarters):
        return "quarter"

    years_complete = all(value is not None for value in years)

    # A year together with a month or quarter in the same label.
    if years_complete and (
        any(value is not None for value in months)
        or any(value is not None for value in quarters)
    ):
        return "period"

    if years_complete and all(_YEAR_TOKEN_RE.search(value) for value in distinct):
        return "year"

    return None


_UNIT_HEADER_RE = re.compile(r"^(um|mu)\s*:", re.IGNORECASE)
_UNIT_PREFIX_RE = re.compile(r"^(um|mu)\s*:\s*", re.IGNORECASE)
_INVALID_NAME_RE = re.compile(r"[^0-9A-Za-z_]")


def _make_names(names: Iterable[Any]) -> list[str]:
    """Turn column names into unique, usable Python identifiers."""
    result: list[str] = []
    taken: set = set()

    for name in names:
        candidate = _INVALID_NAME_RE.sub("_", str(name))
        if not candidate or not (candidate[0].isalpha() or candidate[0] == "_"):
            candidate = "x" + candidate

        unique = candidate
        suffix = 1
        while unique in taken:
            unique = f"{candidate}_{suffix}"
            suffix += 1

        taken.add(unique)
        result.append(unique)

    return result


def _to_text(value: Any) -> str:
    """Render one cell as the service would have sent it."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):  # pragma: no cover - array-like cell
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _as_table(x: Any) -> pd.DataFrame:
    """Accept a data frame, a CSV path or a string of CSV text."""
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(
            {name: x[name].map(_to_text) for name in x.columns},
            index=x.index,
            columns=x.columns,
        )

    if isinstance(x, str):
        if "\r" in x or "\n" in x:
            return read_csv_text(x)
        if os.path.exists(x):
            return read_csv_file(x)
        raise TempoInputError(f"`x` is neither CSV text nor an existing file: {x}")

    if isinstance(x, os.PathLike):
        path = os.fspath(x)
        if os.path.exists(path):
            return read_csv_file(path)
        raise TempoInputError(f"`x` is not an existing file: {path}")

    raise TempoInputError(
        "`x` must be a data frame, the path of a CSV file, or a single string "
        "of CSV text."
    )


def _to_date(year: int | None, month: int | None):
    """A first-of-month date, or ``None`` when either part is unusable."""
    if year is None or month is None:
        return None
    try:
        return pd.Timestamp(year=int(year), month=int(month), day=1)
    except (ValueError, OverflowError):
        return None


def tidy(
    x: Any,
    code: str | None = None,
    periods: bool = True,
    na: str | Sequence[str] = (":", ""),
    drop_notes: bool = True,
    clean_names: bool = False,
    value_name: str = "value",
    unit_name: str = "unit",
) -> pd.DataFrame:
    """Turn a downloaded TEMPO table into analysis-ready data.

    The CSV that TEMPO Online returns is not directly usable: the observation
    column is text, the unit of measure hides in a column *header*, years read
    ``"Anul 2020"`` rather than ``2020``, and missing observations are marked
    ``":"``.  This resolves all of that and leaves one row per observation.

    Args:
        x: A data frame as returned by :func:`tempo_ins.get` with
            ``tidy=False``, the path of a CSV file written by
            :func:`tempo_ins.bulk`, or a single string holding CSV text.
        code: Optional matrix code, recorded on the result as the
            ``tempo_code`` entry of ``DataFrame.attrs``.
        periods: Whether to parse the time dimensions.  See *Time layouts*.
        na: Strings that mark a missing observation.  Defaults to ``(":", "")``,
            the service's own markers.
        drop_notes: Whether to drop rows in which every dimension column is
            empty.  Those are the footnote and copyright lines the service
            appends after the data; rows with a genuine missing observation are
            kept, because their dimension columns are filled.
        clean_names: Whether to pass the column names through a
            cleaner that makes them usable as Python identifiers.
            Defaults to ``False``, which keeps the service's own names.
        value_name: Name for the observation column.
        unit_name: Name for the unit-of-measure column.

    Returns:
        A data frame with one row per observation: the dimension columns as
        received, ``unit`` holding the unit of measure, ``value`` as a numeric
        column, and ``period`` when a sub-annual period could be
        reconstructed.  ``DataFrame.attrs`` carries ``tempo_code`` and
        ``tempo_unit`` when known.

    Whitespace:
        The service separates its CSV fields with ``", "``, so every column
        but the first arrives with a leading space in both its header and its
        values.  Both are stripped, so that the values match the labels
        :func:`tempo_ins.meta` reports and a frame is consistent across its
        columns.  Use ``tidy=False`` on :func:`tempo_ins.get` to see the
        service's own bytes instead.

    Time layouts:
        TEMPO uses two, and they are handled differently.

        Some matrices carry a dedicated ``Ani``/``Years`` dimension, optionally
        alongside ``Luni``/``Months``, ``Trimestre`` or ``Semestre``.  The year
        column becomes an integer, the sub-annual labels are left readable, and
        a ``period`` column of datetimes is added when the two can be combined.

        Others carry a single ``Perioade``/``Periods`` dimension whose labels
        mix frequencies, as in ``"Year 2010"``, ``"January 2010"``,
        ``"February 2010"``.  Here the label is left exactly as received,
        because it is the only thing distinguishing an annual total from a
        monthly figure, and two columns are added instead: ``year``, and
        ``period`` holding the first day of the month, quarter or semester
        named.  ``period`` is ``NaT`` for an annual row, so
        ``frame[frame["period"].notna()]`` selects the sub-annual rows and
        ``frame[frame["period"].isna()]`` the annual ones.

    Unparsed observations:
        Values that are neither numeric nor listed in ``na`` are reported once
        with a :class:`~tempo_ins.errors.TempoTidyWarning`, and the original
        text is kept in a ``value_raw`` column so nothing is lost.  This is how
        an unexpected flag or thousands separator becomes visible instead of
        turning quietly into a missing value.

    Example:
        >>> import tempo_ins as tempo
        >>> csv = "\\n".join([
        ...     "Judete,Ani,UM: Numar persoane,Valoare",
        ...     "Cluj,Anul 2019,Numar persoane,1234",
        ...     "Cluj,Anul 2020,Numar persoane,:",
        ...     ",,,(c) INS 1998-2026",
        ... ])
        >>> frame = tempo.tidy(csv, code="DEMO101A")
        >>> list(frame.columns)
        ['Judete', 'Ani', 'unit', 'value']
        >>> frame["Ani"].tolist()
        [2019, 2020]
    """
    periods = validate_flag(periods, "periods")
    drop_notes = validate_flag(drop_notes, "drop_notes")
    clean_names = validate_flag(clean_names, "clean_names")

    na_values = [na] if isinstance(na, str) else list(na)

    data = _as_table(x)
    if data.shape[1] == 0:
        raise TempoInputError("`x` holds no columns to tidy.")

    data = data.copy()
    data.columns = [str(name).strip() for name in data.columns]

    # The service separates its CSV fields with ", ", so every value but the
    # first column's arrives with a leading space: "Total", " Cluj", " TOTAL".
    # Left alone, a frame is internally inconsistent and its values no longer
    # match the labels in meta(), which the service supplies as JSON and which
    # are already collapsed and stripped -- so the values a caller filters by
    # would not equal the values they get back, and `frame["Counties"] ==
    # "Cluj"` would silently select nothing. The padding is a CSV formatting
    # artifact rather than data, so it goes. Pass tidy=False to get the
    # service's bytes untouched.
    for name in data.columns:
        column = data[name]
        if pd.api.types.is_object_dtype(column) or pd.api.types.is_string_dtype(column):
            data[name] = column.str.strip()

    # Unit of measure ---------------------------------------------------------
    unit: str | None = None
    unit_values: list[str | None] | None = None

    unit_positions = [
        index for index, name in enumerate(data.columns) if _UNIT_HEADER_RE.match(name)
    ]

    if unit_positions:
        position = unit_positions[0]
        unit = _UNIT_PREFIX_RE.sub("", data.columns[position]).strip() or None

        column = [str(value).strip() for value in data.iloc[:, position]]
        if any(column):
            unit_values = [value if value else None for value in column]
        else:
            unit_values = [unit] * len(data)

        keep = [index for index in range(data.shape[1]) if index != position]
        data = data.iloc[:, keep]

    # Observation column ------------------------------------------------------
    lowered = [name.lower() for name in data.columns]
    value_positions = [
        index for index, name in enumerate(lowered) if name in {"valoare", "value"}
    ]
    value_position = value_positions[0] if value_positions else None

    if value_position is None:
        warn_once(
            "tidy-no-value-column",
            'No "Valoare"/"Value" column found, so no numeric observation '
            "column was produced. Pass tidy=False to keep the service layout, "
            f"and please report the table at {_ISSUES}",
            TempoTidyWarning,
        )

    dimension_positions = [
        index for index in range(data.shape[1]) if index != value_position
    ]

    # Footer rows -------------------------------------------------------------
    if drop_notes and len(data) and dimension_positions:
        filled = pd.Series(False, index=data.index)
        for index in dimension_positions:
            column = data.iloc[:, index].map(
                lambda value: bool(_to_text(value).strip())
            )
            filled |= column.astype(bool)

        if unit_values is not None:
            unit_values = [
                value
                for value, keep in zip(unit_values, filled.tolist(), strict=True)
                if keep
            ]
        data = data[filled]

    data = data.reset_index(drop=True)

    # Periods -----------------------------------------------------------------
    period: list[Any] | None = None
    derived_year: list[int | None] | None = None

    if periods and len(data):
        year: list[int | None] | None = None
        year_column: str | None = None
        sub_annual: tuple[list[int | None], int] | None = None
        combined: dict[str, list[int | None]] | None = None

        for index in dimension_positions:
            role = time_role(data.columns[index], data.iloc[:, index].tolist())
            if role is None:
                continue

            column = data.iloc[:, index].tolist()

            if role == "year" and year is None:
                parsed = parse_year(column)
                if any(value is not None for value in parsed):
                    year = parsed
                    # Replace the label with the integer year, as the service's
                    # "Anul 2020" is never what a caller wants to compute with.
                    # Safe here because a year column carries nothing else.
                    if all(value is not None for value in parsed):
                        year_column = data.columns[index]
            elif role == "month" and sub_annual is None:
                parsed = parse_month(column)
                if any(value is not None for value in parsed):
                    sub_annual = (parsed, 1)
            elif role == "quarter" and sub_annual is None:
                parsed = parse_quarter(column)
                if any(value is not None for value in parsed):
                    sub_annual = (parsed, 3)
            elif role == "semester" and sub_annual is None:
                parsed = parse_semester(column)
                if any(value is not None for value in parsed):
                    sub_annual = (parsed, 6)
            elif role == "period" and combined is None:
                parsed_year = parse_year(column)
                if any(value is not None for value in parsed_year):
                    # The label is left exactly as received: it is the only
                    # thing that distinguishes an annual figure from a monthly
                    # one.
                    combined = {
                        "year": parsed_year,
                        "month": parse_month(column),
                        "quarter": parse_quarter(column),
                        "semester": parse_semester(column),
                    }

        if year_column is not None and year is not None:
            data[year_column] = pd.array(year, dtype="Int64")

        if combined is not None:
            months: list[int | None] = []
            for position in range(len(data)):
                if combined["month"][position] is not None:
                    months.append(combined["month"][position])
                elif combined["quarter"][position] is not None:
                    months.append((combined["quarter"][position] - 1) * 3 + 1)
                elif combined["semester"][position] is not None:
                    months.append((combined["semester"][position] - 1) * 6 + 1)
                else:
                    # None where the label names no sub-annual period, which is
                    # how an annual row stays distinguishable from a monthly one.
                    months.append(None)

            period = [
                _to_date(value, month)
                for value, month in zip(combined["year"], months, strict=True)
            ]
            derived_year = combined["year"]
        elif year is not None and sub_annual is not None:
            values, step = sub_annual
            period = [
                _to_date(value, None if number is None else (number - 1) * step + 1)
                for value, number in zip(year, values, strict=True)
            ]

    # Assemble ----------------------------------------------------------------
    if unit_values is not None:
        data[unit_name] = unit_values

    if value_position is not None:
        value_column = data.columns[value_position]
        raw = data[value_column].map(_to_text).str.strip()
        cleaned = raw.str.replace(r"\s", "", regex=True).str.replace(
            " ", "", regex=False
        )
        cleaned = cleaned.mask(raw.isin(na_values))

        numeric = pd.to_numeric(cleaned, errors="coerce")
        unexpected = numeric.isna() & cleaned.notna() & (cleaned != "")

        data = data.drop(columns=[value_column])
        data[value_name] = numeric

        if bool(unexpected.any()):
            examples = list(dict.fromkeys(raw[unexpected].tolist()))[:5]
            warn_once(
                "tidy-unparsed-values",
                f"{int(unexpected.sum())} observation(s) could not be read as "
                "numbers and became missing; the original text is kept in "
                "`value_raw`. Examples: "
                + ", ".join(f'"{example}"' for example in examples)
                + ".",
                TempoTidyWarning,
            )
            data["value_raw"] = raw

    if derived_year is not None and "year" not in data.columns:
        data["year"] = pd.array(derived_year, dtype="Int64")

    if period is not None:
        data["period"] = pd.to_datetime(pd.Series(period, dtype="object"))

    if clean_names:
        data.columns = _make_names(data.columns)

    data = data.reset_index(drop=True)
    data.attrs["tempo_code"] = code
    data.attrs["tempo_unit"] = unit
    return data
