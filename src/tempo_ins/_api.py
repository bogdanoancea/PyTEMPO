"""Service endpoints, metadata and query planning.

Nothing here is part of the public API; the user-facing functions live in
:mod:`tempo_ins.toc`, :mod:`tempo_ins.meta`, :mod:`tempo_ins.get`,
:mod:`tempo_ins.bulk`, :mod:`tempo_ins.hierarchy` and :mod:`tempo_ins.tidy`.

TEMPO Online publishes no API contract, so the accessors below accept the
plausible field spellings rather than assuming one, and report an unexpected
shape instead of guessing.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import date, datetime
from typing import (
    Any,
    NamedTuple,
)

import pandas as pd
import requests

from . import _cache as cache_module
from ._http import Reporter, request, request_many, response_text
from .errors import (
    TempoAPIError,
    TempoDateWarning,
    TempoInputError,
    TempoIOError,
    warn_once,
)
from .options import get_option

_ISSUES = "https://github.com/bogdanoancea/PyTEMPO/issues"


def base_url() -> str:
    """The configured base URL of the TEMPO Online service."""
    return str(get_option("base_url"))


def url(path: str, language: str | None = None) -> str:
    """Build a service URL, adding ``?lang=en`` for English."""
    built = f"{base_url()}/{path}"
    if language == "en":
        built += "?lang=en"
    return built


# Input validation ------------------------------------------------------------

_CODE_PATTERN = re.compile(r"^[A-Za-z0-9]+$")


def validate_language(language: Any) -> str:
    """Require ``"ro"`` or ``"en"``."""
    if not isinstance(language, str) or language not in {"ro", "en"}:
        raise TempoInputError('`language` must be either "ro" or "en".')
    return language


def validate_flag(value: Any, name: str) -> bool:
    """Require a real boolean, not something merely truthy."""
    if not isinstance(value, bool):
        # numpy.bool_ quacks like a bool but is not a subclass of it.
        if type(value).__name__ == "bool_" and hasattr(value, "item"):
            return bool(value.item())
        raise TempoInputError(f"`{name}` must be either True or False.")
    return value


def validate_codes(codes: Any) -> list[str]:
    """Normalise one or many matrix codes to a list of unique, valid codes."""
    if isinstance(codes, str):
        candidates: list[Any] = [codes]
    elif isinstance(codes, (bytes, bytearray)):
        raise TempoInputError("`codes` must contain at least one TEMPO matrix code.")
    elif isinstance(codes, Iterable):
        candidates = list(_flatten(codes))
    else:
        candidates = [codes]

    if not candidates or any(item is None for item in candidates):
        raise TempoInputError("`codes` must contain at least one TEMPO matrix code.")

    if not all(isinstance(item, str) for item in candidates):
        raise TempoInputError("`codes` must contain at least one TEMPO matrix code.")

    stripped = [item.strip() for item in candidates]
    if any(not item or not _CODE_PATTERN.match(item) for item in stripped):
        raise TempoInputError(
            "Each element of `codes` must contain only letters and numbers."
        )

    return list(dict.fromkeys(stripped))


def validate_code(code: Any) -> str:
    """Require exactly one matrix code."""
    resolved = validate_codes(code)
    if len(resolved) != 1:
        raise TempoInputError("`code` must be a single TEMPO matrix code.")
    return resolved[0]


def _flatten(values: Iterable[Any]) -> Iterator[Any]:
    for value in values:
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            yield value
        else:
            yield from _flatten(value)


# JSON helpers ----------------------------------------------------------------


def parse_json(response: requests.Response, context: str) -> Any:
    """Parse a response body as JSON, reporting the context on failure."""
    text = response_text(response, context)

    try:
        return json.loads(text)
    except ValueError as error:
        raise TempoAPIError(
            f"The TEMPO Online service returned invalid {context} JSON: {error}"
        )


def first_field(
    obj: Any,
    names: Sequence[str],
    default: str | None = None,
) -> str | None:
    """Pick the first non-empty scalar field from a list of candidate names.

    The service's field names are not formally documented, so callers pass the
    plausible spellings rather than assuming one.
    """
    if not isinstance(obj, dict):
        return default

    for name in names:
        value = obj.get(name)
        if value is None or isinstance(value, (dict, list, tuple)):
            continue
        text = str(value)
        if text:
            return text

    return default


# HTML in service labels ------------------------------------------------------
#
# The top-level statistical domains returned by the service embed HTML anchors,
# for example
#
#   AGRICULTURE <a href="https://insse.ro/...">Press releases</a>
#
# so the raw value is unusable in a data frame.  strip_html() keeps the text,
# extract_links() keeps the URLs.

_ANCHOR_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_RE = re.compile(r"^[;,\s]+|[;,\s]+$")
_HREF_RE = re.compile(r'href="([^"]*)"')

_ENTITIES = (
    ("&nbsp;", " "),
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
)


def _unescape(value: str) -> str:
    for entity, replacement in _ENTITIES:
        value = value.replace(entity, replacement)
    return value


def _tidy(value: str) -> str:
    value = _TAG_RE.sub("", value)
    value = _unescape(value)
    value = _WHITESPACE_RE.sub(" ", value).strip()
    return _EDGE_RE.sub("", value)


def strip_html_one(value: Any) -> str | None:
    """Reduce one service label to plain text.

    An anchor in a service label is usually navigation chrome appended to the
    real name, as in ``AGRICULTURE <a ...>Press releases</a>``, so the whole
    element goes.  A label that is nothing but a link would then vanish, so in
    that case the link text is kept instead.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value)
    without_anchors = _tidy(_ANCHOR_RE.sub(" ", text))
    if without_anchors:
        return without_anchors

    return _tidy(text)


def strip_html(values: Iterable[Any]) -> list[str | None]:
    """Reduce a sequence of service labels to plain text."""
    return [strip_html_one(value) for value in values]


def extract_links_one(value: Any, collapse: str = " ") -> str | None:
    """The ``href`` targets embedded in one service label, or ``None``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    matches = _HREF_RE.findall(str(value))
    if not matches:
        return None

    return collapse.join(matches)


def extract_links(values: Iterable[Any], collapse: str = " ") -> list[str | None]:
    """The ``href`` targets embedded in a sequence of service labels."""
    return [extract_links_one(value, collapse) for value in values]


# Table of contents -----------------------------------------------------------


def toc_from_response(response: requests.Response) -> pd.DataFrame:
    """Turn the matrix listing into a data frame of names and codes."""
    parsed = parse_json(response, "table-of-contents")

    if not isinstance(parsed, list) or not parsed:
        raise TempoAPIError(
            "The TEMPO Online service returned an unexpected table-of-contents format."
        )

    if not all(
        isinstance(row, dict) and "name" in row and "code" in row for row in parsed
    ):
        raise TempoAPIError(
            "The TEMPO Online service returned an unexpected table-of-contents format."
        )

    frame = pd.DataFrame(
        {
            "name": strip_html([row.get("name") for row in parsed]),
            "code": [str(row.get("code")) for row in parsed],
        }
    )

    comments = strip_html([row.get("comment") for row in parsed])
    comments = [value if value else None for value in comments]
    if any(value is not None for value in comments):
        frame["comment"] = comments

    return frame.reset_index(drop=True)


# Matrix metadata -------------------------------------------------------------


def matrix_metadata(
    code: str,
    language: str,
    cache: bool = True,
) -> dict[str, Any]:
    """Fetch (and optionally cache) the metadata of one matrix."""

    def fetch() -> dict[str, Any]:
        response = request(url(f"matrix/{code}", language=language))
        return validate_metadata(parse_json(response, f"metadata for `{code}`"), code)

    if not cache:
        return fetch()

    return cache_module.cached(
        cache_module.cache_key("matrix", code, language, base_url()),
        fetch,
    )


def validate_metadata(metadata: Any, code: str) -> dict[str, Any]:
    """Require the fields every later step depends on."""
    if (
        not isinstance(metadata, dict)
        or metadata.get("dimensionsMap") is None
        or metadata.get("details") is None
    ):
        raise TempoAPIError(
            f"The TEMPO Online service returned incomplete metadata for `{code}`.",
            code=code,
        )

    return metadata


def matrix_metadata_many(
    codes: Sequence[str],
    language: str,
    progress_label: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch the metadata of several matrices, using the concurrent pool."""
    keys = [
        cache_module.cache_key("matrix", code, language, base_url()) for code in codes
    ]
    metadata: list[dict[str, Any] | None] = [
        cache_module.cache_get(key) for key in keys
    ]
    missing = [index for index, value in enumerate(metadata) if value is None]

    if missing:
        urls = [url(f"matrix/{codes[index]}", language=language) for index in missing]
        responses = request_many(urls, progress_label=progress_label)

        for position, index in enumerate(missing):
            parsed = validate_metadata(
                parse_json(responses[position], f"metadata for `{codes[index]}`"),
                codes[index],
            )
            metadata[index] = parsed
            cache_module.cache_set(keys[index], parsed)

    # A short return here would silently misalign every caller that zips the
    # result against `codes` -- toc(full_description=True) would attach each
    # publication date to the wrong matrix. Refuse instead.
    if any(value is None for value in metadata):
        absent = [codes[i] for i, value in enumerate(metadata) if value is None]
        raise TempoAPIError(
            "The TEMPO Online service returned no metadata for "
            + ", ".join(f"`{code}`" for code in absent)
            + "."
        )

    return [value for value in metadata if value is not None]


def metadata_update_string(metadata: dict[str, Any]) -> str | None:
    """The service's own update string, verbatim."""
    value = metadata.get("ultimaActualizare")
    if value is None or isinstance(value, (dict, list)):
        return None

    return str(value).strip()


#: The service has published "dd-mm-yyyy" historically, but the format is not
#: contractual, so the plausible layouts are all tried.
DATE_FORMATS = ("%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d")


def metadata_date(
    metadata: dict[str, Any],
    code: str | None = None,
    warn: bool = True,
) -> date | None:
    """Parse the matrix's publication date, or report that it cannot be parsed.

    Returns ``None`` when the service reports a layout this package does not
    recognise, after warning once per distinct value rather than silently
    re-downloading for ever.
    """
    value = metadata_update_string(metadata)
    if not value:
        return None

    for candidate in (value, value.split()[0] if value.split() else value):
        for pattern in DATE_FORMATS:
            try:
                return datetime.strptime(candidate, pattern).date()
            except ValueError:
                continue

    if warn:
        where = "" if code is None else f" reported for `{code}`"
        warn_once(
            f"date-format-{value}",
            f'Could not parse the update date "{value}"{where}. Freshness '
            "checks fall back to the service's raw update string, and "
            f"`last_update` is None. Please report this at {_ISSUES}",
            TempoDateWarning,
        )

    return None


def ancestor(metadata: dict[str, Any], position: int) -> str | None:
    """The name of the matrix's ancestor at ``position``, counting from 1."""
    ancestors = metadata.get("ancestors")
    if not isinstance(ancestors, list) or len(ancestors) < position:
        return None

    return first_field(ancestors[position - 1], ("name", "label", "context"))


# Dimensions ------------------------------------------------------------------


def dimensions(metadata: dict[str, Any], code: str) -> list[Any]:
    """The matrix's dimensions, as the service describes them."""
    found = metadata.get("dimensionsMap")
    if not isinstance(found, list) or not found:
        raise TempoAPIError(
            f"Matrix `{code}` has no dimensions to download.", code=code
        )

    return found


def dimension_names(metadata: dict[str, Any], code: str) -> list[str | None]:
    """The dimension labels, as plain text."""
    found = dimensions(metadata, code)
    names = [
        first_field(
            dimension,
            ("label", "name", "dimName", "nomName", "text"),
            default=f"dimension_{index + 1}",
        )
        for index, dimension in enumerate(found)
    ]

    return strip_html(names)


def _options_of(dimension: Any, code: str, position: int) -> list[Any]:
    options = dimension.get("options") if isinstance(dimension, dict) else None
    if not isinstance(options, list) or not options:
        raise TempoAPIError(
            f"Matrix `{code}` has no selectable options in dimension {position}.",
            code=code,
        )
    return options


def option_ids(metadata: dict[str, Any], code: str) -> list[list[str]]:
    """The service's own value identifiers, which the query payload needs."""
    found = dimensions(metadata, code)
    result: list[list[str]] = []

    for index, dimension in enumerate(found):
        position = index + 1
        options = _options_of(dimension, code, position)

        ids: list[str] = []
        valid = True
        for option in options:
            value = option.get("nomItemId") if isinstance(option, dict) else None
            if value is None or isinstance(value, (dict, list)):
                valid = False
                break
            text = str(value).strip()
            if not text:
                valid = False
                break
            ids.append(text)

        if not valid:
            raise TempoAPIError(
                f"Matrix `{code}` contains an invalid option identifier in "
                f"dimension {position}.",
                code=code,
            )

        result.append(ids)

    return result


def option_labels(metadata: dict[str, Any], code: str) -> list[list[str | None]]:
    """The value labels of every dimension, as plain text."""
    found = dimensions(metadata, code)

    return [
        strip_html(
            [
                first_field(option, ("label", "name", "nomItemName", "text"))
                for option in _options_of(dimension, code, index + 1)
            ]
        )
        for index, dimension in enumerate(found)
    ]


# Query planning --------------------------------------------------------------
#
# The service refuses or times out on very large queries.  Two limits matter:
#
#   * a per-dimension cap on how many options one query may select;
#   * the total number of cells the query asks for, which is the product of the
#     selected option counts across all dimensions.
#
# Chunking each dimension independently against a fixed threshold therefore
# splits small matrices needlessly while letting genuinely huge requests
# through untouched: five dimensions of 200 options each is 3.2e11 cells in a
# single request, yet no single dimension crosses a 300-option threshold.
# plan_query() budgets on the product instead, shrinking every dimension to a
# common cap until the query fits.


class QueryPlan(NamedTuple):
    """How a download is split into service queries."""

    chunks: list[list[list[str]]]
    cells: int
    requests: int
    chunk_sizes: list[int]


def chunk(values: Sequence[str], size: int) -> list[list[str]]:
    """Split ``values`` into consecutive chunks of at most ``size`` elements."""
    size = max(1, int(size))
    if len(values) <= size:
        return [list(values)]

    return [list(values[start : start + size]) for start in range(0, len(values), size)]


def plan_query(
    ids: Sequence[Sequence[str]],
    code: str | None = None,
    max_cells: float | None = None,
    max_options: int | None = None,
    max_requests: float | None = None,
) -> QueryPlan:
    """Plan the requests needed to download the given option selection.

    Raises:
        TempoAPIError: A dimension has no selectable options.
        TempoInputError: The download would need more than ``max_requests``
            requests.
    """
    sizes = [len(options) for options in ids]
    if not sizes or any(size == 0 for size in sizes):
        raise TempoAPIError(
            f"Matrix `{code or 'unknown'}` has no selectable options to download.",
            code=code,
        )

    cell_budget = max(
        1.0, float(get_option("max_cells") if max_cells is None else max_cells)
    )
    option_cap = max(
        1,
        int(get_option("max_options_per_dim") if max_options is None else max_options),
    )
    request_budget = float(
        get_option("max_requests") if max_requests is None else max_requests
    )

    caps = [min(size, option_cap) for size in sizes]

    def fits(k: int) -> bool:
        """Whether capping every dimension at ``k`` keeps the query in budget."""
        product = 1.0
        for cap in caps:
            product *= min(cap, k)
            if product > cell_budget:
                return False
        return True

    # Capping every dimension at the same k keeps the split balanced, which
    # needs fewer requests than shrinking one dimension to the bone.
    if fits(max(caps)):
        chunk_sizes = list(caps)
    else:
        low, high = 1, max(caps)
        while low < high:
            middle = (low + high + 1) // 2
            if fits(middle):
                low = middle
            else:
                high = middle - 1
        chunk_sizes = [min(cap, low) for cap in caps]

    requests_needed = 1
    for size, step in zip(sizes, chunk_sizes, strict=True):
        requests_needed *= math.ceil(size / step)

    if requests_needed > request_budget:
        raise TempoInputError(
            f"Downloading matrix `{code or 'unknown'}` would need "
            f"{requests_needed:.0f} requests, above the limit of "
            f"{request_budget:.0f}. Select fewer values with get(), or raise "
            "set_options(max_requests=...) if you are sure.",
            code=code,
            requests=float(requests_needed),
        )

    cells = 1
    for size in sizes:
        cells *= size

    return QueryPlan(
        chunks=[
            chunk(options, step) for options, step in zip(ids, chunk_sizes, strict=True)
        ],
        cells=cells,
        requests=int(requests_needed),
        chunk_sizes=chunk_sizes,
    )


def each_selection(
    chunks: Sequence[Sequence[Sequence[str]]],
) -> Iterator[tuple[Sequence[str], ...]]:
    """Walk the Cartesian product of the per-dimension chunks."""
    return itertools.product(*chunks)


def payload(
    code: str,
    language: str,
    metadata: dict[str, Any],
    selection: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """The JSON body the ``pivot`` endpoint expects."""
    details = metadata.get("details")
    if (
        not isinstance(details, dict)
        or details.get("matMaxDim") is None
        or details.get("matUMSpec") is None
    ):
        raise TempoAPIError(
            f"Matrix `{code}` has incomplete query metadata.", code=code
        )

    return {
        "language": language,
        "encQuery": ":".join(", ".join(ids) for ids in selection),
        "matCode": code,
        "matMaxDim": details["matMaxDim"],
        "matUMSpec": details["matUMSpec"],
    }


def pivot(
    code: str,
    language: str,
    metadata: dict[str, Any],
    selection: Sequence[Sequence[str]],
) -> bytes:
    """Request one CSV chunk from the service."""
    body = json.dumps(payload(code, language, metadata, selection), ensure_ascii=False)
    response = request(url("pivot"), method="POST", body=body)

    content = response.content
    if not content:
        raise TempoAPIError(
            f"The TEMPO Online service returned an empty CSV for `{code}`.",
            code=code,
        )

    return content


# CSV assembly ----------------------------------------------------------------


class CsvParts(NamedTuple):
    """A CSV response split into its header line and the rest."""

    header: bytes
    body: bytes


def csv_split(content: bytes, code: str | None = None) -> CsvParts:
    """Split a CSV response after its first newline."""
    newline = content.find(b"\n")
    if newline == -1:
        raise TempoAPIError(
            "The TEMPO Online service returned a CSV response without a header line.",
            code=code,
        )

    return CsvParts(content[: newline + 1], content[newline + 1 :])


def without_csv_header(content: bytes, code: str | None = None) -> bytes:
    """Everything after the header line of a CSV response."""
    return csv_split(content, code=code).body


def header_text(header: bytes) -> str:
    """The header line as text, without its line ending."""
    return header.decode("utf-8", errors="replace").rstrip("\r\n")


def write_raw(content: bytes, path: str, append: bool = False) -> str:
    """Write bytes to ``path``, appending when asked."""
    try:
        with open(path, "ab" if append else "wb") as handle:
            handle.write(content)
    except OSError as error:
        raise TempoIOError(f"Could not write to {path}: {error}", path=path)

    return path


# Downloading -----------------------------------------------------------------
#
# Chunked responses are concatenated byte-wise, which keeps memory flat for
# very large matrices.  That is only sound if every chunk carries the same
# header, so each chunk's header is compared with the first one's and a mismatch
# is an error rather than a silently misaligned file.


def collect_matrix(
    code: str,
    language: str,
    metadata: dict[str, Any],
    plan: QueryPlan,
    sink: Callable[[bytes, bool], Any],
    progress_label: str | None = None,
) -> int:
    """Fetch every chunk of a matrix, handing each to ``sink``.

    ``sink`` is called as ``sink(content, append)``.  The first call receives
    the full response including its header; later calls receive only the body.

    Returns:
        The number of requests spent.
    """
    reporter = Reporter(plan.requests, progress_label)
    first_header: bytes | None = None
    spent = 0

    for selection in each_selection(plan.chunks):
        content = pivot(code, language, metadata, selection)
        parts = csv_split(content, code=code)
        spent += 1

        if first_header is None:
            first_header = parts.header
            sink(content, False)
        else:
            if header_text(parts.header) != header_text(first_header):
                raise TempoAPIError(
                    f"Matrix `{code}` returned inconsistent CSV headers across "
                    "chunked requests, so the parts cannot be combined "
                    f"safely.\n  first: {header_text(first_header)}\n"
                    f"  later: {header_text(parts.header)}",
                    code=code,
                )
            sink(parts.body, True)

        reporter.tick()

    reporter.done()
    return spent


class DownloadResult(NamedTuple):
    """Where a downloaded matrix landed, and what it cost."""

    path: str
    requests: int


def download_matrix(
    code: str,
    language: str,
    directory: str,
    metadata: dict[str, Any] | None = None,
    progress: bool = True,
) -> DownloadResult:
    """Download a whole matrix to ``directory`` as ``<code>.csv``.

    The file is assembled under a temporary name in the same directory and
    moved into place only once complete, so an interrupted download never
    leaves a half-written CSV behind.
    """
    if metadata is None:
        metadata = matrix_metadata(code, language)

    plan = plan_query(option_ids(metadata, code), code=code)
    destination = os.path.join(directory, f"{code}.csv")

    handle, temporary = tempfile.mkstemp(
        prefix=f".{code}-", suffix=".csv", dir=directory
    )
    os.close(handle)

    label = f"Downloading {code}" if progress and plan.requests > 1 else None

    try:
        spent = collect_matrix(
            code=code,
            language=language,
            metadata=metadata,
            plan=plan,
            sink=lambda content, append: write_raw(content, temporary, append),
            progress_label=label,
        )
        os.replace(temporary, destination)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise

    return DownloadResult(destination, spent)


class FetchResult(NamedTuple):
    """A matrix fetched into memory, and what it cost."""

    text: str
    requests: int


def fetch_matrix(
    code: str,
    language: str,
    metadata: dict[str, Any],
    plan: QueryPlan,
    progress_label: str | None = None,
) -> FetchResult:
    """Fetch a matrix into memory, as one CSV string."""
    pieces: list[bytes] = []

    spent = collect_matrix(
        code=code,
        language=language,
        metadata=metadata,
        plan=plan,
        sink=lambda content, append: pieces.append(content),
        progress_label=progress_label,
    )

    return FetchResult(b"".join(pieces).decode("utf-8", errors="replace"), spent)
