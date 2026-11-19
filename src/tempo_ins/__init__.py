"""Access the Romanian National Institute of Statistics TEMPO Online database.

Tools for the TEMPO Online database: find the statistical tables, inspect
their structure, and download whole tables or selected parts of them.

Finding a table
    :func:`toc` lists every matrix with its code, :func:`search` filters that
    list, and :func:`hierarchy` returns the classification tree.

Looking before downloading
    :func:`meta` reports a matrix's dimensions, the values selectable in each,
    the publication date, and how many cells and requests a full download
    would cost.

Getting the data
    :func:`get` downloads a table, or a filtered part of one, straight into a
    :class:`pandas.DataFrame`.  :func:`bulk` writes whole tables to CSV files
    and keeps them up to date across runs.  :func:`tidy` turns the service's
    CSV layout into analysis-ready data, and is applied by :func:`get` by
    default.

Configuration
    :mod:`tempo_ins.options` documents the options controlling timeouts,
    retries, concurrency and query size; :func:`cache_dir` controls response
    caching; :mod:`tempo_ins.errors` documents the error and warning classes.

The service
    TEMPO Online is a public service of the Romanian National Institute of
    Statistics, reachable only over plain HTTP and offering no documented API
    contract or published rate limit.  This package identifies itself with a
    ``User-Agent``, retries only what is worth retrying, and caps concurrency
    at four connections by default.  Please keep it modest.

Example:
    >>> import tempo_ins as tempo
    >>> info = tempo.meta("ACC101B", language="en")          # doctest: +SKIP
    >>> frame = tempo.get(
    ...     "ACC101B",
    ...     filters={"Counties": "Cluj", "Years": range(2015, 2025)},
    ...     language="en",
    ... )                                                    # doctest: +SKIP
"""

from __future__ import annotations

from ._bulk import BulkResult, bulk
from ._cache import cache_clear, cache_dir
from ._get import get
from ._hierarchy import hierarchy
from ._meta import Meta, meta
from ._tidy import tidy
from ._toc import search, toc
from ._version import __version__
from .errors import (
    TempoAPIError,
    TempoCacheWarning,
    TempoDateWarning,
    TempoError,
    TempoHTTPError,
    TempoInputError,
    TempoIOError,
    TempoOfflineError,
    TempoTidyWarning,
    TempoWarning,
    reset_warnings,
)
from .options import (
    DEFAULTS,
    get_option,
    option_context,
    reset_options,
    set_options,
)

__all__ = [
    # Public API
    "toc",
    "search",
    "meta",
    "hierarchy",
    "get",
    "bulk",
    "tidy",
    "cache_dir",
    "cache_clear",
    # Returned types
    "Meta",
    "BulkResult",
    # Configuration
    "DEFAULTS",
    "get_option",
    "set_options",
    "reset_options",
    "option_context",
    # Errors and warnings
    "TempoError",
    "TempoInputError",
    "TempoOfflineError",
    "TempoHTTPError",
    "TempoAPIError",
    "TempoIOError",
    "TempoWarning",
    "TempoDateWarning",
    "TempoCacheWarning",
    "TempoTidyWarning",
    "reset_warnings",
    "__version__",
]
