"""Options that control :mod:`tempo_ins`.

Every option has a default, can be overridden for the process with
:func:`set_options`, temporarily with :func:`option_context`, and from the
environment with a ``TEMPO_``-prefixed variable:

Each option, its default, and the environment variable that overrides it:

===================== ============ ===============================
Option                Default      Environment variable
===================== ============ ===============================
``base_url``          see below    ``TEMPO_BASE_URL``
``connecttimeout``    ``30``       ``TEMPO_CONNECTTIMEOUT``
``timeout``           ``300``      ``TEMPO_TIMEOUT``
``user_agent``        ``None``     ``TEMPO_USER_AGENT``
``max_attempts``      ``3``        ``TEMPO_MAX_ATTEMPTS``
``backoff``           ``1.0``      ``TEMPO_BACKOFF``
``throttle``          ``0.0``      ``TEMPO_THROTTLE``
``max_connections``   ``4``        ``TEMPO_MAX_CONNECTIONS``
``max_cells``         ``50000``    ``TEMPO_MAX_CELLS``
``max_options_per_dim`` ``300``    ``TEMPO_MAX_OPTIONS_PER_DIM``
``max_requests``      ``2000``     ``TEMPO_MAX_REQUESTS``
``cache_dir``         ``None``     ``TEMPO_CACHE_DIR``
``cache_max_age``     ``86400``    ``TEMPO_CACHE_MAX_AGE``
``progress``          ``None``     ``TEMPO_PROGRESS``
===================== ============ ===============================

``base_url`` defaults to ``http://statistici.insse.ro:8077/tempo-ins``.

Connection
    ``base_url`` is the base URL of the TEMPO Online service, which is only
    reachable over plain HTTP; it does not offer TLS.  ``connecttimeout`` and
    ``timeout`` are seconds to wait for the connection and for a complete
    response.  ``user_agent`` is sent with every request; ``None`` sends a
    string identifying the package and the Python version.

Politeness and resilience
    ``max_attempts`` is how many times a failed request is attempted in total.
    Transport failures and HTTP 408, 425, 429 and 5xx statuses are retried;
    other statuses fail immediately.  ``backoff`` is the base number of
    seconds for the exponential backoff between attempts, and a
    ``Retry-After`` response header takes precedence.  ``throttle`` is the
    minimum number of seconds between two consecutive requests.
    ``max_connections`` caps the concurrent connections used when many
    requests are issued at once, as in :func:`tempo_ins.toc` with
    ``full_description=True``.  TEMPO Online is a small public service; please
    do not raise it casually.

Query size
    ``max_cells`` is the largest number of cells requested in a single query;
    larger downloads are split into several requests.
    ``max_options_per_dim`` is the largest number of options selected from a
    single dimension in one query.  ``max_requests`` refuses to start a
    download that would need more requests than this.

Output and caching
    ``progress`` controls progress reporting for long operations; ``None``
    means report when stderr is a terminal.  ``cache_dir`` is where service
    responses are cached (see :func:`tempo_ins.cache_dir`) and
    ``cache_max_age`` how many seconds an entry stays usable.

Example:
    >>> import tempo_ins as tempo
    >>> with tempo.option_context(throttle=0.2, max_cells=100_000):
    ...     tempo.get_option("throttle")
    0.2
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

from .errors import TempoInputError

__all__ = [
    "DEFAULTS",
    "get_option",
    "set_options",
    "reset_options",
    "option_context",
    "progress_enabled",
]

DEFAULTS: dict[str, Any] = {
    "base_url": "http://statistici.insse.ro:8077/tempo-ins",
    "connecttimeout": 30,
    "timeout": 300,
    "max_attempts": 3,
    "backoff": 1.0,
    "throttle": 0.0,
    "max_connections": 4,
    "max_cells": 50_000,
    "max_options_per_dim": 300,
    "max_requests": 2_000,
    "user_agent": None,
    "cache_dir": None,
    "cache_max_age": 86_400.0,
    "progress": None,
}


def _as_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


_PARSERS: dict[str, Callable[[str], Any]] = {
    "base_url": str,
    "connecttimeout": int,
    "timeout": int,
    "max_attempts": int,
    "backoff": float,
    "throttle": float,
    "max_connections": int,
    "max_cells": float,
    "max_options_per_dim": int,
    "max_requests": float,
    "user_agent": str,
    "cache_dir": str,
    "cache_max_age": float,
    "progress": _as_bool,
}

_overrides: dict[str, Any] = {}


def _check_name(name: str) -> str:
    if name not in DEFAULTS:
        known = ", ".join(sorted(DEFAULTS))
        raise TempoInputError(f"Unknown option {name!r}. Known options: {known}.")
    return name


def _from_environment(name: str) -> Any:
    raw = os.environ.get("TEMPO_" + name.upper())
    if raw is None or not raw.strip():
        return None

    try:
        return _PARSERS[name](raw)
    except (TypeError, ValueError):
        raise TempoInputError(
            f"Environment variable TEMPO_{name.upper()} holds {raw!r}, "
            f"which is not a valid value for the {name!r} option."
        )


def get_option(name: str) -> Any:
    """Return the effective value of one option.

    Resolution order is an explicit :func:`set_options` override, then the
    ``TEMPO_``-prefixed environment variable, then the default.
    """
    _check_name(name)
    if name in _overrides:
        return _overrides[name]

    from_environment = _from_environment(name)
    if from_environment is not None:
        return from_environment

    return DEFAULTS[name]


def set_options(**options: Any) -> dict[str, Any]:
    """Set options for the rest of the process.

    Returns the previous effective values, so they can be passed back to
    :func:`set_options` to restore them.  Prefer :func:`option_context` for a
    temporary change.
    """
    previous = {}
    for name in options:
        _check_name(name)
        previous[name] = get_option(name)

    for name, value in options.items():
        if value is None and name not in {"user_agent", "cache_dir", "progress"}:
            _overrides.pop(name, None)
        else:
            _overrides[name] = value

    return previous


def reset_options(*names: str) -> None:
    """Drop explicit overrides, for the named options or for all of them."""
    if not names:
        _overrides.clear()
        return

    for name in names:
        _overrides.pop(_check_name(name), None)


@contextmanager
def option_context(**options: Any) -> Iterator[None]:
    """Set options for the duration of a ``with`` block, then restore them.

    Unlike :func:`set_options`, this restores the exact previous override
    state, including options that had no override at all.
    """
    for name in options:
        _check_name(name)

    saved = dict(_overrides)
    try:
        set_options(**options)
        yield
    finally:
        _overrides.clear()
        _overrides.update(saved)


def progress_enabled() -> bool:
    """Whether progress should be reported.

    Follows the ``progress`` option, defaulting to whether stderr is a
    terminal.
    """
    value = get_option("progress")
    if value is None:
        try:
            return bool(sys.stderr.isatty())
        except (AttributeError, ValueError):  # pragma: no cover - closed stream
            return False
    return bool(value)
