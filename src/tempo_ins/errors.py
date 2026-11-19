"""Classed errors and warnings raised by :mod:`tempo_ins`.

Every error raised by this package carries its own class, so a caller can react
to the *kind* of failure instead of matching on English message text.  That
matters for unattended work: an unreachable service is worth retrying later,
whereas a rejected matrix code never will be.

The hierarchy is::

    TempoError
      +- TempoInputError    invalid user input, raised before any request
      +- TempoOfflineError  the service could not be reached at all
      +- TempoHTTPError     the service answered with a non-2xx status
      +- TempoAPIError      the service answered, but not with what we expect
      +- TempoIOError       a local file could not be read or written

``TempoInputError`` also inherits from :class:`ValueError` and ``TempoIOError``
from :class:`OSError`, so ordinary ``except ValueError`` / ``except OSError``
handlers keep working.

Warnings all inherit from :class:`TempoWarning`, itself a
:class:`UserWarning`, and are raised at most once per process each, so a loop
over many matrices does not bury the console.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any

__all__ = [
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
]


class TempoError(Exception):
    """Base class for every error raised by :mod:`tempo_ins`.

    Subclasses expose the extra fields documented on each of them as
    attributes; a field the service did not supply is ``None``.
    """

    def __init__(self, message: str, **fields: Any) -> None:
        super().__init__(message)
        self.message = message
        for name, value in fields.items():
            setattr(self, name, value)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class TempoInputError(TempoError, ValueError):
    """Invalid arguments, detected before any request is made.

    Attributes:
        code: Matrix code, when one is involved.
        requests: Number of requests a refused download would have needed.
    """

    code: str | None = None
    requests: float | None = None


class TempoOfflineError(TempoError):
    """The service could not be reached at all.

    Raised after exhausting ``get_option("max_attempts")``.

    Attributes:
        url: The URL that could not be reached.
    """

    url: str | None = None


class TempoHTTPError(TempoError):
    """The service answered with a non-2xx status.

    Attributes:
        status_code: The HTTP status code.
        url: The requested URL.
        body: The first 200 characters of the response body.
    """

    status_code: int | None = None
    url: str | None = None
    body: str | None = None


class TempoAPIError(TempoError):
    """The service answered, but not with the structure this package expects.

    Malformed JSON, a missing field, or inconsistent CSV headers across chunked
    requests.

    Attributes:
        code: Matrix code, when a matrix is involved.
        url: The requested URL, when one is involved.
    """

    code: str | None = None
    url: str | None = None


class TempoIOError(TempoError, OSError):
    """A local path could not be created, written or read.

    Attributes:
        path: The path that could not be used.
        code: Matrix code, when one is involved.
    """

    path: str | None = None
    code: str | None = None

    def __init__(self, message: str, **fields: Any) -> None:
        # OSError treats positional arguments as (errno, strerror), so route
        # initialisation through TempoError explicitly.
        TempoError.__init__(self, message, **fields)


class TempoWarning(UserWarning):
    """Base class for every warning raised by :mod:`tempo_ins`."""


class TempoDateWarning(TempoWarning):
    """The service reported an update date in an unparseable format.

    ``last_update`` becomes ``None``; freshness checks fall back to comparing
    the service's raw update string, so caching keeps working.
    """


class TempoCacheWarning(TempoWarning):
    """A CSV file is reused although its language cannot be verified.

    Raised for files that predate the download manifest.
    """


class TempoTidyWarning(TempoWarning):
    """:func:`tempo_ins.tidy` met something it could not interpret.

    For example an observation that is not a number.
    """


# Registry used to emit a given warning only once per process, so that a loop
# over hundreds of matrices does not produce hundreds of identical warnings.
_warned: dict[str, bool] = {}
_warned_lock = threading.Lock()


def warn_once(
    key: str,
    message: str,
    category: type[Warning] = TempoWarning,
    stacklevel: int = 3,
) -> bool:
    """Emit ``message`` once per process for ``key``.

    Returns ``True`` when the warning was emitted, ``False`` when it had
    already been emitted earlier.
    """
    with _warned_lock:
        if _warned.get(key):
            return False
        _warned[key] = True

    warnings.warn(message, category, stacklevel=stacklevel)
    return True


def reset_warnings() -> None:
    """Forget which warnings have been emitted.

    Mostly useful in tests, where each case wants a clean slate.
    """
    with _warned_lock:
        _warned.clear()
