"""The HTTP layer.

All traffic to TEMPO Online goes through this module.  Relative to a bare
:func:`requests.get` it adds

* a ``User-Agent`` identifying the package, so that INS can see who is calling;
* transparent gzip, which matters for the large CSV payloads;
* retries with exponential backoff for transport failures and for the statuses
  worth retrying, honouring ``Retry-After`` when present;
* an optional throttle between consecutive requests;
* a concurrent variant with a connection cap, used for request fan-out.

Nothing here is part of the public API.
"""

from __future__ import annotations

import platform
import random
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NamedTuple

import requests

from ._version import __version__
from .errors import (
    TempoAPIError,
    TempoError,
    TempoHTTPError,
    TempoOfflineError,
    TempoWarning,
    warn_once,
)
from .options import get_option, progress_enabled

__all__ = ["request", "request_many", "response_text", "Reporter"]

#: Statuses worth another attempt: request timeout, too early and rate limited.
_RETRYABLE_STATUSES = frozenset({408, 425, 429})

_HOMEPAGE = "https://github.com/bogdanoancea/PyTEMPO"


def user_agent() -> str:
    """The ``User-Agent`` sent with every request."""
    configured = get_option("user_agent")
    if configured is not None:
        return str(configured)

    return f"tempo-ins/{__version__} (Python {platform.python_version()}; {_HOMEPAGE})"


# One session per thread: requests.Session is not thread safe, but keeping one
# per worker preserves connection reuse during a fan-out.
_thread_local = threading.local()


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _timeouts() -> tuple[float, float]:
    return (
        float(get_option("connecttimeout")),
        float(get_option("timeout")),
    )


def _headers(method: str) -> dict:
    headers = {
        "User-Agent": user_agent(),
        "Accept-Encoding": "gzip, deflate",
    }
    if method == "POST":
        headers["Content-Type"] = "application/json"
    return headers


def _status_retryable(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code in _RETRYABLE_STATUSES or status_code >= 500


def _retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None

    if seconds < 0:
        return None

    return min(seconds, 60.0)


def _backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Seconds to wait before attempt ``attempt + 1``.

    ``attempt`` is 1-based.  A ``Retry-After`` value takes precedence;
    otherwise the delay is exponential with a little jitter, so that concurrent
    callers do not line up.
    """
    if retry_after is not None:
        return retry_after

    base = float(get_option("backoff"))
    if base <= 0 or base != base:  # non-positive or NaN
        return 0.0

    return base * (2 ** (attempt - 1)) * random.uniform(0.5, 1.0)


_throttle_lock = threading.Lock()
_last_request: float | None = None


def _throttle() -> None:
    """Hold off until ``throttle`` seconds have passed since the last request."""
    global _last_request

    wait = float(get_option("throttle"))
    if not wait > 0:
        return

    with _throttle_lock:
        now = time.monotonic()
        if _last_request is not None:
            elapsed = now - _last_request
            if elapsed < wait:
                time.sleep(wait - elapsed)
                now = time.monotonic()
        _last_request = now


def response_text(response: requests.Response, context: str) -> str:
    """Decode a response body as text.

    The service declares UTF-8.  A body that is not valid UTF-8 is decoded as
    windows-1252 instead, with a warning, rather than failing outright.
    """
    content = response.content
    if content is None:
        raise TempoAPIError(
            f"The TEMPO service returned an invalid {context} response."
        )

    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        warn_once(
            "decode-not-utf8",
            "The TEMPO Online service returned content that is not valid "
            "UTF-8, although it declares UTF-8; it was decoded as "
            "windows-1252 instead. Some characters may be wrong. Please "
            f"report this at {_HOMEPAGE}/issues",
            TempoWarning,
        )
        return content.decode("cp1252", errors="replace")


def _http_error(response: requests.Response, url: str) -> TempoHTTPError:
    try:
        body_text = response_text(response, "error")
    except TempoError:
        body_text = ""

    body_text = " ".join(body_text.split())[:200].strip()

    return TempoHTTPError(
        "The TEMPO Online service returned HTTP "
        f"{response.status_code} for {url}" + (f": {body_text}" if body_text else "."),
        status_code=int(response.status_code),
        url=url,
        body=body_text,
    )


class _Attempt(NamedTuple):
    """The outcome of a single request."""

    response: requests.Response | None
    error: TempoError | None
    retryable: bool
    retry_after: float | None


def _attempt(
    url: str,
    method: str = "GET",
    body: str | None = None,
) -> _Attempt:
    """Make one request, without retrying.

    Exactly one of ``response`` and ``error`` is set on the result.
    """
    _throttle()

    try:
        response = _session().request(
            method,
            url,
            data=body.encode("utf-8") if body is not None else None,
            headers=_headers(method),
            timeout=_timeouts(),
        )
    except (requests.RequestException, OSError) as error:
        return _Attempt(
            None,
            TempoOfflineError(
                f"Could not reach the TEMPO Online service at {url}: {error}",
                url=url,
            ),
            True,
            None,
        )

    status_code = response.status_code
    if status_code is None:  # pragma: no cover - requests always sets one
        return _Attempt(
            None,
            TempoAPIError(
                f"The TEMPO Online service returned no HTTP status for {url}.",
                url=url,
            ),
            False,
            None,
        )

    if 200 <= status_code < 300:
        return _Attempt(response, None, False, None)

    return _Attempt(
        None,
        _http_error(response, url),
        _status_retryable(status_code),
        _retry_after(response),
    )


def request(
    url: str,
    method: str = "GET",
    body: str | None = None,
) -> requests.Response:
    """Fetch one URL, retrying what is worth retrying.

    Args:
        url: Fully qualified request URL.
        method: Either ``"GET"`` or ``"POST"``.
        body: Request body for ``POST``, as a JSON string.

    Raises:
        TempoOfflineError: The service could not be reached at all.
        TempoHTTPError: The service answered with a non-2xx status.
    """
    if method not in {"GET", "POST"}:
        raise TempoAPIError(f"Unsupported HTTP method {method!r}.")

    attempts = max(1, int(get_option("max_attempts")))
    last_error: TempoError | None = None

    for attempt in range(1, attempts + 1):
        outcome = _attempt(url, method, body)
        if outcome.response is not None:
            return outcome.response

        error = outcome.error
        assert error is not None
        last_error = error

        if outcome.retryable and attempt < attempts:
            time.sleep(_backoff_delay(attempt, outcome.retry_after))
            continue

        raise error

    raise last_error or TempoOfflineError(  # pragma: no cover - unreachable
        f"The request to {url} failed.", url=url
    )


def request_many(
    urls: Sequence[str],
    progress_label: str | None = None,
) -> list[requests.Response]:
    """Fetch several URLs concurrently.

    Returns responses in the same order as ``urls``.  Failures are retried in
    later rounds; a URL that never succeeds raises the error it last produced.
    """
    total = len(urls)
    if total == 0:
        return []

    connections = max(1, int(get_option("max_connections")))

    if total == 1 or connections == 1:
        responses: list[Any] = []
        reporter = Reporter(total, progress_label)
        for url in urls:
            responses.append(request(url))
            reporter.tick()
        reporter.done()
        return responses

    attempts = max(1, int(get_option("max_attempts")))
    collected: list[requests.Response | None] = [None] * total
    failures: list[TempoError | None] = [None] * total
    pending = list(range(total))
    reporter = Reporter(total, progress_label)

    for attempt in range(1, attempts + 1):
        if not pending:
            break

        retry: list[int] = []
        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = {pool.submit(_attempt, urls[index]): index for index in pending}
            for future in as_completed(futures):
                index = futures[future]
                outcome = future.result()
                if outcome.response is not None:
                    collected[index] = outcome.response
                    reporter.tick()
                    continue

                failures[index] = outcome.error
                if outcome.retryable:
                    retry.append(index)

        pending = sorted(set(retry))

        if pending and attempt < attempts:
            time.sleep(_backoff_delay(attempt))

    reporter.done()

    unresolved = [index for index, value in enumerate(collected) if value is None]
    if unresolved:
        first = unresolved[0]
        raise failures[first] or TempoOfflineError(
            f"Could not fetch {urls[first]}.", url=urls[first]
        )

    return [response for response in collected if response is not None]


class Reporter:
    """Minimal progress reporting to stderr, deliberately dependency-free."""

    def __init__(self, total: int, label: str | None = None) -> None:
        self.total = total
        self.label = label
        self.active = progress_enabled() and label is not None and total > 1
        self.count = 0
        self._started = time.monotonic()
        self._lock = threading.Lock()

    def tick(self, n: int = 1) -> None:
        with self._lock:
            self.count += n
            count = self.count
        if not self.active:
            return
        sys.stderr.write(f"\r{self.label}: {count}/{self.total}")
        sys.stderr.flush()

    def done(self) -> None:
        if not self.active:
            return
        elapsed = time.monotonic() - self._started
        sys.stderr.write(
            f"\r{self.label}: {self.count}/{self.total} ({elapsed:.1f}s)\n"
        )
        sys.stderr.flush()
