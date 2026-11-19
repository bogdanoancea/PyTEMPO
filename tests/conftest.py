"""A fake TEMPO Online service.

The real service is a public HTTP endpoint, so none of it can be exercised
offline.  This reimplements just enough of it with the standard library to test
the request layer, chunked downloads, caching and the end-to-end functions
without touching the network.  It reproduces the real endpoints closely,
including the ``", "``-padded CSV headers and rows the live service sends.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import pytest

import tempo_ins as tempo
from tempo_ins import errors, options

#: Matrix code to (dimension 1 size, dimension 2 size).
MATRIX_SIZES: dict[str, list[int]] = {
    "SML101A": [2, 3],
    "BIG101A": [7, 5],
    "TAG101A": [2, 2],
    "BADHDR": [4, 1],
    "ODDDATE": [2, 2],
}

_DOMAIN_WITH_HTML = (
    'AGRICULTURE <a href="https://insse.ro/press?x=1" target="_blank">'
    "Press releases</a>"
)


def fake_metadata(
    code: str,
    n1: int,
    n2: int,
    update: str = "05-11-2025",
) -> dict[str, Any]:
    """The metadata document the fake service returns for one matrix."""
    return {
        "matrixName": f"Fake matrix {code}",
        "ultimaActualizare": update,
        "ancestors": [
            {"name": "ROOT"},
            {"name": _DOMAIN_WITH_HTML},
            {"name": "Sub-domain"},
            {"name": "Survey name"},
        ],
        "details": {"matMaxDim": 2, "matUMSpec": 0},
        "dimensionsMap": [
            {
                "label": "Judete",
                "options": [
                    {"nomItemId": 1000 + index, "label": f"County {index}"}
                    for index in range(1, n1 + 1)
                ],
            },
            {
                "label": "Ani",
                "options": [
                    {"nomItemId": 2000 + index, "label": f"Anul {2019 + index}"}
                    for index in range(1, n2 + 1)
                ],
            },
        ],
    }


def _branch(name: str, code: str, kind: str = "context") -> dict[str, Any]:
    return {
        "name": name,
        "code": code,
        "childrenUrl": kind,
        "comment": None,
        "url": kind,
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:  # noqa: D102 - silence the server
        pass

    # Responses ---------------------------------------------------------------

    def _send(self, body: bytes, status: int = 200, content_type: str = "") -> None:
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(
            json.dumps(payload).encode("utf-8"),
            status,
            "application/json;charset=UTF-8",
        )

    def _text(self, text: str, status: int = 200) -> None:
        self._send(text.encode("utf-8"), status, "text/plain;charset=UTF-8")

    # Routing -----------------------------------------------------------------

    @property
    def counters(self) -> dict[str, int]:
        return self.server.counters  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path

        if path == "/tempo-ins/matrix/matrices":
            self.counters["toc"] += 1
            self._json(
                [
                    {
                        "name": "Small fake table",
                        "code": "SML101A",
                        "childrenUrl": "matrix",
                        "comment": None,
                        "url": "matrix",
                    },
                    {
                        "name": "Big fake table",
                        "code": "BIG101A",
                        "childrenUrl": "matrix",
                        "comment": "a comment",
                        "url": "matrix",
                    },
                    {
                        "name": (
                            'Tagged table <a href="https://insse.ro/x">'
                            "Press releases</a>"
                        ),
                        "code": "TAG101A",
                        "childrenUrl": "matrix",
                        "comment": None,
                        "url": "matrix",
                    },
                ]
            )
            return

        # Fails twice with 500, then succeeds: exercises retry and backoff.
        if path == "/tempo-ins/flaky/matrices":
            self.counters["flaky"] += 1
            if self.counters["flaky"] < 3:
                self._text("upstream wobbled", 500)
                return
            self._json(
                [{"name": "Recovered", "code": "REC101A", "childrenUrl": "matrix"}]
            )
            return

        if path == "/tempo-ins/context/":
            self.counters["context"] += 1
            self._json(
                [
                    {
                        "parentCode": "0",
                        "level": 0,
                        "context": _branch(
                            'AGRICULTURE <a href="https://insse.ro/p" '
                            'target="_blank">Press releases</a>',
                            "1",
                        ),
                    },
                    {
                        "parentCode": "1",
                        "level": 1,
                        "context": _branch("Land use", "10"),
                    },
                    {
                        "parentCode": "10",
                        "level": 2,
                        "context": _branch("Areas and yields", "1010"),
                    },
                ]
            )
            return

        if path.startswith("/tempo-ins/context/"):
            code = path[len("/tempo-ins/context/") :]
            if code == "1010":
                children = [
                    _branch("Small fake table", "SML101A", "matrix"),
                    _branch("Big fake table", "BIG101A", "matrix"),
                ]
            elif code == "10":
                children = [_branch("Areas and yields", "1010")]
            elif code == "1":
                children = [_branch("Land use", "10")]
            else:
                children = []

            self._json(
                {
                    "context": _branch("Node", code),
                    "ancestors": [_branch("home", "")],
                    "children": children,
                }
            )
            return

        if path.startswith("/tempo-ins/matrix/"):
            code = path[len("/tempo-ins/matrix/") :]
            sizes = MATRIX_SIZES.get(code)
            if sizes is None:
                self._text("no such matrix", 404)
                return

            update = "2025.11.05" if code == "ODDDATE" else "05-11-2025"
            self._json(fake_metadata(code, sizes[0], sizes[1], update))
            return

        self._text("not found", 404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")

        if path != "/tempo-ins/pivot":
            self._text("not found", 404)
            return

        self.counters["pivot"] += 1

        selections = payload["encQuery"].split(":")
        ids = [
            [part.strip() for part in selection.split(",")] for selection in selections
        ]

        # One CSV row per selected combination, plus a footer line, so that
        # chunk reassembly can be verified by content.
        # The live service separates header fields with ", ", so every column
        # but the first arrives with a leading space, and it leaves the value
        # column named "Valoare" even when language is "en".  Both were
        # confirmed against the real service and are reproduced here so the
        # offline suite exercises the shape callers actually receive.
        if payload["matCode"] == "BADHDR" and self.counters["pivot"] % 2 == 0:
            header = "Judete, Ani, UM: Numar, TOTAL ALTERED"
        else:
            header = "Judete, Ani, UM: Numar, Valoare"

        rows = []
        position = 0
        for first in ids[0]:
            for second in ids[1]:
                position += 1
                value = ":" if position == 1 else str(position)
                rows.append(
                    f"County {int(first) - 1000},"
                    f" Anul {int(second) - 2000 + 2019}, Numar, {value}"
                )

        body = "\n".join([header, *rows, ",,,(c) INS 1998-2026"]) + "\n"
        self._send(body.encode("utf-8"), 200, "text/csv;charset=UTF-8")


class _FakeServer(ThreadingHTTPServer):
    """A threading HTTP server whose handler threads never block shutdown."""

    daemon_threads = True
    allow_reuse_address = True
    counters: dict[str, int]


class FakeService:
    """A running fake TEMPO service, with the request counters it has seen."""

    def __init__(self, base_url: str, counters: dict[str, int]) -> None:
        self.base_url = base_url
        self.counters = counters

    @property
    def pivot_calls(self) -> int:
        return self.counters["pivot"]

    @property
    def toc_calls(self) -> int:
        return self.counters["toc"]

    @property
    def context_calls(self) -> int:
        return self.counters["context"]


@pytest.fixture(autouse=True)
def _clean_state(tmp_path):
    """Give every test a fresh warning registry, options and cache directory."""
    errors.reset_warnings()
    options.reset_options()
    with options.option_context(cache_dir=str(tmp_path / "cache"), progress=False):
        yield
    errors.reset_warnings()
    options.reset_options()


@pytest.fixture
def service(tmp_path):
    """Start the fake service and point tempo_ins at it for one test."""
    server = _FakeServer(("127.0.0.1", 0), _Handler)
    server.counters = {"pivot": 0, "toc": 0, "flaky": 0, "context": 0}
    # A short poll interval keeps shutdown() from costing half a second per
    # test; the default is 0.5s and this fixture runs for every case.
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()

    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}/tempo-ins"

    previous = tempo.set_options(
        base_url=base_url,
        cache_dir=str(tmp_path / "service-cache"),
        progress=False,
        backoff=0.01,
        throttle=0,
    )
    errors.reset_warnings()

    try:
        yield FakeService(base_url, server.counters)
    finally:
        tempo.set_options(**previous)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def download_dir(tmp_path):
    """A fresh directory for bulk downloads."""
    folder = tmp_path / "tempo-dl"
    folder.mkdir()
    return str(folder)
