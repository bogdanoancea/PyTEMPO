"""Downloading complete TEMPO Online tables to CSV files."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator, Mapping, Sequence

import pandas as pd

from ._api import (
    download_matrix,
    matrix_metadata,
    validate_codes,
    validate_flag,
    validate_language,
)
from ._cache import (
    file_is_current,
    manifest_entry,
    manifest_read,
    manifest_write,
    prepare_directory,
)
from .options import progress_enabled

__all__ = ["BulkResult", "bulk"]

_logger = logging.getLogger("tempo_ins")


class BulkResult(Mapping):
    """What :func:`bulk` did, for every requested matrix.

    The object maps a matrix code to the path of its CSV file, so
    ``result["ACC101B"]`` is the path whether or not the file was fetched in
    this call.

    Attributes:
        paths: Matrix code to CSV path.
        downloaded: Matrix code to whether the file was fetched in this call.
        reason: Matrix code to why the file was or was not fetched:
            ``"missing"``, ``"outdated"``, ``"revised"``, ``"language"``,
            ``"modified"``, ``"unverifiable"``, ``"mtime"``, ``"overwrite"`` or
            ``"current"``.
        requests: Matrix code to the number of HTTP requests spent on it.
        directory: The directory the files were written to.
    """

    __slots__ = ("paths", "downloaded", "reason", "requests", "directory")

    def __init__(
        self,
        paths: dict[str, str],
        downloaded: dict[str, bool],
        reason: dict[str, str],
        requests: dict[str, int | None],
        directory: str,
    ) -> None:
        self.paths = paths
        self.downloaded = downloaded
        self.reason = reason
        self.requests = requests
        self.directory = directory

    def __getitem__(self, code: str) -> str:
        return self.paths[code]

    def __iter__(self) -> Iterator[str]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def to_frame(self) -> pd.DataFrame:
        """The same information as a data frame, one row per matrix."""
        codes = list(self.paths)
        return pd.DataFrame(
            {
                "code": codes,
                "path": [self.paths[code] for code in codes],
                "downloaded": [self.downloaded[code] for code in codes],
                "reason": [self.reason[code] for code in codes],
                "requests": pd.array(
                    [self.requests.get(code) for code in codes], dtype="Int64"
                ),
            }
        )

    def __repr__(self) -> str:
        fetched = sum(1 for value in self.downloaded.values() if value)
        return (
            f"<BulkResult {len(self.paths)} matrix(es), {fetched} downloaded, "
            f"in {self.directory}>"
        )


def bulk(
    codes: str | Sequence[str],
    directory: str | os.PathLike[str],
    language: str = "ro",
    overwrite: bool = False,
    progress: bool | None = None,
) -> BulkResult:
    """Download complete TEMPO Online tables as CSV files.

    Downloads one or more complete matrices from the Romanian National
    Institute of Statistics TEMPO Online database and writes each as a CSV
    file.  Use :func:`tempo_ins.get` instead when you want part of a table, or
    the data in memory.

    Args:
        codes: One matrix code, or a sequence of them, such as ``"ACC101B"``.
            Use :func:`tempo_ins.toc` or :func:`tempo_ins.search` to discover
            codes.
        directory: Where to write the CSV files.  It is created when needed,
            and nothing outside it is ever written.
        language: ``"ro"`` for Romanian (the default) or ``"en"`` for English.
        overwrite: Whether to download every requested matrix even when the
            local copy is already current.
        progress: Whether to report progress, or ``None`` to follow the
            ``progress`` option.

    Returns:
        A :class:`BulkResult`, which maps every requested matrix code to its
        CSV path and also records what happened to each.

    Freshness, and why a manifest:
        Alongside the CSV files, this maintains ``.tempo-manifest.json``,
        recording for every file the language, the service's own update string,
        the retrieval time, the number of requests and an MD5 checksum.  A file
        is reused only when the language, the published revision and the
        checksum all agree.

        A CSV file does not record which language it holds, so a
        modification-time test would accept a Romanian file for an English
        request and report it as up to date.  Files written before the manifest
        existed are still accepted on the old modification-time rule, with a
        warning that their language cannot be verified; delete them, or pass
        ``overwrite=True``, to be certain.

    Query size:
        A full matrix can exceed what the service will answer in one query, so
        the request is split into chunks whose product of selected values stays
        within the ``max_cells`` option.  The parts are concatenated, and each
        part's CSV header is checked against the first part's so that
        misaligned columns are an error rather than a corrupt file.

    See Also:
        :func:`tempo_ins.get` for filtered, in-memory downloads,
        :func:`tempo_ins.tidy` to turn a downloaded CSV into analysis-ready
        data, :func:`tempo_ins.meta` to size a table first.

    Example:
        >>> import tempfile, tempo_ins as tempo
        >>> folder = tempfile.mkdtemp()
        >>> files = tempo.bulk("ACC101B", folder, language="en")
        ...                                                  # doctest: +SKIP
        >>> accidents = tempo.tidy(files["ACC101B"], code="ACC101B")
        ...                                                  # doctest: +SKIP
    """
    resolved = validate_codes(codes)
    language = validate_language(language)
    overwrite = validate_flag(overwrite, "overwrite")
    if progress is not None:
        progress = validate_flag(progress, "progress")
    folder = prepare_directory(directory)

    show_progress = progress_enabled() if progress is None else progress

    def announce(message: str) -> None:
        _logger.info(message)
        if show_progress:
            print(message, file=sys.stderr)

    paths: dict[str, str] = {}
    downloaded: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    requests_spent: dict[str, int | None] = {}

    manifest = manifest_read(folder)
    manifest_changed = False

    for code in resolved:
        metadata = matrix_metadata(code, language)
        destination = os.path.join(folder, f"{code}.csv")
        paths[code] = destination
        downloaded[code] = False
        requests_spent[code] = None

        if overwrite:
            reusable, reason = False, "overwrite"
        else:
            reusable, reason = file_is_current(
                folder, code, language, metadata, destination
            )
        reasons[code] = reason

        if reusable:
            announce(f"Skipping `{code}`: the local CSV is up to date.")
            continue

        written = download_matrix(
            code=code,
            language=language,
            directory=folder,
            metadata=metadata,
            progress=show_progress,
        )

        downloaded[code] = True
        requests_spent[code] = written.requests
        manifest[code] = manifest_entry(
            code=code,
            language=language,
            metadata=metadata,
            path=destination,
            requests_spent=written.requests,
        )
        manifest_changed = True

    if manifest_changed:
        manifest_write(folder, manifest)

    if not any(downloaded.values()):
        announce("All requested CSV files are up to date.")

    return BulkResult(paths, downloaded, reasons, requests_spent, folder)
