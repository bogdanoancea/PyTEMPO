"""Response cache and download manifest.

Two separate mechanisms live here.

The **response cache** keeps the table of contents and the per-matrix metadata,
which change rarely but are expensive to fetch.  Entries are plain JSON, so a
cache file can never execute code when it is read back, and they expire after
``cache_max_age`` seconds.

The **download manifest** is what makes :func:`tempo_ins.bulk`'s freshness
check correct.  A CSV file on its own does not say which language it holds, nor
which published revision of the matrix it came from, so a modification-time
heuristic silently accepts a Romanian file for an English request.  Recording
the language and the service's own update string removes both guesses, and an
MD5 checksum additionally detects a truncated or hand-edited file.  The file is
named ``.tempo-manifest.json`` and sits beside the CSV files it describes.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from ._version import __version__
from .errors import TempoCacheWarning, TempoInputError, TempoIOError, warn_once
from .options import get_option, set_options

__all__ = ["cache_dir", "cache_clear"]

_DATAFRAME_MARKER = "__tempo_dataframe__"
MANIFEST_NAME = ".tempo-manifest.json"


def _user_cache_dir() -> str:
    """The conventional per-user cache directory for this platform."""
    if sys.platform == "win32":  # pragma: no cover - platform specific
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local"
        )
        return os.path.join(base, "tempo-ins", "Cache")

    if sys.platform == "darwin":  # pragma: no cover - platform specific
        return os.path.join(os.path.expanduser("~"), "Library", "Caches", "tempo-ins")

    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "tempo-ins")


def cache_dir(path: Any | None = None) -> str:
    """Inspect or change where TEMPO caches service responses.

    The table of contents and the per-matrix metadata change rarely but are
    expensive to fetch, so they are cached.  Call ``cache_dir()`` to see the
    directory in use and ``cache_dir(path)`` to change it; the directory is
    created when it is first written to.

    By default the cache lives in the conventional per-user cache directory
    (``~/.cache/tempo-ins`` on Linux, ``~/Library/Caches/tempo-ins`` on macOS,
    ``%LOCALAPPDATA%\\tempo-ins\\Cache`` on Windows) and therefore persists
    between runs.  Entries older than ``cache_max_age`` seconds, one day by
    default, are ignored and refetched.  Point it somewhere disposable, such as
    a subdirectory of :func:`tempfile.gettempdir`, to confine it to one run.

    Equivalently, set the ``cache_dir`` option or the ``TEMPO_CACHE_DIR``
    environment variable.

    Args:
        path: A directory path, or ``None`` to query the current setting
            without changing it.

    Returns:
        The cache directory in use, or the previous one when setting it.

    Example:
        >>> import tempfile, os, tempo_ins as tempo
        >>> previous = tempo.cache_dir(os.path.join(tempfile.mkdtemp(), "c"))
        >>> tempo.cache_dir().endswith("c")
        True
        >>> _ = tempo.cache_dir(previous)
    """
    if path is None:
        current = get_option("cache_dir")
        if current is None:
            return _user_cache_dir()
        return os.fspath(current)

    if isinstance(path, bytes):
        raise TempoInputError("`path` must be a single directory path.")

    try:
        text = os.fspath(path)
    except TypeError:
        raise TempoInputError("`path` must be a single directory path.")

    if not isinstance(text, str) or not text.strip():
        raise TempoInputError("`path` must be a single directory path.")

    previous = cache_dir()
    set_options(cache_dir=os.path.expanduser(text))
    return previous


def cache_clear() -> int:
    """Delete every cached response from :func:`cache_dir`.

    Use it after the service has published new data and you want to be certain
    that nothing stale is reused.

    Returns:
        The number of cache entries removed.
    """
    directory = cache_dir()
    if not os.path.isdir(directory):
        return 0

    removed = 0
    for name in os.listdir(directory):
        if not name.endswith(".json"):
            continue
        try:
            os.unlink(os.path.join(directory, name))
        except OSError:  # pragma: no cover - racing another process
            continue
        removed += 1

    return removed


def _cache_max_age() -> float:
    return float(get_option("cache_max_age"))


def cache_key(*parts: Any) -> str:
    """A deterministic, filesystem-safe key for the given parts."""
    joined = "\u001e".join("\u001f".join(_key_parts(part)) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _key_parts(part: Any) -> tuple[str, ...]:
    if isinstance(part, (list, tuple)):
        return tuple(str(item) for item in part)
    return (str(part),)


def _encode(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {_DATAFRAME_MARKER: value.to_json(orient="table", index=False)}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and _DATAFRAME_MARKER in value:
        return pd.read_json(io.StringIO(value[_DATAFRAME_MARKER]), orient="table")
    return value


def _entry_path(key: str) -> str:
    return os.path.join(cache_dir(), key + ".json")


def cache_get(key: str) -> Any | None:
    """Return a cached value, or ``None`` when there is no usable entry."""
    max_age = _cache_max_age()
    if max_age <= 0:
        return None

    path = _entry_path(key)
    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as handle:
            entry = json.load(handle)
    except (OSError, ValueError):
        return None

    if not isinstance(entry, dict) or "stored_at" not in entry or "value" not in entry:
        return None

    try:
        age = time.time() - float(entry["stored_at"])
    except (TypeError, ValueError):
        return None

    if age != age or age > max_age:
        return None

    try:
        return _decode(entry["value"])
    except Exception:  # pragma: no cover - corrupt entry
        return None


def cache_set(key: str, value: Any) -> bool:
    """Store a value in the cache.

    Caching is a convenience, never a requirement, so a failure to write is
    reported as ``False`` rather than raised.
    """
    directory = cache_dir()
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return False

    path = _entry_path(key)
    try:
        payload = json.dumps(
            {"stored_at": time.time(), "value": _encode(value)},
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return False

    # Write through a temporary file so that a crash cannot leave a half-written
    # entry that would later be read back as truncated JSON.
    temporary = path + f".{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        return False

    return True


def cached(key: str, compute: Callable[[], Any]) -> Any:
    """Return the cached value for ``key``, computing and storing it if absent."""
    hit = cache_get(key)
    if hit is not None:
        return hit

    value = compute()
    cache_set(key, value)
    return value


# Download directory ----------------------------------------------------------


def prepare_directory(directory: Any) -> str:
    """Validate, create and return the directory downloads may be written to."""
    if directory is None or isinstance(directory, bytes):
        raise TempoInputError(
            "`directory` must be a single path where the CSV files can be written."
        )

    try:
        text = os.fspath(directory)
    except TypeError:
        raise TempoInputError(
            "`directory` must be a single path where the CSV files can be written."
        )

    if not isinstance(text, str) or not text.strip():
        raise TempoInputError(
            "`directory` must be a single path where the CSV files can be written."
        )

    text = os.path.expanduser(text)
    if not os.path.isdir(text):
        try:
            os.makedirs(text, exist_ok=True)
        except OSError as error:
            raise TempoIOError(
                f"Could not create `directory`: {text} ({error})", path=text
            )

    if not os.access(text, os.W_OK):
        raise TempoIOError(f"`directory` is not writable: {text}", path=text)

    return os.path.realpath(text)


# Download manifest -----------------------------------------------------------


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def manifest_path(directory: str) -> str:
    """Path of the download manifest inside ``directory``."""
    return os.path.join(directory, MANIFEST_NAME)


def manifest_read(directory: str) -> dict[str, Any]:
    """Read the per-file entries of the download manifest.

    Returns an empty mapping when the manifest is absent or unreadable.
    """
    path = manifest_path(directory)
    if not os.path.exists(path):
        return {}

    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}

    if not isinstance(manifest, dict):
        return {}

    files = manifest.get("files")
    if not isinstance(files, dict):
        return {}

    return files


def manifest_write(directory: str, files: dict[str, Any]) -> bool:
    """Write the download manifest, reporting whether it could be written."""
    path = manifest_path(directory)
    payload = {
        "manifest_version": 1,
        "package": "tempo-ins",
        "package_version": __version__,
        "written_at": _timestamp(),
        "files": files,
    }

    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (OSError, TypeError, ValueError):
        warn_once(
            "manifest-write",
            f"Could not write the download manifest to {path}; freshness "
            "checks will fall back to file modification times.",
        )
        return False

    return True


def file_digest(path: str) -> str | None:
    """MD5 of a file, or ``None`` when it does not exist.

    MD5 is used as a change detector, not as a security measure: it only has
    to notice a truncated or hand-edited file.
    """
    if not os.path.exists(path):
        return None

    digest = hashlib.md5()  # noqa: S324 - change detection, not security
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:  # pragma: no cover - unreadable file
        return None

    return digest.hexdigest()


def manifest_entry(
    code: str,
    language: str,
    metadata: dict[str, Any],
    path: str,
    requests_spent: int | None,
) -> dict[str, Any]:
    """Build the manifest entry describing one downloaded file."""
    from ._api import metadata_date, metadata_update_string

    parsed = metadata_date(metadata, code=code, warn=True)

    return {
        "code": code,
        "language": language,
        "file": os.path.basename(path),
        "service_update": metadata_update_string(metadata),
        "last_update": parsed.isoformat() if parsed is not None else None,
        "retrieved_at": _timestamp(),
        "requests": int(requests_spent) if requests_spent is not None else None,
        "bytes": os.path.getsize(path) if os.path.exists(path) else None,
        "md5": file_digest(path),
    }


def file_is_current(
    directory: str,
    code: str,
    language: str,
    metadata: dict[str, Any],
    path: str,
) -> tuple[bool, str]:
    """Whether the local CSV for ``code`` can be reused.

    Returns ``(reusable, reason)``, where ``reason`` is one of ``"missing"``,
    ``"outdated"``, ``"revised"``, ``"language"``, ``"modified"``,
    ``"unverifiable"``, ``"mtime"`` or ``"current"``.
    """
    from ._api import metadata_date, metadata_update_string

    if not os.path.exists(path):
        return False, "missing"

    entry = manifest_read(directory).get(code)

    if not isinstance(entry, dict):
        # A file written before the manifest existed, or by something else.  Fall
        # back to the old modification-time test, but say that the language is
        # unverifiable.
        update_date = metadata_date(metadata, code=code, warn=False)
        if update_date is None:
            return False, "unverifiable"

        modified = datetime.fromtimestamp(os.path.getmtime(path)).date()
        if modified >= update_date:
            warn_once(
                "manifest-missing",
                "Reusing CSV files that predate the download manifest. Their "
                "language cannot be verified; delete them or call "
                "bulk(overwrite=True) if you need to be certain.",
                TempoCacheWarning,
            )
            return True, "mtime"

        return False, "outdated"

    if str(entry.get("language")) != language:
        return False, "language"

    recorded_update = entry.get("service_update")
    service_update = metadata_update_string(metadata)
    if _as_optional_str(recorded_update) != _as_optional_str(service_update):
        return False, "revised"

    recorded_md5 = _as_optional_str(entry.get("md5"))
    if recorded_md5 is not None and recorded_md5 != file_digest(path):
        return False, "modified"

    return True, "current"


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
