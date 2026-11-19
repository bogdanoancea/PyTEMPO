"""The response cache and the download manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re

import pandas as pd
import pytest

import tempo_ins as tempo
from tempo_ins._cache import (
    cache_get,
    cache_key,
    cache_set,
    cached,
    file_is_current,
    manifest_entry,
    manifest_path,
    manifest_read,
    manifest_write,
    prepare_directory,
)
from tempo_ins.errors import (
    TempoCacheWarning,
    TempoInputError,
    reset_warnings,
)
from tempo_ins.options import option_context, reset_options


def test_the_default_cache_lives_in_the_user_cache_directory():
    reset_options("cache_dir")

    assert tempo.cache_dir().endswith("tempo-ins") or tempo.cache_dir().endswith(
        os.path.join("tempo-ins", "Cache")
    )
    assert os.path.isabs(tempo.cache_dir())


def test_the_cache_directory_can_be_set_and_returns_the_previous_value(tmp_path):
    with option_context(cache_dir=str(tmp_path / "first")):
        previous = tempo.cache_dir(str(tmp_path / "second"))

        assert previous == str(tmp_path / "first")
        assert tempo.cache_dir() == str(tmp_path / "second")


def test_values_round_trip_through_the_cache_and_are_cleared_on_demand():
    key = cache_key("unit", "test")
    assert cache_get(key) is None

    cache_set(key, {"a": 1})
    assert cache_get(key) == {"a": 1}

    assert tempo.cache_clear() == 1
    assert cache_get(key) is None


def test_data_frames_round_trip_through_the_cache():
    frame = pd.DataFrame(
        {
            "code": ["A", "B"],
            "level": pd.array([1, 2], dtype="Int64"),
            "is_matrix": [True, False],
            "when": pd.to_datetime(["2025-11-05", None]),
        }
    )

    key = cache_key("frame")
    assert cache_set(key, frame) is True

    restored = cache_get(key)
    assert isinstance(restored, pd.DataFrame)
    assert restored["code"].tolist() == ["A", "B"]
    assert restored["is_matrix"].tolist() == [True, False]
    assert restored["when"].iloc[0] == pd.Timestamp("2025-11-05")
    assert pd.isna(restored["when"].iloc[1])


def test_cached_values_expire():
    key = cache_key("expiry")
    cache_set(key, "stale")

    with option_context(cache_max_age=-1):
        assert cache_get(key) is None


def test_cached_computes_once_and_reuses_afterwards():
    calls = []

    def compute():
        calls.append(1)
        return "value"

    key = cache_key("memo")

    assert cached(key, compute) == "value"
    assert cached(key, compute) == "value"
    assert len(calls) == 1


def test_cache_keys_distinguish_language_and_base_url():
    assert cache_key("matrix", "A", "ro") != cache_key("matrix", "A", "en")
    assert cache_key("matrix", "A", "ro", "u1") != cache_key("matrix", "A", "ro", "u2")


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path):
    key = cache_key("corrupt")
    cache_set(key, {"a": 1})
    path = os.path.join(tempo.cache_dir(), key + ".json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{not json")

    assert cache_get(key) is None


# Download manifest -----------------------------------------------------------


def fake_metadata(update: str = "05-11-2025") -> dict:
    return {
        "ultimaActualizare": update,
        "details": {"matMaxDim": 1, "matUMSpec": 0},
        "dimensionsMap": [{"options": [{"nomItemId": 1}]}],
    }


def write_csv_file(directory: str, code: str, text: str = "a,b\n1,2\n") -> str:
    path = os.path.join(directory, f"{code}.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def test_a_missing_file_is_never_current(download_dir):
    reusable, reason = file_is_current(
        download_dir,
        "AAA",
        "ro",
        fake_metadata(),
        os.path.join(download_dir, "AAA.csv"),
    )

    assert reusable is False
    assert reason == "missing"


def test_a_file_recorded_in_the_manifest_is_reused(download_dir):
    metadata = fake_metadata()
    path = write_csv_file(download_dir, "AAA")
    manifest_write(
        download_dir, {"AAA": manifest_entry("AAA", "ro", metadata, path, 1)}
    )

    reusable, reason = file_is_current(download_dir, "AAA", "ro", metadata, path)

    assert reusable is True
    assert reason == "current"


def test_a_romanian_file_is_not_accepted_for_an_english_request(download_dir):
    # This is the bug the manifest exists to fix: the CSV itself does not
    # record its language, so a modification-time test cannot tell them apart.
    metadata = fake_metadata()
    path = write_csv_file(download_dir, "AAA")
    manifest_write(
        download_dir, {"AAA": manifest_entry("AAA", "ro", metadata, path, 1)}
    )

    reusable, reason = file_is_current(download_dir, "AAA", "en", metadata, path)

    assert reusable is False
    assert reason == "language"


def test_a_revised_matrix_invalidates_the_local_file(download_dir):
    path = write_csv_file(download_dir, "AAA")
    manifest_write(
        download_dir,
        {"AAA": manifest_entry("AAA", "ro", fake_metadata("05-11-2025"), path, 1)},
    )

    reusable, reason = file_is_current(
        download_dir, "AAA", "ro", fake_metadata("06-12-2025"), path
    )

    assert reusable is False
    assert reason == "revised"


def test_an_edited_or_truncated_file_invalidates_the_local_copy(download_dir):
    metadata = fake_metadata()
    path = write_csv_file(download_dir, "AAA")
    manifest_write(
        download_dir, {"AAA": manifest_entry("AAA", "ro", metadata, path, 1)}
    )

    write_csv_file(download_dir, "AAA", "a,b\n9,9\n")
    reusable, reason = file_is_current(download_dir, "AAA", "ro", metadata, path)

    assert reusable is False
    assert reason == "modified"


def test_files_predating_the_manifest_are_reused_with_a_warning(download_dir):
    reset_warnings()
    path = write_csv_file(download_dir, "AAA")

    with pytest.warns(TempoCacheWarning):
        reusable, reason = file_is_current(
            download_dir, "AAA", "ro", fake_metadata("05-11-2020"), path
        )

    assert reusable is True
    assert reason == "mtime"


def test_an_unparseable_update_date_on_a_legacy_file_forces_a_download(
    download_dir,
):
    path = write_csv_file(download_dir, "AAA")

    reusable, reason = file_is_current(
        download_dir, "AAA", "ro", fake_metadata("not a date"), path
    )

    assert reusable is False
    assert reason == "unverifiable"


def test_the_manifest_survives_a_round_trip_and_records_provenance(download_dir):
    metadata = fake_metadata()
    path = write_csv_file(download_dir, "AAA")
    manifest_write(
        download_dir, {"AAA": manifest_entry("AAA", "en", metadata, path, 3)}
    )

    entry = manifest_read(download_dir)["AAA"]

    assert entry["language"] == "en"
    assert entry["service_update"] == "05-11-2025"
    assert entry["last_update"] == "2025-11-05"
    assert entry["requests"] == 3
    with open(path, "rb") as handle:
        assert entry["md5"] == hashlib.md5(handle.read()).hexdigest()
    assert re.match(r"^\d{4}-\d{2}-\d{2}T", entry["retrieved_at"])


def test_the_manifest_has_a_stable_shape(download_dir):
    metadata = fake_metadata()
    path = write_csv_file(download_dir, "AAA")
    manifest_write(
        download_dir, {"AAA": manifest_entry("AAA", "ro", metadata, path, 1)}
    )

    assert os.path.basename(manifest_path(download_dir)) == ".tempo-manifest.json"
    with open(manifest_path(download_dir), encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["manifest_version"] == 1
    assert set(payload) == {
        "manifest_version",
        "package",
        "package_version",
        "written_at",
        "files",
    }


def test_a_corrupt_manifest_is_ignored_rather_than_fatal(download_dir):
    with open(manifest_path(download_dir), "w", encoding="utf-8") as handle:
        handle.write("{not json")

    assert manifest_read(download_dir) == {}


def test_an_unusable_download_directory_is_a_classed_error():
    with pytest.raises(TempoInputError):
        prepare_directory(None)

    with pytest.raises(TempoInputError):
        prepare_directory(["a", "b"])

    with pytest.raises(TempoInputError):
        prepare_directory("   ")


def test_a_download_directory_is_created_when_missing(tmp_path):
    target = tmp_path / "deep" / "nested"
    created = prepare_directory(str(target))

    assert os.path.isdir(created)


def test_a_hierarchy_shaped_frame_survives_the_cache_unchanged():
    # hierarchy() caches its result, so the JSON encoding has to preserve a
    # missing parent_code, nullable integer levels and a boolean column.
    frame = pd.DataFrame(
        {
            "code": ["1", "10", "SML101A"],
            "name": ["AGRICULTURE", "Land use", "Small fake table"],
            "parent_code": [None, "1", "10"],
            "level": pd.array([1, 2, 3], dtype="Int64"),
            "is_matrix": [False, False, True],
            "path": [
                "AGRICULTURE",
                "AGRICULTURE > Land use",
                "AGRICULTURE > Land use > Small fake table",
            ],
        }
    )

    key = cache_key("hierarchy", "ro", True, "url")
    assert cache_set(key, frame) is True

    restored = cache_get(key)
    assert restored.equals(frame)
    assert pd.isna(restored["parent_code"].iloc[0])
    assert pd.api.types.is_integer_dtype(restored["level"])


def test_a_manifest_from_another_writer_is_understood(download_dir):
    # A manifest this package did not write itself must still be read, so that
    # an older version's output, or another tool sharing the directory, does
    # not silently force a re-download.
    path = write_csv_file(download_dir, "AAA")
    with open(path, "rb") as handle:
        digest = hashlib.md5(handle.read()).hexdigest()

    foreign_manifest = {
        "manifest_version": 1,
        "package": "some-other-tool",
        "package_version": "1.4.2",
        "written_at": "2026-11-05T10:00:00+0200",
        "files": {
            "AAA": {
                "code": "AAA",
                "language": "ro",
                "file": "AAA.csv",
                "service_update": "05-11-2025",
                "last_update": "2025-11-05",
                "retrieved_at": "2026-11-05T10:00:00+0200",
                "requests": 1,
                "bytes": os.path.getsize(path),
                "md5": digest,
            }
        },
    }
    with open(manifest_path(download_dir), "w", encoding="utf-8") as handle:
        json.dump(foreign_manifest, handle)

    metadata = fake_metadata()

    assert list(manifest_read(download_dir)) == ["AAA"]
    assert file_is_current(download_dir, "AAA", "ro", metadata, path) == (
        True,
        "current",
    )
    # And the language check still protects against reusing the wrong one.
    assert file_is_current(download_dir, "AAA", "en", metadata, path) == (
        False,
        "language",
    )
