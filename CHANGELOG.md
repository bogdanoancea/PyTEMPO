# Changelog

## tempo-ins 0.2.0

* Python 3.10 is now the minimum. Python 3.9 reached end of life in October
  2025 and GitHub's current runner images no longer provide it, so it could
  not be tested honestly; claiming support for a version that is neither
  security-supported nor exercised in CI is worse than dropping it. 0.1.0
  remains installable on 3.9. The test matrix runs 3.10 and 3.13 on Linux,
  macOS and Windows, with 3.11 and 3.12 on Linux.

* Fixed a path by which `toc(full_description=True)` could have attached each
  publication date to the wrong matrix code. The internal concurrent metadata
  fetch filtered missing entries out of its result rather than failing, so a
  short return would have silently misaligned the per-code columns built from
  it. It now raises a `TempoAPIError` naming the codes it could not fetch, and
  every `zip()` over paired sequences passes `strict=True` so that a length
  mismatch is an error instead of a quiet truncation.

## tempo-ins 0.1.0

First release: a Python library for the Romanian National Institute of
Statistics' TEMPO Online database — finding the statistical tables, inspecting
their structure before downloading, and retrieving whole tables or selected
parts of them as pandas data frames or CSV files.

### Requirements

Python 3.9 or newer, pandas 2.0 or newer, and `requests`. The test suite runs
against pandas 2.0, 2.3 and 3.1; pandas 3 returns its new `str` dtype from
`read_csv` and represents a missing string as `nan` rather than `None`, which
this package and its tests accommodate.

### The API

* `toc()` lists every statistical matrix, and with `full_description=True`
  also its statistical domain, sub-domain, survey and publication date.
* `search()` filters that listing by one or more patterns, over chosen columns.
* `hierarchy()` walks the service's own classification tree, with the branch
  structure in a single request and the matrices beneath it on demand.
* `meta()` reports a matrix's dimensions, their selectable values, the
  publication date, the classification path, and how many cells and requests a
  full download would cost — in one request, before any download.
* `get()` downloads a table, or a filtered part of one, straight into a pandas
  data frame. Filters are matched against the value labels exactly, then as
  whole tokens, then against the service's own identifiers; a value that
  matches nothing is an error listing what is available.
* `bulk()` downloads whole matrices to CSV files and skips what is already
  current, returning the path of every requested matrix together with why it
  was or was not fetched.
* `tidy()` turns the service's CSV layout into analysis-ready data: the unit of
  measure lifted out of the column header, a numeric `value` column, `":"` as a
  missing value, parsed time labels and no footer rows.
* `cache_dir()` and `cache_clear()` control the response cache;
  `set_options()`, `option_context()` and `TEMPO_*` environment variables
  control timeouts, retries, throttling, concurrency and query size.

### Behaviour worth knowing

Several things the service does are easy to get wrong, and are handled here
rather than left to the caller:

* A combined `Perioade`/`Periods` dimension is never collapsed to its year.
  Its labels mix frequencies, so `"Year 2010"`, `"January 2010"` and
  `"February 2010"` would otherwise become indistinguishable. The label is
  kept verbatim and `year` and `period` columns are added instead, where
  `period` is missing for an annual row. A dedicated `Ani`/`Years` dimension
  still becomes an integer in place, since it carries nothing else.
* HTML is stripped from every service label. The top-level domains arrive as
  `AGRICULTURE <a href="...">Press releases</a>`; the text is kept and the
  URLs go to a separate `Press_releases` column.
* Freshness is decided from `.tempo-manifest.json`, which records each file's
  language, the service's own update string and an MD5 checksum, because a CSV
  file does not say which language it holds. Files with no manifest entry are
  still accepted on the old modification-time rule, with a warning that their
  language cannot be verified.
* Query size is budgeted on the product of the selected values, capped by
  `max_cells`, with a per-dimension cap retained as `max_options_per_dim`. A
  matrix with five dimensions of 200 values each crosses no per-dimension
  threshold yet asks for 3.2e11 cells in one request.
* Several update-date layouts are recognised, caching falls back to the
  service's raw update string, and an unrecognised layout warns once rather
  than silently re-downloading for ever.
* Each chunk's CSV header is compared with the first chunk's, so misaligned
  columns are an error rather than a corrupt file.

### Conventions

* Parsed years are a nullable `Int64` column, parsed periods a
  `datetime64[ns]` column, and provenance — `tempo_code`, `tempo_language`,
  `tempo_requests`, `tempo_service_update`, `tempo_retrieved_at` — lives in
  `DataFrame.attrs`.
* Errors are exceptions: `TempoError` and its subclasses `TempoInputError`
  (also a `ValueError`), `TempoOfflineError`, `TempoHTTPError`,
  `TempoAPIError` and `TempoIOError` (also an `OSError`). Warnings subclass
  `TempoWarning` and are raised at most once per process each.
* Options are `set_options()`, `option_context()`, `get_option()` and
  `TEMPO_`-prefixed environment variables.
* The response cache stores JSON, never pickles, so a cache file cannot
  execute code when it is read back. It lives in the per-user cache directory,
  so it survives between runs; `cache_max_age`, one day by default, governs
  staleness.
* Request fan-out uses a thread pool capped at `max_connections`.
* `bulk()` returns a `BulkResult`, which maps each code to its path and also
  offers `.to_frame()`, `.downloaded`, `.reason` and `.requests`.
* `clean_names=True` turns `Macroregiuni si judete` into
  `Macroregiuni_si_judete`.
* `tidy()` strips the service's `", "` field padding from cell values, not
  only from column names. TEMPO Online separates CSV fields with `", "`, so
  every column but the first arrives padded: a row reads `"Total"`, `" Cluj"`,
  `" TOTAL"`, `" Year 2023"`. Left alone, a frame is inconsistent across its
  own columns and its values no longer match the labels `meta()` reports,
  which arrive as JSON already collapsed and stripped — so
  `frame["Counties"] == "Cluj"` silently selects nothing. The padding is a
  formatting artifact rather than data, so it is removed; `tidy=False` still
  returns the service's bytes untouched.
* Progress and status messages go to stderr only when progress is enabled, and
  always to the `tempo_ins` logger, so the library stays quiet by default in a
  script.
