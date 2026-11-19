# tempo-ins

[![CI](https://github.com/bogdanoancea/PyTEMPO/actions/workflows/ci.yml/badge.svg)](https://github.com/bogdanoancea/PyTEMPO/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tempo-ins.svg)](https://pypi.org/project/tempo-ins/)
[![Python versions](https://img.shields.io/pypi/pyversions/tempo-ins.svg)](https://pypi.org/project/tempo-ins/)

`tempo-ins` gives Python access to the Romanian National Institute of
Statistics' [TEMPO Online](http://statistici.insse.ro:8077/tempo-online/)
database: roughly 1500 statistical matrices covering population, labour,
agriculture, trade and prices.

## Installation

```sh
pip install tempo-ins
```

Python 3.9 or newer and pandas 2.0 or newer. Tested against pandas 2.0, 2.3
and 3.1; pandas 1.x is not supported, because it neither round-trips the
response cache with its dtypes intact nor tidies without emitting warnings of
its own.

The import name is `tempo_ins`; the distribution is `tempo-ins` because `tempo`
and `pytempo` were already taken on PyPI.

## Usage

```python
import tempo_ins as tempo

# Find a table.
tempo.search("unemployed", language="en")
```

Then look before you download. A TEMPO matrix can be enormous, and
`tempo.meta()` tells you so in one request:

```python
info = tempo.meta("SOM101E", language="en")
print(info)
```

```text
<tempo_meta> SOM101E
  A. SOCIAL STATISTICS > LABOUR FORCE > REGISTERED UNEMPLOYED
  language:    en
  last update: 2026-10-12
  full table:  88,663,248 cells, over 2,000 requests
               too large to fetch whole; select values with get()
  dimensions:  5
    1. Sex (3 values): Total, Male, Female
    2. Counties (43 values): TOTAL, Alba, Arad, ...
    3. Localities (3182 values): TOTAL, 1017 ALBA IULIA MUNICIPALITY, ...
    4. Periods (216 values): Year 2010, January 2010, February 2010, ...
    5. MU: Persons (1 value): Number of persons
```

88 million cells is not a download. Fetch the slice you need instead, taking
the dimension names and value labels from `info`:

```python
periods = info.values["Periods"]["label"]

unemployment = tempo.get(
    "SOM101E",
    filters={
        "Sex": "Total",
        "Counties": "Cluj",
        "Localities": "TOTAL",
        "Periods": periods[periods.str.contains("2024")].tolist(),
    },
    language="en",
)
```

`tempo.get()` returns analysis-ready data: a numeric `value` column, the unit
of measure as a column, `":"` mapped to a missing value, and the service's
footer lines removed. Time labels are parsed according to the matrix's layout —
a dedicated `Years` dimension becomes an integer, while a combined `Periods`
dimension like the one above keeps its labels and gains `year` and `period`
columns, so an annual total stays distinguishable from the monthly figures
beneath it:

```python
unemployment[unemployment["period"].isna()]  # the annual rows
unemployment[unemployment["period"].notna()]  # the monthly rows
```

For whole tables on disk, `tempo.bulk()` writes CSV files and keeps them up to
date between runs, recording the language, published revision and checksum of
each file in a manifest beside them.

```python
files = tempo.bulk(
    ["ACC101B", "ACC101C"],
    directory="tempo-data",
    language="en",
)

accidents = tempo.tidy(files["ACC101B"], code="ACC101B")
```

`files` maps every requested code to its CSV path — including the ones that
were already current — and records what happened to each:

```python
files.to_frame()
#       code                    path  downloaded   reason  requests
# 0  ACC101B  tempo-data/ACC101B.csv        True  missing         1
# 1  ACC101C  tempo-data/ACC101C.csv        True  missing         1
```

| Function | Purpose |
| --- | --- |
| `tempo.toc()` | List every matrix, optionally with its classification |
| `tempo.search()` | Filter the table of contents by pattern |
| `tempo.hierarchy()` | Walk the statistical classification tree |
| `tempo.meta()` | Dimensions, selectable values, size and update date of one matrix |
| `tempo.get()` | Download a table, or a filtered part of it, into a data frame |
| `tempo.bulk()` | Download whole tables to CSV, skipping what is current |
| `tempo.tidy()` | Turn the service's CSV layout into analysis-ready data |
| `tempo.cache_dir()` | Inspect or change where metadata responses are cached |

## Configuration

Options are set for the process with `set_options()`, temporarily with
`option_context()`, or from the environment with a `TEMPO_`-prefixed variable:

```python
import tempo_ins as tempo

tempo.set_options(throttle=0.2)  # be gentler still

with tempo.option_context(max_cells=100_000):  # bigger single queries
    data = tempo.get("ACC101B")
```

```sh
TEMPO_THROTTLE=0.2 TEMPO_CACHE_DIR=~/.cache/tempo python script.py
```

See `help(tempo_ins.options)` for every option, and `tempo.cache_dir()` for
response caching. Metadata and the table of contents are cached for a day in
the usual per-user cache directory; `tempo.cache_clear()` empties it.

## Errors

Every error carries a class, so a script can distinguish a bad argument from
an unreachable service without matching on message text:

```python
from tempo_ins.errors import TempoHTTPError, TempoOfflineError

try:
    data = tempo.get("ACC101B")
except TempoOfflineError:
    ...  # worth retrying later
except TempoHTTPError as error:
    print(error.status_code, error.url)
```

`TempoInputError`, `TempoOfflineError`, `TempoHTTPError`, `TempoAPIError` and
`TempoIOError` all inherit from `TempoError`. `TempoInputError` is also a
`ValueError` and `TempoIOError` also an `OSError`, so the obvious handlers keep
working. Warnings inherit from `TempoWarning` and are raised at most once per
process each, so a loop over many matrices does not bury the console.

## The service

TEMPO Online is a public service of the Romanian National Institute of
Statistics. It is reachable only over plain HTTP and publishes no API contract
or rate limit. This package identifies itself with a `User-Agent`, retries only
what is worth retrying, caches metadata, and caps itself at four concurrent
connections by default. Please keep it modest.

Every function that reaches the network needs an internet connection, and
`tempo.bulk()` writes only to the directory you give it.

## Conventions

A few things worth knowing about what comes back:

- `tidy()` strips the service's `", "` field padding from values as well as
  from column names. The service pads every column but the first, so a frame
  would otherwise hold `" Cluj"` beside `"Total"`, and its values would not
  match the labels `meta()` reports — making `frame["Counties"] == "Cluj"`
  select nothing. Use `tidy=False` for the service's bytes untouched.
- Parsed years are a nullable `Int64` column and parsed periods a
  `datetime64[ns]` column.
- Provenance lives in `DataFrame.attrs`: `tempo_code`, `tempo_language`,
  `tempo_requests`, `tempo_service_update` and `tempo_retrieved_at`.
- `clean_names=True` turns `Macroregiuni si judete` into
  `Macroregiuni_si_judete`.
- The response cache holds JSON, never pickles, so a cache file cannot execute
  code when it is read back. It lives in the per-user cache directory, so it
  survives between runs; `cache_max_age` (one day) governs staleness.
- Progress and status messages go to stderr only when progress is enabled, and
  always to the `tempo_ins` logger, so the library stays quiet in a script.

## Development

```sh
pip install -e ".[dev]"
pytest                               # the offline suite
TEMPO_LIVE_TESTS=true pytest -m live # against the real service
```

The offline suite runs against a fake TEMPO service built on the standard
library, which reproduces the real endpoints — including the `", "`-padded CSV
headers and rows the service actually sends — so no test touches the network.

## Citation

See `CITATION.cff`.

## License

GPL-3.0-or-later OR EUPL-1.1.
