"""Inputs are validated before any request is made."""

from __future__ import annotations

import pandas as pd
import pytest

import tempo_ins as tempo
from tempo_ins.errors import TempoInputError


def test_toc_validates_inputs_before_requesting():
    with pytest.raises(TempoInputError, match="`full_description` must be either"):
        tempo.toc(full_description=None)

    with pytest.raises(TempoInputError, match="`language` must be either"):
        tempo.toc(language="fr")


def test_bulk_validates_its_inputs_before_requesting(tmp_path):
    with pytest.raises(TempoInputError, match="`codes` must contain"):
        tempo.bulk([], str(tmp_path))

    with pytest.raises(TempoInputError, match="only letters and numbers"):
        tempo.bulk("ACC 101B", str(tmp_path))

    with pytest.raises(TempoInputError, match="`directory` must be a single path"):
        tempo.bulk("ACC101B", None)

    with pytest.raises(TempoInputError, match="`overwrite` must be either"):
        tempo.bulk("ACC101B", str(tmp_path), overwrite="yes")


def test_single_code_functions_reject_sequences():
    with pytest.raises(TempoInputError, match="`code` must be a single"):
        tempo.meta(["A1", "B2"])

    with pytest.raises(TempoInputError):
        tempo.get(["A1", "B2"])


def test_search_validates_pattern_fields_and_toc():
    toc = pd.DataFrame({"name": ["x"], "code": ["X1"]})

    with pytest.raises(TempoInputError, match="`pattern` must be"):
        tempo.search([], toc=toc)

    with pytest.raises(TempoInputError, match="`pattern` must be"):
        tempo.search(1, toc=toc)

    with pytest.raises(TempoInputError, match="not in `toc`"):
        tempo.search("x", fields=["nope"], toc=toc)

    with pytest.raises(TempoInputError, match="at least one row"):
        tempo.search("x", toc=toc.iloc[0:0])


def test_tidy_rejects_unusable_input():
    with pytest.raises(TempoInputError):
        tempo.tidy(42)

    with pytest.raises(TempoInputError, match="neither CSV text nor an existing file"):
        tempo.tidy("no-such-file.csv")


def test_cache_dir_validates_its_path():
    with pytest.raises(TempoInputError, match="`path` must be a single directory"):
        tempo.cache_dir(123)


def test_unknown_options_are_rejected():
    with pytest.raises(TempoInputError, match="Unknown option"):
        tempo.set_options(not_an_option=1)

    with pytest.raises(TempoInputError, match="Unknown option"):
        tempo.get_option("not_an_option")


def test_input_errors_are_also_value_errors():
    # So that a caller who does not import the package's classes still catches
    # a bad argument with the obvious handler.
    with pytest.raises(ValueError):
        tempo.toc(language="fr")
