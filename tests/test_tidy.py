"""Turning the service's CSV layout into analysis-ready data."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import tempo_ins as tempo
from tempo_ins._tidy import (
    parse_month,
    parse_quarter,
    parse_semester,
    parse_year,
    read_csv_text,
    time_role,
)
from tempo_ins.errors import TempoTidyWarning, reset_warnings

SERVICE_CSV = "\n".join(
    [
        "Judete,Ani,UM: Numar persoane,Valoare",
        "Cluj,Anul 2019,Numar persoane,1234",
        "Cluj,Anul 2020,Numar persoane,:",
        "Bihor,Anul 2019,Numar persoane,987",
        ",,,(c) INS 1998-2026",
    ]
)

# A single "Periods" dimension mixing frequencies, as SOM101E and many other
# monthly matrices use.  Collapsing these labels to a year would make the
# annual total indistinguishable from the January figure.
MIXED_CSV = "\n".join(
    [
        "Counties,Periods,UM: Persons,Value",
        "Cluj,Year 2010,Number of persons,100",
        "Cluj,January 2010,Number of persons,8",
        "Cluj,February 2010,Number of persons,9",
        "Cluj,Quarter 2 2010,Number of persons,27",
        "Cluj,Year 2011,Number of persons,110",
    ]
)


def test_the_unit_of_measure_is_lifted_out_of_the_column_header():
    frame = tempo.tidy(SERVICE_CSV, code="DEMO")

    assert not any(name.startswith("UM:") for name in frame.columns)
    assert frame["unit"].unique().tolist() == ["Numar persoane"]
    assert frame.attrs["tempo_unit"] == "Numar persoane"
    assert frame.attrs["tempo_code"] == "DEMO"


def test_observations_become_numeric_and_colon_becomes_missing():
    frame = tempo.tidy(SERVICE_CSV)

    assert pd.api.types.is_float_dtype(frame["value"])
    assert frame["value"].tolist()[0] == 1234
    assert pd.isna(frame["value"].tolist()[1])
    assert frame["value"].tolist()[2] == 987
    assert "Valoare" not in frame.columns


def test_footer_lines_are_dropped_but_missing_observations_are_kept():
    frame = tempo.tidy(SERVICE_CSV)

    # Three data rows survive; the copyright line does not.
    assert len(frame) == 3
    assert (
        not frame.astype(str)
        .apply(lambda column: column.str.contains("INS 1998"))
        .to_numpy()
        .any()
    )
    # The row whose value is ":" is a real observation and must remain.
    assert int(frame["value"].isna().sum()) == 1


def test_footer_dropping_can_be_switched_off():
    reset_warnings()

    # Keeping the footer means its text reaches the observation column, which
    # is exactly the case that must warn rather than turn quietly into NA.
    with pytest.warns(TempoTidyWarning):
        frame = tempo.tidy(SERVICE_CSV, drop_notes=False)

    assert len(frame) == 4
    assert pd.isna(frame["value"].iloc[3])
    assert frame["value_raw"].iloc[3] == "(c) INS 1998-2026"


def test_the_year_label_becomes_an_integer():
    frame = tempo.tidy(SERVICE_CSV)

    assert pd.api.types.is_integer_dtype(frame["Ani"])
    assert frame["Ani"].tolist() == [2019, 2020, 2019]


def test_monthly_labels_produce_a_date_period():
    monthly = "\n".join(
        [
            "Luni,Ani,UM: Numar,Valoare",
            "luna ianuarie,Anul 2024,Numar,10",
            "luna februarie,Anul 2024,Numar,12",
            "luna decembrie,Anul 2024,Numar,14",
        ]
    )

    frame = tempo.tidy(monthly)

    assert pd.api.types.is_datetime64_any_dtype(frame["period"])
    assert frame["period"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        pd.Timestamp("2024-12-01"),
    ]
    # The readable label is left alone.
    assert frame["Luni"].iloc[0] == "luna ianuarie"


def test_quarterly_labels_produce_a_date_period():
    quarterly = "\n".join(
        [
            "Trimestre,Ani,UM: Numar,Valoare",
            "Trimestrul I,Anul 2023,Numar,1",
            "Trimestrul III,Anul 2023,Numar,3",
        ]
    )

    frame = tempo.tidy(quarterly)
    assert frame["period"].tolist() == [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-07-01"),
    ]


def test_english_labels_are_parsed_too():
    english = "\n".join(
        [
            "Months,Years,MU: Number,Value",
            "month January,Year 2024,Number,10",
            "month July,Year 2024,Number,20",
        ]
    )

    frame = tempo.tidy(english)
    assert frame["period"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-07-01"),
    ]
    assert frame["unit"].unique().tolist() == ["Number"]
    assert frame["value"].tolist() == [10, 20]


def test_period_parsing_can_be_switched_off():
    frame = tempo.tidy(SERVICE_CSV, periods=False)

    assert not pd.api.types.is_numeric_dtype(frame["Ani"])
    assert frame["Ani"].iloc[0] == "Anul 2019"
    assert "period" not in frame.columns


def test_annual_data_gets_no_period_column_only_an_integer_year():
    frame = tempo.tidy(SERVICE_CSV)
    assert "period" not in frame.columns
    assert "year" not in frame.columns


def test_unreadable_observations_warn_and_are_preserved():
    reset_warnings()
    odd = "\n".join(
        [
            "Judete,Ani,UM: Numar,Valoare",
            "Cluj,Anul 2019,Numar,1234",
            "Bihor,Anul 2019,Numar,confidential",
        ]
    )

    with pytest.warns(TempoTidyWarning):
        frame = tempo.tidy(odd)

    assert "value_raw" in frame.columns
    assert frame["value_raw"].tolist() == ["1234", "confidential"]
    assert frame["value"].iloc[0] == 1234
    assert pd.isna(frame["value"].iloc[1])


def test_thousands_separators_are_tolerated():
    spaced = "\n".join(
        [
            "Judete,Ani,UM: Lei,Valoare",
            'Cluj,Anul 2019,Lei,"1 234 567"',
        ]
    )

    assert tempo.tidy(spaced)["value"].iloc[0] == 1234567


def test_non_breaking_spaces_are_tolerated():
    spaced = "\n".join(
        [
            "Judete,Ani,UM: Lei,Valoare",
            'Cluj,Anul 2019,Lei,"1 234"',
        ]
    )

    assert tempo.tidy(spaced)["value"].iloc[0] == 1234


def test_clean_names_produces_usable_names_on_request():
    spaced = "\n".join(
        [
            "Macroregiuni si judete,Ani,UM: Numar,Valoare",
            "Cluj,Anul 2019,Numar,1",
        ]
    )

    assert "Macroregiuni si judete" in tempo.tidy(spaced).columns
    assert "Macroregiuni_si_judete" in tempo.tidy(spaced, clean_names=True).columns


def test_a_table_with_no_recognised_value_column_warns_once():
    reset_warnings()

    with pytest.warns(TempoTidyWarning):
        frame = tempo.tidy("Judete,Ani\nCluj,Anul 2019\n")

    assert list(frame.columns) == ["Judete", "Ani"]


def test_tidy_accepts_a_data_frame_and_a_file_path(tmp_path):
    frame = read_csv_text(SERVICE_CSV)
    assert len(tempo.tidy(frame)) == 3

    path = tmp_path / "table.csv"
    path.write_text(SERVICE_CSV + "\n", encoding="utf-8")
    assert len(tempo.tidy(str(path))) == 3
    assert len(tempo.tidy(path)) == 3


def test_tidy_does_not_mutate_the_frame_it_was_given():
    frame = read_csv_text(SERVICE_CSV)
    before = frame.copy()

    tempo.tidy(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_period_parsers_handle_the_labels_in_isolation():
    assert parse_year(["Anul 2020", "2021", "Year 1999"]) == [2020, 2021, 1999]
    assert parse_month(["luna mai", "May", "decembrie"]) == [5, 5, 12]
    assert parse_quarter(["Trimestrul IV", "Quarter 2", "Trim. II"]) == [4, 2, 2]
    assert parse_semester(["Semestrul I", "Semester 2"]) == [1, 2]
    assert parse_month(["Total"]) == [None]


def test_a_combined_periods_column_is_recognised_as_such():
    assert time_role("Periods", ["Year 2010", "January 2010"]) == "period"
    assert time_role("Perioade", ["Anul 2010", "luna ianuarie 2010"]) == "period"
    # Detected from content even when the column is named something else.
    assert time_role("Time", ["Year 2010", "March 2010", "Year 2011"]) == "period"
    # A plain year column is still a year column.
    assert time_role("Ani", ["Anul 2019", "Anul 2020"]) == "year"
    assert time_role("Time", ["Year 2019", "Year 2020"]) == "year"
    assert time_role("Judete", ["Cluj", "Bihor"]) is None


def test_a_combined_periods_label_is_never_collapsed_to_its_year():
    frame = tempo.tidy(MIXED_CSV)

    # The label survives verbatim: it is the only thing telling an annual total
    # apart from a monthly figure sharing the same year.
    assert not pd.api.types.is_numeric_dtype(frame["Periods"])
    assert frame["Periods"].iloc[0] == "Year 2010"
    assert frame["Periods"].iloc[1] == "January 2010"
    assert frame["Periods"].nunique() == 5


def test_a_combined_periods_column_yields_year_and_period_columns():
    frame = tempo.tidy(MIXED_CSV)

    assert frame["year"].tolist() == [2010, 2010, 2010, 2010, 2011]
    assert pd.api.types.is_datetime64_any_dtype(frame["period"])
    assert frame["period"].isna().tolist() == [True, False, False, False, True]
    assert frame["period"].iloc[1] == pd.Timestamp("2010-01-01")
    assert frame["period"].iloc[2] == pd.Timestamp("2010-02-01")
    assert frame["period"].iloc[3] == pd.Timestamp("2010-04-01")


def test_annual_and_sub_annual_rows_are_separable_after_tidying():
    frame = tempo.tidy(MIXED_CSV)

    annual = frame[frame["period"].isna()]
    sub_annual = frame[frame["period"].notna()]

    assert annual["value"].tolist() == [100, 110]
    assert sub_annual["value"].tolist() == [8, 9, 27]
    assert len(annual) + len(sub_annual) == len(frame)


def test_a_dedicated_year_column_is_still_converted_in_place():
    # The two layouts must not be confused: this one has no other information
    # in the column, so the integer year is the useful form.
    frame = tempo.tidy(SERVICE_CSV)
    assert pd.api.types.is_integer_dtype(frame["Ani"])
    assert "year" not in frame.columns


def test_a_periods_column_of_years_alone_still_parses():
    annual_only = "\n".join(
        [
            "Counties,Periods,UM: Persons,Value",
            "Cluj,Year 2019,Number of persons,1",
            "Cluj,Year 2020,Number of persons,2",
        ]
    )

    frame = tempo.tidy(annual_only)
    assert frame["year"].tolist() == [2019, 2020]
    assert bool(frame["period"].isna().all())


def test_tidying_is_quiet_on_a_well_formed_table():
    reset_warnings()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tempo.tidy(SERVICE_CSV)

    assert caught == []


def test_custom_value_and_unit_names_are_honoured():
    frame = tempo.tidy(SERVICE_CSV, value_name="obs", unit_name="measure")

    assert "obs" in frame.columns
    assert "measure" in frame.columns
    assert "value" not in frame.columns


def test_padded_values_are_stripped_so_they_match_the_meta_labels():
    # The exact shape the live service sends: ", "-separated, so every column
    # but the first is padded in both the header and the values.
    padded = "\n".join(
        [
            "Sex, Counties, Localities, Periods, UM: Persons, Valoare",
            "Total, Cluj, TOTAL, Year 2023, Number of persons, 4182",
        ]
    )

    frame = tempo.tidy(padded)

    assert list(frame.columns) == [
        "Sex",
        "Counties",
        "Localities",
        "Periods",
        "unit",
        "value",
        "year",
        "period",
    ]
    # Every cell matches the label meta() would report, with no stray padding.
    assert frame["Sex"].iloc[0] == "Total"
    assert frame["Counties"].iloc[0] == "Cluj"
    assert frame["Localities"].iloc[0] == "TOTAL"
    assert frame["Periods"].iloc[0] == "Year 2023"
    assert frame["unit"].iloc[0] == "Number of persons"
    assert frame["value"].iloc[0] == 4182
    assert frame["year"].iloc[0] == 2023
    # An annual label names no sub-annual period.
    assert pd.isna(frame["period"].iloc[0])


def test_selecting_a_padded_value_by_its_label_works():
    # The failure mode this prevents: frame["Counties"] == "Cluj" selecting
    # nothing because the value is really " Cluj".
    padded = "\n".join(
        [
            "Sex, Counties, UM: Persons, Valoare",
            "Total, Cluj, Number of persons, 1",
            "Total, Bihor, Number of persons, 2",
        ]
    )

    frame = tempo.tidy(padded)

    assert len(frame[frame["Counties"] == "Cluj"]) == 1
    assert frame.groupby("Counties")["value"].sum().to_dict() == {
        "Bihor": 2,
        "Cluj": 1,
    }
