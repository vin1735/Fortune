from __future__ import annotations

from pathlib import Path

import pytest

from fortune.loader import REQUIRED_COLUMNS, load_transactions


def test_loads_both_accounts(sample_transactions):
    assert set(sample_transactions["account"].unique()) == {"chequing", "savings"}


def test_required_columns_present(sample_transactions):
    for col in REQUIRED_COLUMNS:
        assert col in sample_transactions.columns


def test_row_counts_match_generated_sample(sample_transactions):
    assert (sample_transactions["account"] == "chequing").sum() == 61
    assert (sample_transactions["account"] == "savings").sum() == 33


def test_sorted_by_date(sample_transactions):
    assert sample_transactions["Date"].is_monotonic_increasing


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path)


def test_custom_filenames_are_respected(tmp_path: Path, sample_transactions):
    import shutil

    from fortune.loader import DEFAULT_CHEQUING_FILENAME, DEFAULT_SAVINGS_FILENAME

    data_dir = Path(__file__).resolve().parent.parent / "data"
    shutil.copy(data_dir / DEFAULT_CHEQUING_FILENAME, tmp_path / "my_chequing_export.csv")
    shutil.copy(data_dir / DEFAULT_SAVINGS_FILENAME, tmp_path / "my_savings_export.csv")

    loaded = load_transactions(tmp_path, "my_chequing_export.csv", "my_savings_export.csv")
    assert len(loaded) == len(sample_transactions)
