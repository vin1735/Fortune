from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fortune.loader import load_transactions

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SCHEMA_COLUMNS = ["Date", "Description", "Sub-description", "Type of Transaction", "Amount", "Balance", "account"]


@pytest.fixture(scope="session")
def sample_transactions() -> pd.DataFrame:
    """The committed synthetic sample dataset (see scripts/generate_sample_data.py).

    This is what the test suite exercises end-to-end — never real
    organizational data, which is never committed to this repo.
    """
    return load_transactions(DATA_DIR)


def make_transactions(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic transactions DataFrame matching loader's schema.

    Each row dict may omit any column; sensible defaults are filled in so
    tests can specify only what matters for the case under test.
    """
    defaults = {
        "Description": "",
        "Sub-description": "",
        "Type of Transaction": "Debit",
        "Balance": 0.0,
        "account": "chequing",
    }
    filled = []
    for row in rows:
        merged = {**defaults, **row}
        filled.append(merged)
    df = pd.DataFrame(filled, columns=SCHEMA_COLUMNS)
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Amount"] = df["Amount"].astype(float)
    return df


@pytest.fixture
def empty_transactions() -> pd.DataFrame:
    return make_transactions([])
