"""CSV ingestion, date parsing, and schema validation for bank exports.

Both accounts share a schema (Date, Description, Sub-description, Type of
Transaction, Amount, Balance) plus an optional leading ``Filter`` column that
some export tools populate only on the first data row with a metadata string
describing the export filter (e.g. "All available transactions (up to 2
years), From date=..."). It is dropped here if present; it carries no
per-transaction information.

Default filenames point at the committed synthetic sample dataset
(scripts/generate_sample_data.py). Point ``--chequing-file``/``--savings-file``
(see cli.py) at your own local, non-committed export filenames to run this
against real data without ever needing to rename your bank exports or edit
tracked source.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "Date",
    "Description",
    "Sub-description",
    "Type of Transaction",
    "Amount",
    "Balance",
)

DEFAULT_CHEQUING_FILENAME = "chequing.csv"
DEFAULT_SAVINGS_FILENAME = "savings.csv"


def _load_single_csv(path: Path, account: str) -> pd.DataFrame:
    """Load one bank export and normalize it into the canonical schema.

    Raises FileNotFoundError if ``path`` does not exist, and ValueError if
    the file is missing an expected column.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected transaction export at {path}, but it does not exist.")

    df = pd.read_csv(path)
    if "Filter" in df.columns:
        df = df.drop(columns=["Filter"])

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Amount"] = df["Amount"].astype(float)
    df["Sub-description"] = df["Sub-description"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["account"] = account
    return df[["Date", "Description", "Sub-description", "Type of Transaction", "Amount", "Balance", "account"]]


def load_transactions(
    data_dir: str | Path,
    chequing_filename: str = DEFAULT_CHEQUING_FILENAME,
    savings_filename: str = DEFAULT_SAVINGS_FILENAME,
) -> pd.DataFrame:
    """Load and concatenate the chequing and savings exports from ``data_dir``.

    Returns a single DataFrame sorted by Date with an added ``account``
    column ("chequing" or "savings"). Raises FileNotFoundError if either
    expected CSV is absent from ``data_dir``. Filenames default to the
    committed synthetic sample; pass your own local filenames to point this
    at real, non-committed exports.
    """
    data_dir = Path(data_dir)
    chequing = _load_single_csv(data_dir / chequing_filename, "chequing")
    savings = _load_single_csv(data_dir / savings_filename, "savings")
    combined = pd.concat([chequing, savings], ignore_index=True)
    combined = combined.sort_values("Date", kind="stable").reset_index(drop=True)
    return combined
