"""Generate the committed synthetic sample dataset.

Writes data/chequing.csv and data/savings.csv: entirely fabricated
transactions that are structurally faithful to a real two-account bank
export (same schema, same classification-relevant patterns) but contain no
real dates, amounts, account numbers, or vendor names. This is the dataset
anyone cloning this repo actually runs against — the real organization's
real banking history is never committed (see README's confidentiality
note).

Seeded and reproducible: same --seed produces byte-identical CSVs. Run with:

    python scripts/generate_sample_data.py

Structural features exercised (mirrors config.py's default assumptions):
  - Rent-equivalent fixed obligation paid on an irregular cadence: some
    months individually, one cluster of three payments within days, and
    several quarterly lump-sum payments (3x the monthly amount).
  - A recurring small subscription-style debit.
  - An NSF reversal: a debit, a same-day equal-and-opposite
    "returned cheque - nsf" credit, and a small fee.
  - Internal transfer pairs of both kinds: one referencing the counterpart
    account number in the sub-description, and one same-day
    equal-and-opposite cross-account pair with no account-number reference
    (the case the generalized structural rule in classify.py catches).
  - One large disbursement, pre-funded by transfers in shortly before.
  - Three idiosyncratic one-off operating costs of differing magnitude (one
    with its identifying text in Sub-description rather than Description,
    regression-testing the same real-world parsing bug this project found).
  - Two zero-activity months on the chequing side.
  - Occasional revenue inflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

CHEQUING_ACCOUNT_ID = "555000011122"
SAVINGS_ACCOUNT_ID = "555000033344"

ZERO_ACTIVITY_MONTHS = {(2023, 11), (2024, 11)}
ALL_MONTHS = [(y, m) for y in (2023, 2024) for m in range(1, 13)]
ACTIVE_MONTHS = [ym for ym in ALL_MONTHS if ym not in ZERO_ACTIVITY_MONTHS]

NOISE_POOL = [
    ("service charge", "Interac E-Transfer Fee"),
    ("pos purchase", "Card Purchase - Office Supplies"),
    ("pos purchase", "Card Purchase - Print Shop"),
    ("pos purchase", "Card Purchase - Transit"),
    ("service charge", ""),
]


def _row(date: str, description: str, sub_description: str, amount: float) -> dict:
    return {
        "Date": date,
        "Description": description,
        "Sub-description": sub_description,
        "Type of Transaction": "Credit" if amount > 0 else "Debit",
        "Amount": round(amount, 2),
    }


def build_chequing_rows(rng: np.random.Generator) -> list[dict]:
    rows = [
        # Rent-equivalent obligation, irregular cadence.
        _row("2023-01-10", "cheque  301", "", -1200.00),
        _row("2023-02-14", "cheque  302", "", -1200.00),
        _row("2023-03-14", "debit memo", "Interac E-Transfer", -1200.00),
        # A cluster of several payments within a few days.
        _row("2023-04-02", "cheque  304", "", -1200.00),
        _row("2023-04-04", "cheque  305", "", -1200.00),
        _row("2023-04-06", "cheque  306", "", -1200.00),
        # NSF reversal: debit, same-day equal-and-opposite reversal, small fee.
        _row("2023-05-20", "cheque  307", "", -1200.00),
        _row("2023-05-20", "returned cheque - nsf", "", 1200.00),
        _row("2023-05-20", "service charge", "NSF Fee", -35.00),
        _row("2023-06-01", "cheque  309", "", -1200.00),
        # Quarterly lump-sum payments (3x the monthly amount).
        _row("2023-07-01", "cheque  310", "", -3600.00),
        _row("2023-10-01", "cheque  313", "", -3600.00),
        _row("2024-01-05", "cheque  316", "", -1200.00),
        _row("2024-04-01", "cheque  319", "", -3600.00),
        _row("2024-07-08", "cheque  322", "", -1200.00),
        _row("2024-10-02", "cheque  325", "", -1200.00),
        # Idiosyncratic one-offs, differing magnitude.
        _row("2023-03-25", "cheque  303", "", -18000.00),
        # Identifying text lives in Sub-description, not Description — mirrors
        # a real parsing bug this project's classification rules must catch.
        _row("2023-08-14", "debit memo", "Draft Purchase", -9000.00),
        _row("2024-03-02", "cheque  318", "", -2200.00),
        # Pre-funding transfers in, then a large disbursement.
        _row("2024-06-10", "customer transfer cr.", f"Pc From {SAVINGS_ACCOUNT_ID}", 32500.00),
        _row("2024-06-12", "customer transfer cr.", f"Pc From {SAVINGS_ACCOUNT_ID}", 49999.00),
        _row("2024-06-24", "cheque  320", "", -80000.00),
        # Internal transfer pair with NO account-number reference (twice).
        _row("2023-09-15", "customer transfer cr.", "Mobile Transfer", 5000.00),
        _row("2024-02-01", "customer transfer cr.", "Mobile Transfer", 5000.00),
        # Surplus sweep, chequing -> savings.
        _row("2024-08-01", "customer transfer dr.", f"Pc To {SAVINGS_ACCOUNT_ID}", -15000.00),
        # Revenue inflows.
        _row("2023-02-20", "miscellaneous payment", "Membership Fees", 20000.00),
        _row("2024-04-25", "miscellaneous payment", "Grant Payment", 15000.00),
    ]

    # Recurring subscription-style debit (a handful of instances, jittered).
    for d in ("2023-01-12", "2023-04-12", "2023-07-12", "2024-01-12", "2024-07-12"):
        amount = -round(float(rng.uniform(38.0, 42.0)), 2)
        rows.append(_row(d, "pos purchase", "Opos 40.00 NimbusForms Inc", amount))

    # Small noise transactions to pad out realistic density, skipping the
    # two zero-activity months entirely.
    for year, month in ACTIVE_MONTHS:
        # At least one noise transaction per active month, so randomness
        # never accidentally creates a third zero-activity month beyond the
        # two deliberately designated ones.
        n_noise = int(rng.integers(1, 3))
        for _ in range(n_noise):
            day = int(rng.integers(1, 27))
            desc, sub = NOISE_POOL[int(rng.integers(0, len(NOISE_POOL)))]
            amount = -round(float(rng.uniform(1.0, 60.0)), 2)
            rows.append(_row(f"{year:04d}-{month:02d}-{day:02d}", desc, sub, amount))

    return rows


def build_savings_rows(rng: np.random.Generator) -> list[dict]:
    rows = [
        _row("2024-06-10", "customer transfer dr.", f"Pc To {CHEQUING_ACCOUNT_ID}", -32500.00),
        _row("2024-06-12", "customer transfer dr.", f"Pc To {CHEQUING_ACCOUNT_ID}", -49999.00),
        _row("2023-09-15", "customer transfer dr.", "Mobile Transfer", -5000.00),
        _row("2024-02-01", "customer transfer dr.", "Mobile Transfer", -5000.00),
        _row("2024-08-01", "customer transfer cr.", f"Pc From {CHEQUING_ACCOUNT_ID}", 15000.00),
    ]

    # Monthly interest credit, every month including chequing's zero months.
    for year, month in ALL_MONTHS:
        amount = round(float(rng.uniform(50.0, 300.0)), 2)
        rows.append(_row(f"{year:04d}-{month:02d}-28", "interest credit", "", amount))

    # A few small savings-side service charges.
    fee_months = ((2023, 3), (2023, 10), (2024, 5), (2024, 9))
    for year, month in fee_months:
        amount = -round(float(rng.uniform(1.0, 2.0)), 2)
        rows.append(_row(f"{year:04d}-{month:02d}-05", "service charge", "Interac E-Transfer Fee", amount))

    return rows


def _with_running_balance(rows: list[dict], starting_balance: float) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    df["Balance"] = (starting_balance + df["Amount"].cumsum()).round(2)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df[["Date", "Description", "Sub-description", "Type of Transaction", "Amount", "Balance"]]


def generate(seed: int, output_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)

    chequing = _with_running_balance(build_chequing_rows(rng), starting_balance=25_000.00)
    savings = _with_running_balance(build_savings_rows(rng), starting_balance=130_000.00)

    output_dir.mkdir(parents=True, exist_ok=True)
    chequing_path = output_dir / "chequing.csv"
    savings_path = output_dir / "savings.csv"
    chequing.to_csv(chequing_path, index=False)
    savings.to_csv(savings_path, index=False)
    return chequing_path, savings_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2024, help="RNG seed for the noise/jitter transactions.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data", help="Directory to write CSVs to.")
    args = parser.parse_args()

    chequing_path, savings_path = generate(args.seed, args.output_dir)
    print(f"Wrote {chequing_path}")
    print(f"Wrote {savings_path}")


if __name__ == "__main__":
    main()
