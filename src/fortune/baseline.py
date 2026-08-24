"""Mechanism 1: deterministic operating baseline.

Fixed, known outflows only — rent, recurring subscriptions, a seasonal
adjustment for confirmed zero-activity months, and the historical average of
small miscellaneous operating costs (OTHER_OPERATING). Nothing here is
sampled: this mechanism produces a single scalar per run, not a
distribution, per the IPS's framing of "projected disbursements" as a
deterministic operating forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fortune.classify import TransactionBucket, outflow_total
from fortune.config import BaselineConfig

_MIN_WINDOW_YEARS = 1 / 12  # floor to avoid divide-by-zero on a single-day window


@dataclass(frozen=True)
class BaselineResult:
    """Output of Mechanism 1.

    ``zero_activity_months`` and ``zero_activity_fraction`` are data-derived
    from whichever transactions were supplied; they are not hardcoded to any
    specific calendar months. ``other_operating_annual_average`` is the
    historical OTHER_OPERATING outflow total annualized over the observed
    window — also data-derived.
    """

    rent_monthly: float
    subscription_monthly: float
    zero_activity_months: tuple[str, ...]
    zero_activity_fraction: float
    other_operating_annual_average: float
    annual_committed_spend: float


def compute_zero_activity_months(
    classified_df: pd.DataFrame, account: str = "chequing"
) -> tuple[tuple[str, ...], int]:
    """Find calendar months with no genuine operating activity in ``account``.

    A month counts as zero-activity if it has no rows outside
    INTERNAL_TRANSFER/RETURNED_ITEM in the observed window (a month with
    only an internal transfer between the org's own accounts is treated as
    zero operating activity, since no money actually left the org). Returns
    (missing_months, total_months_in_window); both are 0-length/0 for an
    empty or single-account-missing input.
    """
    if classified_df.empty:
        return (), 0

    sub = classified_df[classified_df["account"] == account]
    if sub.empty:
        return (), 0

    operating = sub[~sub["bucket"].isin([TransactionBucket.INTERNAL_TRANSFER, TransactionBucket.RETURNED_ITEM])]
    months_with_activity = set(operating["Date"].dt.to_period("M"))

    start, end = sub["Date"].min().to_period("M"), sub["Date"].max().to_period("M")
    full_range = pd.period_range(start, end, freq="M")
    missing = tuple(str(m) for m in full_range if m not in months_with_activity)
    return missing, len(full_range)


def compute_other_operating_annual_average(classified_df: pd.DataFrame) -> float:
    """Annualize historical OTHER_OPERATING outflows over the observed window."""
    if classified_df.empty:
        return 0.0
    total = outflow_total(classified_df, TransactionBucket.OTHER_OPERATING)
    window_days = (classified_df["Date"].max() - classified_df["Date"].min()).days
    window_years = max(window_days / 365.25, _MIN_WINDOW_YEARS)
    return total / window_years


def compute_annual_baseline(classified_df: pd.DataFrame, config: BaselineConfig) -> BaselineResult:
    """Compute Mechanism 1's deterministic annual committed spend.

    ``classified_df`` must already carry a ``bucket`` column (see
    classify.classify_transactions). Rent and subscriptions are charged at
    their fixed monthly rate for the fraction of the year with confirmed
    operating activity, plus the historical OTHER_OPERATING average.
    """
    zero_months, total_months = compute_zero_activity_months(classified_df)
    zero_fraction = len(zero_months) / total_months if total_months else 0.0
    active_fraction = 1.0 - zero_fraction

    fixed_monthly = config.rent_monthly_amount + config.subscription_monthly_amount
    annual_fixed = 12.0 * active_fraction * fixed_monthly

    other_avg = compute_other_operating_annual_average(classified_df)

    return BaselineResult(
        rent_monthly=config.rent_monthly_amount,
        subscription_monthly=config.subscription_monthly_amount,
        zero_activity_months=zero_months,
        zero_activity_fraction=zero_fraction,
        other_operating_annual_average=other_avg,
        annual_committed_spend=annual_fixed + other_avg,
    )
