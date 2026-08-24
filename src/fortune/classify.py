"""Transaction classification rules.

Pure functions, no I/O. Every transaction is assigned to exactly one
TransactionBucket. Rules are applied in a fixed precedence order so that a
transaction matching more than one pattern (e.g. an exact-confirmed date
that also happens to look like a round rent amount) resolves unambiguously:

    RETURNED_ITEM > INTERNAL_TRANSFER > CASE_FUNDING > IDIOSYNCRATIC_ONEOFF
    > DETERMINISTIC_RECURRING > OTHER_OPERATING (default)
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd

from fortune.config import ClassificationRules

_AMOUNT_TOL = 0.005  # cents-level float tolerance for exact-amount matches


class TransactionBucket(StrEnum):
    DETERMINISTIC_RECURRING = "DETERMINISTIC_RECURRING"
    CASE_FUNDING = "CASE_FUNDING"
    INTERNAL_TRANSFER = "INTERNAL_TRANSFER"
    RETURNED_ITEM = "RETURNED_ITEM"
    IDIOSYNCRATIC_ONEOFF = "IDIOSYNCRATIC_ONEOFF"
    OTHER_OPERATING = "OTHER_OPERATING"


def _close(a: pd.Series | float, b: float, tol: float = _AMOUNT_TOL) -> pd.Series | bool:
    return (a - b).abs() < tol


def _mark_returned_items(df: pd.DataFrame, rules: ClassificationRules) -> pd.Series:
    """Identify NSF/returned-item reversal pairs.

    A "returned cheque" credit on a given date is paired with any
    same-date transaction whose amount is the exact negative of it and
    whose description contains "cheque" (the original item that bounced).
    Both legs are marked; the NSF fee itself is left unclassified here (it
    falls through to OTHER_OPERATING as a genuine, if small, cost).
    """
    is_returned = df["Description"].str.contains(rules.returned_item_keyword, case=False, na=False)
    mask = is_returned.copy()
    for idx in df.index[is_returned]:
        row = df.loc[idx]
        candidates = df.index[
            (df["Date"] == row["Date"])
            & (df["account"] == row["account"])
            & _close(df["Amount"], -row["Amount"])
            & df["Description"].str.contains("cheque", case=False, na=False)
            & (~is_returned)
        ]
        mask.loc[candidates] = True
    return mask


def _mark_internal_transfers(df: pd.DataFrame, rules: ClassificationRules) -> pd.Series:
    """Internal transfers between the org's own accounts.

    Two ways a transaction qualifies: (1) the description carries the
    internal-transfer keyword and the sub-description explicitly references
    one of the org's own account numbers, or (2) the description carries the
    keyword and there is a same-day, equal-and-opposite-amount transaction
    in the *other* account also carrying the keyword — a structural
    signature of a self-transfer even when the sub-description doesn't name
    the counterpart account (e.g. a mobile-transfer label with no account
    number, that nonetheless moves identical amounts between chequing and
    savings on the same day).
    """
    has_keyword = df["Description"].str.contains(rules.internal_transfer_keyword, case=False, na=False)
    references_counterpart = df["Sub-description"].apply(
        lambda s: any(acct in s for acct in rules.counterpart_account_ids)
    )
    explicit = has_keyword & references_counterpart

    mask = explicit.copy()
    for idx in df.index[has_keyword & ~explicit]:
        row = df.loc[idx]
        paired = df.index[
            has_keyword
            & (df["Date"] == row["Date"])
            & (df["account"] != row["account"])
            & _close(df["Amount"], -row["Amount"])
        ]
        if len(paired) > 0:
            mask.loc[idx] = True
            mask.loc[paired] = True
    return mask


def _mark_confirmed_events(df: pd.DataFrame, events: tuple) -> pd.Series:
    """Match confirmed events by date + exact amount, requiring the keyword
    to appear in either Description or Sub-description — bank exports don't
    consistently put identifying text (e.g. "Draft Purchase") in the same
    column across transaction types.
    """
    mask = pd.Series(False, index=df.index)
    for event in events:
        keyword_match = df["Description"].str.contains(
            event.description_contains, case=False, na=False
        ) | df["Sub-description"].str.contains(event.description_contains, case=False, na=False)
        match = (df["Date"].dt.date == event.txn_date) & _close(df["Amount"], event.amount) & keyword_match
        mask |= match
    return mask


def _mark_deterministic_recurring(df: pd.DataFrame, rules: ClassificationRules) -> pd.Series:
    is_subscription = df["Sub-description"].str.contains(rules.subscription_vendor_keyword, case=False, na=False)
    is_rent = _close(df["Amount"], rules.rent_monthly_amount) | _close(df["Amount"], rules.rent_quarterly_amount)
    return is_subscription | is_rent


def classify_transactions(df: pd.DataFrame, rules: ClassificationRules | None = None) -> pd.DataFrame:
    """Assign a ``bucket`` column to every row of ``df``.

    ``df`` must have the schema produced by loader.load_transactions
    (Date, Description, Sub-description, Type of Transaction, Amount,
    Balance, account). Returns a copy; the input is not mutated. Handles an
    empty input DataFrame by returning it unchanged with an empty ``bucket``
    column.
    """
    if rules is None:
        rules = ClassificationRules()

    out = df.copy()
    if out.empty:
        out["bucket"] = pd.Series(dtype=object)
        return out

    bucket = pd.Series(TransactionBucket.OTHER_OPERATING, index=out.index, dtype=object)

    returned = _mark_returned_items(out, rules)
    internal = _mark_internal_transfers(out, rules) & ~returned
    case_funding = _mark_confirmed_events(out, rules.confirmed_case_fundings) & ~returned & ~internal
    idiosyncratic = (
        _mark_confirmed_events(out, rules.confirmed_idiosyncratic) & ~returned & ~internal & ~case_funding
    )
    deterministic = (
        _mark_deterministic_recurring(out, rules)
        & ~returned
        & ~internal
        & ~case_funding
        & ~idiosyncratic
    )

    bucket[deterministic] = TransactionBucket.DETERMINISTIC_RECURRING
    bucket[idiosyncratic] = TransactionBucket.IDIOSYNCRATIC_ONEOFF
    bucket[case_funding] = TransactionBucket.CASE_FUNDING
    bucket[internal] = TransactionBucket.INTERNAL_TRANSFER
    bucket[returned] = TransactionBucket.RETURNED_ITEM

    out["bucket"] = bucket
    return out


def outflow_total(df: pd.DataFrame, bucket: TransactionBucket) -> float:
    """Sum of |Amount| for debit (Amount < 0) rows in the given bucket.

    Returns 0.0 if no rows match (including on an empty/degenerate frame).
    """
    if df.empty or "bucket" not in df.columns:
        return 0.0
    rows = df[(df["bucket"] == bucket) & (df["Amount"] < 0)]
    return float(-rows["Amount"].sum())
