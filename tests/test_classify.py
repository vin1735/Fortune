from __future__ import annotations

import pandas as pd

from fortune.classify import TransactionBucket, classify_transactions, outflow_total
from tests.conftest import make_transactions

FAKE_CHEQUING_ACCT = "000111222333"
FAKE_SAVINGS_ACCT = "000111444555"


def _bucket_for(df: pd.DataFrame, date: str, amount: float) -> str:
    row = df[(df["Date"] == pd.Timestamp(date)) & (df["Amount"].round(2) == round(amount, 2))]
    assert len(row) == 1, f"expected exactly one match for {date} {amount}, found {len(row)}"
    return row["bucket"].iloc[0]


class TestConfirmedTransactionsLandInCorrectBucket:
    """These assertions target the committed synthetic sample dataset's own
    hand-placed confirmed events (see scripts/generate_sample_data.py), not
    any real organization's data.
    """

    def test_rent_quarterly_lump_payments(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        assert _bucket_for(c, "2023-07-01", -3600.00) == TransactionBucket.DETERMINISTIC_RECURRING
        assert _bucket_for(c, "2024-04-01", -3600.00) == TransactionBucket.DETERMINISTIC_RECURRING

    def test_bounced_rent_cheque_is_returned_item_not_rent(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        assert _bucket_for(c, "2023-05-20", -1200.00) == TransactionBucket.RETURNED_ITEM
        assert _bucket_for(c, "2023-05-20", 1200.00) == TransactionBucket.RETURNED_ITEM

    def test_monthly_rent_debit_memos_are_recurring(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        for date in ["2023-01-10", "2023-02-14", "2023-03-14", "2023-04-02", "2023-04-04", "2023-04-06"]:
            assert _bucket_for(c, date, -1200.00) == TransactionBucket.DETERMINISTIC_RECURRING

    def test_subscription_vendor(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        subscription_rows = c[c["Sub-description"].str.contains("NimbusForms", na=False)]
        assert len(subscription_rows) == 5
        assert (subscription_rows["bucket"] == TransactionBucket.DETERMINISTIC_RECURRING).all()

    def test_case_funding_disbursement(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        assert _bucket_for(c, "2024-06-24", -80000.00) == TransactionBucket.CASE_FUNDING

    def test_case_prefunding_transfers_are_internal(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        assert _bucket_for(c, "2024-06-10", 32500.00) == TransactionBucket.INTERNAL_TRANSFER
        assert _bucket_for(c, "2024-06-12", 49999.00) == TransactionBucket.INTERNAL_TRANSFER

    def test_surplus_sweep_is_internal(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        assert _bucket_for(c, "2024-08-01", -15000.00) == TransactionBucket.INTERNAL_TRANSFER

    def test_confirmed_idiosyncratic_oneoffs(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        assert _bucket_for(c, "2023-03-25", -18000.00) == TransactionBucket.IDIOSYNCRATIC_ONEOFF
        assert _bucket_for(c, "2023-08-14", -9000.00) == TransactionBucket.IDIOSYNCRATIC_ONEOFF
        assert _bucket_for(c, "2024-03-02", -2200.00) == TransactionBucket.IDIOSYNCRATIC_ONEOFF

    def test_idiosyncratic_oneoff_identified_via_sub_description(self, sample_transactions):
        """Regression test: the $9,000 one-off's identifying text ("Draft
        Purchase") lives in Sub-description, not Description — this is the
        exact shape of a real classification bug this project found and
        fixed (matching must check both columns).
        """
        c = classify_transactions(sample_transactions)
        row = c[(c["Date"] == pd.Timestamp("2023-08-14")) & (c["Amount"] == -9000.00)]
        assert row["Description"].iloc[0] == "debit memo"
        assert "Draft Purchase" in row["Sub-description"].iloc[0]
        assert row["bucket"].iloc[0] == TransactionBucket.IDIOSYNCRATIC_ONEOFF

    def test_exactly_three_idiosyncratic_oneoffs(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        assert (c["bucket"] == TransactionBucket.IDIOSYNCRATIC_ONEOFF).sum() == 3

    def test_exactly_one_case_funding_event(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        assert (c["bucket"] == TransactionBucket.CASE_FUNDING).sum() == 1


class TestInternalTransferExclusion:
    def test_internal_transfers_excluded_from_outflow_totals(self):
        df = make_transactions(
            [
                {"Date": "2026-01-01", "Description": "customer transfer dr.", "Sub-description": f"Pc To {FAKE_CHEQUING_ACCT}", "Amount": -50000.0},
                {"Date": "2026-01-01", "Description": "customer transfer cr.", "Sub-description": f"Pc From {FAKE_SAVINGS_ACCT}", "Amount": 50000.0, "account": "savings"},
            ]
        )
        c = classify_transactions(df)
        assert (c["bucket"] == TransactionBucket.INTERNAL_TRANSFER).all()
        assert outflow_total(c, TransactionBucket.INTERNAL_TRANSFER) == 50000.0  # bucket total itself is nonzero...
        # ...but it must never be included in any of the three outflow mechanisms' inputs.
        assert outflow_total(c, TransactionBucket.OTHER_OPERATING) == 0.0
        assert outflow_total(c, TransactionBucket.DETERMINISTIC_RECURRING) == 0.0
        assert outflow_total(c, TransactionBucket.IDIOSYNCRATIC_ONEOFF) == 0.0
        assert outflow_total(c, TransactionBucket.CASE_FUNDING) == 0.0

    def test_unreferenced_but_structurally_paired_transfer_is_internal(self, sample_transactions):
        """The sample dataset's "Mobile Transfer" entries: no account-number
        cross-reference in the sub-description, but a same-day,
        equal-and-opposite, cross-account 'customer transfer' pair is a
        structural signature of a self-transfer.
        """
        c = classify_transactions(sample_transactions)
        for txn_date in ("2023-09-15", "2024-02-01"):
            rows = c[c["Date"] == pd.Timestamp(txn_date)]
            assert len(rows) == 2
            assert (rows["bucket"] == TransactionBucket.INTERNAL_TRANSFER).all()
            assert set(rows["account"]) == {"chequing", "savings"}
            assert "Mobile Transfer" in rows["Sub-description"].iloc[0]

    def test_unreferenced_pair_synthetic_fixture(self):
        """Same property, isolated in a minimal ad-hoc fixture."""
        df = make_transactions(
            [
                {"Date": "2026-01-09", "Description": "customer transfer cr.", "Sub-description": "Mobile Transfer", "Amount": 10000.0, "account": "chequing"},
                {"Date": "2026-01-09", "Description": "customer transfer dr.", "Sub-description": "Mobile Transfer", "Amount": -10000.0, "account": "savings"},
            ]
        )
        c = classify_transactions(df)
        assert (c["bucket"] == TransactionBucket.INTERNAL_TRANSFER).all()


class TestCaseFundingDoesNotNetAgainstPrefunding:
    def test_disbursement_and_prefunding_both_counted_separately(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        case_funding_total = outflow_total(c, TransactionBucket.CASE_FUNDING)
        assert case_funding_total == 80000.00

        prefunding_inflow = c[
            (c["Date"].isin([pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-12")]))
            & (c["bucket"] == TransactionBucket.INTERNAL_TRANSFER)
            & (c["account"] == "chequing")
        ]
        assert prefunding_inflow["Amount"].sum() == 82499.00
        # the two must not have been netted into one row or offset against each other
        assert len(c[c["bucket"] == TransactionBucket.CASE_FUNDING]) == 1
        assert len(prefunding_inflow) == 2


class TestEmptyAndDegenerateInputs:
    def test_empty_dataframe_does_not_crash(self, empty_transactions):
        c = classify_transactions(empty_transactions)
        assert c.empty
        assert "bucket" in c.columns

    def test_outflow_total_on_empty_is_zero(self, empty_transactions):
        c = classify_transactions(empty_transactions)
        assert outflow_total(c, TransactionBucket.OTHER_OPERATING) == 0.0

    def test_single_transaction(self):
        df = make_transactions([{"Date": "2026-01-01", "Description": "pos purchase", "Amount": -20.0}])
        c = classify_transactions(df)
        assert len(c) == 1
        assert c["bucket"].iloc[0] == TransactionBucket.OTHER_OPERATING
