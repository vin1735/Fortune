from __future__ import annotations

from fortune.baseline import compute_annual_baseline, compute_zero_activity_months
from fortune.classify import classify_transactions
from fortune.config import BaselineConfig
from tests.conftest import make_transactions


class TestZeroActivityMonths:
    def test_sample_data_finds_confirmed_zero_months(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        missing, total_months = compute_zero_activity_months(c)
        # Chequing genuinely has no operating-activity rows in these months
        # (see scripts/generate_sample_data.py's ZERO_ACTIVITY_MONTHS).
        assert "2023-11" in missing
        assert "2024-11" in missing
        assert total_months > 0

    def test_month_with_only_internal_transfer_counts_as_zero_activity(self):
        df = make_transactions(
            [
                {"Date": "2026-01-01", "Description": "pos purchase", "Amount": -20.0, "account": "chequing"},
                {"Date": "2026-02-15", "Description": "customer transfer dr.", "Sub-description": "Pc To 000111222333", "Amount": -1000.0, "account": "chequing"},
                {"Date": "2026-02-15", "Description": "customer transfer cr.", "Sub-description": "Pc From 000111444555", "Amount": 1000.0, "account": "savings"},
                {"Date": "2026-03-01", "Description": "pos purchase", "Amount": -20.0, "account": "chequing"},
            ]
        )
        c = classify_transactions(df)
        missing, total_months = compute_zero_activity_months(c)
        assert "2026-02" in missing
        assert total_months == 3


class TestComputeAnnualBaseline:
    def test_no_nan_or_negative_on_empty_input(self, empty_transactions):
        c = classify_transactions(empty_transactions)
        result = compute_annual_baseline(c, BaselineConfig())
        assert result.annual_committed_spend >= 0
        assert result.annual_committed_spend == result.annual_committed_spend  # not NaN

    def test_no_crash_on_single_transaction(self):
        df = make_transactions([{"Date": "2026-01-01", "Description": "pos purchase", "Amount": -20.0}])
        c = classify_transactions(df)
        result = compute_annual_baseline(c, BaselineConfig())
        assert result.annual_committed_spend >= 0

    def test_six_consecutive_zero_months_reduces_but_does_not_break_baseline(self):
        # Active Jan-Mar and Oct-Dec 2025; Apr-Sep (6 consecutive months) have
        # zero chequing activity within the Jan-Dec observation window.
        rows = [{"Date": f"2025-{m:02d}-05", "Description": "pos purchase", "Amount": -20.0} for m in (1, 2, 3, 10, 11, 12)]
        c = classify_transactions(make_transactions(rows))
        result = compute_annual_baseline(c, BaselineConfig())
        assert result.annual_committed_spend >= 0
        assert len(result.zero_activity_months) == 6
        assert result.zero_activity_fraction == 0.5

    def test_zero_activity_fraction_reduces_committed_spend(self):
        config = BaselineConfig(rent_monthly_amount=1000.0, subscription_monthly_amount=0.0)
        full_activity = make_transactions(
            [{"Date": f"2025-{m:02d}-05", "Description": "pos purchase", "Amount": -1.0} for m in range(1, 13)]
        )
        partial_activity = make_transactions(
            [{"Date": f"2025-{m:02d}-05", "Description": "pos purchase", "Amount": -1.0} for m in range(1, 7)]
            + [{"Date": f"2025-{m:02d}-05", "Description": "pos purchase", "Amount": -1.0} for m in range(10, 13)]
        )
        full_result = compute_annual_baseline(classify_transactions(full_activity), config)
        partial_result = compute_annual_baseline(classify_transactions(partial_activity), config)
        assert partial_result.zero_activity_fraction > 0
        assert partial_result.annual_committed_spend < full_result.annual_committed_spend
