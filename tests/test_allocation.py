from __future__ import annotations

from fortune.allocation import check_compliance, recommend_allocation
from fortune.config import AllocationConfig


class TestRecommendAllocation:
    def test_excess_split_per_target_weights(self):
        config = AllocationConfig(
            fixed_income_target_weight=0.6,
            equity_target_weight=0.4,
            fixed_income_expected_return=0.04,
            equity_expected_return=0.05,
        )
        rec = recommend_allocation(total_liquid=500_000.0, reserve_floor=300_000.0, config=config)

        assert rec.cash_floor_amount == 300_000.0
        assert rec.shortfall == 0.0
        assert rec.excess_amount == 200_000.0
        assert rec.fixed_income_amount == 120_000.0
        assert rec.equity_amount == 80_000.0
        assert rec.expected_annual_return == 120_000.0 * 0.04 + 80_000.0 * 0.05
        assert abs(rec.expected_annual_return_pct - (rec.expected_annual_return / 200_000.0)) < 1e-9

    def test_shortfall_against_floor_has_nothing_to_allocate(self):
        config = AllocationConfig()
        rec = recommend_allocation(total_liquid=100_000.0, reserve_floor=315_143.0, config=config)

        assert rec.cash_floor_amount == 100_000.0
        assert rec.shortfall == 215_143.0
        assert rec.excess_amount == 0.0
        assert rec.fixed_income_amount == 0.0
        assert rec.equity_amount == 0.0
        assert rec.expected_annual_return == 0.0
        assert rec.expected_annual_return_pct == 0.0

    def test_exact_floor_match_is_neither_shortfall_nor_excess(self):
        config = AllocationConfig()
        rec = recommend_allocation(total_liquid=315_143.0, reserve_floor=315_143.0, config=config)

        assert rec.shortfall == 0.0
        assert rec.excess_amount == 0.0
        assert rec.cash_floor_amount == 315_143.0

    def test_zero_liquid_degrades_gracefully(self):
        config = AllocationConfig()
        rec = recommend_allocation(total_liquid=0.0, reserve_floor=315_143.0, config=config)

        assert rec.cash_floor_amount == 0.0
        assert rec.shortfall == 315_143.0
        assert rec.excess_amount == 0.0


class TestComplianceCheck:
    def test_nothing_invested_returns_no_flags(self):
        config = AllocationConfig()
        assert check_compliance(current_fixed_income=0.0, current_equity=0.0, config=config) == []

    def test_within_band_is_in_range(self):
        config = AllocationConfig(fixed_income_target_weight=0.6, equity_target_weight=0.4, target_band=0.05)
        flags = check_compliance(current_fixed_income=58_000.0, current_equity=42_000.0, config=config)

        by_bucket = {f.bucket: f for f in flags}
        assert by_bucket["Fixed income"].in_range
        assert by_bucket["Equity"].in_range

    def test_outside_band_is_flagged(self):
        config = AllocationConfig(fixed_income_target_weight=0.6, equity_target_weight=0.4, target_band=0.05)
        # 80/20 actual vs. 60/40 target with a 5-point band -> well outside range
        flags = check_compliance(current_fixed_income=80_000.0, current_equity=20_000.0, config=config)

        by_bucket = {f.bucket: f for f in flags}
        assert not by_bucket["Fixed income"].in_range
        assert not by_bucket["Equity"].in_range
