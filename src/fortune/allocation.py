"""Phase 2: allocate liquid holdings across cash / fixed income / equity
under the org's investment policy.

The Phase 1 reserve requirement is a hard cash floor: money that must stay
fully liquid for case funding and operating needs. Only holdings held in
EXCESS of that floor are eligible to be allocated toward return-seeking
fixed income / equity, per ``AllocationConfig``'s target weights. If total
liquid holdings don't even reach the floor, there is no excess to allocate
— the recommendation degrades to "hold everything as cash" rather than
raising, since a shortfall against the floor is a Phase 1 finding, not a
Phase 2 one.
"""

from __future__ import annotations

from dataclasses import dataclass

from fortune.config import AllocationConfig


@dataclass(frozen=True)
class AllocationRecommendation:
    """How to split current liquid holdings between the reserve floor
    (always cash) and an excess tranche (fixed income / equity per policy).
    """

    total_liquid: float
    reserve_floor: float
    cash_floor_amount: float
    shortfall: float
    excess_amount: float
    fixed_income_amount: float
    equity_amount: float
    expected_annual_return: float  # in dollars, on the excess tranche only

    @property
    def expected_annual_return_pct(self) -> float:
        if self.excess_amount <= 0:
            return 0.0
        return self.expected_annual_return / self.excess_amount


def recommend_allocation(
    total_liquid: float, reserve_floor: float, config: AllocationConfig
) -> AllocationRecommendation:
    cash_floor_amount = min(total_liquid, reserve_floor)
    shortfall = max(0.0, reserve_floor - total_liquid)
    excess_amount = max(0.0, total_liquid - reserve_floor)

    fixed_income_amount = excess_amount * config.fixed_income_target_weight
    equity_amount = excess_amount * config.equity_target_weight
    expected_annual_return = (
        fixed_income_amount * config.fixed_income_expected_return
        + equity_amount * config.equity_expected_return
    )

    return AllocationRecommendation(
        total_liquid=total_liquid,
        reserve_floor=reserve_floor,
        cash_floor_amount=cash_floor_amount,
        shortfall=shortfall,
        excess_amount=excess_amount,
        fixed_income_amount=fixed_income_amount,
        equity_amount=equity_amount,
        expected_annual_return=expected_annual_return,
    )


@dataclass(frozen=True)
class ComplianceFlag:
    """Whether a currently-held asset bucket's weight (of the invested
    excess tranche, not of total holdings) sits within policy's target
    band.
    """

    bucket: str
    current_weight: float
    target_weight: float
    target_low: float
    target_high: float
    in_range: bool


def _compliance_flag(bucket: str, current_weight: float, target_weight: float, band: float) -> ComplianceFlag:
    low, high = target_weight - band, target_weight + band
    return ComplianceFlag(
        bucket=bucket,
        current_weight=current_weight,
        target_weight=target_weight,
        target_low=low,
        target_high=high,
        in_range=low <= current_weight <= high,
    )


def check_compliance(
    current_fixed_income: float, current_equity: float, config: AllocationConfig
) -> list[ComplianceFlag]:
    """Flag any currently-invested bucket that falls outside its policy
    target band. Returns an empty list if nothing is currently invested
    (nothing to check yet) rather than dividing by zero.
    """
    invested_total = current_fixed_income + current_equity
    if invested_total <= 0:
        return []

    fi_weight = current_fixed_income / invested_total
    eq_weight = current_equity / invested_total
    return [
        _compliance_flag("Fixed income", fi_weight, config.fixed_income_target_weight, config.target_band),
        _compliance_flag("Equity", eq_weight, config.equity_target_weight, config.target_band),
    ]
