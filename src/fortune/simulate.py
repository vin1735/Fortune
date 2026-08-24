"""Combination, orchestration, and result objects.

Total simulated annual outflow = deterministic baseline (Mechanism 1) +
case-funding draw (Mechanism 2) + idiosyncratic draw (Mechanism 3), per
simulated year. The three mechanisms are independent: they are simulated
separately and summed, never blended into a single resampled historical
series (see the package README for why naive bootstrapping over raw monthly
cash flow is the wrong approach for this org).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fortune.baseline import BaselineResult, compute_annual_baseline
from fortune.case_funding import historical_case_funding_events, simulate_annual_case_funding
from fortune.classify import classify_transactions
from fortune.config import FortuneConfig, LiquidHoldings
from fortune.tail import (
    MIN_RELIABLE_SAMPLE_SIZE,
    BootstrapCIResult,
    bootstrap_percentile_ci,
    historical_idiosyncratic_amounts,
    simulate_annual_idiosyncratic_shocks,
)


@dataclass(frozen=True)
class DistributionSummary:
    mean: float
    median: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    max: float


def summarize_distribution(arr: np.ndarray) -> DistributionSummary:
    if arr.size == 0:
        return DistributionSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return DistributionSummary(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
        max=float(np.max(arr)),
    )


@dataclass(frozen=True)
class CoverageResult:
    confidence_level: float
    required_reserve: float
    liquid_holdings: float
    covered: bool
    margin: float  # liquid - required; negative magnitude is the shortfall


def evaluate_coverage(required_reserve: float, liquid_holdings: float, confidence_level: float) -> CoverageResult:
    margin = liquid_holdings - required_reserve
    return CoverageResult(
        confidence_level=confidence_level,
        required_reserve=required_reserve,
        liquid_holdings=liquid_holdings,
        covered=margin >= 0,
        margin=margin,
    )


@dataclass(frozen=True)
class AssumptionRecord:
    name: str
    value: str
    provenance: str


def describe_returned_items(classified_df: pd.DataFrame) -> list[str]:
    """Human-readable one-liners for RETURNED_ITEM rows, surfaced as a
    liquidity-stress signal rather than silently netted away.
    """
    from fortune.classify import TransactionBucket

    if classified_df.empty or "bucket" not in classified_df.columns:
        return []
    rows = classified_df[classified_df["bucket"] == TransactionBucket.RETURNED_ITEM]
    events = []
    for txn_date, group in rows.groupby(rows["Date"].dt.date):
        gross = group.loc[group["Amount"] < 0, "Amount"].abs().sum()
        if gross > 0:
            events.append(f"{txn_date}: a ${gross:,.2f} item was returned/reversed (NSF) the same day")
    return events


@dataclass(frozen=True)
class SimulationResult:
    """Full output of one simulation run."""

    seed: int
    n_iterations: int
    baseline: BaselineResult
    total_annual_outflow: np.ndarray
    baseline_component: np.ndarray
    case_funding_component: np.ndarray
    idiosyncratic_component: np.ndarray
    summary: DistributionSummary
    ips_minimum_reserve: float
    model_recommended_buffer: dict[float, float]
    liquid_holdings: LiquidHoldings
    coverage: dict[float, CoverageResult]
    tail_ci_p95: BootstrapCIResult
    n_case_funding_history: int
    n_idiosyncratic_history: int
    returned_item_events: list[str] = field(default_factory=list)
    assumptions: list[AssumptionRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _build_assumption_manifest(config: FortuneConfig, baseline: BaselineResult) -> list[AssumptionRecord]:
    fs = config.case_funding.frequency_sampler
    ss = config.case_funding.severity_sampler
    return [
        AssumptionRecord("Rent (monthly)", f"${baseline.rent_monthly:,.2f}", "data-derived"),
        AssumptionRecord("Subscriptions (monthly)", f"${baseline.subscription_monthly:,.2f}", "data-derived"),
        AssumptionRecord(
            "Zero-activity months",
            ", ".join(baseline.zero_activity_months) or "none observed",
            "data-derived",
        ),
        AssumptionRecord(
            "Case-funding frequency", f"{fs!r}", config.case_funding.frequency_provenance
        ),
        AssumptionRecord(
            "Case-funding severity", f"{ss!r}", config.case_funding.severity_provenance
        ),
        AssumptionRecord(
            "Idiosyncratic shocks/year", str(config.tail.shocks_per_year), config.tail.shocks_per_year_provenance
        ),
        AssumptionRecord(
            "IPS minimum reserve", f"{config.ips.minimum_reserve_months} months", "policy (ratified IPS)"
        ),
        AssumptionRecord(
            "Liquid holdings (total)", f"${config.liquid_holdings.total:,.2f}", "board-provided"
        ),
        AssumptionRecord("Monte Carlo iterations", str(config.simulation.n_iterations), "engineering choice"),
        AssumptionRecord("Random seed", str(config.simulation.seed), "engineering choice (reproducibility)"),
    ]


def run_simulation(config: FortuneConfig, transactions: pd.DataFrame) -> SimulationResult:
    """Run the full three-mechanism Monte Carlo simulation.

    ``transactions`` should be the output of loader.load_transactions (or an
    equivalent/synthetic DataFrame with the same schema). Deterministic given
    ``config.simulation.seed``: identical config + data + seed reproduces
    identical results, including array contents, bit for bit.
    """
    rng = np.random.default_rng(config.simulation.seed)
    n = config.simulation.n_iterations

    classified = classify_transactions(transactions, config.classification)

    baseline = compute_annual_baseline(classified, config.baseline)
    baseline_component = np.full(n, baseline.annual_committed_spend)

    case_funding_component = simulate_annual_case_funding(rng, n, config.case_funding)

    idiosyncratic_amounts = historical_idiosyncratic_amounts(classified)
    idiosyncratic_component = simulate_annual_idiosyncratic_shocks(rng, n, idiosyncratic_amounts, config.tail)

    total = baseline_component + case_funding_component + idiosyncratic_component
    summary = summarize_distribution(total)

    ips_minimum_reserve = baseline.annual_committed_spend  # already a 12-month figure

    confidence_levels = config.simulation.confidence_levels
    model_recommended_buffer = {
        level: float(np.percentile(total, level * 100)) for level in confidence_levels
    }
    liquid_total = config.liquid_holdings.total
    coverage = {
        level: evaluate_coverage(model_recommended_buffer[level], liquid_total, level)
        for level in confidence_levels
    }

    tail_ci_p95 = bootstrap_percentile_ci(rng, idiosyncratic_amounts, 0.95, config.tail)

    n_case_funding_history = len(historical_case_funding_events(classified))
    n_idiosyncratic_history = int(idiosyncratic_amounts.size)
    returned_item_events = describe_returned_items(classified)

    warnings: list[str] = []
    if n_idiosyncratic_history < MIN_RELIABLE_SAMPLE_SIZE:
        warnings.append(
            f"Idiosyncratic tail bucket (Mechanism 3) is bootstrap-resampled from only "
            f"{n_idiosyncratic_history} historical observation(s). Any percentile derived from this "
            f"few points carries enormous uncertainty — see the bootstrap confidence interval below."
        )
    if n_case_funding_history <= 1:
        warnings.append(
            f"Case-funding frequency and severity (Mechanism 2) are forward-looking board planning "
            f"assumptions, not fit to history: only {n_case_funding_history} historical case-funding "
            f"event exists, far too few to estimate a rate or severity distribution empirically."
        )
    for event in returned_item_events:
        warnings.append(f"Liquidity signal: {event}. Excluded from all outflow mechanisms.")
    if baseline.zero_activity_months:
        warnings.append(
            f"{len(baseline.zero_activity_months)} historical month(s) had zero confirmed operating "
            f"activity ({', '.join(baseline.zero_activity_months)}); modeled as a deterministic "
            f"reduction to the operating baseline, not a random low draw."
        )

    assumptions = _build_assumption_manifest(config, baseline)

    return SimulationResult(
        seed=config.simulation.seed,
        n_iterations=n,
        baseline=baseline,
        total_annual_outflow=total,
        baseline_component=baseline_component,
        case_funding_component=case_funding_component,
        idiosyncratic_component=idiosyncratic_component,
        summary=summary,
        ips_minimum_reserve=ips_minimum_reserve,
        model_recommended_buffer=model_recommended_buffer,
        liquid_holdings=config.liquid_holdings,
        coverage=coverage,
        tail_ci_p95=tail_ci_p95,
        n_case_funding_history=n_case_funding_history,
        n_idiosyncratic_history=n_idiosyncratic_history,
        returned_item_events=returned_item_events,
        assumptions=assumptions,
        warnings=warnings,
    )
