from __future__ import annotations

from dataclasses import replace

import numpy as np

from fortune.config import (
    CaseFundingConfig,
    FortuneConfig,
    SimulationConfig,
    UniformContinuousSeverity,
    UniformDiscreteFrequency,
)
from fortune.simulate import run_simulation
from tests.conftest import make_transactions


def _small_config(**overrides) -> FortuneConfig:
    config = FortuneConfig(simulation=SimulationConfig(n_iterations=5_000, seed=42))
    return replace(config, **overrides)


class TestReproducibility:
    def test_same_seed_identical_results_bit_for_bit(self, sample_transactions):
        config = _small_config()
        r1 = run_simulation(config, sample_transactions)
        r2 = run_simulation(config, sample_transactions)
        assert np.array_equal(r1.total_annual_outflow, r2.total_annual_outflow)
        assert np.array_equal(r1.case_funding_component, r2.case_funding_component)
        assert np.array_equal(r1.idiosyncratic_component, r2.idiosyncratic_component)
        assert r1.summary == r2.summary

    def test_different_seed_gives_different_results(self, sample_transactions):
        r1 = run_simulation(_small_config(simulation=SimulationConfig(n_iterations=5_000, seed=1)), sample_transactions)
        r2 = run_simulation(_small_config(simulation=SimulationConfig(n_iterations=5_000, seed=2)), sample_transactions)
        assert not np.array_equal(r1.total_annual_outflow, r2.total_annual_outflow)


class TestMonotonicity:
    def test_increasing_case_frequency_does_not_decrease_reserve(self, sample_transactions):
        low_freq = _small_config(
            simulation=SimulationConfig(n_iterations=50_000, seed=42),
            case_funding=CaseFundingConfig(
                frequency_sampler=UniformDiscreteFrequency(low=1, high=2),
                severity_sampler=UniformContinuousSeverity(low=50_000, high=100_000),
            ),
        )
        high_freq = _small_config(
            simulation=SimulationConfig(n_iterations=50_000, seed=42),
            case_funding=CaseFundingConfig(
                frequency_sampler=UniformDiscreteFrequency(low=3, high=5),
                severity_sampler=UniformContinuousSeverity(low=50_000, high=100_000),
            ),
        )
        r_low = run_simulation(low_freq, sample_transactions)
        r_high = run_simulation(high_freq, sample_transactions)
        assert r_high.summary.mean > r_low.summary.mean
        assert r_high.summary.p95 >= r_low.summary.p95
        assert r_high.model_recommended_buffer[0.99] >= r_low.model_recommended_buffer[0.99]

    def test_increasing_confidence_level_does_not_decrease_required_reserve(self, sample_transactions):
        result = run_simulation(_small_config(), sample_transactions)
        levels = sorted(result.model_recommended_buffer)
        values = [result.model_recommended_buffer[level] for level in levels]
        assert values == sorted(values)

    def test_increasing_severity_cap_does_not_decrease_reserve(self, sample_transactions):
        low_sev = _small_config(
            simulation=SimulationConfig(n_iterations=50_000, seed=42),
            case_funding=CaseFundingConfig(severity_sampler=UniformContinuousSeverity(low=50_000, high=100_000)),
        )
        high_sev = _small_config(
            simulation=SimulationConfig(n_iterations=50_000, seed=42),
            case_funding=CaseFundingConfig(severity_sampler=UniformContinuousSeverity(low=50_000, high=200_000)),
        )
        r_low = run_simulation(low_sev, sample_transactions)
        r_high = run_simulation(high_sev, sample_transactions)
        assert r_high.summary.p99 >= r_low.summary.p99


class TestDegenerateInputs:
    def test_empty_dataframe_does_not_crash_and_yields_sane_reserve(self, empty_transactions):
        result = run_simulation(_small_config(), empty_transactions)
        assert result.ips_minimum_reserve >= 0
        assert all(v >= 0 for v in result.model_recommended_buffer.values())
        assert np.isfinite(result.total_annual_outflow).all()

    def test_single_transaction_does_not_crash(self):
        df = make_transactions([{"Date": "2026-01-01", "Description": "pos purchase", "Amount": -20.0}])
        result = run_simulation(_small_config(), df)
        assert result.ips_minimum_reserve >= 0
        assert np.all(result.total_annual_outflow >= 0)

    def test_all_zero_amount_months(self):
        rows = [{"Date": f"2025-{m:02d}-05", "Description": "service charge", "Amount": 0.0} for m in range(1, 13)]
        result = run_simulation(_small_config(), make_transactions(rows))
        assert result.ips_minimum_reserve >= 0
        assert np.isfinite(result.total_annual_outflow).all()

    def test_month_with_only_internal_transfers_does_not_crash(self):
        rows = [
            {"Date": "2026-01-01", "Description": "pos purchase", "Amount": -20.0, "account": "chequing"},
            {"Date": "2026-02-15", "Description": "customer transfer dr.", "Sub-description": "Pc To 000111222333", "Amount": -1000.0, "account": "chequing"},
            {"Date": "2026-02-15", "Description": "customer transfer cr.", "Sub-description": "Pc From 000111444555", "Amount": 1000.0, "account": "savings"},
        ]
        result = run_simulation(_small_config(), make_transactions(rows))
        assert result.ips_minimum_reserve >= 0
        assert np.isfinite(result.total_annual_outflow).all()


class TestCoverageArithmetic:
    def test_covered_when_liquid_exceeds_required(self, sample_transactions):
        from fortune.config import LiquidHoldings

        config = _small_config(liquid_holdings=LiquidHoldings(chequing=10**9, savings=0, investment_cash_sleeve=0))
        result = run_simulation(config, sample_transactions)
        assert all(cov.covered for cov in result.coverage.values())
        assert all(cov.margin > 0 for cov in result.coverage.values())

    def test_shortfall_when_liquid_is_zero(self, sample_transactions):
        from fortune.config import LiquidHoldings

        config = _small_config(liquid_holdings=LiquidHoldings(chequing=0, savings=0, investment_cash_sleeve=0))
        result = run_simulation(config, sample_transactions)
        assert all(not cov.covered for cov in result.coverage.values())
        assert all(cov.margin < 0 for cov in result.coverage.values())


class TestWarnings:
    def test_low_history_warnings_present_for_real_data(self, sample_transactions):
        result = run_simulation(_small_config(), sample_transactions)
        assert any("idiosyncratic" in w.lower() for w in result.warnings)
        assert any("case-funding" in w.lower() or "case funding" in w.lower() for w in result.warnings)

    def test_returned_item_surfaced_as_warning(self, sample_transactions):
        result = run_simulation(_small_config(), sample_transactions)
        assert any("2023-05-20" in event for event in result.returned_item_events)
        assert any("2023-05-20" in w for w in result.warnings)
