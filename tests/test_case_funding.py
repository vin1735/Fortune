from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fortune.case_funding import simulate_annual_case_funding
from fortune.config import CaseFundingConfig, UniformContinuousSeverity, UniformDiscreteFrequency


class _ConstantFrequency:
    def __init__(self, n: int):
        self.n = n

    def sample(self, rng, n_years):
        return np.full(n_years, self.n, dtype=int)


class _ConstantSeverity:
    def __init__(self, value: float):
        self.value = value

    def sample(self, rng, n_draws):
        return np.full(n_draws, self.value)


class TestNoPrematureAveraging:
    def test_compound_distribution_matches_analytic_discrete_distribution(self):
        """1-3 cases/year at a fixed severity c must produce totals in
        {c, 2c, 3c} each with probability ~1/3 — not a single averaged value
        of 2c. This is the test that would fail if frequency and severity
        were collapsed into an expected value before simulating.
        """
        c = 10_000.0
        config = CaseFundingConfig(
            frequency_sampler=UniformDiscreteFrequency(low=1, high=3),
            severity_sampler=_ConstantSeverity(c),
        )
        rng = np.random.default_rng(0)
        totals = simulate_annual_case_funding(rng, 60_000, config)

        observed = set(np.round(totals / c).astype(int))
        assert observed == {1, 2, 3}, "collapsing to an expected value would produce a single constant total"

        for k in (1, 2, 3):
            frac = np.mean(np.isclose(totals, k * c))
            assert abs(frac - 1 / 3) < 0.02

    def test_zero_years_returns_empty(self):
        config = CaseFundingConfig()
        rng = np.random.default_rng(0)
        totals = simulate_annual_case_funding(rng, 0, config)
        assert totals.shape == (0,)


class TestStatisticalCorrectnessAgainstKnownGroundTruth:
    def test_severity_percentiles_converge_to_analytic_uniform_quantiles(self):
        """With exactly one case per year, the annual total *is* the
        severity draw, i.e. Uniform(50_000, 100_000). Its quantiles are
        known analytically via scipy.stats.uniform — the simulated
        percentile machinery must converge to them.
        """
        low, high = 50_000.0, 100_000.0
        config = CaseFundingConfig(
            frequency_sampler=_ConstantFrequency(1),
            severity_sampler=UniformContinuousSeverity(low=low, high=high),
        )
        rng = np.random.default_rng(42)
        totals = simulate_annual_case_funding(rng, 200_000, config)

        dist = stats.uniform(loc=low, scale=high - low)
        for q in (0.10, 0.50, 0.90, 0.99):
            expected = dist.ppf(q)
            observed = np.percentile(totals, q * 100)
            assert abs(observed - expected) < 0.02 * (high - low), (q, expected, observed)


class TestStressScenario:
    def test_three_cases_at_200k_ceiling(self):
        config = CaseFundingConfig(
            frequency_sampler=_ConstantFrequency(3),
            severity_sampler=_ConstantSeverity(200_000.0),
        )
        rng = np.random.default_rng(0)
        totals = simulate_annual_case_funding(rng, 100, config)
        assert np.all(totals == 600_000.0)
        assert np.isfinite(totals).all()


@pytest.mark.parametrize("freq_high", [3, 5])
def test_injectable_frequency_sampler_is_used(freq_high):
    config = CaseFundingConfig(
        frequency_sampler=UniformDiscreteFrequency(low=1, high=freq_high),
        severity_sampler=_ConstantSeverity(1.0),
    )
    rng = np.random.default_rng(1)
    totals = simulate_annual_case_funding(rng, 20_000, config)
    assert totals.max() <= freq_high
