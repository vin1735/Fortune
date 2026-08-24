from __future__ import annotations

import numpy as np

from fortune.config import TailConfig
from fortune.tail import (
    bootstrap_percentile_ci,
    historical_idiosyncratic_amounts,
    simulate_annual_idiosyncratic_shocks,
)
from fortune.classify import classify_transactions


class TestHistoricalAmounts:
    def test_sample_data_has_exactly_three_confirmed_amounts(self, sample_transactions):
        c = classify_transactions(sample_transactions)
        amounts = historical_idiosyncratic_amounts(c)
        assert sorted(amounts) == [2200.0, 9000.0, 18000.0]

    def test_empty_input_returns_empty_array(self, empty_transactions):
        c = classify_transactions(empty_transactions)
        assert historical_idiosyncratic_amounts(c).size == 0


class TestBootstrapResampling:
    def test_resampled_values_only_come_from_base_pool(self):
        base = np.array([2200.0, 9000.0, 18000.0])
        config = TailConfig(shocks_per_year=1)
        rng = np.random.default_rng(0)
        draws = simulate_annual_idiosyncratic_shocks(rng, 10_000, base, config)
        assert set(np.unique(draws)) <= set(base)

    def test_empty_base_amounts_returns_zeros_not_crash(self):
        config = TailConfig()
        rng = np.random.default_rng(0)
        draws = simulate_annual_idiosyncratic_shocks(rng, 100, np.zeros(0), config)
        assert np.all(draws == 0.0)


class TestBootstrapConfidenceInterval:
    def test_ci_widens_relative_to_point_estimate_with_n3(self):
        base = np.array([2200.0, 9000.0, 18000.0])
        config = TailConfig(ci_n_bootstrap=500, ci_sample_size=2000)
        rng = np.random.default_rng(0)
        result = bootstrap_percentile_ci(rng, base, 0.95, config)
        assert result.low_sample_warning is True
        assert result.n_source_observations == 3
        assert result.ci_low <= result.point_estimate <= result.ci_high

    def test_empty_base_amounts_flags_low_sample_and_does_not_crash(self):
        config = TailConfig()
        rng = np.random.default_rng(0)
        result = bootstrap_percentile_ci(rng, np.zeros(0), 0.95, config)
        assert result.low_sample_warning is True
        assert result.point_estimate == 0.0

    def test_larger_source_sample_narrows_low_sample_warning(self):
        """Sanity check on the MIN_RELIABLE_SAMPLE_SIZE threshold itself:
        a synthetic 20-observation pool should not trip the low-sample flag.
        """
        rng = np.random.default_rng(0)
        base = rng.uniform(1000, 5000, size=20)
        config = TailConfig(ci_n_bootstrap=200, ci_sample_size=1000)
        result = bootstrap_percentile_ci(rng, base, 0.95, config)
        assert result.low_sample_warning is False
