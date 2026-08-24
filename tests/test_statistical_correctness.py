"""The single most important test file: verifies the percentile/summary
machinery against analytically known ground truth, independent of any
domain-specific mechanism. If this fails, every percentile reported
anywhere in the package (CLI, figures, coverage check) is suspect.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from fortune.simulate import summarize_distribution


def test_percentiles_converge_to_known_normal_quantiles():
    rng = np.random.default_rng(7)
    mu, sigma = 100_000.0, 15_000.0
    sample = rng.normal(mu, sigma, size=500_000)

    summary = summarize_distribution(sample)
    dist = stats.norm(loc=mu, scale=sigma)

    checks = {
        "median": (0.50, summary.median),
        "p50": (0.50, summary.p50),
        "p75": (0.75, summary.p75),
        "p90": (0.90, summary.p90),
        "p95": (0.95, summary.p95),
        "p99": (0.99, summary.p99),
    }
    for name, (q, observed) in checks.items():
        expected = dist.ppf(q)
        assert abs(observed - expected) < 0.01 * sigma, f"{name}: expected~{expected}, got {observed}"

    assert abs(summary.mean - mu) < 0.01 * sigma
    assert summary.max >= summary.p99


def test_percentiles_converge_to_known_uniform_quantiles():
    rng = np.random.default_rng(11)
    low, high = 200_000.0, 400_000.0
    sample = rng.uniform(low, high, size=500_000)

    summary = summarize_distribution(sample)
    dist = stats.uniform(loc=low, scale=high - low)

    for q, observed in [(0.50, summary.p50), (0.90, summary.p90), (0.99, summary.p99)]:
        expected = dist.ppf(q)
        assert abs(observed - expected) < 0.01 * (high - low)


def test_empty_array_returns_zeros_not_nan_or_crash():
    summary = summarize_distribution(np.zeros(0))
    assert summary.mean == 0.0
    assert summary.p99 == 0.0
    for field_value in (summary.mean, summary.median, summary.p50, summary.p75, summary.p90, summary.p95, summary.p99, summary.max):
        assert field_value == field_value  # not NaN
