"""Mechanism 3: idiosyncratic tail bucket.

Bootstrap-resamples the n=3 confirmed one-off operating costs to estimate
unpredictable operating shocks. n=3 is a severe limitation: any percentile
derived from three observations carries enormous uncertainty, so this
module reports a bootstrap confidence interval around percentile estimates
rather than a bare point estimate, and every result carries an explicit,
programmatically-checkable low-sample-size warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.random import Generator

from fortune.classify import TransactionBucket
from fortune.config import TailConfig

MIN_RELIABLE_SAMPLE_SIZE = 10


@dataclass(frozen=True)
class BootstrapCIResult:
    """A percentile point estimate plus a bootstrap confidence interval that
    quantifies how much that estimate would move if the underlying n=3 (or
    however many) source observations had come out slightly differently.
    """

    percentile: float
    point_estimate: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_source_observations: int
    n_bootstrap: int
    low_sample_warning: bool


def historical_idiosyncratic_amounts(classified_df: pd.DataFrame) -> np.ndarray:
    """Return |Amount| for the confirmed IDIOSYNCRATIC_ONEOFF rows."""
    if classified_df.empty or "bucket" not in classified_df.columns:
        return np.zeros(0)
    rows = classified_df[classified_df["bucket"] == TransactionBucket.IDIOSYNCRATIC_ONEOFF]
    return rows["Amount"].abs().to_numpy()


def simulate_annual_idiosyncratic_shocks(
    rng: Generator, n_years: int, base_amounts: np.ndarray, config: TailConfig
) -> np.ndarray:
    """Bootstrap-resample ``config.shocks_per_year`` shocks per simulated year
    from ``base_amounts`` (with replacement) and sum them.

    Returns an all-zero array if ``base_amounts`` is empty (no idiosyncratic
    history to draw from) rather than raising, so degenerate inputs degrade
    gracefully instead of crashing.
    """
    if n_years == 0 or base_amounts.size == 0 or config.shocks_per_year == 0:
        return np.zeros(n_years)
    draws = rng.choice(base_amounts, size=(n_years, config.shocks_per_year), replace=True)
    return draws.sum(axis=1)


def bootstrap_percentile_ci(
    rng: Generator,
    base_amounts: np.ndarray,
    percentile: float,
    config: TailConfig,
) -> BootstrapCIResult:
    """Bootstrap-the-bootstrap confidence interval for a percentile of the
    simulated annual idiosyncratic-shock distribution.

    Repeatedly resamples ``base_amounts`` itself (with replacement, same
    size as the original) to form alternate "what if history had been
    slightly different" populations, regenerates a simulated annual-shock
    sample from each, and takes the requested percentile of each. The
    spread of those percentile estimates is the confidence interval: it
    directly answers "how much would our estimate move if we'd observed a
    different set of 3 one-off costs."
    """
    n_source = int(base_amounts.size)
    if n_source == 0:
        return BootstrapCIResult(
            percentile=percentile,
            point_estimate=0.0,
            ci_low=0.0,
            ci_high=0.0,
            ci_level=config.ci_level,
            n_source_observations=0,
            n_bootstrap=config.ci_n_bootstrap,
            low_sample_warning=True,
        )

    point_sample = simulate_annual_idiosyncratic_shocks(rng, config.ci_sample_size, base_amounts, config)
    point_estimate = float(np.percentile(point_sample, percentile * 100))

    estimates = np.empty(config.ci_n_bootstrap)
    for b in range(config.ci_n_bootstrap):
        alt_population = rng.choice(base_amounts, size=n_source, replace=True)
        sim = simulate_annual_idiosyncratic_shocks(rng, config.ci_sample_size, alt_population, config)
        estimates[b] = np.percentile(sim, percentile * 100)

    tail = (1.0 - config.ci_level) / 2.0
    ci_low = float(np.percentile(estimates, tail * 100))
    ci_high = float(np.percentile(estimates, (1.0 - tail) * 100))

    return BootstrapCIResult(
        percentile=percentile,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=config.ci_level,
        n_source_observations=n_source,
        n_bootstrap=config.ci_n_bootstrap,
        low_sample_warning=n_source < MIN_RELIABLE_SAMPLE_SIZE,
    )
