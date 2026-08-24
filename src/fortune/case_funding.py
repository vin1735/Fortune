"""Mechanism 2: case-funding frequency-severity model (the load-bearing piece).

An actuarial-style compound distribution, not a bootstrap over historical
months: only one historical case-funding event exists (n=1), which is
insufficient to estimate a rate empirically. Frequency and severity are
forward-looking board planning assumptions (see config.CaseFundingConfig),
sampled independently per simulated year and then summed.

CRITICAL: frequency and severity are never averaged into a single expected
value before simulating. ``simulate_annual_case_funding`` draws a number of
cases per year, then that many independent severities, then sums — per
year, per iteration. Collapsing "1-3 cases at $50K-$100K" into one expected
value would destroy exactly the tail information this mechanism exists to
capture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.random import Generator

from fortune.classify import TransactionBucket
from fortune.config import CaseFundingConfig


def simulate_annual_case_funding(rng: Generator, n_years: int, config: CaseFundingConfig) -> np.ndarray:
    """Draw ``n_years`` independent annual case-funding totals.

    For each simulated year: draw a case count from
    ``config.frequency_sampler``, draw that many independent severities from
    ``config.severity_sampler``, and sum them. Vectorized: draws a
    (n_years, max_cases_drawn) matrix of severities and masks out entries
    beyond each year's drawn case count, rather than looping in Python.
    """
    if n_years == 0:
        return np.zeros(0)

    n_cases = config.frequency_sampler.sample(rng, n_years)
    max_cases = int(n_cases.max()) if n_cases.size else 0
    if max_cases == 0:
        return np.zeros(n_years)

    severities = config.severity_sampler.sample(rng, n_years * max_cases).reshape(n_years, max_cases)
    mask = np.arange(max_cases)[None, :] < n_cases[:, None]
    return (severities * mask).sum(axis=1)


def historical_case_funding_events(classified_df: pd.DataFrame) -> pd.DataFrame:
    """Return the confirmed historical CASE_FUNDING rows (n=1), for reporting
    and validation only. Never used to fit the simulated distribution — see
    module docstring.
    """
    if classified_df.empty or "bucket" not in classified_df.columns:
        return classified_df
    return classified_df[classified_df["bucket"] == TransactionBucket.CASE_FUNDING]
