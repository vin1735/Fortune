"""Configuration dataclasses for every assumption used by the simulation engine.

Every numeric assumption in this package lives here, not scattered through
mechanism logic. Each field's docstring or the enclosing dataclass docstring
states its provenance:

- "data-derived": computed or read directly off the two bank exports in
  ``data/``. Verifiable by re-running the extraction.
- "board-assumed": a forward-looking planning assumption supplied by the
  org's board/treasurer because the historical record is too thin to
  estimate it empirically (e.g. case-funding frequency, n=1).
- "policy": mandated by the org's ratified Investment Policy Statement (IPS),
  independent of any statistical estimate.

IMPORTANT — confidentiality: the defaults below describe the committed
synthetic sample dataset (see scripts/generate_sample_data.py), not any real
organization's real figures. Real transaction history and real assumption
values must never be hardcoded here, since this file is public. To run this
engine against real, non-committed data, supply your own values via a local
overrides file (see ``load_overrides`` below and the README) rather than
editing these defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.random import Generator


# ---------------------------------------------------------------------------
# Classification rules (Mechanism-agnostic; consumed by classify.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfirmedEvent:
    """A single historical transaction confirmed by the treasurer, used as an
    exact-match classification anchor. Provenance: data-derived (treasurer-
    confirmed against bank records).
    """

    txn_date: date
    amount: float
    description_contains: str
    label: str


@dataclass(frozen=True)
class ClassificationRules:
    """Parameters driving classify.py's rule-based bucketing.

    Defaults below match the committed synthetic sample dataset. All values
    are data-derived: either read directly from the CSV exports or confirmed
    by the org's treasurer against specific transactions — for real data,
    supply your own via a local overrides file (see ``load_overrides``).
    """

    # Rent: a fixed monthly amount, sometimes paid as a monthly debit memo,
    # sometimes bundled into a quarterly cheque (3x the monthly amount), and
    # occasionally as a short burst of several payments within a few days.
    # Both exact amounts are treated as rent regardless of cadence.
    rent_monthly_amount: float = -1200.00
    rent_quarterly_amount: float = -3600.00

    # Recurring subscription vendor keyword.
    subscription_vendor_keyword: str = "NimbusForms"

    # Internal transfers: description contains this keyword AND the
    # sub-description references one of the org's own account numbers.
    internal_transfer_keyword: str = "customer transfer"
    counterpart_account_ids: tuple[str, ...] = ("555000033344", "555000011122")

    # Returned/NSF items: description contains this keyword. The matching
    # reversed cheque (same date, opposite sign, equal magnitude) is paired
    # automatically by classify.py. Excluded from all outflow mechanisms but
    # surfaced as a liquidity-stress signal in the report.
    returned_item_keyword: str = "returned cheque"

    # The org's confirmed historical case-funding disbursement(s).
    confirmed_case_fundings: tuple[ConfirmedEvent, ...] = field(
        default_factory=lambda: (
            ConfirmedEvent(date(2024, 6, 24), -80000.00, "cheque", "cheque 320 — program disbursement"),
        )
    )

    # The confirmed idiosyncratic one-off operating costs (n=3).
    confirmed_idiosyncratic: tuple[ConfirmedEvent, ...] = field(
        default_factory=lambda: (
            ConfirmedEvent(date(2023, 3, 25), -18000.00, "cheque", "cheque 303"),
            ConfirmedEvent(date(2023, 8, 14), -9000.00, "draft purchase", "draft purchase"),
            ConfirmedEvent(date(2024, 3, 2), -2200.00, "cheque", "cheque 318"),
        )
    )


# ---------------------------------------------------------------------------
# Mechanism 1: deterministic baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineConfig:
    """Parameters for the deterministic operating baseline (Mechanism 1).

    Rent and subscription amounts are data-derived fixed contractual costs
    and are never sampled. The zero-activity-month adjustment and the
    other-operating average are computed at runtime from whatever
    transaction data is supplied (also data-derived, not hardcoded), since
    they depend on the observation window.
    """

    rent_monthly_amount: float = 1200.00
    subscription_monthly_amount: float = 40.00


# ---------------------------------------------------------------------------
# Mechanism 2: case-funding frequency-severity model
# ---------------------------------------------------------------------------


class FrequencySampler(Protocol):
    """Strategy interface for sampling the number of case-funding events per
    simulated year. Injectable so the frequency distribution can be swapped
    (e.g. for Poisson or triangular) without touching simulate.py or
    case_funding.py call sites.
    """

    def sample(self, rng: Generator, n_years: int) -> np.ndarray: ...


class SeveritySampler(Protocol):
    """Strategy interface for sampling per-case severities. Injectable for
    the same reason as FrequencySampler.
    """

    def sample(self, rng: Generator, n_draws: int) -> np.ndarray: ...


@dataclass(frozen=True)
class UniformDiscreteFrequency:
    """Default frequency sampler: uniform over the integers [low, high].

    Provenance: board-assumed. Only one historical case-funding event
    exists (n=1) — nowhere near enough to fit a rate empirically — so the
    board adopted 1-3 cases/year as a deliberately conservative forward
    planning range rather than an estimated Poisson rate.
    """

    low: int = 1
    high: int = 3

    def sample(self, rng: Generator, n_years: int) -> np.ndarray:
        return rng.integers(self.low, self.high + 1, size=n_years)


@dataclass(frozen=True)
class UniformContinuousSeverity:
    """Default severity sampler: continuous uniform over [low, high].

    Provenance: board-assumed. $50K-$100K per case is the board's currently
    adopted planning range, deliberately narrower than an earlier, more
    aggressive assumption set (2-5 cases/yr, $50K-$200K) that produced a
    99th-percentile annual disbursement far in excess of true liquid
    holdings — see the README for the specific finding computed against
    real (non-committed) data. $150K and $200K caps are retained as
    sensitivity scenarios (see the sensitivity-grid figure), not as the
    default.
    """

    low: float = 50_000.0
    high: float = 100_000.0

    def sample(self, rng: Generator, n_draws: int) -> np.ndarray:
        if n_draws == 0:
            return np.zeros(0)
        return rng.uniform(self.low, self.high, size=n_draws)


@dataclass(frozen=True)
class ShiftedLogNormalSeverity:
    """Default severity sampler: a hard floor plus a lognormal draw.

    Provenance: board-assumed shape, parameterized against real confirmed
    case-funding amounts found across the org's 2015-2026 archive: ordinary
    cases ran $435-$35K (lawyer-referral subsidies up through multi-year
    litigation tranches), and one real, confirmed disbursement of $150,000
    (paid 2026-03-30) that no fixed-ceiling uniform band could represent.
    Real severity is heavily right-skewed — frequent small-to-mid cases, an
    occasional enormous one — not a flat band, so the sampler shape changed
    along with the numbers: a uniform distribution puts zero probability
    mass below its floor and a hard wall at its ceiling; this one does
    neither.

    ``floor`` is set at the low end of the board's $5K-$15K target (see
    README) rather than down at $435, since that smallest historical case
    was a lawyer-referral consultation subsidy — a different, much smaller
    category of assistance than full case-funding litigation support, and
    belongs in the idiosyncratic/other-operating bucket instead.

    ``mu``/``sigma`` are chosen, not fit: median total severity (floor +
    median lognormal draw) lands around $15K, matching the real cluster of
    ordinary cases ($12.5K-$17K), while P(severity > $150,000) ≈ 3% — rare
    but not vanishingly so, since it has already happened once in the ~2
    years of detailed transaction data this engine has.
    """

    floor: float = 5_000.0
    mu: float = 9.210340371976184  # ln(10_000)
    sigma: float = 1.4

    def sample(self, rng: Generator, n_draws: int) -> np.ndarray:
        if n_draws == 0:
            return np.zeros(0)
        return self.floor + rng.lognormal(mean=self.mu, sigma=self.sigma, size=n_draws)


@dataclass(frozen=True)
class CaseFundingConfig:
    """Mechanism 2 configuration.

    CRITICAL: frequency and severity are sampled independently per
    simulated year and summed — never averaged into a single point value
    before simulating. Averaging first would destroy the tail information
    that is the entire purpose of this mechanism.
    """

    frequency_sampler: FrequencySampler = field(default_factory=UniformDiscreteFrequency)
    severity_sampler: SeveritySampler = field(default_factory=UniformContinuousSeverity)
    frequency_provenance: str = (
        "board-assumed (forward-looking planning assumption; informed by a 2015-2026 "
        "archive review showing ~1 funded case/yr long-run but bursty, with an uptick "
        "to 2 in 2026 alone — still far too thin and irregular to fit a rate empirically)"
    )
    severity_provenance: str = (
        "board-assumed (adopted as a solvency constraint; see docstring). A heavier-"
        "tailed ShiftedLogNormalSeverity alternative was evaluated (see its docstring) "
        "and would raise the p99 requirement from $315,143 to $469,867 — the board "
        "reviewed both and elected to keep this uniform band as the live default."
    )


# ---------------------------------------------------------------------------
# Mechanism 3: idiosyncratic tail bucket
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TailConfig:
    """Mechanism 3 configuration: bootstrap resampling of n=3 confirmed
    one-off operating costs.

    ``shocks_per_year`` is a simplifying frequency assumption. Provenance:
    board-assumed (frequency) layered on top of data-derived severities (the
    resampling pool itself).
    """

    shocks_per_year: int = 1
    shocks_per_year_provenance: str = (
        "board-assumed simplifying default; see README for the empirical rate "
        "observed in the underlying (real, non-committed) data"
    )
    ci_level: float = 0.90
    ci_n_bootstrap: int = 2000
    ci_sample_size: int = 5000


# ---------------------------------------------------------------------------
# Liquid holdings / coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiquidHoldings:
    """Current true liquid reserve, broken out by account.

    Defaults below are fabricated figures matching the synthetic sample
    dataset. Provenance for real use: board-provided (bank statement
    balances plus the cash sleeve reported by any managed investment
    account) — supply your own via a local overrides file, never by editing
    this default.
    """

    chequing: float = 18_750.00
    savings: float = 92_340.00
    investment_cash_sleeve: float = 150_000.00

    @property
    def total(self) -> float:
        return self.chequing + self.savings + self.investment_cash_sleeve


# ---------------------------------------------------------------------------
# IPS policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IPSConfig:
    """Investment Policy Statement requirements. Provenance: policy (ratified
    IPS document), not a statistical estimate.
    """

    minimum_reserve_months: int = 12
    cash_equivalent_fraction_required: float = 1.0


# ---------------------------------------------------------------------------
# Phase 2: allocation of holdings in excess of the reserve floor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllocationConfig:
    """Target weights for the portion of liquid holdings held in EXCESS of
    the Phase 1 reserve floor. The floor itself is never allocated by this
    config — it stays 100% cash per
    ``IPSConfig.cash_equivalent_fraction_required``, regardless of these
    weights.

    Provenance: board-assumed, informed by real current market data and the
    org's own history: a 2016-17 100% GIC / 0% equity era, then a 2021
    50/50 "diversified low-carbon" balanced-fund era. A board review (Oct
    2026) weighed real GIC-ladder rates (multi-issuer, ~3.5%-4.1% across
    1-5yr terms, vs. Scotiabank's own in-house 1yr rate of just 2.45%)
    against the fund actually used historically (Scotia Low Carbon Global
    Balanced Fund: 1yr return 3.92%, but 5yr only 2.15% annualized,
    confirming it carries real volatility, not free return). The board
    chose 60% fixed income / 40% equity — more conservative than 2021's
    50/50, less conservative than the 2016-17 GIC-only era — consistent
    with "low return, low risk is fine, we are not an investment club."

    ``target_band`` is a +/- tolerance around each target weight for the
    compliance check (e.g. 0.05 = flag anything more than 5 points off
    target). Not independently board-ratified — a reasonable default
    pending the real IPS document, which specifies the reserve-floor cash
    requirement but was never found to specify an excess-tranche split.

    Expected returns are real, current, sourced figures (not fabricated),
    not guarantees: fixed-income uses a laddered multi-issuer GIC rate
    rather than relying on Scotiabank's own below-market rate; equity uses
    the existing low-carbon fund's trailing 1-year return.
    """

    fixed_income_target_weight: float = 0.60
    equity_target_weight: float = 0.40
    target_band: float = 0.05
    fixed_income_expected_return: float = 0.038
    equity_expected_return: float = 0.0392
    provenance: str = (
        "board-assumed (Oct 2026): 60/40 fixed-income/equity for holdings above "
        "the reserve floor, chosen over a 70/30 alternative and the 2016-era "
        "100/0 GIC-only split, after reviewing real GIC-ladder rates against the "
        "Scotia Low Carbon Global Balanced Fund's actual return/volatility."
    )


# ---------------------------------------------------------------------------
# Top-level simulation configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo run parameters."""

    n_iterations: int = 50_000
    seed: int = 42
    confidence_levels: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class FortuneConfig:
    """Full assumption set for one simulation run."""

    classification: ClassificationRules = field(default_factory=ClassificationRules)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    case_funding: CaseFundingConfig = field(default_factory=CaseFundingConfig)
    tail: TailConfig = field(default_factory=TailConfig)
    liquid_holdings: LiquidHoldings = field(default_factory=LiquidHoldings)
    ips: IPSConfig = field(default_factory=IPSConfig)
    allocation: AllocationConfig = field(default_factory=AllocationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# ---------------------------------------------------------------------------
# Local overrides: the only sanctioned way to point this engine at a real
# organization's real figures, without ever committing them.
# ---------------------------------------------------------------------------


def load_overrides(config: FortuneConfig, path: str | Path) -> FortuneConfig:
    """Apply a local, non-committed JSON overrides file on top of ``config``.

    This is how a treasurer runs this engine against real data: keep real
    account numbers, confirmed transaction amounts, rent, subscription
    vendor, and liquid holdings in a local JSON file (see
    ``local_config.example.json`` for the shape) that is listed in
    .gitignore and never staged. Every key is optional; only supplied keys
    are overridden. Unknown keys raise ValueError rather than being silently
    ignored, to catch typos.
    """
    with open(path) as f:
        overrides = json.load(f)

    known_keys = {
        "rent_monthly_amount",
        "rent_quarterly_amount",
        "subscription_vendor_keyword",
        "subscription_monthly_amount",
        "counterpart_account_ids",
        "confirmed_case_fundings",
        "confirmed_idiosyncratic",
        "liquid_chequing",
        "liquid_savings",
        "liquid_investment_cash_sleeve",
    }
    unknown = set(overrides) - known_keys
    if unknown:
        raise ValueError(f"Unknown override key(s): {sorted(unknown)}. Known keys: {sorted(known_keys)}")

    classification_updates: dict = {}
    baseline_updates: dict = {}
    if "rent_monthly_amount" in overrides:
        classification_updates["rent_monthly_amount"] = overrides["rent_monthly_amount"]
        baseline_updates["rent_monthly_amount"] = abs(overrides["rent_monthly_amount"])
    if "rent_quarterly_amount" in overrides:
        classification_updates["rent_quarterly_amount"] = overrides["rent_quarterly_amount"]
    if "subscription_vendor_keyword" in overrides:
        classification_updates["subscription_vendor_keyword"] = overrides["subscription_vendor_keyword"]
    if "subscription_monthly_amount" in overrides:
        baseline_updates["subscription_monthly_amount"] = abs(overrides["subscription_monthly_amount"])
    if "counterpart_account_ids" in overrides:
        classification_updates["counterpart_account_ids"] = tuple(overrides["counterpart_account_ids"])
    if "confirmed_case_fundings" in overrides:
        classification_updates["confirmed_case_fundings"] = tuple(
            ConfirmedEvent(date.fromisoformat(e["date"]), e["amount"], e["description_contains"], e["label"])
            for e in overrides["confirmed_case_fundings"]
        )
    if "confirmed_idiosyncratic" in overrides:
        classification_updates["confirmed_idiosyncratic"] = tuple(
            ConfirmedEvent(date.fromisoformat(e["date"]), e["amount"], e["description_contains"], e["label"])
            for e in overrides["confirmed_idiosyncratic"]
        )

    new_classification = replace(config.classification, **classification_updates) if classification_updates else config.classification
    new_baseline = replace(config.baseline, **baseline_updates) if baseline_updates else config.baseline

    liquid_updates = {}
    if "liquid_chequing" in overrides:
        liquid_updates["chequing"] = overrides["liquid_chequing"]
    if "liquid_savings" in overrides:
        liquid_updates["savings"] = overrides["liquid_savings"]
    if "liquid_investment_cash_sleeve" in overrides:
        liquid_updates["investment_cash_sleeve"] = overrides["liquid_investment_cash_sleeve"]
    new_liquid = replace(config.liquid_holdings, **liquid_updates) if liquid_updates else config.liquid_holdings

    return replace(config, classification=new_classification, baseline=new_baseline, liquid_holdings=new_liquid)
