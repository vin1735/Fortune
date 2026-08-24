"""Human-readable stdout report.

Renders a SimulationResult into a plain-text board report: assumption
manifest, annual disbursement distribution, the IPS-mandated minimum kept
visually and textually separate from the model-recommended statistical
buffer, and a coverage check against current liquid holdings.
"""

from __future__ import annotations

from fortune.simulate import SimulationResult

_RULE = "=" * 78
_SUBRULE = "-" * 78


def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def render_report(result: SimulationResult) -> str:
    lines: list[str] = []
    w = lines.append

    w(_RULE)
    w("FORTUNE — CASH FLOW & DISBURSEMENT RISK SIMULATION")
    w(_RULE)
    w(
        "This is a decision-support tool for a fiduciary body that retains full "
        "responsibility for reserve decisions. It is not investment advice and "
        "does not forecast markets."
    )
    w(f"Seed: {result.seed}   Iterations: {result.n_iterations:,}")
    w("")

    w(_SUBRULE)
    w("ASSUMPTION MANIFEST")
    w(_SUBRULE)
    for a in result.assumptions:
        w(f"  {a.name:<28} {a.value:<40} [{a.provenance}]")
    w("")

    w(_SUBRULE)
    w("MECHANISM 1 — DETERMINISTIC OPERATING BASELINE")
    w(_SUBRULE)
    w(f"  Rent (monthly, fixed):            {_fmt_money(result.baseline.rent_monthly)}")
    w(f"  Subscriptions (monthly, fixed):   {_fmt_money(result.baseline.subscription_monthly)}")
    w(
        f"  Zero-activity months observed:   {len(result.baseline.zero_activity_months)} "
        f"({result.baseline.zero_activity_fraction:.1%} of the observed window)"
    )
    w(f"  Other operating (annualized avg): {_fmt_money(result.baseline.other_operating_annual_average)}")
    w(f"  => Annual committed spend:        {_fmt_money(result.baseline.annual_committed_spend)}")
    w("")

    w(_SUBRULE)
    w("ANNUAL DISBURSEMENT DISTRIBUTION (baseline + case funding + idiosyncratic tail)")
    w(_SUBRULE)
    s = result.summary
    w(f"  Mean:    {_fmt_money(s.mean)}")
    w(f"  Median:  {_fmt_money(s.median)}")
    w(f"  P50:     {_fmt_money(s.p50)}")
    w(f"  P75:     {_fmt_money(s.p75)}")
    w(f"  P90:     {_fmt_money(s.p90)}")
    w(f"  P95:     {_fmt_money(s.p95)}")
    w(f"  P99:     {_fmt_money(s.p99)}")
    w(f"  Max:     {_fmt_money(s.max)}")
    w("")
    w(
        f"  Mechanism 2 (case funding) history: {result.n_case_funding_history} confirmed event(s). "
        f"Frequency/severity are board-assumed, not fit to this count (see manifest)."
    )
    ci = result.tail_ci_p95
    w(
        f"  Mechanism 3 (idiosyncratic tail) history: {result.n_idiosyncratic_history} confirmed "
        f"observation(s). Bootstrap 95th-percentile estimate: {_fmt_money(ci.point_estimate)} "
        f"[{ci.ci_level:.0%} CI: {_fmt_money(ci.ci_low)} - {_fmt_money(ci.ci_high)}]"
        + (" — LOW SAMPLE SIZE, wide uncertainty" if ci.low_sample_warning else "")
    )
    w("")

    w(_SUBRULE)
    w("OPERATING RESERVE REQUIREMENT")
    w(_SUBRULE)
    w("  These are two different numbers with two different justifications. Do not merge them.")
    w("")
    w(
        f"  IPS-MANDATED MINIMUM (policy-driven, deterministic):    "
        f"{_fmt_money(result.ips_minimum_reserve)}"
    )
    w("    12 months of projected deterministic operating disbursements (Mechanism 1 only).")
    w("")
    w("  MODEL-RECOMMENDED BUFFER (statistical, risk-driven):")
    for level, value in sorted(result.model_recommended_buffer.items()):
        w(f"    at {level:.0%} confidence:  {_fmt_money(value)}")
    w("    Full simulated annual disbursement distribution, including case-funding and tail risk.")
    w("")
    effective = max(result.ips_minimum_reserve, max(result.model_recommended_buffer.values()))
    w(f"  Effective planning minimum (greater of the two above): {_fmt_money(effective)}")
    w("")

    w(_SUBRULE)
    w("COVERAGE CHECK vs. CURRENT LIQUID HOLDINGS")
    w(_SUBRULE)
    lh = result.liquid_holdings
    w(f"  Chequing:                 {_fmt_money(lh.chequing)}")
    w(f"  Savings:                  {_fmt_money(lh.savings)}")
    w(f"  Investment cash sleeve:   {_fmt_money(lh.investment_cash_sleeve)}")
    w(f"  Total true liquid:        {_fmt_money(lh.total)}")
    w("")
    for level, cov in sorted(result.coverage.items()):
        status = "COVERED" if cov.covered else "SHORTFALL"
        w(
            f"  p{int(level*100):<3} requires {_fmt_money(cov.required_reserve):>14}  ->  {status:<9} "
            f"by {_fmt_money(abs(cov.margin))}"
        )
    w("")

    if result.warnings:
        w(_SUBRULE)
        w("WARNINGS")
        w(_SUBRULE)
        for wmsg in result.warnings:
            w(f"  - {wmsg}")
        w("")

    w(_RULE)
    return "\n".join(lines)


def print_report(result: SimulationResult) -> None:
    print(render_report(result))
