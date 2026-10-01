"""Matplotlib figures for the README and board report.

Four required figures, generated with the ``--figures`` CLI flag:
  1. Distribution histogram (headline)
  2. Coverage curve
  3. Mechanism contribution
  4. Sensitivity grid (severity cap x frequency assumption)

A consistent two-color "covered" / "shortfall" palette is used across all
four; no default matplotlib rainbow colormaps and no 3D effects.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from fortune.config import FortuneConfig, UniformContinuousSeverity, UniformDiscreteFrequency
from fortune.simulate import SimulationResult, run_simulation

BG_COLOR = "#0d1117"
FG_COLOR = "#c9d1d9"
MUTED_COLOR = "#8b949e"
GRID_COLOR = "#21262d"
COVERED_COLOR = "#1f6feb"
SHORTFALL_COLOR = "#39d0ff"
NEUTRAL_COLOR = MUTED_COLOR

_COVERAGE_CMAP = LinearSegmentedColormap.from_list("coverage", [SHORTFALL_COLOR, GRID_COLOR, COVERED_COLOR])

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "font.size": 10,
        "font.family": "monospace",
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "savefig.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": FG_COLOR,
        "text.color": FG_COLOR,
        "xtick.color": MUTED_COLOR,
        "ytick.color": MUTED_COLOR,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def _caption(fig, text: str) -> None:
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=8, color=MUTED_COLOR, wrap=True)


def plot_distribution_histogram(result: SimulationResult, path: Path) -> Path:
    """Figure 1: headline histogram of simulated annual disbursements."""
    total = result.total_annual_outflow
    liquid = result.liquid_holdings.total

    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bin_edges = np.histogram(total, bins=60)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    widths = np.diff(bin_edges)
    colors = [COVERED_COLOR if edge < liquid else SHORTFALL_COLOR for edge in bin_edges[:-1]]
    ax.bar(centers, counts, width=widths * 0.95, color=colors, edgecolor=BG_COLOR, linewidth=0.3)

    ax.axvline(liquid, color="white", linestyle="--", linewidth=1.5)
    ax.text(
        liquid,
        ax.get_ylim()[1] * 0.97,
        f"  current liquid: ${liquid:,.0f}",
        rotation=0,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.set_xlabel("Simulated annual disbursement ($)")
    ax.set_ylabel("Simulated years (count)")
    ax.set_title("Simulated Annual Disbursement Distribution")
    ax.xaxis.set_major_formatter(lambda x, _: f"${x/1000:,.0f}k")

    cf = result.assumptions
    freq = next(a.value for a in cf if a.name == "Case-funding frequency")
    sev = next(a.value for a in cf if a.name == "Case-funding severity")
    _caption(
        fig,
        f"n={result.n_iterations:,} simulated years, seed={result.seed}. Case funding: {freq}, {sev} "
        f"(board-assumed). Blue = years covered by current liquid holdings; glowing cyan = years it would "
        f"not be.",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coverage_curve(result: SimulationResult, path: Path) -> Path:
    """Figure 2: reserve level vs. coverage probability."""
    total = result.total_annual_outflow
    liquid = result.liquid_holdings.total
    max_x = max(total.max(), liquid) * 1.05

    levels = np.linspace(0, max_x, 400)
    coverage_prob = np.array([(total <= level).mean() for level in levels])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(levels, coverage_prob * 100, color=NEUTRAL_COLOR, linewidth=2)
    ax.fill_between(levels, 0, coverage_prob * 100, color=NEUTRAL_COLOR, alpha=0.08)

    current_coverage = (total <= liquid).mean() * 100
    ax.scatter([liquid], [current_coverage], color="white", zorder=5, s=40)
    ax.axvline(liquid, color="white", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"current liquid\n${liquid:,.0f} -> {current_coverage:.1f}% covered",
        xy=(liquid, current_coverage),
        xytext=(10, -30),
        textcoords="offset points",
        fontsize=8,
    )

    for target, color in ((0.95, COVERED_COLOR), (0.99, SHORTFALL_COLOR)):
        req = float(np.percentile(total, target * 100))
        ax.axhline(target * 100, color=color, linestyle=":", linewidth=1)
        ax.axvline(req, color=color, linestyle=":", linewidth=1)
        ax.annotate(f"{target:.0%} needs ${req:,.0f}", xy=(req, target * 100), xytext=(6, 6), textcoords="offset points", fontsize=8, color=color)

    ax.set_xlabel("Reserve level ($)")
    ax.set_ylabel("Coverage probability (%)")
    ax.set_title("Coverage Curve: How Much Reserve Buys What Probability")
    ax.xaxis.set_major_formatter(lambda x, _: f"${x/1000:,.0f}k")
    ax.set_ylim(0, 102)

    cf = result.assumptions
    freq = next(a.value for a in cf if a.name == "Case-funding frequency")
    sev = next(a.value for a in cf if a.name == "Case-funding severity")
    _caption(
        fig,
        f"n={result.n_iterations:,} simulated years, seed={result.seed}. Case funding: {freq}, {sev} "
        f"(board-assumed); rent/subscriptions/other-operating baseline (data-derived). Dotted lines mark "
        f"the reserve levels needed for 95% and 99% coverage.",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mechanism_contribution(result: SimulationResult, path: Path) -> Path:
    """Figure 3: decompose mean and p95 by mechanism."""
    mechanisms = ["Deterministic\nbaseline", "Case funding", "Idiosyncratic\ntail"]
    components = [result.baseline_component, result.case_funding_component, result.idiosyncratic_component]
    means = [float(np.mean(c)) for c in components]
    p95s = [float(np.percentile(c, 95)) for c in components]

    x = np.arange(len(mechanisms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, means, width, label="Mean", color=NEUTRAL_COLOR)
    ax.bar(x + width / 2, p95s, width, label="P95", color=COVERED_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(mechanisms)
    ax.set_ylabel("Annual outflow contribution ($)")
    ax.set_title("Mechanism Contribution to Annual Disbursement Risk")
    ax.yaxis.set_major_formatter(lambda v, _: f"${v/1000:,.0f}k")
    ax.legend(frameon=False)

    cf = result.assumptions
    freq = next(a.value for a in cf if a.name == "Case-funding frequency")
    sev = next(a.value for a in cf if a.name == "Case-funding severity")
    _caption(
        fig,
        f"n={result.n_iterations:,} simulated years, seed={result.seed}. Deterministic baseline: rent + "
        f"subscriptions + other-operating average (data-derived). Case funding: {freq}, {sev} "
        f"(board-assumed). Idiosyncratic tail: bootstrap-resampled from 3 historical one-offs "
        f"(data-derived pool, board-assumed shock frequency). Each mechanism's mean/P95 is its own "
        f"marginal distribution — percentiles do not sum across independent mechanisms, so these bars "
        f"will not add up to the combined distribution's P95. They show which mechanism dominates the "
        f"org's risk, which is why a three-mechanism model was used instead of one bootstrap over raw "
        f"historical cash flow.",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sensitivity_grid(
    result: SimulationResult, config: FortuneConfig, transactions: pd.DataFrame, path: Path
) -> Path:
    """Figure 4: coverage probability across severity cap x frequency assumption."""
    severity_caps = [100_000, 150_000, 200_000]
    frequency_ranges = [(1, 3), (2, 5)]
    liquid = config.liquid_holdings.total

    grid = np.zeros((len(frequency_ranges), len(severity_caps)))
    for i, (f_lo, f_hi) in enumerate(frequency_ranges):
        for j, cap in enumerate(severity_caps):
            cf = replace(
                config.case_funding,
                frequency_sampler=UniformDiscreteFrequency(low=f_lo, high=f_hi),
                severity_sampler=UniformContinuousSeverity(low=50_000, high=cap),
            )
            scenario_config = replace(config, case_funding=cf)
            scenario_result = run_simulation(scenario_config, transactions)
            grid[i, j] = (scenario_result.total_annual_outflow <= liquid).mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, cmap=_COVERAGE_CMAP, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(severity_caps)))
    ax.set_xticklabels([f"${c/1000:,.0f}k" for c in severity_caps])
    ax.set_yticks(range(len(frequency_ranges)))
    ax.set_yticklabels([f"{lo}-{hi}/yr" for lo, hi in frequency_ranges])
    ax.set_xlabel("Severity cap per case")
    ax.set_ylabel("Frequency assumption")
    ax.set_title("Sensitivity: Coverage Probability at Current Liquid Holdings")
    ax.grid(False)

    for i in range(len(frequency_ranges)):
        for j in range(len(severity_caps)):
            value = grid[i, j]
            r, g, b, _ = im.cmap(im.norm(value))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text_color = "black" if luminance > 0.5 else "white"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=11)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Coverage probability (%)")

    _caption(
        fig,
        f"n={result.n_iterations:,} simulated years per cell, seed={result.seed}. Liquid holdings held fixed "
        f"at \\${liquid:,.0f} (board-provided). Severity floor fixed at \\$50k in all cells; the severity "
        f"cap and frequency range shown on each axis are the varied assumptions (board-assumed).",
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_figures(
    result: SimulationResult, config: FortuneConfig, transactions: pd.DataFrame, outputs_dir: Path
) -> list[Path]:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_distribution_histogram(result, outputs_dir / "figure1_distribution_histogram.png"),
        plot_coverage_curve(result, outputs_dir / "figure2_coverage_curve.png"),
        plot_mechanism_contribution(result, outputs_dir / "figure3_mechanism_contribution.png"),
        plot_sensitivity_grid(result, config, transactions, outputs_dir / "figure4_sensitivity_grid.png"),
    ]
