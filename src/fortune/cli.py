"""CLI entry point: python -m fortune.cli --data data/ --seed 42 [--figures].

By default this reads the committed synthetic sample dataset. To run against
real, non-committed data: point --chequing-file/--savings-file at your local
export filenames and --overrides-json at a local (gitignored) overrides file
containing your real assumptions (see local_config.example.json).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from fortune.config import (
    FortuneConfig,
    UniformContinuousSeverity,
    UniformDiscreteFrequency,
    load_overrides,
)
from fortune.loader import DEFAULT_CHEQUING_FILENAME, DEFAULT_SAVINGS_FILENAME, load_transactions
from fortune.report import print_report
from fortune.simulate import run_simulation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fortune.cli",
        description="Cash-flow risk simulation for a legal-aid fund: reserve recommendation report.",
    )
    parser.add_argument("--data", type=Path, default=Path("data"), help="Directory containing the two CSV exports.")
    parser.add_argument(
        "--chequing-file",
        type=str,
        default=DEFAULT_CHEQUING_FILENAME,
        help="Chequing export filename within --data (default: the committed synthetic sample).",
    )
    parser.add_argument(
        "--savings-file",
        type=str,
        default=DEFAULT_SAVINGS_FILENAME,
        help="Savings export filename within --data (default: the committed synthetic sample).",
    )
    parser.add_argument(
        "--overrides-json",
        type=Path,
        default=None,
        help="Path to a local, non-committed JSON file overriding assumption defaults (see local_config.example.json).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument("--iterations", type=int, default=None, help="Monte Carlo iterations (default: 50,000).")
    parser.add_argument(
        "--severity-min", type=float, default=None, help="Override case-funding severity lower bound."
    )
    parser.add_argument(
        "--severity-max", type=float, default=None, help="Override case-funding severity upper bound."
    )
    parser.add_argument(
        "--frequency-min", type=int, default=None, help="Override case-funding frequency lower bound (cases/yr)."
    )
    parser.add_argument(
        "--frequency-max", type=int, default=None, help="Override case-funding frequency upper bound (cases/yr)."
    )
    parser.add_argument(
        "--figures", action="store_true", help="Also generate the four required PNGs under outputs/."
    )
    parser.add_argument(
        "--outputs", type=Path, default=Path("outputs"), help="Directory to write figures to (with --figures)."
    )
    return parser


def build_config(args: argparse.Namespace) -> FortuneConfig:
    config = FortuneConfig()

    if args.overrides_json is not None:
        config = load_overrides(config, args.overrides_json)

    sim = config.simulation
    if args.seed is not None:
        sim = replace(sim, seed=args.seed)
    if args.iterations is not None:
        sim = replace(sim, n_iterations=args.iterations)
    config = replace(config, simulation=sim)

    cf = config.case_funding
    freq = cf.frequency_sampler
    sev = cf.severity_sampler
    if args.frequency_min is not None or args.frequency_max is not None:
        freq = UniformDiscreteFrequency(
            low=args.frequency_min if args.frequency_min is not None else freq.low,
            high=args.frequency_max if args.frequency_max is not None else freq.high,
        )
    if args.severity_min is not None or args.severity_max is not None:
        sev = UniformContinuousSeverity(
            low=args.severity_min if args.severity_min is not None else sev.low,
            high=args.severity_max if args.severity_max is not None else sev.high,
        )
    if freq is not cf.frequency_sampler or sev is not cf.severity_sampler:
        config = replace(config, case_funding=replace(cf, frequency_sampler=freq, severity_sampler=sev))

    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = build_config(args)
    transactions = load_transactions(args.data, args.chequing_file, args.savings_file)
    result = run_simulation(config, transactions)
    print_report(result)

    if args.figures:
        from fortune.figures import generate_all_figures

        args.outputs.mkdir(parents=True, exist_ok=True)
        paths = generate_all_figures(result, config, transactions, args.outputs)
        print(f"\nWrote {len(paths)} figure(s) to {args.outputs}/:")
        for p in paths:
            print(f"  - {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
