"""Fortune: cash-flow risk simulation engine for a legal-aid fund.

This package estimates how much cash a case-funding organization must hold
in liquid reserve to safely fund its cases, by modeling three independent
mechanisms (deterministic operating baseline, case-funding
frequency-severity, and idiosyncratic tail shocks) rather than
bootstrap-resampling raw historical cash flow.

This is a decision-support tool for a fiduciary body that retains full
responsibility for reserve decisions. It is not investment advice and does
not forecast markets.
"""

__version__ = "0.1.0"
