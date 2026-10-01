# Fortune

This tool sizes the cash reserve for a legal-aid non-profit.

It's built and validated on the organisation's real banking data. That data isn't in this repo; `data/` is a synthetic copy with the same structure.

This tool currently has the following features:

- Three-mechanism Monte Carlo engine (baseline + case funding + bootstrapped tail)
- Transaction classifier that strips out internal transfers and bounced payments
- Confidence intervals on thin-sample tail estimates
- Seeded, reproducible runs from a CLI
- Figures, regenerable on demand
- Full test suite against analytic ground truth

Future features that I would like to include:

- Net cash flow modelling, tracking inflows as well as outflows
- Re-estimate case funding once more real events accrue (board-assumed off n=1 right now)
- Reserve allocation across cash, fixed income and equity (Black-Litterman, Ledoit-Wolf)
- Flag holdings that fall outside policy ranges
- Web UI instead of CLI
- LLM-drafted board memo
- Full board packet as one document

## Services Used

- Language - [Python](https://www.python.org)
- Monte Carlo Sampling - [NumPy](https://numpy.org)
- Transaction Loading - [pandas](https://pandas.pydata.org)
- Ground-Truth Stats - [SciPy](https://scipy.org)
- Figures - [Matplotlib](https://matplotlib.org)
- Tests - [pytest](https://docs.pytest.org)
- Helpful Partner - [Claude](https://claude.ai)

## Dataset

Two bank accounts, chequing and savings.

Real data withheld. `scripts/generate_sample_data.py` builds a synthetic sample with the same edge cases.

TODO: generator doesn't model inflows yet.

## Simulate

Use `simulate.py` to:

- Bucket every transaction (`classify.py`)
- Model the baseline, case funding and tail (`baseline.py`, `case_funding.py`, `tail.py`)
- Run 50,000 simulated years against current liquid holdings

## Results

![Distribution](docs/sample_figures/figure1_distribution_histogram.png)

On the real data, current reserves cover about 97.7% of simulated years, short about $10K at the 99th percentile ($315K needed). Figure above is from the synthetic sample.

## Run it

```bash
pip install -r requirements.txt
python -m fortune.cli --data data/ --seed 42 --figures
pytest
```

Against real data: fill in `local_config.example.json` and pass it with `--overrides-json`.
