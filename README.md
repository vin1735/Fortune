# Fortune

Cash flow risk simulation for a legal-aid non-profit. Figures out how much cash reserve is actually needed to safely fund cases, instead of guessing off a gut number.

Built and validated on a real org's real banking data. That data isn't in this repo — `data/` is synthetic, generated to match its structure. See `scripts/generate_sample_data.py`.

This currently has:

- Three-mechanism Monte Carlo engine (deterministic baseline + case-funding frequency/severity model + bootstrap idiosyncratic tail), instead of one naive bootstrap over raw cash flow
- Rule-based transaction classifier that excludes internal transfers and bounced payments from spend
- Bootstrap confidence intervals on thin-sample tail estimates, not just a point number
- IPS reserve minimum kept separate from the model's statistical buffer
- CLI with seeded, reproducible runs, and a local-overrides file so real org data never touches tracked code
- Four matplotlib figures, regenerable on demand
- Full test suite: statistical correctness, classification, reproducibility, degenerate inputs, monotonicity, stress cases

Future:

- Net cash flow (inflows − outflows), not outflows only
- Re-estimate case-funding rate once more real events happen (n=1 right now)
- Portfolio / reserve allocation
- Web UI instead of CLI
- LLM layer to draft the board memo
- Auto-generated board packet

## Stack

- Python 3.11+
- numpy - Monte Carlo sampling
- pandas - transaction loading/classification
- scipy - ground-truth stats for tests
- matplotlib - figures
- pytest - tests

No web framework, no database, no external API. Just a CLI.

## Dataset

Two bank accounts (chequing + savings), same schema: date, description, sub-description, type, amount, balance.

Real data withheld. `scripts/generate_sample_data.py` builds a synthetic sample with the same edge cases: irregular rent cadence, an NSF reversal, both kinds of internal transfer, a pre-funded disbursement, thin-sample one-offs, zero-activity months.

TODO: generator doesn't model inflows yet.

## How it works

1. `loader.py` - load + validate the CSVs
2. `classify.py` - bucket every transaction (recurring / case funding / internal transfer / returned item / one-off / other)
3. `baseline.py`, `case_funding.py`, `tail.py` - the three mechanisms
4. `simulate.py` - combine mechanisms, run the Monte Carlo
5. `report.py` / `figures.py` - output

## Results

![Distribution](docs/sample_figures/figure1_distribution_histogram.png)

On the real data: current reserves cover ~97.7% of simulated years, short ~$10K at the 99th percentile (~$315K needed). Figure above is from the synthetic sample, not the real numbers.

## Run it

```bash
pip install -r requirements.txt
python -m fortune.cli --data data/ --seed 42 --figures
pytest
```

Against real data: fill in `local_config.example.json`, pass it via `--overrides-json`.
