# BTC Regime Reproduction Repo

Research-first reproduction scaffold for the Bitcoin regime article using a K-Means initialized HMM pipeline.

Current default dataset:

- external input path: `${BTC_BINANCE_SOURCE_PATH}`
- market approximation: `Binance BTCUSDT futures`
- fidelity caveat: this is not the exact `BTC/USD 1m 2012-2021` dataset referenced by the article

## Environment Setup

This repo reads local paths from `.env` automatically when you run the CLI.

Copy `.env.example` to `.env` and set the paths for your machine:

```bash
cp .env.example .env
```

Required variables:

- `BTC_REGIME_OUTPUT_ROOT`
- `BTC_BINANCE_SOURCE_PATH`
- `BTC_PAPER_SOURCE_PATH`
- `BTC_PAPER_LIKE_SOURCE_PATH`

## What this repo does

- Validates and resamples raw 1-minute BTC OHLCV into a canonical 5-minute dataset
- Builds `log_return` and `rolling_volatility` features
- Runs elbow diagnostics for `K=2..6`
- Compares:
  - `hmm_random_init`
  - `kmeans_initialized_hmm`
- Exports artifacts per run under `runs/`

## CLI

```bash
python scripts/prepare_btc_data.py --config configs/local_btc_binance_2021_2026.yaml
python scripts/run_regime_reproduction.py --config configs/local_btc_binance_2021_2026.yaml
python scripts/compare_models.py --run-dir runs/<run_id>
```

## Repo layout

- `src/btc_regime_repro/`: reusable pipeline code
- `configs/`: example configs
- `scripts/`: CLI entrypoints
- `tests/`: unit tests
- `runs/`: timestamped artifacts
