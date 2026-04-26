# AMGLA

AMGLA is a compact workflow project for cold-start financial product sales
forecasting. It focuses on the practical processing pipeline:

1. read historical complete products and cold-start target products;
2. extract product representations from TimesFM 2.5 hidden states;
3. retrieve similar historical products;
4. build AMGLA-style CA/SA signals;
5. blend TimesFM forecasts with AMGLA future proxies;
6. export prediction and neighbor inspection files.

This repository intentionally does not include example data. Use your own
financial product CSV files that follow the schema below.

## TimesFM

TimesFM is used as the time-series foundation model. The default checkpoint is:

```text
google/timesfm-2.5-200m-pytorch
```

The adapter follows the public loading and forecasting API from
[google-research/timesfm](https://github.com/google-research/timesfm), and
uses TimesFM 2.5 internal hidden states as the AMGLA representation:

```python
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(...))
point_forecast, quantile_forecast = model.forecast(horizon=7, inputs=batch)
```

For retrieval, AMGLA calls the internal torch module exposed by the
TimesFM 2.5 wrapper and pools the last valid patch hidden state. The direct
`forecast()` output is still used for `yhat_timesfm`.

## Input CSV Schema

Required columns:

| column | description |
| --- | --- |
| `product_id` | financial product identifier |
| `date` | observation date |
| `sales` | daily sales amount, subscription amount, or normalized target |

Additional columns can be kept in the CSV for future covariate extensions, but
the current workflow uses the univariate `sales` sequence.

## Run

```bash
python -m scripts.predict \
  --history-csv path/to/history.csv \
  --target-csv path/to/cold_start.csv \
  --output-dir runs/my_run \
  --horizon 7
```

All artifacts are written under the selected `runs/` directory:

- `prediction.csv`
- `neighbors.csv`
- `config.json`
- `run_summary.md`

## Output Columns

`prediction.csv` contains:

| column | description |
| --- | --- |
| `product_id` | target cold-start product |
| `date` | forecast date |
| `horizon_step` | 1-based forecast step |
| `yhat` | blended final forecast |
| `yhat_timesfm` | direct TimesFM forecast |
| `yhat_amgla_proxy` | AMGLA similar-product future proxy |
| `neighbor_ids` | retrieved historical products |

`neighbors.csv` contains one row per retrieved neighbor, including similarity
and aggregation weight.

## Project Layout

```text
amgla/                 Core workflow package
scripts/predict.py     CLI entrypoint
docs/                  Workflow and TimesFM representation notes
tests/                 Unit tests with in-memory mock data only
runs/                  Local run outputs, ignored by Git
```

## Scope

This project is a clean GitHub-ready workflow implementation. It is not a
full reproduction of private MYBank experiments, online A/B tests, or the full
benchmark matrix from the AMGLA paper.
