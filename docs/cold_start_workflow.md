# Cold-Start Financial Product Forecasting Workflow

This document describes the AMGLA workflow for cold-start financial product
sales forecasting. The project does not include example data. Users should
provide their own historical financial product sales CSV files.

## 1. Input Data

Required columns:

| Column | Meaning |
| --- | --- |
| `product_id` | Financial product ID |
| `date` | Sales date |
| `sales` | Daily sales amount, subscription amount, or normalized target |

Optional business columns can remain in the CSV, such as channel, risk level,
product term, yield range, or holiday indicator. The current workflow uses the
univariate `sales` sequence, while these optional columns can be connected to
TimesFM covariates or downstream supervised predictors in later extensions.

## 2. Historical and Cold-Start Split

`history_df` contains historical products that have completed their sales
cycles. Each product needs at least `history_length + forecast_horizon`
observations. The default setting is `21 + 7`.

`cold_start_df` contains target products in the early sales stage. A target
product can have fewer than 21 observations. The matrix representation is
left-padded to `history_length`, while TimesFM hidden-state extraction uses the
real observed sequence before adapter-level masking.

## 3. TimesFM Representation Extraction

For each product, AMGLA builds three lookback windows: `21/14/7`. Each window
is passed into the internal TimesFM 2.5 torch module, and AMGLA reads the
hidden state of the last valid patch.

The hidden states from all windows are concatenated into a product-level
representation. This representation is used to measure similarity between
cold-start target products and historical reference products. The public
TimesFM `forecast()` output is still used as `yhat_timesfm`.

## 4. AMGLA-CA

CA performs cosine top-k retrieval over TimesFM hidden-state representations.
For each cold-start product, AMGLA finds the most similar historical products
and aggregates their observed future segments into `yhat_amgla_proxy`.

The retrieved neighbor IDs, similarities, and aggregation weights are written
to `neighbors.csv` for inspection.

## 5. AMGLA-SA

SA decomposes historical product contexts into trend, seasonal, and residual
components. It then applies trend mixup, seasonal phase alignment, and residual
perturbation. In the current workflow, SA records augmented contexts in
metadata and provides a clean extension point for a downstream supervised
predictor.

## 6. Final Prediction

The final prediction blends the direct TimesFM forecast and the AMGLA
similar-product future proxy:

```text
yhat = blend_weight * yhat_timesfm + (1 - blend_weight) * yhat_amgla_proxy
```

The default value is `blend_weight=0.5`. The workflow writes:

- `prediction.csv`: final forecasts by product and horizon step;
- `neighbors.csv`: retrieved historical products for each cold-start product;
- `config.json`: run configuration;
- `run_summary.md`: run summary.
