# TimesFM Hidden-State Representation

AMGLA uses TimesFM as the time-series foundation model. To match the paper's
setting, the default representation is extracted from internal TimesFM 2.5
hidden states rather than from forecast signatures alone.

## TimesFM Interface

Model loading and forecasting still follow the public TimesFM 2.5 PyTorch API:

```python
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(...))
point_forecast, quantile_forecast = model.forecast(horizon=F, inputs=series_batch)
```

Representation extraction uses the internal `model.model` module exposed by
the TimesFM 2.5 wrapper. The internal torch module returns `input_embeddings`
and `output_embeddings`; AMGLA takes the hidden state from `output_embeddings`
at the last valid patch.

## Hidden-State Workflow

For each product, AMGLA builds three windows: `21/14/7`. For each window:

1. Use the real observed tail sequence without business-level left padding.
2. Front-pad zeros inside the TimesFM adapter so the length is a multiple of
   the patch length.
3. Mark padded positions with a mask.
4. Normalize only observed values and keep padded positions as zero.
5. Call `model.model(patched_inputs, patched_masks)`.
6. Apply `last_patch` pooling to read the final valid patch hidden state.

The hidden states from all windows are concatenated and standardized. This
representation is then used for cosine top-k retrieval and CA.

## Forecast Signature Fallback

`representation_mode="forecast_signature"` remains available as a debugging
fallback. It builds statistical features from TimesFM point forecasts and
quantile forecasts. It is not the default path and should not be described as
the paper-aligned representation.

## Version Assumption

Hidden states rely on an internal TimesFM interface. This project targets the
TimesFM 2.5 PyTorch checkpoint:

```text
google/timesfm-2.5-200m-pytorch
```

If a future TimesFM release changes the internal forward return structure, the
adapter raises a clear error and asks the user to check the installed TimesFM
version.
