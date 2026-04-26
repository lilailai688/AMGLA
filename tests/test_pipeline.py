import numpy as np
import pandas as pd

from amgla import AMGLAColdStartPipeline, AMGLAConfig
from amgla.timesfm_adapter import TimesFMForecast


class MockTimesFMAdapter:
    def encode_hidden(self, inputs) -> np.ndarray:
        embeddings = []
        for values in inputs:
            arr = np.asarray(values, dtype=float)
            embeddings.append(
                [
                    float(arr.mean()),
                    float(arr.std()),
                    float(arr[0]),
                    float(arr[-1]),
                    float(arr[-1] - arr[0]),
                    float(arr.shape[0]),
                ]
            )
        return np.asarray(embeddings, dtype=float)

    def forecast(self, inputs, horizon: int) -> TimesFMForecast:
        point = []
        quantile = []
        for values in inputs:
            arr = np.asarray(values, dtype=float)
            last = arr[-1]
            slope = (arr[-1] - arr[0]) / max(arr.shape[0] - 1, 1)
            base = last + slope * np.arange(1, horizon + 1)
            point.append(base)
            quantile.append(np.stack([base, base - 1.0, base, base + 1.0], axis=1))
        return TimesFMForecast(np.asarray(point), np.asarray(quantile))


def _frame(products: list[str], periods: int) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=periods)
    rows = []
    for offset, product in enumerate(products):
        for i, date in enumerate(dates):
            rows.append(
                {
                    "product_id": product,
                    "date": date,
                    "sales": float(10 + offset * 5 + i),
                }
            )
    return pd.DataFrame(rows)


def test_pipeline_predicts_required_columns() -> None:
    config = AMGLAConfig(forecast_horizon=7, ca_top_k=2, sa_top_k=2)
    history = _frame(["h1", "h2", "h3"], 35)
    target = _frame(["t1", "t2"], 8)
    pipeline = AMGLAColdStartPipeline(config, timesfm_adapter=MockTimesFMAdapter())
    result = pipeline.fit(history).predict(target)
    assert result.predictions.shape[0] == 14
    assert set(
        [
            "product_id",
            "date",
            "horizon_step",
            "yhat",
            "yhat_timesfm",
            "yhat_amgla_proxy",
            "neighbor_ids",
        ]
    ).issubset(result.predictions.columns)
    assert result.neighbors.shape[0] == 4
    assert result.metadata["num_augmented_contexts"] == 9


def test_timesfm_representation_shape() -> None:
    config = AMGLAConfig(forecast_horizon=7)
    history = _frame(["h1", "h2"], 35)
    pipeline = AMGLAColdStartPipeline(config, timesfm_adapter=MockTimesFMAdapter()).fit(history)
    assert pipeline._reference_repr is not None
    assert pipeline._reference_repr.shape[0] == 2
    assert pipeline._reference_repr.shape[1] == 18
    assert pipeline.config.representation_mode == "hidden_state"
