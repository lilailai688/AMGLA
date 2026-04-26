"""End-to-end cold-start forecasting pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .augmentation import SampleAugmenter, build_future_proxy, build_neighbor_covariates
from .config import AMGLAConfig
from .data import ProductPanel, build_reference_panel, build_target_panel, future_dates
from .representation import TimesFMRepresentationExtractor
from .retrieval import RetrievalResult, cosine_topk
from .timesfm_adapter import ForecastAdapter


@dataclass
class PredictionResult:
    predictions: pd.DataFrame
    neighbors: pd.DataFrame
    metadata: dict[str, Any]


class AMGLAColdStartPipeline:
    """Fit on complete historical products and predict cold-start products."""

    def __init__(self, config: AMGLAConfig | None = None, timesfm_adapter: ForecastAdapter | None = None) -> None:
        self.config = config or AMGLAConfig()
        self.config.validate()
        self.extractor = TimesFMRepresentationExtractor(self.config, adapter=timesfm_adapter)
        self.sample_augmenter = SampleAugmenter(
            k=self.config.sa_top_k,
            period=7,
            seed=self.config.seed,
        )
        self._reference_panel: ProductPanel | None = None
        self._reference_repr: np.ndarray | None = None
        self._augmented_contexts: np.ndarray | None = None

    def fit(self, history_df: pd.DataFrame) -> "AMGLAColdStartPipeline":
        reference_panel = build_reference_panel(
            history_df,
            history_length=self.config.history_length,
            horizon=self.config.forecast_horizon,
        )
        self._reference_panel = reference_panel
        self._reference_repr = self.extractor.transform(reference_panel.observed_contexts or reference_panel.contexts)
        self._augmented_contexts = self.sample_augmenter.augment(reference_panel.contexts)
        return self

    def predict(self, cold_start_df: pd.DataFrame) -> PredictionResult:
        if self._reference_panel is None or self._reference_repr is None:
            raise RuntimeError("AMGLAColdStartPipeline must be fitted before predict().")
        target_panel = build_target_panel(cold_start_df, self.config.history_length)
        target_repr = self.extractor.transform(target_panel.observed_contexts or target_panel.contexts)
        retrieval = cosine_topk(target_repr, self._reference_repr, self.config.ca_top_k)
        future_proxy = build_future_proxy(
            self._reference_panel.futures,
            retrieval.indices,
            retrieval.weights,
        )
        _ = build_neighbor_covariates(self._reference_panel.contexts, retrieval.indices)
        timesfm_forecast = self.extractor.forecast(target_panel.observed_contexts or target_panel.contexts).point_forecast
        yhat = self.config.blend_weight * timesfm_forecast + (1.0 - self.config.blend_weight) * future_proxy

        predictions = self._build_prediction_frame(target_panel, yhat, timesfm_forecast, future_proxy, retrieval)
        neighbors = self._build_neighbor_frame(target_panel, retrieval)
        metadata = {
            "config": asdict(self.config),
            "num_reference_products": len(self._reference_panel.product_ids),
            "num_target_products": len(target_panel.product_ids),
            "num_augmented_contexts": int(self._augmented_contexts.shape[0]) if self._augmented_contexts is not None else 0,
            "representation": f"timesfm_{self.config.representation_mode}",
        }
        return PredictionResult(predictions=predictions, neighbors=neighbors, metadata=metadata)

    def _build_prediction_frame(
        self,
        target_panel: ProductPanel,
        yhat: np.ndarray,
        yhat_timesfm: np.ndarray,
        yhat_proxy: np.ndarray,
        retrieval: RetrievalResult,
    ) -> pd.DataFrame:
        assert target_panel.last_dates is not None
        assert self._reference_panel is not None
        rows: list[dict[str, Any]] = []
        for i, product_id in enumerate(target_panel.product_ids):
            neighbor_ids = "|".join(self._reference_panel.product_ids[int(idx)] for idx in retrieval.indices[i])
            for step, date in enumerate(future_dates(target_panel.last_dates[i], self.config.forecast_horizon), start=1):
                rows.append(
                    {
                        "product_id": product_id,
                        "date": date.date().isoformat(),
                        "horizon_step": step,
                        "yhat": float(yhat[i, step - 1]),
                        "yhat_timesfm": float(yhat_timesfm[i, step - 1]),
                        "yhat_amgla_proxy": float(yhat_proxy[i, step - 1]),
                        "neighbor_ids": neighbor_ids,
                    }
                )
        return pd.DataFrame(rows)

    def _build_neighbor_frame(self, target_panel: ProductPanel, retrieval: RetrievalResult) -> pd.DataFrame:
        assert self._reference_panel is not None
        rows: list[dict[str, Any]] = []
        for i, product_id in enumerate(target_panel.product_ids):
            for rank, ref_idx in enumerate(retrieval.indices[i], start=1):
                rows.append(
                    {
                        "product_id": product_id,
                        "neighbor_rank": rank,
                        "neighbor_id": self._reference_panel.product_ids[int(ref_idx)],
                        "similarity": float(retrieval.similarities[i, rank - 1]),
                        "weight": float(retrieval.weights[i, rank - 1]),
                    }
                )
        return pd.DataFrame(rows)
