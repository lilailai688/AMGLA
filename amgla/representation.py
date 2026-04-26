"""TimesFM hidden states for multi-window product representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import AMGLAConfig
from .data import left_pad
from .timesfm_adapter import ForecastAdapter, TimesFMAdapter, TimesFMForecast


@dataclass
class TimesFMRepresentationExtractor:
    config: AMGLAConfig
    adapter: ForecastAdapter | None = None

    def __post_init__(self) -> None:
        self.config.validate()
        if self.adapter is None:
            self.adapter = TimesFMAdapter(self.config)

    def transform(self, contexts: np.ndarray) -> np.ndarray:
        sequences = _as_context_sequences(contexts)
        view_features: list[np.ndarray] = []
        for window in self.config.lookback_windows:
            inputs = [_window_tail(row, window) for row in sequences]
            if self.config.representation_mode == "hidden_state":
                view_features.append(self.adapter.encode_hidden(inputs))
            else:
                padded = [left_pad(row, window) for row in inputs]
                forecast = self.adapter.forecast(padded, self.config.forecast_horizon)
                view_features.append(forecast_signature(forecast))
        return standardize(np.concatenate(view_features, axis=1))

    def forecast(self, contexts: np.ndarray) -> TimesFMForecast:
        inputs = _as_context_sequences(contexts)
        return self.adapter.forecast(inputs, self.config.forecast_horizon)


def forecast_signature(forecast: TimesFMForecast) -> np.ndarray:
    point = _as_forecast_matrix(forecast.point_forecast)
    horizon = point.shape[1]
    time = np.arange(horizon, dtype=float)
    time = time - time.mean()
    denom = float(np.sum(time**2)) or 1.0
    slope = ((point - point.mean(axis=1, keepdims=True)) @ time) / denom
    features = [
        point.mean(axis=1),
        point.std(axis=1),
        point.min(axis=1),
        point.max(axis=1),
        point[:, 0],
        point[:, -1],
        point[:, -1] - point[:, 0],
        slope,
    ]

    if forecast.quantile_forecast is not None:
        quantile = np.asarray(forecast.quantile_forecast, dtype=float)
        if quantile.ndim != 3 or quantile.shape[:2] != point.shape:
            raise ValueError("quantile_forecast must have shape (N, H, Q).")
        low = quantile[:, :, 1] if quantile.shape[2] > 1 else quantile[:, :, 0]
        high = quantile[:, :, -1]
        spread = high - low
        features.extend(
            [
                quantile.mean(axis=(1, 2)),
                spread.mean(axis=1),
                spread.max(axis=1),
                low.mean(axis=1),
                high.mean(axis=1),
            ]
        )

    result = np.column_stack(features)
    return standardize(result)


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.where(std < 1e-8, 1.0, std)


def _as_context_matrix(contexts: np.ndarray) -> np.ndarray:
    matrix = np.asarray(contexts, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("contexts must have shape (N, T).")
    if matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("contexts must contain at least one product and two steps.")
    if not np.isfinite(matrix).all():
        raise ValueError("contexts contain NaN or infinite values.")
    return matrix


def _as_context_sequences(contexts: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(contexts, np.ndarray) and contexts.ndim == 2:
        return [row.astype(float, copy=False) for row in _as_context_matrix(contexts)]
    sequences: list[np.ndarray] = []
    for row in contexts:
        arr = np.asarray(row, dtype=float)
        if arr.ndim != 1 or arr.shape[0] == 0:
            raise ValueError("contexts must contain non-empty 1D series.")
        if not np.isfinite(arr).all():
            raise ValueError("contexts contain NaN or infinite values.")
        sequences.append(arr)
    if not sequences:
        raise ValueError("contexts cannot be empty.")
    return sequences


def _window_tail(values: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("window must be at least 2.")
    return values[-window:].astype(float, copy=False)


def _as_forecast_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("point_forecast must have shape (N, H).")
    if not np.isfinite(matrix).all():
        raise ValueError("point_forecast contains NaN or infinite values.")
    return matrix
