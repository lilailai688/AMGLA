"""TimesFM 2.5 adapter used by the AMGLA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .config import AMGLAConfig


@dataclass(frozen=True)
class TimesFMForecast:
    point_forecast: np.ndarray
    quantile_forecast: np.ndarray | None


class ForecastAdapter(Protocol):
    def forecast(self, inputs: Sequence[np.ndarray], horizon: int) -> TimesFMForecast:
        """Forecast a batch of univariate histories."""

    def encode_hidden(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        """Encode histories into TimesFM hidden-state embeddings."""


class TimesFMAdapter:
    """Lazy wrapper around the official TimesFM 2.5 PyTorch API."""

    def __init__(self, config: AMGLAConfig) -> None:
        self.config = config
        self._model = None
        self._compiled_horizon: int | None = None

    def forecast(self, inputs: Sequence[np.ndarray], horizon: int) -> TimesFMForecast:
        if horizon < 1:
            raise ValueError("horizon must be positive.")
        model = self._load_model(horizon)
        arrays = [np.asarray(values, dtype=float) for values in inputs]
        point, quantile = model.forecast(horizon=horizon, inputs=arrays)
        return TimesFMForecast(
            point_forecast=np.asarray(point, dtype=float),
            quantile_forecast=None if quantile is None else np.asarray(quantile, dtype=float),
        )

    def encode_hidden(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        model = self._load_model(self.config.forecast_horizon)
        inner_model = getattr(model, "model", None)
        if inner_model is None:
            raise RuntimeError(
                "TimesFM 2.5 wrapper does not expose `model.model`; "
                "check that google/timesfm-2.5-200m-pytorch is installed."
            )

        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise ImportError("TimesFM hidden-state extraction requires PyTorch.") from exc

        patch_len = _timesfm_patch_length(inner_model)
        values, masks = front_pad_to_patch_multiple(
            inputs,
            patch_len=patch_len,
            max_context=self.config.timesfm_max_context,
        )
        forecast_config = getattr(model, "forecast_config", None)
        if getattr(forecast_config, "normalize_inputs", True):
            values = normalize_observed_values(values, masks)

        device = getattr(inner_model, "device", None)
        if device is None:
            device = torch.device(self.config.resolved_device)
        tensor_values = torch.tensor(values, dtype=torch.float32, device=device)
        tensor_masks = torch.tensor(masks, dtype=torch.bool, device=device)
        patched_inputs = tensor_values.reshape(tensor_values.shape[0], -1, patch_len)
        patched_masks = tensor_masks.reshape(tensor_masks.shape[0], -1, patch_len)

        with torch.no_grad():
            forward_result = inner_model(patched_inputs, patched_masks)
        output_embeddings = _extract_output_embeddings(forward_result)
        if self.config.hidden_state_pooling != "last_patch":
            raise ValueError("Only last_patch pooling is supported.")
        hidden = output_embeddings[:, -1, :]
        result = hidden.detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("TimesFM hidden-state embeddings contain NaN or infinite values.")
        return result

    def _load_model(self, horizon: int):
        if self._model is not None and self._compiled_horizon is not None and horizon <= self._compiled_horizon:
            return self._model
        try:
            import torch  # type: ignore
            import timesfm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "TimesFM is not installed. Follow the official google-research/timesfm "
                "installation guide, then install this project with `pip install -e .`."
            ) from exc

        if self.config.resolved_device == "cuda":
            torch.set_float32_matmul_precision("high")

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.config.timesfm_checkpoint)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.config.timesfm_max_context,
                max_horizon=max(horizon, self.config.timesfm_max_horizon),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._model = model
        self._compiled_horizon = max(horizon, self.config.timesfm_max_horizon)
        return model


def front_pad_to_patch_multiple(
    inputs: Sequence[np.ndarray],
    patch_len: int,
    max_context: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if patch_len < 1:
        raise ValueError("patch_len must be positive.")
    series = [_as_1d(values, "inputs") for values in inputs]
    if not series:
        raise ValueError("inputs cannot be empty.")
    if max_context is not None:
        series = [values[-max_context:] for values in series]
    max_length = max(values.shape[0] for values in series)
    padded_length = int(np.ceil(max_length / patch_len) * patch_len)
    values = np.zeros((len(series), padded_length), dtype=np.float32)
    masks = np.ones((len(series), padded_length), dtype=bool)
    for row, item in enumerate(series):
        start = padded_length - item.shape[0]
        values[row, start:] = item
        masks[row, start:] = False
    return values, masks


def normalize_observed_values(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    observed = ~masks
    counts = np.maximum(observed.sum(axis=1, keepdims=True), 1)
    mean = np.sum(np.where(observed, values, 0.0), axis=1, keepdims=True) / counts
    variance = np.sum(np.where(observed, (values - mean) ** 2, 0.0), axis=1, keepdims=True) / counts
    std = np.sqrt(np.maximum(variance, 1e-6))
    normalized = (values - mean) / std
    normalized[masks] = 0.0
    return normalized.astype(np.float32, copy=False)


def _timesfm_patch_length(inner_model) -> int:
    patch_len = getattr(inner_model, "p", None)
    if patch_len is None:
        patch_len = getattr(inner_model, "input_patch_len", None)
    if patch_len is None and getattr(inner_model, "config", None) is not None:
        patch_len = getattr(inner_model.config, "input_patch_len", None)
    return int(patch_len or 32)


def _extract_output_embeddings(forward_result):
    outputs = forward_result[0] if isinstance(forward_result, tuple) else forward_result
    if isinstance(outputs, tuple) and len(outputs) >= 2:
        return outputs[1]
    output_embeddings = getattr(outputs, "output_embeddings", None)
    if output_embeddings is not None:
        return output_embeddings
    raise RuntimeError(
        "TimesFM internal forward did not return output_embeddings. "
        "This adapter targets TimesFM 2.5 PyTorch; please check the installed version."
    )


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"{name} must contain 1D time series.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} cannot contain empty time series.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr
