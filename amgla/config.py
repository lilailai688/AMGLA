"""Configuration for the AMGLA pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DeviceName = Literal["auto", "cpu", "cuda"]
RepresentationMode = Literal["forecast_signature", "hidden_state"]
HiddenStatePooling = Literal["last_patch"]


@dataclass(frozen=True)
class AMGLAConfig:
    """Runtime options for cold-start AMGLA forecasting."""

    history_length: int = 21
    forecast_horizon: int = 7
    lookback_windows: tuple[int, ...] = (21, 14, 7)
    ca_top_k: int = 2
    sa_top_k: int = 3
    timesfm_checkpoint: str = "google/timesfm-2.5-200m-pytorch"
    device: DeviceName = "auto"
    blend_weight: float = 0.5
    representation_mode: RepresentationMode = "hidden_state"
    hidden_state_pooling: HiddenStatePooling = "last_patch"
    timesfm_max_context: int = 1024
    timesfm_max_horizon: int = 256
    seed: int = 42

    def validate(self) -> None:
        if self.history_length < 2:
            raise ValueError("history_length must be at least 2.")
        if self.forecast_horizon < 1:
            raise ValueError("forecast_horizon must be positive.")
        if not self.lookback_windows:
            raise ValueError("lookback_windows cannot be empty.")
        if any(window < 2 for window in self.lookback_windows):
            raise ValueError("lookback_windows must contain values >= 2.")
        if max(self.lookback_windows) > self.history_length:
            raise ValueError("lookback_windows cannot exceed history_length.")
        if self.ca_top_k < 1:
            raise ValueError("ca_top_k must be positive.")
        if self.sa_top_k < 0:
            raise ValueError("sa_top_k cannot be negative.")
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be in [0, 1].")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")
        if self.representation_mode not in {"forecast_signature", "hidden_state"}:
            raise ValueError("unsupported representation_mode.")
        if self.hidden_state_pooling != "last_patch":
            raise ValueError("hidden_state_pooling must be 'last_patch'.")

    @property
    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
