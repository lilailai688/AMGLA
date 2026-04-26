"""Simplified CA and SA components for the AMGLA workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def build_future_proxy(reference_future: np.ndarray, neighbor_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    future = np.asarray(reference_future, dtype=float)
    indices = np.asarray(neighbor_indices, dtype=int)
    w = np.asarray(weights, dtype=float)
    if future.ndim != 2:
        raise ValueError("reference_future must have shape (N, H).")
    if indices.shape != w.shape:
        raise ValueError("neighbor_indices and weights must have the same shape.")
    proxy = np.zeros((indices.shape[0], future.shape[1]), dtype=float)
    for row in range(indices.shape[0]):
        proxy[row] = w[row] @ future[indices[row]]
    return proxy


def build_neighbor_covariates(reference_contexts: np.ndarray, neighbor_indices: np.ndarray) -> np.ndarray:
    contexts = np.asarray(reference_contexts, dtype=float)
    indices = np.asarray(neighbor_indices, dtype=int)
    if contexts.ndim != 2:
        raise ValueError("reference_contexts must have shape (N, T).")
    output = np.zeros((indices.shape[0], contexts.shape[1], indices.shape[1]), dtype=float)
    for i in range(indices.shape[0]):
        for rank in range(indices.shape[1]):
            output[i, :, rank] = contexts[indices[i, rank]]
    return output


@dataclass
class SampleAugmenter:
    """Trend-seasonal-residual mixup for process completeness."""

    k: int = 3
    period: int = 7
    seed: int = 42

    def augment(self, contexts: np.ndarray) -> np.ndarray:
        matrix = np.asarray(contexts, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("contexts must have shape (N, T).")
        if self.k == 0 or matrix.shape[0] <= 1:
            return matrix.copy()
        rng = np.random.default_rng(self.seed)
        similar = _topk_by_shape(matrix, self.k)
        synthetic: list[np.ndarray] = []
        for i in range(matrix.shape[0]):
            for j in similar[i]:
                synthetic.append(self._mix_pair(matrix[i], matrix[int(j)], rng))
        if not synthetic:
            return matrix.copy()
        return np.concatenate([matrix, np.stack(synthetic, axis=0)], axis=0)

    def _mix_pair(self, primary: np.ndarray, neighbor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        p_trend, p_season, p_residual = _decompose(primary, self.period)
        n_trend, n_season, n_residual = _decompose(neighbor, self.period)
        lam = rng.uniform(0.3, 0.7)
        trend = lam * p_trend + (1.0 - lam) * n_trend
        seasonal = lam * p_season + (1.0 - lam) * _phase_align(n_season, p_season)
        residual = 0.5 * p_residual + 0.5 * n_residual
        residual += rng.normal(0.0, 0.03 * max(np.std(primary), np.std(neighbor), 1e-6), size=primary.shape)
        return np.nan_to_num(trend + seasonal + residual, nan=float(np.mean(primary)))


def _topk_by_shape(matrix: np.ndarray, k: int) -> np.ndarray:
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.where(norms < 1e-12, 1.0, norms)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    k_eff = min(k, matrix.shape[0] - 1)
    indices = np.argpartition(-similarity, kth=k_eff - 1, axis=1)[:, :k_eff]
    scores = np.take_along_axis(similarity, indices, axis=1)
    order = np.argsort(-scores, axis=1)
    return np.take_along_axis(indices, order, axis=1)


def _decompose(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = np.asarray(values, dtype=float)
    if series.shape[0] < max(period * 2, 4) or np.std(series) < 1e-8:
        return series.copy(), np.zeros_like(series), np.zeros_like(series)
    trend = _moving_average(series, period)
    detrended = series - trend
    seasonal = np.zeros_like(series)
    for offset in range(period):
        mask = np.arange(series.shape[0]) % period == offset
        seasonal[mask] = detrended[mask].mean()
    residual = series - trend - seasonal
    return trend, seasonal, residual


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")[: values.shape[0]]


def _phase_align(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if values.shape != reference.shape or values.shape[0] < 2:
        return values.copy()
    best_shift = 0
    best_score = -np.inf
    for shift in range(values.shape[0]):
        shifted = np.roll(values, shift)
        score = float(np.dot(shifted, reference))
        if score > best_score:
            best_score = score
            best_shift = shift
    return np.roll(values, best_shift)
