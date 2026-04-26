"""Similarity retrieval for TimesFM representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalResult:
    indices: np.ndarray
    similarities: np.ndarray
    weights: np.ndarray


def cosine_topk(query: np.ndarray, reference: np.ndarray, k: int) -> RetrievalResult:
    q = _normalize(_as_2d(query, "query"))
    r = _normalize(_as_2d(reference, "reference"))
    if k < 1:
        raise ValueError("k must be positive.")
    if r.shape[0] == 0:
        raise ValueError("reference cannot be empty.")
    k_eff = min(k, r.shape[0])
    similarity = q @ r.T
    indices = np.argpartition(-similarity, kth=k_eff - 1, axis=1)[:, :k_eff]
    scores = np.take_along_axis(similarity, indices, axis=1)
    order = np.argsort(-scores, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    scores = np.take_along_axis(scores, order, axis=1)
    return RetrievalResult(indices=indices, similarities=scores, weights=softmax(scores))


def softmax(scores: np.ndarray) -> np.ndarray:
    stable = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(stable)
    denom = exp.sum(axis=1, keepdims=True)
    return exp / np.where(denom < 1e-12, 1.0, denom)


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms < 1e-12, 1.0, norms)


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (N, D).")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return matrix
