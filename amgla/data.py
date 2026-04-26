"""Input validation and panel shaping utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("product_id", "date", "sales")


@dataclass(frozen=True)
class ProductPanel:
    product_ids: list[str]
    contexts: np.ndarray
    futures: np.ndarray | None = None
    last_dates: list[pd.Timestamp] | None = None
    observed_contexts: list[np.ndarray] | None = None


def load_sales_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_sales_frame(frame)


def validate_sales_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    result = frame.copy()
    result["product_id"] = result["product_id"].astype(str)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["sales"] = pd.to_numeric(result["sales"], errors="raise").astype(float)
    if result["product_id"].isna().any():
        raise ValueError("product_id contains missing values.")
    if not np.isfinite(result["sales"].to_numpy()).all():
        raise ValueError("sales contains NaN or infinite values.")
    return result.sort_values(["product_id", "date"]).reset_index(drop=True)


def build_reference_panel(frame: pd.DataFrame, history_length: int, horizon: int) -> ProductPanel:
    validated = validate_sales_frame(frame)
    product_ids: list[str] = []
    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    observed_contexts: list[np.ndarray] = []

    required_length = history_length + horizon
    for product_id, group in validated.groupby("product_id", sort=False):
        values = group["sales"].to_numpy(dtype=float)
        if values.shape[0] < required_length:
            continue
        tail = values[-required_length:]
        context = tail[:history_length]
        product_ids.append(str(product_id))
        contexts.append(context)
        observed_contexts.append(context.copy())
        futures.append(tail[history_length:])

    if not contexts:
        raise ValueError(
            "history data must contain at least one product with "
            f"{required_length} observations."
        )
    return ProductPanel(
        product_ids=product_ids,
        contexts=np.stack(contexts, axis=0),
        futures=np.stack(futures, axis=0),
        observed_contexts=observed_contexts,
    )


def build_target_panel(frame: pd.DataFrame, history_length: int) -> ProductPanel:
    validated = validate_sales_frame(frame)
    product_ids: list[str] = []
    contexts: list[np.ndarray] = []
    last_dates: list[pd.Timestamp] = []
    observed_contexts: list[np.ndarray] = []

    for product_id, group in validated.groupby("product_id", sort=False):
        values = group["sales"].to_numpy(dtype=float)
        if values.shape[0] == 0:
            continue
        product_ids.append(str(product_id))
        observed = values[-history_length:].copy()
        contexts.append(left_pad(observed, history_length))
        observed_contexts.append(observed)
        last_dates.append(pd.Timestamp(group["date"].iloc[-1]))

    if not contexts:
        raise ValueError("target data must contain at least one observed product.")
    return ProductPanel(
        product_ids=product_ids,
        contexts=np.stack(contexts, axis=0),
        futures=None,
        last_dates=last_dates,
        observed_contexts=observed_contexts,
    )


def left_pad(values: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if arr.shape[0] >= length:
        return arr[-length:].copy()
    if arr.shape[0] == 0:
        raise ValueError("cannot pad an empty series.")
    pad = np.repeat(arr[:1], length - arr.shape[0])
    return np.concatenate([pad, arr])


def future_dates(last_date: pd.Timestamp, horizon: int) -> list[pd.Timestamp]:
    return [last_date + pd.Timedelta(days=step) for step in range(1, horizon + 1)]
