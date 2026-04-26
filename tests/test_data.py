import pandas as pd
import pytest

from amgla.data import build_reference_panel, build_target_panel, validate_sales_frame


def test_validate_sales_frame_requires_schema() -> None:
    with pytest.raises(ValueError):
        validate_sales_frame(pd.DataFrame({"product_id": ["a"]}))


def test_build_panels() -> None:
    dates = pd.date_range("2026-01-01", periods=30)
    rows = []
    for product in ["a", "b"]:
        for i, date in enumerate(dates):
            rows.append({"product_id": product, "date": date, "sales": float(i + 1)})
    frame = pd.DataFrame(rows)
    reference = build_reference_panel(frame, history_length=21, horizon=7)
    target = build_target_panel(frame.groupby("product_id").head(5), history_length=21)
    assert reference.contexts.shape == (2, 21)
    assert reference.futures.shape == (2, 7)
    assert target.contexts.shape == (2, 21)
