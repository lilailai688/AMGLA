import importlib.util
import os

import numpy as np
import pytest

from amgla import AMGLAConfig
from amgla.timesfm_adapter import TimesFMAdapter


@pytest.mark.skipif(
    importlib.util.find_spec("timesfm") is None
    or os.environ.get("AMGLA_RUN_TIMESFM_INTEGRATION") != "1",
    reason="timesfm integration test requires timesfm and AMGLA_RUN_TIMESFM_INTEGRATION=1",
)
def test_real_timesfm_adapter_hidden_smoke() -> None:
    adapter = TimesFMAdapter(AMGLAConfig(forecast_horizon=2, timesfm_max_horizon=2))
    hidden = adapter.encode_hidden([np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 3.0, 5.0])])
    assert hidden.ndim == 2
    assert hidden.shape[0] == 2
    assert np.isfinite(hidden).all()
