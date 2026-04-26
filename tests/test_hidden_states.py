import numpy as np

from amgla import AMGLAConfig
from amgla.representation import TimesFMRepresentationExtractor
from amgla.timesfm_adapter import TimesFMForecast, front_pad_to_patch_multiple, normalize_observed_values


class MockHiddenAdapter:
    def encode_hidden(self, inputs) -> np.ndarray:
        rows = []
        for values in inputs:
            arr = np.asarray(values, dtype=float)
            rows.append([arr.mean(), arr[-1], arr.shape[0]])
        return np.asarray(rows, dtype=float)

    def forecast(self, inputs, horizon: int) -> TimesFMForecast:
        point = np.stack([np.repeat(np.asarray(values, dtype=float)[-1], horizon) for values in inputs])
        return TimesFMForecast(point, None)


def test_hidden_representation_uses_all_windows() -> None:
    config = AMGLAConfig(lookback_windows=(21, 14, 7))
    extractor = TimesFMRepresentationExtractor(config, MockHiddenAdapter())
    contexts = [np.arange(1, 22, dtype=float), np.arange(3, 24, dtype=float)]
    representation = extractor.transform(contexts)
    assert representation.shape == (2, 9)
    assert np.isfinite(representation).all()


def test_front_padding_marks_prefix_and_keeps_last_patch_valid() -> None:
    values, masks = front_pad_to_patch_multiple([np.arange(5, dtype=float), np.arange(35, dtype=float)], patch_len=32)
    assert values.shape == (2, 64)
    assert masks.shape == (2, 64)
    assert masks[0, :59].all()
    assert not masks[0, 59:].any()
    assert masks[1, :29].all()
    assert not masks[1, 29:].any()


def test_normalization_ignores_padding_values() -> None:
    values, masks = front_pad_to_patch_multiple([np.array([10.0, 11.0, 12.0])], patch_len=32)
    normalized = normalize_observed_values(values, masks)
    assert np.allclose(normalized[masks], 0.0)
    observed = normalized[~masks]
    assert abs(float(observed.mean())) < 1e-6
