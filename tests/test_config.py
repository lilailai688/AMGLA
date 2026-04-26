import pytest

from amgla import AMGLAConfig


def test_default_config_is_valid() -> None:
    AMGLAConfig().validate()


def test_invalid_blend_weight_fails() -> None:
    with pytest.raises(ValueError):
        AMGLAConfig(blend_weight=1.5).validate()
