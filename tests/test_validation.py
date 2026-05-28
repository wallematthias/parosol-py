import numpy as np
import pytest

from parosol_py.validation import compare_field


def test_compare_field_passes_within_tolerance():
    result = compare_field(
        np.array([1, 2, 3]),
        np.array([1, 2.001, 2.999]),
        rtol=1e-2,
        atol=1e-3,
    )

    assert result.passed is True
    assert result.max_abs_error == pytest.approx(0.001)


def test_compare_field_fails_outside_tolerance():
    result = compare_field(
        np.array([1, 2, 3]),
        np.array([1, 2.5, 3]),
        rtol=1e-3,
        atol=1e-6,
    )

    assert result.passed is False
    assert result.max_abs_error == pytest.approx(0.5)
