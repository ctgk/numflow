import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, axis, keepdims, name, error', [
    ([1, 2, 3], 0, True, 'mean', 'NoError'),
    ([1, 2, 3], 'a', True, 'mean', TypeError),
    ([1, 2, 3], 0, 1, 'mean', TypeError),
    ([1, 2, 3], 0, True, 1, TypeError),
    ([1, 2, 3], 0, True, 's,m', ValueError),
])
def test_forward_error(x, axis, keepdims, name, error):
    if error == 'NoError':
        gd.mean(x, axis, keepdims, name=name)
    else:
        with pytest.raises(error):
            gd.mean(x, axis, keepdims, name=name)


@pytest.mark.parametrize('x, axis, keepdims, name, expected', [
    ([1, 3, 0, -2], None, True, 'mean', np.array([0.5])),
])
def test_forward(x, axis, keepdims, name, expected):
    actual = gd.mean(x, axis, keepdims, name=name)
    assert np.allclose(actual.data, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, axis, keepdims, dy, expected', [
    (
        gd.Array([1., 3, -2], is_variable=True),
        None, True, np.array([2]), np.array([2 / 3] * 3)
    ),
    (
        gd.Array([[2., 1], [-2, 5]], is_variable=True),
        0, False, np.array([1, 2]),
        np.array([[1, 2], [1, 2]]) * 0.5,
    ),
    (
        gd.Array([[2., 1], [-2, 5]], is_variable=True),
        1, False, np.array([1, 2]),
        np.array([[1, 1], [2, 2]]) * 0.5,
    ),
    (
        gd.Array([[2., 1], [-2, 5]], is_variable=True),
        1, True, np.array([[1], [2]]),
        np.array([[1, 1], [2, 2]]) * 0.5,
    ),
])
def test_backward(x, axis, keepdims, dy, expected):
    y = gd.mean(x, axis, keepdims)
    y.backward(_grad=dy)
    assert np.allclose(x.grad, expected)


@pytest.mark.parametrize('x, axis, keepdims', [
    (gd.Array(np.random.rand(2, 3), is_variable=True), 1, False),
    (gd.Array(np.random.rand(4, 2, 3), is_variable=True), 0, True),
])
def test_numerical_grad(x, axis, keepdims):
    x.mean().backward()
    dx = _numerical_grad(gd.mean, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
