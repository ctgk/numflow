import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, name, expected', [
    ([1, -1, 5], 'negate', [-1, 1, -5]),
])
def test_forward(x, name, expected):
    actual = pg.negate(x, name=name)
    assert np.allclose(actual.value, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, dy, expected', [
    (pg.Array([1., -1, 5], is_differentiable=True), None, [-1, -1, -1]),
    (pg.Array([-7., 3], is_differentiable=True), np.array([1., -2]), [-1, 2]),
])
def test_backward(x, dy, expected):
    if dy is None:
        pg.negate(x).backward()
    else:
        pg.negate(x).backward(_grad=dy)
    assert np.allclose(x.grad, expected)


@pytest.mark.parametrize('x', [
    pg.Array(np.random.rand(2, 3), is_differentiable=True),
    pg.Array(np.random.rand(4, 2, 3), is_differentiable=True),
])
def test_numerical_grad(x):
    pg.negate(x).backward()
    dx = _numerical_grad(pg.negate, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])