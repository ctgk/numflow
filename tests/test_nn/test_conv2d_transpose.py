import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x, w, strides, pad, shape', [
    (
        gd.Array(np.random.rand(1, 4, 4, 2), is_variable=True),
        gd.Array(np.random.rand(3, 3, 5, 2), is_variable=True),
        1,
        0,
        None,
    ),
    (
        gd.Array(np.random.rand(2, 4, 4, 2), is_variable=True),
        gd.Array(np.random.rand(3, 3, 5, 2), is_variable=True),
        2,
        0,
        (9, 9),
    ),
    (
        gd.Array(np.random.rand(2, 4, 4, 2), is_variable=True),
        gd.Array(np.random.rand(3, 3, 5, 2), is_variable=True),
        2,
        0,
        (10, 10),
    ),
    (
        gd.Array(np.random.rand(2, 4, 4, 2), is_variable=True),
        gd.Array(np.random.rand(3, 3, 5, 2), is_variable=True),
        2,
        1,
        (7, 7),
    ),
    (
        gd.Array(np.random.rand(2, 4, 4, 2), is_variable=True),
        gd.Array(np.random.rand(3, 3, 5, 2), is_variable=True),
        2,
        1,
        (8, 8),
    ),
])
def test_numerical_grad(x, w, strides, pad, shape):
    with gd.Graph() as g:
        gd.nn.conv2d_transpose(x, w, strides, pad, shape)
    g.backward()
    dx, dw = _numerical_grad(
        lambda a, b: gd.nn.conv2d_transpose(a, b, strides, pad, shape), x, w)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
    assert np.allclose(dw, w.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
