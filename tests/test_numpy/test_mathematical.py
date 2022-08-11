import numpy as np
import pytest

from tests.test_differentiation import (  # noqa:I202
    _test_egrad,
    _test_graph_backward,
    _test_graph_backward_custom_grad,
)


@pytest.fixture(params=[
    # https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions
    (np.cos, np.random.uniform(-10, 10, (3, 2))),
    (np.sin, np.random.uniform(-10, 10, (2, 5))),
    (np.tan, np.arctan(np.random.uniform(-2, 2, (4, 1)))),
    (np.arcsin, np.random.uniform(-0.9, 0.9, (3, 4))),
    (np.arccos, np.random.uniform(-1, 1, (3, 4))),
    (np.arctan, np.random.uniform(-10, 10, (5, 3))),
    (lambda a: np.hypot(a, 4), 3),
    (lambda a: np.hypot([2, 1], a), [[1], [-2]]),
    (np.hypot, (np.random.normal(size=(3,)), np.random.normal(size=(4, 1)))),
    (lambda a: np.arctan2(a, 1), np.random.uniform(-10, 10, (5, 3))),
    (np.arctan2, ([1, 2, 3], [3, -2, -1])),
    (np.degrees, np.random.uniform(-10, 10, (2, 3))),
    (np.radians, np.random.uniform(-1000, 1000, (5, 2))),
    (np.rad2deg, np.random.uniform(-10, 10, (4, 2))),
    (np.deg2rad, np.random.uniform(-1000, 1000, (3, 4))),

    # https://numpy.org/doc/stable/reference/routines.math.html#hyperbolic-functions
    (np.cosh, np.random.uniform(-10, 10, (3, 4))),
    (np.sinh, np.random.uniform(-10, 10, (1, 5))),
    (np.tanh, np.random.uniform(-10, 10, (4, 2))),
    (np.arcsinh, np.random.uniform(-10, 10, (4, 2, 3))),
    (np.arccosh, np.random.uniform(1, 10, (5, 2))),
    (np.arctanh, np.random.uniform(-0.9, 0.9, (2,))),

    # https://numpy.org/doc/stable/reference/routines.math.html#sums-products-differences
    (lambda a: np.prod(a), 1),
    (lambda a: a.prod(), [1, -1]),
    (lambda a: np.prod(a, 1), np.random.rand(2, 3, 2)),
    (lambda a: a.prod((0, 2), keepdims=True), np.random.rand(2, 3, 2)),
    (lambda a: np.sum(a), -1),
    (lambda a: np.sum(a), [-1, 1]),
    (lambda a: a.sum(axis=1), np.random.rand(3, 2)),
    (lambda a: np.sum(a, (0, 2), keepdims=True), np.random.rand(4, 2, 3)),
    (lambda a: np.nanprod(a), 1),
    (lambda a: np.nanprod(a), np.nan),
    (lambda a: np.nanprod(a), [np.nan, -1]),
    (lambda a: np.nanprod(a, 1), [[1, 2, np.nan], [np.nan, np.nan, np.nan]]),
    (
        lambda a: np.nanprod(a, 0, keepdims=True),
        [[1, 2, np.nan], [np.nan, np.nan, np.nan]],
    ),
    (lambda a: np.nanprod(a, (0, 2), keepdims=True), np.random.rand(2, 3, 2)),
    (lambda a: np.nansum(a), 1),
    (lambda a: np.nansum(a), np.nan),
    (lambda a: np.nansum(a), [np.nan, -1]),
    (lambda a: np.nansum(a, 1), [[1, 2, np.nan], [np.nan, np.nan, np.nan]]),
    (
        lambda a: np.nansum(a, 0, keepdims=True),
        [[1, 2, np.nan], [np.nan, np.nan, np.nan]],
    ),
    (lambda a: np.nansum(a, (0, 2), keepdims=True), np.random.rand(2, 3, 2)),
    (lambda a: np.cumprod(a), 1),
    (lambda a: a.cumprod(), [1, -1]),
    (lambda a: np.cumprod(a), np.random.rand(2, 3, 2)),
    (lambda a: a.cumprod(0), np.random.rand(2, 3, 2)),
    (lambda a: np.cumsum(a), 1),
    (lambda a: a.cumsum(), [1, -1]),
    (lambda a: np.cumsum(a), np.random.rand(2, 3, 2)),
    (lambda a: a.cumsum(0), np.random.rand(2, 3, 2)),
    (lambda a: np.nancumprod(a), 1),
    (lambda a: np.nancumprod(a), np.nan),
    (lambda a: np.nancumprod(a), [np.nan, -1]),
    (lambda a: np.nancumprod(a), [np.nan, np.nan]),
    (lambda a: np.nancumprod(a, 1), [[2, np.nan, 3], [np.nan, 4, np.nan]]),
    (lambda a: np.nancumprod(a), [[2, np.nan, 3], [np.nan, 4, np.nan]]),
    (lambda a: np.nancumsum(a), 1),
    (lambda a: np.nancumsum(a), np.nan),
    (lambda a: np.nancumsum(a), [np.nan, -1]),
    (lambda a: np.nancumsum(a), [np.nan, np.nan]),
    (lambda a: np.nancumsum(a, 1), [[2, np.nan, 3], [np.nan, 4, np.nan]]),
    (lambda a: np.nancumsum(a), [[2, np.nan, 3], [np.nan, 4, np.nan]]),
    (lambda a: np.ediff1d(a), [1, 2, 4, 7, 0]),
    (lambda a: np.ediff1d(a, to_end=99), [1, 2, 4, 7, 0]),
    (lambda a: np.ediff1d(a, to_begin=99), [1, 2, 4, 7, 0]),
    (lambda a: np.ediff1d(a, to_begin=-99, to_end=[88, 99]), [1, 2, 4, 7, 0]),
    (
        lambda a, to_begin, to_end: np.ediff1d(a, to_begin, to_end),
        ([1, 2, 4, 7, 0], np.eye(2), np.eye(3)),
    ),

    # https://numpy.org/doc/stable/reference/routines.math.html#exponents-and-logarithms
    (np.exp, [-1, -0.2, 0.5, 2]),
    (np.expm1, [-1, -0.2, 0.5, 2]),
    (np.exp2, [-1, -0.2, 0.5, 2]),
    (np.log, [1, 0.2, 0.5, 2]),
    (np.log10, [1, 0.2, 0.5, 2]),
    (np.log2, [1, 0.2, 0.5, 2]),
    (np.log1p, [1, 0.2, 0.5, 2, -0.9]),
    (lambda a: np.logaddexp(a, [1, 2]), np.random.rand(4, 2)),
    (lambda a: np.logaddexp([1, 2], a), np.random.rand(4, 2)),
    (
        np.logaddexp,
        (np.random.normal(size=(3, 4)), np.random.normal(size=(5, 1, 4))),
    ),
    (lambda a: np.logaddexp2(a, [1, 2]), np.random.rand(4, 2)),
    (lambda a: np.logaddexp2([1, 2], a), np.random.rand(4, 2)),
    (
        np.logaddexp2,
        (np.random.normal(size=(3, 4)), np.random.normal(size=(5, 1, 4))),
    ),


    # https://numpy.org/doc/stable/reference/routines.math.html#arithmetic-operations
    (lambda a: np.add(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a + [[1, 2], [3, 4]], [1, 2]),
    (lambda a, b: a + b, ([[1, 2]], [[1], [2]])),
    (np.add, ([[1, 2]], [[1], [2]])),
    (np.reciprocal, [1, -2]),
    (np.positive, -3),
    (lambda a: +a, -3),
    (np.negative, -3),
    (lambda a: -a, -3),
    (lambda a: np.multiply(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a * [[1, 2], [3, 4]], [1, 2]),
    (lambda a: np.float64(1) * a, [1, 2]),
    (lambda a, b: a * b, ([[1, 2]], [[1], [2]])),
    (np.multiply, ([[1, 2]], [[1], [2]])),
    (lambda a: np.divide(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a / [[1, 2], [3, 4]], [1, 2]),
    (lambda a, b: a / b, ([[1, 2]], [[1], [2]])),
    (np.divide, ([[1, 2]], [[1], [2]])),
    (np.true_divide, ([[1, 2]], [[1], [2]])),
    (lambda a: np.power(a, [[1], [-2]]), [[1, 2]]),
    (lambda a: a ** [[1], [-2]], [[1, 2]]),
    (np.power, ([[1, 2]], [[1], [-2]])),
    (lambda a: np.subtract(a, [[1, 2], [3, 4]]), [1, 2]),
    (lambda a: a - [[1, 2], [3, 4]], [1, 2]),
    (lambda a, b: a - b, ([[1, 2]], [[1], [2]])),
    (np.subtract, ([[1, 2]], [[1], [2]])),
    (np.true_divide, ([[1, 2]], [[1], [2]])),
    (lambda a: np.float_power(a, [[1], [-2]]), [[1, 2]]),
    (np.float_power, ([[1, 2]], [[1], [-2]])),
    (lambda a: np.fmod(a, 2), [-3, -2.5, -1, 1, 2.5, 3]),
    (lambda b: np.fmod([-3, -2.5, -1, 1, 2.5, 3], b), 2),
    (lambda a, b: np.fmod(a, b), ([-3, -2.5, -1, 1, 2.5, 3], 2)),
    (lambda a: np.mod(a, 2), [-3, -2.5, -1, 1, 2.5, 3]),
    (lambda b: np.mod([-3, -2.5, -1, 1, 2.5, 3], b), 2),
    (lambda a, b: np.mod(a, b), ([-3, -2.5, -1, 1, 2.5, 3], 2)),
    (lambda a: np.remainder(a, 2), [-3, -2.5, -1, 1, 2.5, 3]),
    (lambda b: np.remainder([-3, -2.5, -1, 1, 2.5, 3], b), 2),
    (lambda a, b: np.remainder(a, b), ([-3, -2.5, -1, 1, 2.5, 3], 2)),

    # https://numpy.org/doc/stable/reference/routines.math.html#extrema-finding
    (np.maximum, (3, -1)),
    (np.maximum, (0.5, np.random.rand(3, 2))),
    (np.maximum, (np.random.rand(4, 3), 0.5)),
    (np.maximum, (np.random.rand(2, 3, 4), np.random.rand(1, 4))),
    (np.fmax, (np.nan, 3)),
    (np.fmax, (3, np.nan)),
    (np.fmax, ([1, np.nan, -1], [[-0.5], [0.5]])),
    (np.fmax, ([1, np.nan, -1], [[-0.5], [np.nan]])),
    (np.amax, 9),
    (np.amax, [1, 2]),
    (np.max, 9),
    (np.max, [1, 2]),
    (lambda a: a.max(axis=1), np.random.rand(2, 3) * 10),
    (lambda a: a.max(axis=(0, 2), keepdims=True), np.random.rand(2, 4, 3)),
    (lambda a: np.nanmax(a), np.nan),
    (lambda a: np.nanmax(a), [np.nan, 1]),
    (lambda a: np.nanmax(a, axis=0, keepdims=True), [np.nan, 1]),
    (np.minimum, (3, -1)),
    (np.minimum, (0.5, np.random.rand(3, 2))),
    (np.minimum, (np.random.rand(4, 3), 0.5)),
    (np.minimum, (np.random.rand(2, 3, 4), np.random.rand(1, 4))),
    (np.fmin, (np.nan, 3)),
    (np.fmin, (3, np.nan)),
    (np.fmin, ([1, np.nan, -1], [[-0.5], [0.5]])),
    (np.fmin, ([1, np.nan, -1], [[-0.5], [np.nan]])),
    (np.amin, 9),
    (np.amin, [1, 2]),
    (np.min, 9),
    (np.min, [1, 2]),
    (lambda a: a.min(axis=1), np.random.rand(2, 3) * 10),
    (lambda a: a.min(axis=(0, 2), keepdims=True), np.random.rand(2, 4, 3)),
    (lambda a: np.nanmin(a), np.nan),
    (lambda a: np.nanmin(a), [np.nan, 1]),
    (lambda a: np.nanmin(a, axis=0, keepdims=True), [np.nan, 1]),

    # https://numpy.org/doc/stable/reference/routines.math.html#miscellaneous
    (lambda a: np.convolve(a, [0, 1, 0.5], mode='full'), [1, 2, 3]),
    (lambda a: np.convolve(a, [0, 1, 0.5], mode='same'), [1, 2, 3]),
    (lambda a: np.convolve(a, [0, 1, 0.5], mode='valid'), [1, 2, 3]),
    (lambda a, v: np.convolve(a, v, mode='full'), ([1, 2, 3], [0, 1, 0.5])),
    (lambda a, v: np.convolve(a, v, mode='same'), ([1, 2, 3], [0, 1, 0.5])),
    (lambda a, v: np.convolve(a, v, mode='valid'), ([1, 2, 3], [0, 1, 0.5])),
    (lambda a, v: np.convolve(a, v, mode='valid'), ([1, 2, 3, 4], [0, 1, -1])),
    (lambda a, v: np.convolve(a, v, mode='full'), ([1, 2, 3], [1, 0.5])),
    (lambda a, v: np.convolve(a, v, mode='same'), ([1, 2, 3], [1, 0.5])),
    (lambda a, v: np.convolve(a, v, mode='valid'), ([1, 2, 3], [1, 0.5])),
    (lambda a: a.clip(4.5), np.arange(10)),
    (lambda a: a.clip(np.arange(20).reshape(2, 10) - 0.1), np.random.rand(10)),
    (lambda a: a.clip(max=7.5), np.arange(10)),
    (lambda a, b, c: a.clip(b, c), (np.arange(10), 1.2, 8.8)),
    (lambda a, b, c: a.clip(min=b, max=c), (np.arange(10), 7.7, 2.1)),
    (lambda a: np.clip(a, 2.8, None), np.arange(10)),
    (lambda a: np.clip(a, None, 8.2), np.arange(10)),
    (lambda a, b: np.clip(range(10), a, b), (1.5, 4.4)),
    (lambda a, b, c: np.clip(a, b, c), (np.arange(10), 8.2, 1.8)),
    (np.sqrt, [3, 0.5]),
    (np.cbrt, [3, 0.5]),
    (np.square, [2, -1]),
    (lambda a: abs(a), [2, -1]),
    (np.abs, [2, -1]),
    (np.absolute, [2, -1]),
    (np.fabs, [2, -1]),
    (np.nan_to_num, 1),
    (np.nan_to_num, np.nan),
    (np.nan_to_num, [1, np.nan]),
])
def parameters(request):
    return request.param


def test_graph_backward(parameters):
    f = parameters[0]
    args = parameters[1] if isinstance(
        parameters[1], tuple) else (parameters[1],)
    _test_egrad(f, *args)
    _test_graph_backward(f, *args)
    _test_graph_backward_custom_grad(f, *args)


if __name__ == '__main__':
    pytest.main([__file__])
