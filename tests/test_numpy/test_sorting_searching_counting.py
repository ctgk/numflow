import numpy as np
import pytest

from tests.test_differentiation import (  # noqa:I202
    _test_egrad,
    _test_graph_backward,
    _test_graph_backward_custom_grad,
)


@pytest.fixture(params=[
    (lambda a: np.sort(a), [[3, 2, 1], [4, 6, -1]]),
    (lambda a: np.sort(a, axis=0), [[3, 2, 1], [4, 6, -1]]),
    (lambda a: np.sort(a, axis=None), [[3, 2, 1], [4, 6, -1]]),
])
def parameters(request):
    return request.param


def test_differentiation(parameters):
    f = parameters[0]
    args = parameters[1] if isinstance(
        parameters[1], tuple) else (parameters[1],)
    _test_graph_backward(f, *args)
    _test_graph_backward_custom_grad(f, *args)
    _test_egrad(f, *args)


if __name__ == '__main__':
    pytest.main([__file__])
