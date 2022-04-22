import numpy as np
import pytest

import pygrad as gd


def test_enter_exit():
    assert gd.config.graph is None
    with gd.Graph() as g:
        assert gd.config.graph is g
    assert gd.config.graph is None


def test_enter_exit_error():
    with gd.Graph():
        with pytest.raises(ValueError):
            with gd.Graph():
                pass


def test_automatic_operation_storing():
    a = gd.Tensor(-1., is_variable=True)
    with gd.Graph() as g:
        b = gd.square(a)
    assert len(g._operations) == 1
    assert g._operations[0]._child is b
    assert g._operations[0]._args[0] is a


def test_no_automatic_operation_storing():
    a = gd.Tensor(-1)
    with gd.Graph() as g:
        gd.square(a)
    assert len(g._operations) == 0


def test_gradient():
    a = gd.Tensor(-1., is_variable=True)
    with gd.Graph() as g:
        b = gd.square(a)
    grads = g.gradient(b, [a])
    assert len(grads) == 1
    assert np.allclose(grads[0], -2)


def test_gradient_2():
    a = gd.Tensor(-1, is_variable=True)
    b = gd.Tensor(1)
    with gd.Graph() as g:
        c = a + b
    grads = g.gradient(c, [a, b])
    assert len(grads) == 2
    assert np.allclose(grads[0], 1)
    assert grads[1] is None


def test_gradient_error():
    a = gd.Tensor(-1)
    with gd.Graph() as g:
        b = gd.square(a)
    with pytest.raises(gd.DifferentiationError):
        g.gradient(b, [a])


if __name__ == '__main__':
    pytest.main([__file__])
