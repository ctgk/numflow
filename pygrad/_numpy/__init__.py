import numpy as np

from pygrad._decorators import register_gradient
from pygrad._utils._unbroadcast import _unbroadcast_to


@register_gradient(np.add)
def _add_gradient(doutput, output, x, y):
    return (
        _unbroadcast_to(doutput, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(doutput, y.shape) if hasattr(y, 'shape') else None,
    )


@register_gradient(np.square)
def _square_gradient(dy, y, x):
    return 2 * x * dy


@register_gradient(np.cos)
def _cos_gradient(doutput, output, x):
    return -np.sin(x) * doutput


@register_gradient(np.sin)
def _sin_gradient(doutput, output, x):
    return np.cos(x) * doutput


@register_gradient(np.tan)
def _tan_gradient(doutput, output, x):
    return (1 + np.square(output)) * doutput