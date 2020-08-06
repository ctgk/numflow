import numpy as np

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args


class _Sinh(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        return np.sinh(x)

    def _backward_numpy(self, dy, x):
        return dy * np.cosh(x)


@_typecheck_args
def sinh(x, *, name: str = None) -> Array:
    r"""Return hyperbolic sine of each element.

    .. math::
        \sinh x &= {e^{x} - e^{-x}\over 2}

        {\partial\over\partial x}\sinh x &= \cosh x

    Parameters
    ----------
    x
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Hyperbolic sine of each element

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.sinh([0, 1, 2])
    array([0.        , 1.17520119, 3.62686041])
    """
    return _Sinh(x, name=name).forward()
