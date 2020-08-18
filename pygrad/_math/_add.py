import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


class _Add(_Operator):

    def __init__(self, x: Array, y: Array, name: str = None):
        super().__init__(x, y, name=name)

    @staticmethod
    def _forward_numpy(x, y):
        return x + y

    @staticmethod
    def _backward_numpy(delta: np.ndarray, x: np.ndarray, y: np.ndarray):
        return _unbroadcast_to(delta, x.shape), _unbroadcast_to(delta, y.shape)


@_typecheck(exclude=('x', 'y'))
def add(x: Array, y: Array, *, name: str = None) -> Array:
    """Return element-wise addition of two arrays.

    Parameters
    ----------
    x : Array
        Input array.
    y : Array
        Another input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Element-wise addition of two arrays.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.add([[1, 2], [2, 3]], [-1, 3])
    array([[0., 5.],
           [1., 6.]])
    """
    return _Add(x, y, name=name).forward()
