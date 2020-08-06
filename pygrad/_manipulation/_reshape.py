import typing as tp

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args


class _Reshape(_Operator):

    def __init__(self, x, newshape: tp.Iterable[int], name: str = None):
        super().__init__(x, name=name)
        self._newshape = newshape

    def _forward_numpy(self, x):
        return x.reshape(*self._newshape)

    def _backward_numpy(self, delta, x):
        return delta.reshape(*x.shape)


@_typecheck_args
def reshape(x, newshape: tp.Iterable[int], *, name: str = None) -> Array:
    """Return a reshaped array.

    Parameters
    ----------
    x
        Input array.
    newshape : tp.Iterable[int]
        New shape
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Reshaped array.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.reshape([1, 2, 3, 4, 5, 6], (2, -1))
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    return _Reshape(x, newshape, name=name).forward()
