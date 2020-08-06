from pygrad._array import Array
from pygrad._errors import DifferentiationError
from pygrad._types import (
    DataType, Int8, Int16, Int32, Int64, Float16, Float32, Float64, Float128
)

from pygrad._manipulation._reshape import reshape
from pygrad._manipulation._transpose import transpose

from pygrad._math._add import add
from pygrad._math._divide import divide
from pygrad._math._exp import exp
from pygrad._math._log import log
from pygrad._math._matmul import matmul
from pygrad._math._mean import mean
from pygrad._math._multiply import multiply
from pygrad._math._negate import negate
from pygrad._math._sqrt import sqrt
from pygrad._math._square import square
from pygrad._math._subtract import subtract
from pygrad._math._sum import sum


def _reshape(x, *newshape):
    return reshape(x, newshape)


def _transpose(x, *axes):
    return transpose(x, axes) if axes else transpose(x)


Array.__add__ = add
Array.__matmul__ = matmul
Array.__mul__ = multiply
Array.__neg__ = negate
Array.__sub__ = subtract
Array.__truediv__ = divide
Array.__radd__ = add
Array.__rmatmul__ = lambda x, y: matmul(y, x)
Array.__rmul__ = multiply
Array.__rsub__ = lambda x, y: subtract(y, x)
Array.__rtruediv__ = lambda x, y: divide(y, x)
Array.reshape = _reshape
Array.mean = mean
Array.sum = sum
Array.transpose = _transpose
Array.T = property(lambda self: transpose(self))


_classes = [
    Array,
    DifferentiationError,
    DataType, Int8, Int16, Int32, Int64, Float16, Float32, Float64, Float128,
]

for _cls in _classes:
    _cls.__module__ = 'pygrad'


__all__ = [_cls.__name__ for _cls in _classes] + [
    'reshape',
    'transpose',

    'add',
    'divide',
    'exp',
    'log',
    'matmul',
    'mean',
    'multiply',
    'negate',
    'sqrt',
    'square',
    'subtract',
    'sum',
]