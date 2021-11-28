"""Statistics module."""

from pygrad.stats._bernoulli import Bernoulli
from pygrad.stats._categorical import Categorical
from pygrad.stats._exponential import Exponential
from pygrad.stats._log_softmax import log_softmax
from pygrad.stats._normal import Normal
from pygrad.stats._relaxed_bernoulli import RelaxedBernoulli
from pygrad.stats._relaxed_categorical import RelaxedCategorical
from pygrad.stats._sigmoid import sigmoid
from pygrad.stats._sigmoid_cross_entropy import sigmoid_cross_entropy
from pygrad.stats._softmax import softmax
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy
from pygrad.stats._sparse_softmax_cross_entropy import (
    sparse_softmax_cross_entropy,
)
from pygrad.stats._statistics import Statistics


_classes = [
    Bernoulli,
    Categorical,
    Exponential,
    Normal,
    RelaxedBernoulli,
    RelaxedCategorical,
    Statistics,
]


for _cls in _classes:
    _cls.__module__ = __name__


_functions = [
    log_softmax,
    sigmoid,
    sigmoid_cross_entropy,
    softmax,
    softmax_cross_entropy,
    sparse_softmax_cross_entropy,
]


__all__ = (
    [_cls.__name__ for _cls in _classes]
    + [_func.__name__ for _func in _functions]
)
