from collections import namedtuple
import typing as tp

import numpy as np
import numpy
import scipy.special  # noqa: F401

from pygrad._config import config
from pygrad._decorators import _PATCHED_FUNCTION, _REGISTERED_GRADIENT_FUNCTION
from pygrad._variable import _ndarray_views, Variable


Node = namedtuple('Node', ('result', 'function', 'inputs', 'kwargs'))


class Graph(object):
    """Computational graph."""

    def __init__(self):
        """Construct computational graph."""
        super().__init__()
        self._node_list: tp.List[Node] = []

    def __enter__(self) -> 'Graph':
        """Return new computation graph to construct.

        Returns
        -------
        Graph
            New computational graph to construct.

        Raises
        ------
        ValueError
            Another graph is under construction.
        """
        if config._graph is not None:
            raise ValueError('There is already a graph under construction')
        config._graph = self
        for original, patched in _PATCHED_FUNCTION.items():
            setattr(
                eval('.'.join(
                    m for m in original.__module__.split('.')
                    if not m.startswith('_'))),
                original.__name__, patched,
            )
        return self

    def __exit__(self, *args, **kwargs):
        """Exit from the graph under construction."""
        config._graph = None
        for original in _PATCHED_FUNCTION.keys():
            setattr(
                eval('.'.join(
                    m for m in original.__module__.split('.')
                    if not m.startswith('_'))),
                original.__name__, original,
            )

    def _add_node(self, result, function, *inputs, **kwargs):
        if any(result is node.result for node in self._node_list):
            raise ValueError('The result already exists in the graph')
        self._node_list.append(Node(result, function, inputs, kwargs))

    def gradient(
        self,
        target: Variable,
        sources: tp.Union[tp.List[Variable], tp.Tuple[Variable, ...]],
    ) -> tp.Tuple[np.ndarray]:
        """Return gradients of target with respect to each source.

        Parameters
        ----------
        target : Variable
            Target to be differentiated.
        sources : tp.Union[tp.List[Variable], tp.Tuple[Variable, ...]]
            Source tensors to differentiated against.
        Returns
        -------
        tp.Tuple[np.ndarray]
            Gradients of target with respect to each source.
        """
        tensor_id_to_grad: tp.Dict[int, np.ndarray] = {}
        tensor_id_to_grad[id(target)] = np.ones_like(target.view(np.ndarray))
        for node in reversed(self._node_list):
            if id(node.result) not in tensor_id_to_grad:
                continue
            if node.function not in _REGISTERED_GRADIENT_FUNCTION:
                raise NotImplementedError(
                    f'Gradient of {node.function} is not registered yet.')
            dargs = _REGISTERED_GRADIENT_FUNCTION[node.function](
                tensor_id_to_grad[id(node.result)],
                node.result.view(np.ndarray),
                *_ndarray_views(*node.inputs),
                **node.kwargs,
            )
            if not isinstance(dargs, tuple):
                dargs = (dargs,)
            for x, dx in zip(node.inputs, dargs):
                if dx is None:
                    continue
                if id(x) in tensor_id_to_grad:
                    tensor_id_to_grad[id(x)] += dx
                elif isinstance(x, Variable):
                    tensor_id_to_grad[id(x)] = np.ones_like(
                        x.view(np.ndarray)) * dx
        return tuple(tensor_id_to_grad.get(id(s), None) for s in sources)
