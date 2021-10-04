import numpy as np

from pygrad._core._array import Array
from pygrad._math._log import log
from pygrad._math._square import square
from pygrad.random._normal import normal
from pygrad.stats._statistics import Statistics


_ln2pi_hf = 0.5 * np.log(2 * np.pi)


class Normal(Statistics):
    """Statistics of a normal distribution."""

    def __init__(self, loc: Array, scale: Array):
        """Initialize the statistics object.

        Parameters
        ----------
        loc : Array
            Location parameter.
        scale : Array
            Scale parameter.
        """
        super().__init__()
        self._loc = loc
        self._scale = scale

    @property
    def loc(self) -> Array:
        """Return location parameter of the statistics.

        Returns
        -------
        Array
            Location parameter.
        """
        return self._loc

    @property
    def scale(self) -> Array:
        """Return scale parameter of the statistics.

        Returns
        -------
        Array
            Scale parameter.
        """
        return self._scale

    def logpdf(self, x) -> Array:
        """Return logarithm of pdf.

        Parameters
        ----------
        x : Array
            Observed data

        Returns
        -------
        Array
            Logarithm of pdf.
        """
        return (
            -0.5 * (square((x - self._loc) / self._scale))
            - log(self._scale) - _ln2pi_hf)

    def sample(self) -> Array:
        """Return random sample.

        Returns
        -------
        Array
            Random sample.
        """
        return normal(self._loc, self._scale)
