from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal

from ev_simulation_model.models.multivariate_normal import MultivariateNormal


def _select_mixture_index(weights: npt.NDArray[np.float_]) -> int:
    rng = np.random.default_rng()
    # We draw once from the multinomial distribution with the given weights.
    # This returns an array like [0, 0, 1, 0, 0].
    draws = rng.multinomial(1, weights)
    # Returns an array with the indices where the condition is true. Exactly one entry is always one because we draw
    # one time from the multinomial distribution. For the example above this would be [2].
    mixture_index: npt.NDArray[np.int_] = np.flatnonzero(draws == 1)

    return mixture_index[0]  # type: ignore


class ConditionalMvnMixture:
    """Class to represent a mixture of multiple multivariate normal distributions.
    Allows computation and sampling of a conditional mixtures.
    """

    def __init__(self, mvns: List[MultivariateNormal], weights: npt.ArrayLike):
        """Initializes a ConditionalMvnMixture of weighted MultivariateNormal instances.

        :param mvns: List of MultivariateNormal instances.
        :param weights: Vector of weights for each MultivariateNormal instance. Must sum up to one.
        """
        if not mvns:
            raise ValueError("Must be given at least one MultivariateNormal instance.")
        if not all(isinstance(mvn, MultivariateNormal) for mvn in mvns):
            raise ValueError(f"Please provide only {MultivariateNormal.__class__} instances for the mvns argument.")
        dim = mvns[0].mean.shape[0]
        if not all(mvn.mean.shape[0] == dim for mvn in mvns):
            raise ValueError(f"All MultivariateNormal instances have to have the same dimension. Detected dim={dim}.")

        self._mvns = mvns
        self._weights: npt.NDArray[np.float_] = np.array(weights)

        if not len(self._mvns) == len(self._weights):
            raise ValueError("Number of MultivariateNormal objects must equal the number of weights provided.")
        if not np.isclose(self._weights.sum(), 1):
            raise ValueError("Provided weights for mixture components are invalid. Must sum up to one.")

    @property
    def mvns(self) -> List[MultivariateNormal]:
        return self._mvns

    @property
    def weights(self) -> npt.NDArray[np.float_]:
        return self._weights

    def partition(self, k: int) -> None:
        """Partitions all MVNs at index k.

        Refer to :func:`ev_simulation_model.models.multivariate_normal.MultivariateNormal.partition` for details.

        :param k: Partition index
        """
        for mvn in self._mvns:
            mvn.partition(k)

    def calc_cond_weights(self, ind: int, z: Union[float, int, npt.NDArray[np.float_]]) -> npt.NDArray[np.float_]:
        """Calculates the conditional mixture weights using Bayes' rule for the partitioned MVNs at index ind given
        the conditional z.

        :param ind: Index (either zero or one) to specify which weights for which conditional should be calculated.
        :param z: Conditional value or value vector. E.g. z = {x2, x3} if calculating p(x1 | x2, x3) where ind=0.
        :return: Numpy array of new conditional weights.
        """
        den_components = np.zeros(len(self._mvns))
        for i in range(len(self._mvns)):
            marg_mean, marg_cov = self._mvns[i].marg_dist(1 - ind)
            den_components[i] = multivariate_normal(marg_mean, marg_cov).pdf(z) * self._weights[i]

        den = np.sum(den_components)

        cond_weights = np.zeros(len(self._mvns))
        for i in range(len(self._mvns)):
            cond_weights[i] = den_components[i] / den

        return cond_weights

    def cond_dist(
        self, ind: int, z: Union[float, int, npt.ArrayLike]
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Calculates new mixture weights using Bayes' rules and selects a component according to the new conditional
        weights. Return the conditional distribution for this selected component.

        This method should be called each time a new sample should be drawn to make sure that samples are drawn from
        different mixture components according to the conditional weights.

        Calculation of conditional distribution is according to
        :func:`ev_simulation_model.models.multivariate_normal.MultivariateNormal.cond_dist`.

        :param ind: Index (either zero or one) to specify which conditional should be calculated.
        :param z: Conditional value or value vector. E.g. x2, x3 if calculating p(x1 | x2, x3) where ind=0.
        :return: Tuple of mean vectors and variance-covariances matrices as numpy arrays.
        """
        if isinstance(z, float) or isinstance(z, int):
            z = np.array([z])
        else:
            z = np.array(z)
        if self._mvns[0].means is not None and z.shape != self._mvns[0].means[1 - ind].shape:
            raise ValueError(f"z has invalid shape. Expected {self._mvns[0].means[1 - ind].shape}.")

        cond_weights = self.calc_cond_weights(ind, z)
        mixture_index = _select_mixture_index(cond_weights)
        mixture_component = self._mvns[mixture_index]

        return mixture_component.cond_dist(ind, z)
