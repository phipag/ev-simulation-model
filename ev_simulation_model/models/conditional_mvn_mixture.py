from typing import List, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import multivariate_normal

from ev_simulation_model.models.multivariate_normal import MultivariateNormal


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

    def sample_cond(self, ind: int, z: Union[float, int, npt.ArrayLike], n_samples: int = 1) -> npt.NDArray[np.float_]:
        if isinstance(z, (float, int)):
            z = np.array([z])
        else:
            z = np.array(z)
        if self._mvns[0].means is not None and z.shape != self._mvns[0].means[1 - ind].shape:
            raise ValueError(f"z has invalid shape. Expected {self._mvns[0].means[1 - ind].shape}.")

        cond_weights = self.calc_cond_weights(ind, z)
        # Number of samples per component based on the conditional mixture weights
        rng = np.random.default_rng()
        n_samples_comp = rng.multinomial(n_samples, cond_weights)

        return np.vstack(
            [
                # *mvn.cond_dist(ind, z) unpacks the mean, covariance tuple
                rng.multivariate_normal(*mvn.cond_dist(ind, z), int(n_samples_comp))
                for (mvn, n_samples_comp) in zip(self._mvns, n_samples_comp)
            ]
        )

    def sample_marg(self, ind: int, n_samples: int = 1) -> npt.NDArray[np.float_]:
        # Number of samples per component based on the mvn mixture weights
        rng = np.random.default_rng()
        n_samples_comp = rng.multinomial(n_samples, self._weights)

        return np.vstack(
            [
                # *mvn.marg_dist(ind) unpacks the mean, covariance tuple
                rng.multivariate_normal(*mvn.marg_dist(ind), int(n_samples_comp))
                for (mvn, n_samples_comp) in zip(self._mvns, n_samples_comp)
            ]
        )
