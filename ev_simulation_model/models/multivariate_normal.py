from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


class MultivariateNormal:
    """Class to represent a multivariate normal distribution. Allows computation
    of a conditional normal distribution.
    """

    def __init__(self, mean: npt.ArrayLike, cov: npt.ArrayLike) -> None:
        """
        Initializes a MultivariateNormal object with mean vector mean and covariance matrix cov.

        :param mean: Mean vector.
        :param cov: Variance-Covariance matrix.
        """
        self.mean: npt.NDArray[np.float_] = np.array(mean)
        self.cov: npt.NDArray[np.float_] = np.atleast_2d(cov)

        if self.mean.shape[0] != self.cov.shape[0] or self.mean.shape[0] != self.cov.shape[1]:
            raise ValueError(
                f"Shape of mean and covariance incompatible. Got covariance of shape {self.cov.shape} but expected "
                f"shape ({self.mean.shape[0]}, {self.mean.shape[0]})."
            )

        self.means: Optional[List[npt.NDArray[np.float_]]] = None
        self.covs: Optional[List[List[npt.NDArray[np.float_]]]] = None
        self.betas: Optional[List[npt.NDArray[np.float_]]] = None

    def partition(self, k: int) -> None:
        """Given k, partition the random vector z into a size k vector z1
        and a size N-k vector z2.

        For the mean vector and the cov matrix this works as follows:
        Partition the mean vector mean into mean1 and mean2
        and the covariance matrix cov into cov11, cov12, cov21, cov22
        correspondingly.

        Compute the regression coefficients beta1 and beta2
        using the partitioned arrays. The regression coefficients (betas)
        will be used to calculate the conditional distribution z1 | z2 or
        z2 | z1.

        :param k: Partition index
        """
        if not (1 <= k < self.mean.shape[0]):
            raise ValueError(f"k must be between 1 and {self.mean.shape[0] - 1}.")

        self.means = [self.mean[:k], self.mean[k:]]
        self.covs = [[self.cov[:k, :k], self.cov[:k, k:]], [self.cov[k:, :k], self.cov[k:, k:]]]

        self.betas = [
            self.covs[0][1] @ np.linalg.inv(self.covs[1][1]),
            self.covs[1][0] @ np.linalg.inv(self.covs[0][0]),
        ]

    def cond_dist(
        self, ind: int, z: Union[float, int, npt.ArrayLike]
    ) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Computes the conditional distribution of z1 given z2, or reversely.
        Argument ind determines whether we compute the conditional
        distribution of z1 (ind=0) or z2 (ind=1).

        Example: Let p(x1, x2, x3) be the joint multivariate normal.
        If ind=1 the conditional distribution p(x2, x3 | x1) will be returned.
        if ind=0 the marginal distribution p(x1 | x2, x3) will be returned.

        :param ind: Index (either zero or one) to specify which conditional should be calculated.
        :param z: Conditional value or value vector. E.g. x2, x3 if calculating p(x1 | x2, x3).
        :return:
            mean_hat: The conditional mean of z1 or z2.
            cov_hat: The conditional covariance matrix of z1 or z2.
        """
        if self.betas is None or self.means is None or self.covs is None:
            raise ValueError(
                "Call self.partition(k) first to specify how to partition the joint multivariate normal distribution. "
            )

        if ind != 0 and ind != 1:
            raise ValueError("ind must be zero or one.")

        if isinstance(z, float) or isinstance(z, int):
            z = np.array([z])
        else:
            z = np.array(z)
        if z.shape != self.means[1 - ind].shape:
            raise ValueError(f"z has invalid shape. Expected {self.means[1 - ind].shape}.")

        beta = self.betas[ind]

        mean_hat = self.means[ind] + beta @ (z - self.means[1 - ind])
        cov_hat = self.covs[ind][ind] - beta @ self.covs[1 - ind][1 - ind] @ beta.T

        return mean_hat, cov_hat

    def marg_dist(self, ind: int) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Computes the marginal distribution for the variable at
        index given by ind. Returns the mean and covariance for
        a multivariate normal distribution.

        Example: Let p(x1, x2, x3) be the joint multivariate normal.
        If ind=1 the marginal distribution p(x2, x3) will be returned.
        if ind=0 the marginal distribution p(x1) will be returned.

        :param ind: Index (either zero or one) to specify which marginal should be calculated.
        :return:
            mean: The conditional mean of z1 or z2.
            cov_hat: The conditional covariance matrix of z1 or z2.
        """
        if self.means is None or self.covs is None:
            raise ValueError(
                "Call self.partition(k) first to specify how to partition the joint multivariate normal distribution. "
            )

        if not (ind == 0 or ind == 1):
            raise ValueError("ind must be zero or one.")

        return self.means[ind], self.covs[ind][ind]

    def __str__(self) -> str:  # pragma: nocover
        return f"MultivariateNormal(mean={self.mean}, cov={self.cov})"
