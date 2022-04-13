from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
from jinja2 import Environment, PackageLoader

from ev_simulation_model.io.exporter import Exporter
from ev_simulation_model.models import ConditionalMvnMixture


class ConditionalMvnMixtureExporter(Exporter):
    """Exports a ConditionalMvnMixture model to a XML file.

    Exports all multivariate normal mixture components for the given sequence of conditionals while re-calculating the
    weights for each conditional.

    Additionally, allows including marginal distributions if marg_ind and marg_name is specified.

    See :func:ev_simulation_model.models.conditional_mvn_mixture.ConditionalMvnMixture"""

    _TEMPLATE_NAME = "conditional_mvn_mixture.xml.jinja"

    def __init__(
        self,
        cond_mvn_mixture: ConditionalMvnMixture,
        conditionals: List[Union[float, int, npt.NDArray[np.float_]]],
        conditional_names: List[str],
        cond_ind: int,
        marg_ind: Optional[int] = None,
        marg_name: Optional[str] = None,
        root_name: Optional[str] = "conditionalMvnMixtureExport",
    ) -> None:
        """Initializes the exporter.

        :param cond_mvn_mixture: Partitioned ConditionalMvnMixture instance.
        :param conditionals: Vector of conditionals.
        :param conditional_names: Name of XML tags for each conditional
        :param cond_ind: Index (either zero or one) to indicate the partition to condition on.
        :param marg_ind: Optional index (either zero or one) indicating if the marginal distribution at index
            marg_ind shall be exported as well.
        :param marg_name: XML tag name for the marginal distribution. Will be ignored if marg_ind is not specified.
        """
        if len(conditionals) != len(conditional_names):
            raise ValueError(
                f"Need the same number of conditionals and conditional_names. "
                f"Given {len(conditionals)} conditionals and {len(conditional_names)} conditional_names."
            )

        super().__init__()
        self._cond_mvn_mixture = cond_mvn_mixture
        self._conditionals = conditionals
        self._conditional_names = conditional_names
        self._cond_ind = cond_ind
        self._marg_ind = marg_ind
        self._marg_name = marg_name or f"marginalDist{marg_ind}"
        self._root_name = root_name
        env = Environment(
            loader=PackageLoader(__name__),
        )
        self._template = env.get_template(ConditionalMvnMixtureExporter._TEMPLATE_NAME)

    def export(self, destination: str) -> None:
        """Exports the ConditionalMvnMixture to a XML file at the specified destination.

        :param destination: Filepath as str indicating where to save the XML file.
        """
        cond_weights: Dict[str, npt.NDArray[np.float_]] = {}
        cond_means: Dict[str, List[npt.NDArray[np.float_]]] = {}
        cond_covs: Dict[str, List[npt.NDArray[np.float_]]] = {}
        for i, conditional in enumerate(self._conditionals):
            cond_weights[self._conditional_names[i]] = self._cond_mvn_mixture.calc_cond_weights(
                self._cond_ind, conditional
            )
            cond_means[self._conditional_names[i]] = [
                mvn.cond_dist(self._cond_ind, conditional)[0] for mvn in self._cond_mvn_mixture.mvns
            ]
            # We flatten the cov matrix because the dimension is clear (square matrix of dimension of the mean vector)
            cond_covs[self._conditional_names[i]] = [
                mvn.cond_dist(self._cond_ind, conditional)[1].reshape(-1) for mvn in self._cond_mvn_mixture.mvns
            ]

        marg_weights, marg_means, marg_covs = None, None, None
        if self._marg_ind:
            marg_weights = self._cond_mvn_mixture.weights
            marg_means = [mvn.marg_dist(self._marg_ind)[0] for mvn in self._cond_mvn_mixture.mvns]
            # We flatten the cov matrix because the dimension is clear (square matrix of dimension of the mean vector)
            marg_covs = [mvn.marg_dist(self._marg_ind)[1].reshape(-1) for mvn in self._cond_mvn_mixture.mvns]

        self._template.stream(
            root_name=self._root_name,
            conditional_names=self._conditional_names,
            cond_weights=cond_weights,
            cond_means=cond_means,
            cond_covs=cond_covs,
            marg_name=self._marg_name,
            marg_weights=marg_weights,
            marg_means=marg_means,
            marg_covs=marg_covs,
        ).dump(destination)
