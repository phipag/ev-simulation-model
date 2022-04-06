from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt
from jinja2 import Environment, PackageLoader

from ev_simulation_model.io.exporter import Exporter
from ev_simulation_model.models import ConditionalMvnMixture


class ConditionalMvnMixtureExporter(Exporter):
    """Exports a ConditionalMvnMixture model to a XML file.

    See: :func:ev_simulation_model.models.conditional_mvn_mixture.ConditionalMvnMixture"""

    _TEMPLATE_NAME = "conditional_mvn_mixture.xml.jinja"

    def __init__(
        self,
        cond_mvn_mixture: ConditionalMvnMixture,
        conditionals: List[Union[float, int, npt.NDArray[np.float_]]],
        conditional_names: List[str],
        ind: int,
    ) -> None:
        """Initializes the exporter.

        :param cond_mvn_mixture: Partitioned ConditionalMvnMixture instance.
        :param conditionals: Vector of conditionals.
        :param conditional_names: Name of XML tags for each conditional
        :param ind: Index (either zero or one) to indicate the partition to condition on.
        """
        super().__init__()
        self._cond_mvn_mixture = cond_mvn_mixture
        self._conditionals = conditionals
        self._conditional_names = conditional_names
        self._ind = ind
        env = Environment(
            loader=PackageLoader(__name__),
        )
        self._template = env.get_template(ConditionalMvnMixtureExporter._TEMPLATE_NAME)

    def export(self, destination: str) -> None:
        """Exports the ConditionalMvnMixture to a XML file at the specified destination.

        :param destination: Filepath indicating where to save the XML file.
        """
        weights: Dict[str, npt.NDArray[np.float_]] = {}
        means: Dict[str, List[npt.NDArray[np.float_]]] = {}
        covs: Dict[str, List[npt.NDArray[np.float_]]] = {}
        for i, conditional in enumerate(self._conditionals):
            weights[self._conditional_names[i]] = self._cond_mvn_mixture.calc_cond_weights(self._ind, conditional)
            means[self._conditional_names[i]] = [
                mvn.cond_dist(self._ind, conditional)[0] for mvn in self._cond_mvn_mixture.mvns
            ]
            covs[self._conditional_names[i]] = [
                mvn.cond_dist(self._ind, conditional)[1].reshape(-1) for mvn in self._cond_mvn_mixture.mvns
            ]
        self._template.stream(
            conditional_names=self._conditional_names,
            weights=weights,
            means=means,
            covs=covs,
        ).dump(destination)
