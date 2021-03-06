import os.path
from pathlib import Path
from xml.sax import make_parser
from xml.sax.handler import ContentHandler

import numpy as np
import pytest

from ev_simulation_model.io import ConditionalMvnMixtureExporter
from ev_simulation_model.models import ConditionalMvnMixture, MultivariateNormal


@pytest.fixture
def mvn1():
    return MultivariateNormal(
        mean=[16.27807559, 10.31377236, 18.43250934],
        cov=np.array(
            [
                [1.87081887e01, -4.15188229e00, -8.20127096e00],
                [-4.15188229e00, 3.38159314e01, 3.73923506e00],
                [-8.20127096e00, 3.73923506e00, 8.18433728e00],
            ]
        ),
    )


@pytest.fixture
def mvn2():
    return MultivariateNormal(
        mean=[12.18176491, 14.77768893, 18.80089312],
        cov=np.array(
            [
                [6.36278929e00, 5.80137105e-01, -5.99239467e00],
                [5.80137105e-01, 1.03502991e02, -3.50890884e-01],
                [-5.99239467e00, -3.50890884e-01, 6.03992805e00],
            ]
        ),
    )


def test_conditional_mvn_mixture_xml_exporter_ok(tmpdir, mvn1, mvn2):
    cond_mvn_mixture = ConditionalMvnMixture(mvns=[mvn1, mvn2], weights=[0.2, 0.8])
    cond_mvn_mixture.partition(2)
    exporter = ConditionalMvnMixtureExporter(
        cond_mvn_mixture,
        conditionals=[1, 2, 3, 4, 9],
        conditional_names=list(map(lambda cond: f"hod{cond}", [1, 2, 3, 4, 9])),
        cond_ind=0,
    )
    destination = str(Path(tmpdir) / "residental_ev.xml")
    exporter.export(destination)
    assert os.path.exists(destination)
    # Assert that XML can be parsed
    parser = make_parser()
    parser.setContentHandler(ContentHandler())
    parser.parse(destination)


def test_conditional_mvn_mixture_xml_exporter_marg_dist_ok(tmpdir, mvn1, mvn2):
    cond_mvn_mixture = ConditionalMvnMixture(mvns=[mvn1, mvn2], weights=[0.2, 0.8])
    cond_mvn_mixture.partition(2)
    exporter = ConditionalMvnMixtureExporter(
        cond_mvn_mixture,
        conditionals=[1, 2, 3, 4, 9],
        conditional_names=list(map(lambda cond: f"hod{cond}", [1, 2, 3, 4, 9])),
        cond_ind=0,
        marg_ind=1,
        marg_name="marginalName",
        root_name="rootName",
    )
    destination = str(Path(tmpdir) / "residental_ev.xml")
    exporter.export(destination)
    assert os.path.exists(destination)
    # Assert that XML can be parsed
    parser = make_parser()
    parser.setContentHandler(ContentHandler())
    parser.parse(destination)


def test_conditional_mvn_mixture_xml_exporter_init_wrong_number_of_conditional_names_ko(mvn1, mvn2):
    cond_mvn_mixture = ConditionalMvnMixture(mvns=[mvn1, mvn2], weights=[0.2, 0.8])
    cond_mvn_mixture.partition(2)
    with pytest.raises(ValueError):
        ConditionalMvnMixtureExporter(
            cond_mvn_mixture,
            conditionals=[1, 2, 3, 4, 9],
            conditional_names=list(map(lambda cond: f"hod{cond}", [1, 2, 3, 4])),
            cond_ind=0,
        )


def test_conditional_mvn_mixture_xml_exporter_unpartitioned_cond_mvn_mixture_ko(tmpdir, mvn1, mvn2):
    cond_mvn_mixture = ConditionalMvnMixture(mvns=[mvn1, mvn2], weights=[0.2, 0.8])
    # Note that we do not partition the ConditionalMvnMixture instance
    exporter = ConditionalMvnMixtureExporter(
        cond_mvn_mixture,
        conditionals=[1, 2, 3, 4, 9],
        conditional_names=list(map(lambda cond: f"hod{cond}", [1, 2, 3, 4, 9])),
        cond_ind=0,
    )
    destination = str(Path(tmpdir) / "residental_ev.xml")
    with pytest.raises(ValueError):
        exporter.export(destination)
