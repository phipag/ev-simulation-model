import numpy as np
import pytest

from ev_simulation_model.models import ConditionalMvnMixture, MultivariateNormal
from ev_simulation_model.models.conditional_mvn_mixture import _select_mixture_index


@pytest.fixture
def mvn2d():
    mean = [1, 2]
    cov = [[1, 2], [2, 1]]
    return MultivariateNormal(mean, cov)


@pytest.fixture
def mvn3d1():
    mean = [16.27807559, 10.31377236, 18.43250934]
    cov = [
        [1.87081887e01, -4.15188229e00, -8.20127096e00],
        [-4.15188229e00, 3.38159314e01, 3.73923506e00],
        [-8.20127096e00, 3.73923506e00, 8.18433728e00],
    ]
    return MultivariateNormal(mean, cov)


@pytest.fixture
def mvn3d2():
    mean = [12.18176491, 14.77768893, 18.80089312]
    cov = [
        [6.36278929e00, 5.80137105e-01, -5.99239467e00],
        [5.80137105e-01, 1.03502991e02, -3.50890884e-01],
        [-5.99239467e00, -3.50890884e-01, 6.03992805e00],
    ]
    return MultivariateNormal(mean, cov)


@pytest.fixture
def mvn3d3():
    mean = [1.15435747, 4.12474982, 15.55221238]
    cov = [
        [7.77495585e-01, 2.80003561e00, -4.49958574e-02],
        [2.80003561e00, 1.00921998e01, -1.52154721e-01],
        [-4.49958574e-02, -1.52154721e-01, 1.14469990e01],
    ]
    return MultivariateNormal(mean, cov)


@pytest.fixture
def weights():
    return [0.2, 0.3, 0.5]


@pytest.fixture
def weights_invalid():
    return [0.5, 0.4, 0.2]  # Sum is larger than one


def test_cond_mvn_mixture_init_ok(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)
    np.testing.assert_equal(mvn_mixture._mvns, [mvn3d1, mvn3d2, mvn3d3])
    np.testing.assert_equal(mvn_mixture._weights, weights)


def test_cond_mvn_mixture_init_invalid_weights_ko(mvn3d1, mvn3d2, mvn3d3, weights_invalid):
    with pytest.raises(ValueError):
        ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights_invalid)


def test_cond_mvn_mixture_init_no_mvns_ko():
    with pytest.raises(ValueError):
        ConditionalMvnMixture(mvns=[], weights=[])


def test_cond_mvn_mixture_init_number_mvns_weights_not_equal_ko(mvn3d1, mvn3d2, weights):
    with pytest.raises(ValueError):
        ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2], weights=weights)


def test_cond_mvn_mixture_init_invalid_type_ko(mvn3d1, mvn3d2, weights):
    with pytest.raises(ValueError):
        ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, "Not a MultivariateNormal instance"], weights=weights)


def test_cond_mvn_mixture_init_inconsistent_dimension_ko(mvn3d1, mvn3d2, mvn2d, weights):
    with pytest.raises(ValueError):
        ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn2d], weights=weights)


def test_cond_mvn_mixture_cond_dist_ok(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)
    mvn_mixture.partition(1)
    # We cannot check the correctness, but we know the mean and cov has to be different
    mean1, cov1 = mvn_mixture.cond_dist(0, [2, 2])
    mean2, cov2 = mvn_mixture.cond_dist(0, [200, 200])
    with pytest.raises(AssertionError):
        np.testing.assert_equal(mean1, mean2)
    with pytest.raises(AssertionError):
        np.testing.assert_equal(cov1, cov2)

    # We cannot check the correctness, but we know the mean and cov has to be different
    mean1, cov1 = mvn_mixture.cond_dist(1, 2)
    mean2, cov2 = mvn_mixture.cond_dist(1, 200)
    with pytest.raises(AssertionError):
        np.testing.assert_equal(mean1, mean2)
    with pytest.raises(AssertionError):
        np.testing.assert_equal(cov1, cov2)


def test_cond_mvn_mixture_cond_dist_invalid_z_ko(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)
    mvn_mixture.partition(1)

    with pytest.raises(ValueError):
        mvn_mixture.cond_dist(0, 2)


def test_select_mixture_index():
    n = 1000
    indices = np.empty(n)
    for i in range(n):
        indices[i] = _select_mixture_index(np.array([0.1, 0.3, 0.5]))

    assert len(indices[indices == 0]) < len(indices[indices == 1]) < len(indices[indices == 2])
