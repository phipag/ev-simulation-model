import numpy as np
import pytest

from ev_simulation_model.models import ConditionalMvnMixture, MultivariateNormal


@pytest.fixture
def mvn2d():
    mean = [1, 2]
    cov = np.array([[1, 2], [2, 1]])
    return MultivariateNormal(mean, cov)


@pytest.fixture
def mvn3d1():
    mean = [16.27807559, 10.31377236, 18.43250934]
    cov = np.array(
        [
            [1.87081887e01, -4.15188229e00, -8.20127096e00],
            [-4.15188229e00, 3.38159314e01, 3.73923506e00],
            [-8.20127096e00, 3.73923506e00, 8.18433728e00],
        ]
    )
    return MultivariateNormal(mean, cov)


@pytest.fixture
def mvn3d2():
    mean = [12.18176491, 14.77768893, 18.80089312]
    cov = np.array(
        [
            [6.36278929e00, 5.80137105e-01, -5.99239467e00],
            [5.80137105e-01, 1.03502991e02, -3.50890884e-01],
            [-5.99239467e00, -3.50890884e-01, 6.03992805e00],
        ]
    )
    return MultivariateNormal(mean, cov)


@pytest.fixture
def mvn3d3():
    mean = [1.15435747, 4.12474982, 15.55221238]
    cov = np.array(
        [
            [7.77495585e-01, 2.80003561e00, -4.49958574e-02],
            [2.80003561e00, 1.00921998e01, -1.52154721e-01],
            [-4.49958574e-02, -1.52154721e-01, 1.14469990e01],
        ]
    )
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
    np.testing.assert_equal(mvn_mixture.mvns, [mvn3d1, mvn3d2, mvn3d3])
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


def test_cond_mvn_mixture_sample_cond_ok(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)
    mvn_mixture.partition(1)
    samples_1d = mvn_mixture.sample_cond(0, [2, 2])
    assert samples_1d.shape == (1, 1)
    samples_2d = mvn_mixture.sample_cond(1, 2)
    assert samples_2d.shape == (1, 2)


def test_cond_mvn_mixture_sample_cond_multiple_ok(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)
    mvn_mixture.partition(1)
    samples_1d = mvn_mixture.sample_cond(0, [2, 2], 1000)
    assert samples_1d.shape == (1000, 1)
    samples_2d = mvn_mixture.sample_cond(1, 2, 1000)
    assert samples_2d.shape == (1000, 2)


def test_cond_mvn_mixture_sample_cond_invalid_z_ko(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)
    mvn_mixture.partition(1)

    with pytest.raises(ValueError):
        mvn_mixture.sample_cond(0, 2)


def test_cond_mvn_mixture_sample_cond_not_partitioned_ko(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)

    with pytest.raises(ValueError):
        mvn_mixture.sample_cond(0, [2, 2])


def test_cond_mvn_mixture_sample_marg_ok(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)
    mvn_mixture.partition(1)
    samples_1d = mvn_mixture.sample_marg(0)
    assert samples_1d.shape == (1, 1)
    samples_2d = mvn_mixture.sample_marg(1)
    assert samples_2d.shape == (1, 2)


def test_cond_mvn_mixture_sample_marg_multiple_ok(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)
    mvn_mixture.partition(1)
    samples_1d = mvn_mixture.sample_marg(0, 1000)
    assert samples_1d.shape == (1000, 1)
    samples_2d = mvn_mixture.sample_marg(1, 1000)
    assert samples_2d.shape == (1000, 2)


def test_cond_mvn_mixture_sample_marg_not_partitioned_ko(mvn3d1, mvn3d2, mvn3d3, weights):
    mvn_mixture = ConditionalMvnMixture(mvns=[mvn3d1, mvn3d2, mvn3d3], weights=weights)

    with pytest.raises(ValueError):
        mvn_mixture.sample_marg(0)
