import numpy as np
import pytest

from ev_simulation_model.models import MultivariateNormal


@pytest.fixture
def mvn2d():
    mean = [1, 2]
    cov = [[1, 2], [2, 1]]
    return MultivariateNormal(mean, cov)


@pytest.fixture
def mvn3d():
    mean = [1, 2, 3]
    cov = [[1, 2, 3], [2, 1, 3], [3, 2, 1]]
    return MultivariateNormal(mean, cov)


def test_mvn_init_ok():
    mean = [1, 2]
    cov = [[1, 2], [2, 1]]
    mvn = MultivariateNormal(mean, cov)

    np.testing.assert_equal(mvn.mean, mean)
    np.testing.assert_equal(mvn.cov, cov)
    assert isinstance(mvn.mean, np.ndarray)
    assert isinstance(mvn.cov, np.ndarray)


def test_mvn_init_incompatible_shapes_ko():
    mean = [1, 2]
    cov = [[1, 2], [2, 1], [1, 2]]
    with pytest.raises(ValueError):
        MultivariateNormal(mean, cov)


def test_mvn_partitioning_ok(mvn3d):
    mvn3d.partition(1)
    np.testing.assert_equal(mvn3d.means[0], mvn3d.mean[0])
    np.testing.assert_equal(mvn3d.means[1], mvn3d.mean[1:])
    np.testing.assert_equal(mvn3d.covs[0][0], mvn3d.cov[0, 0])
    np.testing.assert_equal(mvn3d.covs[1][1], mvn3d.cov[1:, 1:])


def test_mvn_partitioning_invalid_k_ko(mvn3d):
    with pytest.raises(ValueError):
        mvn3d.partition(0)

    with pytest.raises(ValueError):
        mvn3d.partition(3)


def test_mvn_cond_ok(mvn2d, mvn3d):
    mvn2d.partition(1)
    mean_hat, cov_hat = mvn2d.cond_dist(0, 2)
    np.testing.assert_equal(mean_hat, 1)
    np.testing.assert_equal(cov_hat, -3)

    mvn3d.partition(1)
    mean_hat, cov_hat = mvn3d.cond_dist(0, [2, 1])
    np.testing.assert_almost_equal(mean_hat, [-0.2])
    np.testing.assert_almost_equal(cov_hat, [[-2.4]])

    mean_hat, cov_hat = mvn3d.cond_dist(1, 2)
    np.testing.assert_almost_equal(mean_hat, [4, 6])
    np.testing.assert_almost_equal(cov_hat, [[-3.0, -3.0], [-4.0, -8.0]])


def test_mvn_cond_invalid_z_shape_ko(mvn3d):
    mvn3d.partition(1)
    with pytest.raises(ValueError):
        mvn3d.cond_dist(0, 1)
    with pytest.raises(ValueError):
        mvn3d.cond_dist(0, [1])
    with pytest.raises(ValueError):
        mvn3d.cond_dist(0, [[1]])


def test_mvn_cond_invalid_ind_shape_ko(mvn3d):
    mvn3d.partition(1)
    with pytest.raises(ValueError):
        mvn3d.cond_dist(2, 1)


def test_mvn_cond_not_partitioned_ko(mvn3d):
    with pytest.raises(ValueError):
        mvn3d.cond_dist(1, 1)


def test_mvn_marg_ok(mvn2d, mvn3d):
    mvn2d.partition(1)
    mean, cov = mvn2d.marg_dist(0)
    assert mean.shape == (1,)
    assert cov.shape == (1, 1)
    np.testing.assert_equal(mean, mvn2d.mean[0])
    np.testing.assert_equal(cov, mvn2d.cov[0, 0])

    mvn3d.partition(1)
    mean, cov = mvn3d.marg_dist(0)
    assert mean.shape == (1,)
    assert cov.shape == (1, 1)
    np.testing.assert_equal(mean, mvn3d.mean[0])
    np.testing.assert_equal(cov, mvn3d.cov[0, 0])

    mean, cov = mvn3d.marg_dist(1)
    assert mean.shape == (2,)
    assert cov.shape == (2, 2)
    np.testing.assert_equal(mean, mvn3d.mean[1:])
    np.testing.assert_equal(cov, mvn3d.cov[1:, 1:])


def test_mvn_marg_not_partitioned_ko(mvn2d):
    with pytest.raises(ValueError):
        mvn2d.marg_dist(1)


def test_mvn_marg_invalid_ind_shape_ko(mvn3d):
    mvn3d.partition(1)
    with pytest.raises(ValueError):
        mvn3d.marg_dist(2)
