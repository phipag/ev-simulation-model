import pkg_resources

from ev_simulation_model import __version__


def test_version():
    assert __version__ == pkg_resources.get_distribution("ev-simulation-model").version
