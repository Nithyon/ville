import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ville import EProcessMonitor, clopper_pearson_upper, response_events  # noqa: E402
from ville.simulate import bootstrap_streams, make_pool  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(123)
    monitor = EProcessMonitor().fit(make_pool("benign", 125, rng), make_pool("benign", 125, rng))
    return monitor, rng


def test_clopper_pearson_bounds():
    assert clopper_pearson_upper(10, 10, 0.01) == 1.0
    q = clopper_pearson_upper(5, 125, 0.005 / 7)
    assert 5 / 125 < q < 1.0


def test_events_are_nested():
    r = {"mismatch": np.array([False, True, True]), "margin": np.array([3.0, 0.05, 0.5])}
    A = response_events(r, [0.0, 0.1, 0.4, 1.0])
    assert A.tolist() == [1, 1, 1, 0]


def test_factor_lower_bound(fitted):
    monitor, rng = fitted
    monitor.reset()
    for r in make_pool("seed_blind", 50, rng) + make_pool("benign", 50, rng):
        E, _ = monitor.factors(r)
        assert np.all(E >= 1 - monitor.c - 1e-12)
        monitor.update(r)


def test_benign_rarely_alarms(fitted):
    monitor, rng = fitted
    pool = make_pool("benign", 250, rng)
    alarms = [monitor.run(s)[1] for s in bootstrap_streams(pool, 200, 250, rng)]
    assert np.mean([a is not None for a in alarms]) <= 0.02


def test_seed_blind_detected(fitted):
    monitor, rng = fitted
    pool = make_pool("seed_blind", 250, rng)
    alarms = [monitor.run(s)[1] for s in bootstrap_streams(pool, 200, 250, rng)]
    assert np.mean([a is not None for a in alarms]) >= 0.95
