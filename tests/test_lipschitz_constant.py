import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lipschitz_constant import compute_lipschitz_vectorized


def test_compute_lipschitz_vectorized_scaling_channel_last():
    rng = np.random.default_rng(42)
    base = rng.normal(size=(8, 8, 3))
    scaled = 1.5 * base

    lipschitz = compute_lipschitz_vectorized(base, scaled, num_samples=200)

    assert np.isclose(lipschitz, 1.5)


def test_compute_lipschitz_vectorized_scaling_channel_first():
    rng = np.random.default_rng(123)
    base = rng.normal(size=(3, 6, 5))
    scaled = 0.75 * base

    lipschitz = compute_lipschitz_vectorized(base, scaled, num_samples=200)

    assert np.isclose(lipschitz, 0.75)
