import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lipschitz_constant import compute_lipschitz_vectorized


def test_compute_lipschitz_vectorized_scaling():
    rng = np.random.default_rng(42)
    base = rng.normal(size=(8, 8, 3))
    scaled = 1.5 * base

    lipschitz = compute_lipschitz_vectorized(base, scaled, num_samples=200)

    assert np.isclose(lipschitz, 1.5)
