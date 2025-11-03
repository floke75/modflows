"""Tests for the Lipschitz constant estimation utilities."""

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


def test_compute_lipschitz_vectorized_detects_infinite_lipschitz():
    base = np.zeros((4, 4, 3), dtype=np.float32)
    output = np.zeros_like(base)
    output[0, 0] = np.array([1.0, 0.0, 0.0])

    lipschitz = compute_lipschitz_vectorized(base, output, num_samples=64)

    assert np.isinf(lipschitz)


def test_compute_lipschitz_vectorized_constant_maps_to_constant():
    base = np.ones((4, 4, 3), dtype=np.float32)
    output = np.ones_like(base)

    lipschitz = compute_lipschitz_vectorized(base, output, num_samples=64)

    assert lipschitz == 0.0
