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
    content = rng.normal(size=(8, 8, 3))
    stylized = 1.5 * content

    lipschitz = compute_lipschitz_vectorized(content, stylized, num_samples=200)

    assert np.isclose(lipschitz, 1.5)


def test_compute_lipschitz_vectorized_scaling_channel_first():
    rng = np.random.default_rng(123)
    content = rng.normal(size=(3, 6, 5))
    stylized = 0.75 * content

    lipschitz = compute_lipschitz_vectorized(content, stylized, num_samples=200)

    assert np.isclose(lipschitz, 0.75)


def test_compute_lipschitz_vectorized_detects_infinite_lipschitz():
    content = np.zeros((4, 4, 3), dtype=np.float32)
    stylized = np.zeros_like(content)
    stylized[0, 0] = np.array([1.0, 0.0, 0.0])

    lipschitz = compute_lipschitz_vectorized(content, stylized, num_samples=64)

    assert np.isinf(lipschitz)


def test_compute_lipschitz_vectorized_constant_maps_to_constant():
    content = np.ones((4, 4, 3), dtype=np.float32)
    stylized = np.ones_like(content)

    lipschitz = compute_lipschitz_vectorized(content, stylized, num_samples=64)

    assert lipschitz == 0.0
