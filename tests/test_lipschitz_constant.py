"""Tests for the Lipschitz constant estimation utilities."""

import pathlib
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.lipschitz_constant import compute_lipschitz_vectorized


class _FixedChoiceRNG:
    """Deterministic RNG stub that returns a pre-defined choice matrix."""

    def __init__(self, result: np.ndarray):
        self._result = np.asarray(result)
        self.calls = 0

    def choice(self, *args, **kwargs):  # noqa: D401
        """Mimic ``np.random.Generator.choice`` with a fixed result."""

        size = kwargs.get("size")
        if size is None and len(args) >= 2:
            size = args[1]

        assert size == self._result.shape
        self.calls += 1
        return self._result


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


def test_compute_lipschitz_vectorized_requires_positive_samples():
    content = np.ones((2, 2, 3), dtype=np.float32)
    stylized = np.ones_like(content)

    with pytest.raises(ValueError, match="positive integer"):
        compute_lipschitz_vectorized(content, stylized, num_samples=0)


def test_compute_lipschitz_vectorized_requires_matching_shapes():
    content = np.ones((2, 2, 3), dtype=np.float32)
    stylized = np.ones((3, 3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="identical spatial dimensions"):
        compute_lipschitz_vectorized(content, stylized, num_samples=4)


def test_compute_lipschitz_vectorized_requires_matching_aspect_ratio():
    content = np.ones((2, 6, 3), dtype=np.float32)
    stylized = np.ones((6, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="identical spatial dimensions"):
        compute_lipschitz_vectorized(content, stylized, num_samples=8)


def test_compute_lipschitz_vectorized_requires_integer_samples():
    content = np.ones((2, 2, 3), dtype=np.float32)
    stylized = np.ones_like(content)

    with pytest.raises(TypeError, match="integer count"):
        compute_lipschitz_vectorized(content, stylized, num_samples=2.5)

    with pytest.raises(TypeError, match="integer count"):
        compute_lipschitz_vectorized(content, stylized, num_samples=True)


def test_compute_lipschitz_vectorized_handles_uint8_inputs():
    content = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    stylized = np.array([[[0, 0, 0], [128, 128, 128]]], dtype=np.uint8)

    fixed_indices = _FixedChoiceRNG(np.array([[0, 1]]))
    lipschitz = compute_lipschitz_vectorized(
        content, stylized, num_samples=1, rng=fixed_indices
    )

    expected = np.linalg.norm([128, 128, 128]) / np.linalg.norm([255, 255, 255])
    assert np.isclose(lipschitz, expected)
    assert fixed_indices.calls == 1


def test_compute_lipschitz_vectorized_requires_rng_with_choice():
    class NoChoice:
        pass

    with pytest.raises(TypeError, match="choice"):
        compute_lipschitz_vectorized(
            np.zeros((1, 1, 3)),
            np.zeros((1, 1, 3)),
            num_samples=1,
            rng=NoChoice(),
        )
