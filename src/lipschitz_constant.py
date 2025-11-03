"""Utilities for estimating Lipschitz constants between color transforms."""

import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.encoder import enc_preprocess


def _to_pixel_matrix(image: np.ndarray) -> np.ndarray:
    """Converts an image tensor to an ``(num_pixels, 3)`` RGB matrix.

    Args:
        image (np.ndarray): The image in channel-last ``(H, W, 3)`` or channel-first
            ``(3, H, W)`` format.

    Returns:
        np.ndarray: Matrix containing RGB values for each pixel.

    Raises:
        ValueError: If the image does not have three dimensions or three channels.
    """

    if image.ndim != 3:
        raise ValueError(
            "image must be a 3D array with explicit channel information; received "
            f"shape {image.shape}"
        )

    if image.shape[-1] == 3:  # (H, W, 3)
        pixels = image
    elif image.shape[0] == 3:  # (3, H, W)
        pixels = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(
            "image must provide exactly three channels in either the first or last "
            f"dimension; received shape {image.shape}"
        )

    return pixels.reshape(-1, 3)


def compute_lipschitz_vectorized(x, y, num_samples):
    """Estimate an upper bound on the Lipschitz constant between two RGB clouds.

    Args:
        x (np.ndarray): Array representing RGB samples from the stylized image.
            Accepts either channel-last ``(H, W, 3)`` or channel-first ``(3, H, W)``
            layout and converts it internally to a pixel matrix.
        y (np.ndarray): Array with RGB samples from the reference image using the
            same supported layouts as ``x``.
        num_samples (int): The number of random pixel pairs to evaluate.

    Returns:
        float: Maximum ratio of pairwise distances found across the sampled
            pairs. Returns ``np.inf`` when the sampled output differences are
            non-zero while the corresponding input differences are zero.
    """
    input_flat = _to_pixel_matrix(x)
    output_flat = _to_pixel_matrix(y)

    indices = np.random.choice(len(input_flat), (num_samples, 2), replace=True)

    dist_input = np.linalg.norm(
        input_flat[indices[:, 0]] - input_flat[indices[:, 1]], axis=1
    )
    dist_output = np.linalg.norm(
        output_flat[indices[:, 0]] - output_flat[indices[:, 1]], axis=1
    )

    lipschitz_values = np.zeros_like(dist_output)
    nonzero_mask = dist_input != 0
    if np.any(nonzero_mask):
        lipschitz_values[nonzero_mask] = np.divide(
            dist_output[nonzero_mask], dist_input[nonzero_mask]
        )

    zero_mask = ~nonzero_mask
    if np.any(zero_mask):
        lipschitz_values[zero_mask] = np.where(
            np.isclose(dist_output[zero_mask], 0.0),
            0.0,
            np.inf,
        )

    return float(np.max(lipschitz_values))



if __name__ == "__main__":
    num_samples = 50000
    path_dir = "V7_encoder_epoch_700000"  # Your directory name
    print("\n", path_dir)

    # Setup paths
    stylization_path = f"data/results_unsplash/{path_dir}/"  # YOUR DIR NAME
    content_path = "data/test_imgs/content/"  # YOUR DIR NAME

    # Get sorted lists of image paths
    stylization_results = sorted(
        [
            os.path.join(stylization_path, f)
            for f in os.listdir(stylization_path)
            if f.endswith(".png")
        ]
    )

    content_images = sorted(
        [
            os.path.join(content_path, f)
            for f in os.listdir(content_path)
            if f.endswith(".png")
        ]
    )

    lipschitz_constants = []
    for idx in tqdm(range(len(stylization_results))):
        stylized_tensor = enc_preprocess(Image.open(stylization_results[idx]))
        reference_tensor = enc_preprocess(Image.open(content_images[idx]))
        L_max = compute_lipschitz_vectorized(
            stylized_tensor.numpy(), reference_tensor.numpy(), num_samples
        )
        lipschitz_constants.append(L_max)

    lipschitz_constants = np.array(lipschitz_constants)

    average_L = np.mean(lipschitz_constants)
    std_L = np.std(
        [
            lipschitz_constants[
                np.random.randint(0, len(lipschitz_constants), len(lipschitz_constants))
            ].mean()
            for _ in range(1000)
        ]
    )  # bootstrap std

    print("Average Lipschitz constant:", average_L)
    print("Standard deviation of Lipschitz constant:", std_L)
