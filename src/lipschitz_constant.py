import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.encoder import enc_preprocess


def compute_lipschitz_vectorized(inputs, outputs, num_samples):
    """Computes the Lipschitz constant between paired input and output pixels.

    Args:
        inputs (np.ndarray): The input image used to drive the color transformation.
        outputs (np.ndarray): The resulting stylized image after the transformation.
        num_samples (int): The number of random pixel pairs to evaluate.

    Returns:
        float: The maximum sampled Lipschitz constant.
    """
    input_flat = inputs.reshape(-1, 3)
    output_flat = outputs.reshape(-1, 3)

    indices = np.random.choice(len(input_flat), (num_samples, 2), replace=True)

    dist_input = np.linalg.norm(
        input_flat[indices[:, 0]] - input_flat[indices[:, 1]], axis=1
    )
    dist_output = np.linalg.norm(
        output_flat[indices[:, 0]] - output_flat[indices[:, 1]], axis=1
    )

    lipschitz_values = np.divide(
        dist_output, dist_input, out=np.zeros_like(dist_output), where=dist_input != 0
    )

    return np.max(lipschitz_values)


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
        content_tensor = enc_preprocess(Image.open(content_images[idx]))
        stylized_tensor = enc_preprocess(Image.open(stylization_results[idx]))
        L_max = compute_lipschitz_vectorized(
            content_tensor.numpy(), stylized_tensor.numpy(), num_samples
        )
        lipschitz_constants.append(L_max)

    lipschitz_constants = np.array(lipschitz_constants)

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
