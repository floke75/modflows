import pathlib
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import encoder


def test_enc_preprocess_preserves_channel_order():
    size = encoder.INPUT_SIZE
    ramp = np.arange(size * size, dtype=np.uint32).reshape(size, size) % 256
    green = np.flipud(ramp).astype(np.uint8)
    blue = np.fliplr(ramp).astype(np.uint8)
    red = ramp.astype(np.uint8)
    image = Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")

    tensor = encoder.enc_preprocess(image)
    array = np.array(image, dtype=np.float32) / 255.0

    assert tensor.shape == (3, size, size)
    np.testing.assert_allclose(tensor.numpy()[0], array[..., 0])
    np.testing.assert_allclose(tensor.numpy()[1], array[..., 1])
    np.testing.assert_allclose(tensor.numpy()[2], array[..., 2])
