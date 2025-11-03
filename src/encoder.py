import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.transforms import v2
from einops import einsum


# INPUT_SIZE = 256
INPUT_SIZE = 528


def enc_preprocess(pil_image, crop=False, image=False, rand_trans=False):
    """Prepare an RGB image so it can be consumed by :class:`Encoder`.

    Depending on ``image`` the function either returns the transformed PIL
    image (to allow manual inspection) or a tensor with the channel-first layout
    that the encoder expects. The pixel values are scaled to ``[0, 1]`` but no
    normalization specific to EfficientNet weights is applied here—those
    statistics are injected later through :meth:`Encoder.forward` when the
    backbone's native preprocessing pipeline is executed.

    Args:
        pil_image (PIL.Image.Image): Input image in RGB mode.
        crop (bool, optional): If ``True`` the shorter side is centrally
            cropped before resizing. Defaults to ``False``.
        image (bool, optional): When ``True`` the transformed PIL image is
            returned instead of a tensor. Defaults to ``False``.
        rand_trans (bool, optional): Apply random horizontal flips and a random
            90-degree rotation.  Used only for data augmentation during
            training. Defaults to ``False``.

    Returns:
        torch.Tensor or PIL.Image.Image: When ``image`` is ``False`` a
        ``float32`` tensor on the CPU with shape ``(3, INPUT_SIZE, INPUT_SIZE)``
        and values in ``[0, 1]``. Otherwise the transformed PIL image.
    """
    im = pil_image
    im_size = (INPUT_SIZE, INPUT_SIZE)
    # v2 currently accepts only PIL Images
    if crop:
        crop = min(im.size)
        im = v2.CenterCrop(crop)(im)
    im = v2.Resize(im_size)(im)
    if rand_trans:
        im = v2.RandomHorizontalFlip(p=0.5)(im)
        im = v2.functional.rotate(im, angle=np.random.choice([0, 90, -90, 180]))
    if image:
        return im
    im = np.array(im, dtype=np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1)).copy()
    return torch.from_numpy(im)


class Encoder(nn.Module):
    """EfficientNet-based encoder that predicts flow parameters.

    The encoder projects an RGB image onto the parameter space of the
    :class:`~src.neural_ode.NeuralODE` color flow.  The ``input_dim``,
    ``hidden`` and ``output_dim`` arguments describe the architecture of that
    downstream flow network and are used to reshape the flattened EfficientNet
    logits into layer-wise weights.  ``input_dim`` must match the
    ``NeuralODE.input_dim`` attribute (which already accounts for the appended
    time channel), i.e. it is typically ``color_channels + 1``.

    Args:
        k_dim (int): Number of parameters the encoder should output (equal to
            the total parameters of the target flow network).
        input_dim (int): Number of channels in the flow input (typically three
            RGB channels plus the time component).
        hidden (int): Width of the hidden layer in the flow network.
        output_dim (int): Number of channels produced by the flow network (for
            color transfer this is ``3``).
        device (torch.device): Device that stores the encoder and intermediate
            tensors.
        encoder_type (str, optional): Name of the EfficientNet backbone. Must
            be ``"B6"`` or ``"B0"``. Defaults to ``"B6"``.
    """

    def __init__(self, k_dim, input_dim, hidden, output_dim, device, encoder_type="B6"):
        super().__init__()
        self.k_dim = k_dim
        if encoder_type == "B6":
            self.model = torchvision.models.efficientnet_b6(num_classes=k_dim)
            self.resize = (
                torchvision.models.efficientnet.EfficientNet_B6_Weights.IMAGENET1K_V1.transforms()
            )  # INPUT_SIZE = 528
        elif encoder_type == "B0":
            self.model = torchvision.models.efficientnet_b0(num_classes=k_dim)
            self.resize = (
                torchvision.models.efficientnet.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
            )  # INPUT_SIZE = 256

        self.device = device
        #       # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
        #       # self.conv = nn.Conv2d(6, 3, (3, 3), stride=1, padding=1)

        self.input_dim = input_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.splits = [
            0,
            input_dim * hidden,
            input_dim * hidden + hidden,
            input_dim * hidden + hidden + output_dim * hidden,
            input_dim * hidden + hidden + output_dim * hidden + output_dim,
        ]
        self.to(device)

    def forward(self, im1):
        """Encode a batch of RGB images.

        Args:
            im1 (torch.Tensor): Batch of images with shape ``(N, 3, H, W)`` and
                values in ``[0, 1]`` on any device supported by torchvision.

        Returns:
            torch.Tensor: Flattened parameter vectors with shape ``(N, k_dim)``.
        """
        # the input should be [batch_size, channels, height, width]
        with torch.no_grad():
            im1 = self.resize(im1)
        return self.model(im1)

    def apply_e(self, e, x, t):
        """Evaluate the decoded flow parameters on sample points.

        Args:
            e (torch.Tensor): Batch of flattened parameter vectors produced by
                :meth:`forward` with shape ``(N, k_dim)``.
            x (torch.Tensor): Sampled RGB values of shape ``(N, M, C)`` where
                ``C = self.input_dim - 1`` (the ``-1`` accounts for the time
                channel that is concatenated internally).
            t (torch.Tensor): Time values with shape ``(N, M, 1)`` associated
                with each sample in ``x``.

        Returns:
            torch.Tensor: Flow predictions with shape ``(N, M, output_dim)`` on
                the same device as ``x`` and ``t``.
        """
        splits = self.splits
        batch_size = e.shape[0]
        shapes = [
            torch.Size([batch_size, self.hidden, self.input_dim]),
            torch.Size([batch_size, self.hidden]),
            torch.Size([batch_size, self.output_dim, self.hidden]),
            torch.Size([batch_size, self.output_dim]),
        ]
        # e = [batch_size, params_size]
        e0 = e[:, splits[0]:splits[1]].reshape(shapes[0])
        e1 = e[:, splits[1]:splits[2]].reshape(shapes[1])
        e2 = e[:, splits[2]:splits[3]].reshape(shapes[2])
        e3 = e[:, splits[3]:splits[4]].reshape(shapes[3])
        e1 = e1.unsqueeze(1)
        e3 = e3.unsqueeze(1)
        # x = [batch_size, sample_size, channels]
        # t = [batch_size, sample_size, 1]
        xt = torch.cat([x, t], dim=-1)
        xt = einsum(xt, e0, 'i j k, i n k -> i j n') + e1
        xt = torch.tanh(xt)
        xt = einsum(xt, e2, 'i j k, i n k -> i j n') + e3
        return xt
