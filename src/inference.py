import os
from typing import Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from PIL import Image

from src.neural_ode import NeuralODE, train_ode
from src.encoder import Encoder, enc_preprocess


def load_filenames(path):
    """Enumerate every file contained in ``path``.

    The function prints a short summary for each directory it visits to make it
    obvious how many files were discovered. Hidden ``.ipynb_checkpoints``
    folders are included in the counts; callers should filter them out if
    needed.

    Args:
        path (str): Root directory that should be traversed recursively.

    Returns:
        list[str]: Full paths of all discovered files.
    """
    dataset_filenames = []
    for dirpath, dirnames, filenames in os.walk(path):
        print(f"{dirpath} : {len(filenames)}")
        for fname in filenames:
            dataset_filenames.append(dirpath + "/" + fname)
    print(f"total: {len(dataset_filenames)}")
    return dataset_filenames


def print_images(pil_imgs, with_density=False, points=200, titles=None, s=10):
    """Render one or more images (and optionally their color distributions).

    Args:
        pil_imgs (Sequence[PIL.Image.Image]): Images to render.
        with_density (bool, optional): If ``True`` append a 3D RGB scatter plot
            for each image. Defaults to ``False``.
        points (int, optional): Number of pixels to sample when ``with_density``
            is enabled. Defaults to ``200``.
        titles (Sequence[str], optional): Titles to display above each image.
            Defaults to ``None``.
        s (int, optional): Size of the scatter plot markers. Defaults to ``10``.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing all subplots.
    """
    fig = plt.figure(figsize=(20, 10))
    n = len(pil_imgs)
    azim = 236 #np.random.uniform(0,360)
    nrows = 2 if with_density else 1
#     plt.subplots_adjust(hspace=-0.2)
    for i in range(n):
        img = pil_imgs[i]
        ax = fig.add_subplot(nrows, n, i+1)
        ax.imshow(img)
        if titles is not None:
            ax.set_title(titles[i])
        ax.axis('off')
        
        if with_density:
            img = img.convert('RGB')
            w, h = img.size
            arr = np.array(img).reshape(w*h, 3) / 255
            idx = np.random.permutation(w*h)[:points]
            
            ax = fig.add_subplot(nrows, n, n+i+1, projection='3d')
            r = arr[idx, 0]
            g = arr[idx, 1]
            b = arr[idx, 2]
            colors = arr[idx, :]
            ax.scatter(r, g, b, c=colors, alpha=0.5, s=s)
            ax.view_init(elev=30, azim=azim)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_zlim(0, 1)
            ax.axes.get_xaxis().set_ticklabels([])
            ax.axes.get_yaxis().set_ticklabels([])
            ax.axes.get_zaxis().set_ticklabels([])
    return fig


def tensor_to_im(tensor, w, h):
    """Convert a flattened tensor back into a PIL image.

    Args:
        tensor (torch.Tensor): Flattened pixel data with shape ``(w * h, 3)``.
            The tensor may reside on any device; it will be detached and moved
            to the CPU before conversion.
        w (int): Target width in pixels.
        h (int): Target height in pixels.

    Returns:
        PIL.Image.Image: RGB image with size ``(w, h)``.
    """
    tensor = tensor.detach().cpu()
    tensor = torch.clip(tensor, 0, 1)
    tensor = tensor.reshape((h, w, 3)) * 255
    array = np.array(tensor, dtype=np.uint8)
    return Image.fromarray(array)


# @torch.no_grad()
def run_inference(encoder, device, content_im_path, style_im_path,
                  compress=False, enc_steps=3, strength=1.0, crop=False):
    """Transfer the color palette of ``style_im_path`` onto ``content_im_path``.

    Args:
        encoder (Encoder): Pre-trained encoder that predicts flow parameters.
        device (torch.device): Device used for all tensor computations.
        content_im_path (str): Path to the content image.
        style_im_path (str): Path to the style reference image.
        compress (float or bool, optional): When set to a positive, non-zero
            value the content image is downscaled by this factor (e.g. ``2``
            halves both spatial dimensions) before encoding. Use ``False`` to
            disable resizing. Defaults to ``False``.
        enc_steps (int, optional): Number of Euler steps used when integrating
            the flows. Defaults to ``3``.
        strength (float, optional): Relative integration length. Values below
            ``1`` yield a more subtle transfer by stopping early. Defaults to
            ``1.0``.
        crop (bool, optional): Centrally crop both images to a square before
            encoding. Defaults to ``False``.

    Returns:
        Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
        Possibly resized content image, latent reconstruction, stylized output,
        and the untouched style reference image.
    """
    with torch.no_grad():
        encoder.eval()
        
        content_im = Image.open(content_im_path).convert('RGB')
        style_im = Image.open(style_im_path).convert('RGB')
        
        if crop:
            content_im = v2.CenterCrop(min(content_im.size))(content_im)
            style_im = v2.CenterCrop(min(style_im.size))(style_im)
        
        base_im = enc_preprocess(content_im, crop).to(device)
        base_im = base_im.unsqueeze(0)
        base_e = encoder(base_im).flatten()
        base_encoded_flow = NeuralODE(input_dim=3, hidden=encoder.hidden, device=device)
        base_encoded_flow.set_weights(base_e)
        
        targ_im = enc_preprocess(style_im, crop).to(device)
        targ_im = targ_im.unsqueeze(0)
        targ_e = encoder(targ_im).flatten()
        targ_encoded_flow = NeuralODE(input_dim=3, hidden=encoder.hidden, device=device)
        targ_encoded_flow.set_weights(targ_e)
        
        w = content_im.width
        h = content_im.height

        if compress:
            w = int(w / compress)
            h = int(h / compress)
            content_im = content_im.resize((w, h)) 

        print(f"im size: {w, h}")
        
        base_x = np.array(content_im, dtype=np.float32) / 255
        base_x = base_x.reshape(w * h, 3)
        base_x = torch.tensor(base_x).to(device)
        
        latent_x = base_encoded_flow.sample(base_x, N=enc_steps, strength=strength)
        styled_x = targ_encoded_flow.inv_sample(latent_x, N=enc_steps, strength=strength)
        
        latent_im = tensor_to_im(latent_x, w, h)
        styled_im = tensor_to_im(styled_x, w, h)
        return content_im, latent_im, styled_im, style_im


# @torch.no_grad()
def run_inference_flow(device, content_flow_path, target_flow_path, content_im_path,
                       hidden=64, enc_steps=3, strength=1.0, compress=None):
    """Stylize an image using pre-trained :class:`NeuralODE` checkpoints.

    Args:
        device (torch.device): Device used for loading checkpoints and running
            inference.
        content_flow_path (str): Path to the checkpoint describing the content
            flow.
        target_flow_path (str): Path to the checkpoint of the style flow.
        content_im_path (str): Path to the content image.
        hidden (int, optional): Hidden layer width used when instantiating the
            flow networks. Defaults to ``64``.
        enc_steps (int, optional): Number of Euler integration steps. Defaults
            to ``3``.
        strength (float, optional): Relative integration length. Defaults to
            ``1.0``.
        compress (float, optional): Positive resize factor applied before
            stylization. When ``None`` the original resolution is preserved.
            Defaults to ``None``.

    Returns:
        Tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]: Possibly
        resized content image, latent reconstruction, and stylized output.
    """
    with torch.no_grad():
        content_im = Image.open(content_im_path).convert('RGB')
    
        base_flow_params = torch.load(content_flow_path, map_location=device)
        base_flow = NeuralODE(input_dim=3, hidden=hidden, device=device)
        base_flow.load_state_dict(base_flow_params)
        
        targ_flow_params = torch.load(target_flow_path, map_location=device)
        targ_flow = NeuralODE(input_dim=3, hidden=hidden, device=device)
        targ_flow.load_state_dict(targ_flow_params)
        
        w = content_im.width
        h = content_im.height

        print(f"im size: {w, h}")
        if compress is not None:
            w = int(w / compress)
            h = int(h / compress)
            im_size = (h, w)
            print(f"compressed size: {w, h}")
            content_im = v2.Resize(im_size)(content_im)
        
        base_x = np.array(content_im, dtype=np.float32) / 255
        base_x = base_x.reshape(w * h, 3)
        base_x = torch.tensor(base_x, dtype=torch.float32).to(device)
        
        latent_x = base_flow.sample(base_x, N=enc_steps, strength=strength)
        styled_x = targ_flow.inv_sample(latent_x, N=enc_steps, strength=strength)   
        latent_im = tensor_to_im(latent_x, w, h)
        styled_im = tensor_to_im(styled_x, w, h)
        return content_im, latent_im, styled_im


