import os
import shutil
from typing import Iterable

import numpy as np
import torch
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from torchvision.transforms import v2
from PIL import Image


def clean_dirs(dir_list):
    """Remove Jupyter checkpoint folders from the provided directories.

    Args:
        dir_list (Iterable[str]): Directories whose ``.ipynb_checkpoints``
            folders should be deleted if present.
    """
    for dir_name in dir_list:
        if os.path.exists(dir_name + '.ipynb_checkpoints'):
            shutil.rmtree(dir_name + '.ipynb_checkpoints')


def load_filenames(path, verbose=False):
    """List files contained in ``path`` while optionally logging progress.

    Args:
        path (str): Directory that should be traversed recursively.
        verbose (bool, optional): When ``True`` print the number of files found
            in each visited directory. Defaults to ``False``.

    Returns:
        list[str]: Paths to all discovered files.
    """
    dataset_filenames = []
    for dirpath, dirnames, filenames in os.walk(path):
        if verbose:
            print(f"{dirpath} : {len(filenames)}")
        if '.ipynb_checkpoints' not in dirpath:
            for fname in filenames:
                dataset_filenames.append(dirpath + "/" + fname)
    if verbose:
        print(f"total: {len(dataset_filenames)}")
    return dataset_filenames


def create_save_path(filepath, dataset_root, flows_root):
    """Construct the directory path used to store a flow checkpoint.

    Args:
        filepath (str): Absolute path of the dataset image.
        dataset_root (str): Root directory of the dataset.
        flows_root (str): Root directory for storing flow checkpoints.

    Returns:
        str: Directory mirroring ``filepath`` under ``flows_root``.
    """
    filename = filepath.split("/")[-1]
    start_char = len(dataset_root)
    last_char = len(filepath) - len(filename)
    savepath = flows_root + filepath[start_char:last_char]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return savepath


def get_flow_id(filepath):
    """Derive a deterministic checkpoint filename from an image path.

    Args:
        filepath (str): Path to the dataset image.

    Returns:
        str: Filename of the associated flow checkpoint.
    """
    filename = filepath.split("/")[-1].split(".")[0]
    return filename + "_model.pt"


def get_flow_path(filepath, dataset_root, flows_root):
    """Return the full checkpoint path for an image.

    Args:
        filepath (str): Absolute path of the image in the dataset.
        dataset_root (str): Root directory that contains the dataset.
        flows_root (str): Root directory containing the generated flows.

    Returns:
        str: Full path to the checkpoint file for ``filepath``.
    """
    filename = filepath.split("/")[-1]
    start_char = len(dataset_root)
    last_char = len(filepath) - len(filename)
    savepath = flows_root + filepath[start_char:last_char]
    flow_id = get_flow_id(filepath)    
    return savepath + flow_id


def print_images(pil_imgs, with_density=False, points=200, titles=None, s=10):
    """Render images with optional RGB density scatter plots.

    Args:
        pil_imgs (Iterable[PIL.Image.Image]): Images to render.
        with_density (bool, optional): Append an RGB scatter plot for each
            image when ``True``. Defaults to ``False``.
        points (int, optional): Number of pixels to subsample for the density
            plot. Defaults to ``200``.
        titles (Iterable[str], optional): Titles to display above each image.
            Defaults to ``None``.
        s (int, optional): Scatter point size. Defaults to ``10``.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure with all rendered content.
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