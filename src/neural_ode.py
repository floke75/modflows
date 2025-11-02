import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

from PIL import Image
from torchvision.transforms import v2
from torch.distributions.multivariate_normal import MultivariateNormal


class NeuralODE(nn.Module):
    """Two-layer MLP that parameterizes the color transport ODE.

    Args:
        input_dim (int): Dimensionality of the color space that will be
            transformed (typically ``3`` for RGB). A scalar time dimension is
            internally appended which is why ``self.input_dim`` becomes
            ``input_dim + 1``.
        device (torch.device): Compute device used for parameters and
            integration.
        hidden (int, optional): Width of the hidden layer used to approximate
            the dynamics. Defaults to ``32``.
    """
    
    def __init__(self, input_dim, device, hidden=32):
        super().__init__()
        self.device = device
        self.hidden = hidden
        self.input_dim = input_dim + 1
        self.output_dim = input_dim
        self.activation = nn.Tanh()
        self.layer_1 = nn.Linear(self.input_dim, self.hidden, bias=1)
        self.layer_2 = nn.Linear(self.hidden, self.output_dim, bias=1)
        self.shapes = [
          self.layer_1.weight.shape,
          self.layer_1.bias.shape,
          self.layer_2.weight.shape,
          self.layer_2.bias.shape,
         ]
        self.splits = [0,
          self.input_dim * hidden, 
          self.input_dim * hidden + hidden, 
          self.input_dim * hidden + hidden + self.output_dim * hidden,
          self.input_dim * hidden + hidden + self.output_dim * hidden + self.output_dim,
         ]
        self.total_params = sum(p.numel() for p in self.parameters())
        self.to(self.device)
        
    def set_weights(self, e):
        """Load a flattened parameter vector produced by the encoder.

        Args:
            e (torch.Tensor): One-dimensional tensor with ``self.total_params``
                elements that contains ``layer_1`` and ``layer_2`` weights and
                biases in row-major order. The tensor can live on any device;
                it is reshaped and loaded directly into ``self``.
        """
        assert len(e) == self.total_params
        splits = self.splits
        shapes = self.shapes
        e0 = e[splits[0]:splits[1]].reshape(shapes[0])
        e1 = e[splits[1]:splits[2]].reshape(shapes[1])
        e2 = e[splits[2]:splits[3]].reshape(shapes[2])
        e3 = e[splits[3]:splits[4]].reshape(shapes[3])
        mask_dict = {
            'layer_1.weight': e0,
            'layer_1.bias': e1,
            'layer_2.weight': e2,
            'layer_2.bias': e3
        }
        self.load_state_dict(mask_dict)
        self.to(self.device)

    def forward(self, x, t):
        """Evaluate the velocity field of the learned flow.

        Args:
            x (torch.Tensor): Batch of sample points with shape ``(N, input_dim)``
                where ``input_dim`` matches the spatial color dimensionality
                supplied to the constructor.
            t (torch.Tensor): Matching time values with shape ``(N, 1)``.

        Returns:
            torch.Tensor: Estimated flow velocities with shape ``(N, output_dim)``.
        """
        xt = torch.cat([x, t], dim=1)
        xt = self.layer_1(xt)
        xt = self.activation(xt)
        xt = self.layer_2(xt)
        return xt

    @torch.no_grad()
    def sample(self, x0, N=10_000, strength=1.0):
        """Integrate the learned ODE forward in time using Euler steps.

        Args:
            x0 (torch.Tensor): Initial sample locations with shape
                ``(num_samples, input_dim)``. ``x0`` is cloned and moved to the
                model's device during integration.
            N (int, optional): Number of Euler steps to take. Defaults to
                ``10_000``.
            strength (float, optional): Fraction of the trajectory to trace.
                ``1.0`` follows the full integration horizon while values
                smaller than ``1`` perform an early stop based on
                ``int(strength * N)``. Defaults to ``1.0``.

        Returns:
            torch.Tensor: Final sample positions with the same shape as ``x0``.
        """
        sample_size = len(x0) 
        z = x0.detach().clone()
        dt = 1.0 / N
        for i in range(N):
            t = torch.ones((sample_size, 1)) * i / N
            t = t.to(self.device)
            z = z.to(self.device)
            pred = self.forward(z, t)
            #eps = 0.03
            #noise = torch.randn_like(pred)
            #z = z.detach().clone() + pred * dt + eps*noise*torch.sqrt(t*(1-t))
            z = z.detach().clone() + pred * dt
            if i > int(strength * N):
                break
        return z.detach().clone()

    @torch.no_grad()
    def inv_sample(self, x0, N=10_000, strength=1.0):
        """Integrate the learned ODE backwards in time.

        Args:
            x0 (torch.Tensor): Initial sample locations with shape
                ``(num_samples, input_dim)``. The tensor is cloned and moved to
                the model's device during integration.
            N (int, optional): Number of Euler steps used during integration.
                Defaults to ``10_000``.
            strength (float, optional): Fraction of the trajectory to evaluate
                before terminating once ``i`` exceeds ``int(strength * N)``.
                Defaults to ``1.0``.

        Returns:
            torch.Tensor: Final sample positions after reverse integration.
        """
        sample_size = len(x0)
        z = x0.detach().clone()
        dt = 1.0 / N
        for i in range(N):
            t = torch.ones((sample_size, 1)) * i / N
            t = t.to(self.device)
            z = z.to(self.device)
            pred = self.forward(z, 1-t)
            z = z.detach().clone() - pred * dt
            if i > int(strength * N):
                break
        return z.detach().clone()

def train_ode(model, lr, base_x, targ_x, samples, sample_size, tt=None, text=None, shuffle=True):
    """Optimize the flow model to match a target point cloud.

    Args:
        model (NeuralODE): Flow network to be optimized in-place.
        lr (float): Learning rate used by the Adam optimizer.
        base_x (torch.Tensor): Source samples with shape ``(dataset, dim)``.
        targ_x (torch.Tensor): Target samples with the same shape as ``base_x``.
        samples (int): Number of optimization iterations.
        sample_size (int): Mini-batch size drawn from the datasets each
            iteration.
        tt (tqdm.std.tqdm or None): Progress bar for logging. Defaults to
            ``None``.
        text (str or None): Prefix appended to the progress bar description.
            Defaults to ``None``.
        shuffle (bool, optional): Whether to resample ``base_x`` on every
            iteration (``targ_x`` is always resampled). Defaults to ``True``.
    """
    data_size = base_x.shape[0]
    device = model.device
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for sn in range(samples):
        optim.zero_grad()

        indx = torch.randperm(data_size)[:sample_size]
        x_1 = targ_x[indx].to(device)
        
        if shuffle:
            indx = torch.randperm(data_size)[:sample_size]
        x_0 = base_x[indx].to(device)
        
        t = torch.rand((sample_size, 1), device=device)
        
        z_t = t * x_1 + (1.0 - t) * x_0
        v = x_1 - x_0
        v_pred = model(z_t, t)
        loss = torch.mean((v - v_pred)**2)
        
        loss.backward()
        optim.step()
        if tt is not None and sn % 100 == 0:
            tt.set_description(f"{text} {sn}/{samples} || loss: {loss.detach().cpu()}")
            tt.refresh()


def normal_to_uniform(x):
    """Map standard normal samples to the unit cube.

    Args:
        x (torch.Tensor): Tensor of standard normal samples.

    Returns:
        torch.Tensor: Tensor with values in ``[0, 1]`` obtained via the normal
        cumulative distribution function applied element-wise.
    """
    return (torch.special.erf(x / np.sqrt(2)) + 1) / 2


def uniform_latent(dim, data_size):
    """Sample latent codes from a factorized uniform distribution.

    Args:
        dim (int): Dimensionality of the latent space.
        data_size (int): Number of samples to draw.

    Returns:
        torch.Tensor: Tensor with shape ``(data_size, dim)`` whose entries lie
        in ``[0, 1]``.
    """
    base_mu = torch.zeros(dim)
    base_cov = torch.eye(dim)
    latent_dist = MultivariateNormal(base_mu, base_cov)
    x = latent_dist.rsample(sample_shape=(data_size,))
    return normal_to_uniform(x)
    

def create_save_path(filepath, dataset_root, flows_root):
    """Derive the directory used to store a trained flow model.

    Args:
        filepath (str): Absolute path of an input image.
        dataset_root (str): Root directory that contains the dataset images.
        flows_root (str): Root directory where the trained flow checkpoints are
            written.

    Returns:
        str: Directory that mirrors the dataset structure under ``flows_root``.
            The directory is created on disk if it does not yet exist.
    """
    filename = filepath.split("/")[-1]
    start_char = len(dataset_root)
    last_char = len(filepath) - len(filename)
    savepath = flows_root + filepath[start_char:last_char]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return savepath



