import os
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm, trange
from PIL import Image
from torch.distributions.multivariate_normal import MultivariateNormal

from src.neural_ode import NeuralODE, train_ode
from src.encoder import Encoder, enc_preprocess

from src.encoder import INPUT_SIZE

from src.utils import clean_dirs
 
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

LOAD = 'data/merged/'
SAVE = 'check_points/latent_flow_merged_8195/'

# LOAD = 'data/test_imgs_unsplash/'
# SAVE = 'check_points/latent_flow_unsplash_8195/'

# LOAD = 'data/test_imgs_unsplash/'
# SAVE = 'check_points/latent_flow_unsplash_515/' 


def load_filenames(path):
    """Enumerate all images that should be converted into flows.

    Args:
        path (str): Root directory that contains the dataset images.

    Returns:
        list[str]: Absolute paths to every discovered file. This helper performs
        no filtering, so non-image files need to be excluded downstream if
        present.
    """
    dataset_filenames = []
    for dirpath, dirnames, filenames in os.walk(path):
        print(f"{dirpath} : {len(filenames)}")
        for fname in filenames:
            dataset_filenames.append(dirpath + "/" + fname)
    print(f"total: {len(dataset_filenames)}")
    return dataset_filenames
    

def normal_to_uniform(x):
    """Map samples from a standard normal to the unit cube.

    Args:
        x (torch.Tensor): Tensor of standard normal samples.

    Returns:
        torch.Tensor: Tensor with values in ``[0, 1]`` after applying the CDF
        element-wise.
    """
    return (torch.special.erf(x / np.sqrt(2)) + 1) / 2


def uniform_latent(dim, data_size):
    """Sample a set of latent codes from a factorized uniform distribution.

    Args:
        dim (int): Dimensionality of the latent space.
        data_size (int): Number of samples to draw.

    Returns:
        torch.Tensor: Tensor with shape ``(data_size, dim)`` and values in
        ``[0, 1]``.
    """
    base_mu = torch.zeros(dim)
    base_cov = torch.eye(dim)
    latent_dist = MultivariateNormal(base_mu, base_cov)
    x = latent_dist.rsample(sample_shape=(data_size,))
    return normal_to_uniform(x)


def get_flow_id(filepath):
    """Derive the filename suffix used for storing a flow checkpoint.

    Args:
        filepath (str): Absolute path to the dataset image.

    Returns:
        str: Checkpoint filename component without directory information. The
        ``"_model_515"`` suffix encodes the flow dimensionality.
    """
    filename = filepath.split("/")[-1].split(".")[0]
    return filename + "_model_515"


def create_save_path(filepath, dataset_root, flows_root):
    """Construct the directory path used to store the generated flow.

    Args:
        filepath (str): Absolute path of the dataset image.
        dataset_root (str): Root directory of the dataset.
        flows_root (str): Root directory where flow checkpoints will be stored.

    Returns:
        str: Directory path under ``flows_root`` matching the dataset structure.
        The directory is created if it does not already exist.
    """
    filename = filepath.split("/")[-1]
    start_char = len(dataset_root)
    last_char = len(filepath) - len(filename)
    savepath = flows_root + filepath[start_char:last_char]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    return savepath 


## ------------ hyper-parameters
lr = 5e-4
sample_size = 512 * 100
samples = 50_000

clean_dirs([LOAD, ])
dataset_filenames = load_filenames(LOAD)

n_img = len(dataset_filenames)
tt = trange(n_img, desc='>>> Next train step <<<', leave=True)
for n in tt:
    # model = NeuralODE(input_dim=3, device=device, hidden=1024)
    model = NeuralODE(input_dim=3, device=device, hidden=64)
    print("total_params ", model.total_params)
    
    filepath = dataset_filenames[n]
    savepath = create_save_path(filepath, LOAD, SAVE)
    flow_id  = get_flow_id(filepath)
    
    image = Image.open(filepath).convert("RGB")
    
    # ## coupling(X_0, X_1) = fixed pairs!
    base_density = enc_preprocess(image)  #.squeeze(0)  ~  enc_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
    base_density = base_density.reshape(INPUT_SIZE**2, 3)
    target_density = uniform_latent(3, INPUT_SIZE**2)
    
    ## --------- initial train
    text = f'{n}:'
    train_ode(model, lr, base_density, target_density, samples, sample_size, tt, text, shuffle=True)
            
    ## --------- rectify the model
    text = f'{n} rectify:'
    target_density_2 = model.sample(base_density, N=100)
    train_ode(model, lr, base_density, target_density_2, samples, sample_size, tt, text, shuffle=False)
    
    ## save the trained model:
    model_path = savepath + flow_id
    torch.save(model.state_dict(), model_path)
    print(f"{model_path}  SAVED")
    
print("\n FINISHED")