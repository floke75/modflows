# Color Transfer with Modulated Flows (AAAI 2025)

arXiv: https://arxiv.org/abs/2503.19062

This is the official implementation of AAAI 2025 paper "Color Transfer with Modulated Flows". 

<p align="center">
     <img src="./img/results_unsplash.png" style="width: 1000px"/>
</p>

The paper was also presented at ["Workshop SPIGM @ ICML 2024"](https://openreview.net/forum?id=Lztt4WVusu).

Please refer to the
- <strong>[ModFlows_demo.ipynb](https://github.com/maria-larchenko/modflows/blob/main/ModFlows_demo.ipynb)</strong> to use the pretrained model for color transfer on your own images with the demo jupyter notebook
- <strong>[ModFlows_demo_batched.ipynb](https://github.com/maria-larchenko/modflows/blob/main/ModFlows_demo_batched.ipynb)</strong> to use the pretrained model for color transfer for large images
- <strong>[HuggingFace](https://huggingface.co/MariaLarchenko/modflows_color_encoder)</strong> for the model checkpoints
- <strong>src</strong> directory for models definitions
- <strong>generate_flows_v2</strong> script for training the dataset of rectified flows
- <strong>train_encoder_v2</strong> script for training the encoder

## Getting Started

### Prerequisites

* Python 3.9–3.13 (newer releases are recommended for the best CUDA 13 support)
* PyTorch with CUDA 13.0 wheels (see installation notes below)
* torchvision
* NumPy
* Matplotlib
* Pillow
* tqdm
* einops

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/maria-larchenko/modflows.git
   ```
2. Navigate to the project directory:
   ```bash
   cd modflows
   ```
3. Install the required dependencies. The `requirements.txt` file now targets NVIDIA GPUs such as the RTX 5090 by pulling the CUDA 13.0 (cu130) wheels from the official PyTorch index. If you prefer to install PyTorch manually, run the first command below before installing the remaining packages:
   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   pip install -r requirements.txt
   ```

### Usage

1. **Download the pre-trained weights:**
   ```bash
   sudo apt install git-lfs
   git lfs install
   git clone https://huggingface.co/MariaLarchenko/modflows_color_encoder
   ```
2. **Run inference:**
   ```bash
   python3 run_inference.py --content <path_to_content_images> --style <path_to_style_images> --output <path_to_output_directory>
   ```

   For a full list of arguments, run:
   ```bash
   python3 run_inference.py --help
   ```

## How to clone and download pre-trained weights:
```
git clone https://github.com/maria-larchenko/modflows.git
cd modflows;
sudo apt install git-lfs; git lfs install
git clone https://huggingface.co/MariaLarchenko/modflows_color_encoder
```

Call `python3 run_inference.py --help` to see a full list of arguments for inference.
`Ctrl+C` cancels the execution.

<p align="center">
     <img src="./img/SPIGM_visual_abstract.png" style="width: 500px"/>
</p>

## Citation
If you use this code in your research, please cite our work:
```
@article{Larchenko_Lobashev_Guskov_Palyulin_2025, title={Color Transfer with Modulated Flows}, volume={39}, url={https://ojs.aaai.org/index.php/AAAI/article/view/32470},
DOI={10.1609/aaai.v39i4.32470},  number={4},
journal={Proceedings of the AAAI Conference on Artificial Intelligence},
author={Larchenko, Maria and Lobashev, Alexander and Guskov, Dmitry and Palyulin, Vladimir Vladimirovich}, year={2025}, month={Apr.}, pages={4464-4472} }
```
