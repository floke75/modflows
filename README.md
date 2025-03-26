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

How to clone and download pre-trained weights:
```
git clone https://github.com/maria-larchenko/modflows.git
cd modflows; git clone https://huggingface.co/MariaLarchenko/modflows_color_encoder
```

Call `python3 run_inference.py --help` to see a full list of arguments for inference.
`Ctrl+C` cancels the execution.

<p align="center">
     <img src="./img/SPIGM_visual_abstract.png" style="width: 500px"/>
</p>

## Citation
If you use this code in your research, please cite our work:
```
@inproceedings{larchenko2024color,
  title={Color Style Transfer with Modulated Flows},
  author={Larchenko, Maria and Lobashev, Alexander and Guskov, Dmitry and Palyulin, Vladimir Vladimirovich},
  booktitle={ICML 2024 Workshop on Structured Probabilistic Inference $\{$$\backslash$\&$\}$ Generative Modeling}
}
```
