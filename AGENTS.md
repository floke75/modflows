## Project Overview

This repository contains the implementation of "Color Transfer with Modulated Flows," a neural color-transfer pipeline based on neural ordinary differential equations (NODEs). The system first trains color transport flows for each image and then trains an encoder to predict the flow parameters directly from RGB pixels so the effect can be applied at inference time.

### Repository layout

* **`src/encoder.py`** – EfficientNet-based encoder that predicts flattened NODE weights. The helper `enc_preprocess` mirrors the data preparation used throughout training and analysis scripts.
* **`src/neural_ode.py`** – Lightweight NODE implementation, Euler integration utilities, and helpers for sampling latent color codes.
* **`src/inference.py`** – High-level helpers for stylizing an image pair (encoder-driven or via saved flow checkpoints) plus visualization utilities.
* **`generate_flows_v2.py`** – Offline rectified-flow generator. Uses `enc_preprocess` to build the base distribution and trains a `NeuralODE` per image.
* **`train_encoder_v2.py`** – Offline encoder training loop that distills pre-computed flows into a single predictor.
* **`run_inference.py`** – Command-line interface that stylizes a batch of image pairs using a trained encoder.

Training data is expected under `data/` (mirroring the original experiments). Flow checkpoints live under `check_points/` and encoder checkpoints under `checkpoints/` by default. Adjust the constants at the top of the scripts when targeting a different layout.

### Workflow summary

1. Generate rectified flows with `python generate_flows_v2.py`. This script enumerates every file beneath `LOAD` and writes checkpoints under `SAVE`, mirroring the dataset structure.
2. Train the encoder with `python train_encoder_v2.py`. Ensure the `FLOW` directory points to the flow checkpoints produced in step 1. The script periodically writes encoder checkpoints with timestamped filenames.
3. Stylize new images either via `python run_inference.py` (encoder-driven; expects parallel content/style directories) or by calling `run_inference_flow` from `src.inference` when working with individual flow checkpoints.

### Development guidelines

* Use Google Style docstrings for any new public functions, methods, or modules. Keep shape/dtype descriptions accurate—``NeuralODE`` uses `input_dim + 1` internally to account for time.
* Prefer deterministic RNG seeds when adding new training or evaluation utilities.
* Format Python code with `black` and keep imports sorted (``isort`` compatible).
* Run `python -m compileall src` before submitting a PR to catch syntax errors. Add targeted unit tests under `tests/` when extending the core library.
* Large training scripts log progress with `tqdm`; follow the existing patterns when adding new loops so notebook and CLI usage stay consistent.
* Avoid introducing mocks or stubs in tests unless they are absolutely necessary, demonstrably safe, and clearly justified in accompanying documentation or comments.
* Ensure the test environment has ``numpy``, ``pillow``, ``torch``, ``torchvision``, and ``einops`` installed (matching the versions pinned in ``requirements.txt``) before running the test suite to prevent dependency-related failures.

### Communication expectations

* When updating documentation, ensure references to tensor shapes, devices, and directory conventions mirror the actual code paths.
* Prefer raising informative `ValueError`s over silent failure for CLI additions.
* Keep PR descriptions concise but explicit about affected scripts (`encoder`, `neural_ode`, etc.).
