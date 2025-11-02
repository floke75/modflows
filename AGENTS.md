## Project Overview

This repository contains the implementation of "Color Transfer with Modulated Flows," a deep learning model for color transfer between images. The model is based on neural ordinary differential equations (NODEs) and uses a modulated flow to learn the color distribution of a style image and apply it to a content image.

### Key Components

*   **`src/encoder.py`**: Defines the `Encoder` class, which is a convolutional neural network (CNN) that learns a compact representation of the color distribution of an image.
*   **`src/neural_ode.py`**: Defines the `NeuralODE` class, which is a neural ordinary differential equation model that learns the color flow between two images.
*   **`src/inference.py`**: Contains functions for running inference with the model, including `run_inference` and `run_inference_flow`.
*   **`generate_flows_v2.py`**: A script for training the dataset of rectified flows.
*   **`train_encoder_v2.py`**: A script for training the encoder.
*   **`run_inference.py`**: A script for running inference with the model.

### How it Works

The model works by first training a set of rectified flows on a dataset of images. Each flow learns to transform the color distribution of one image to another. Then, an encoder is trained to predict the parameters of the flow that will transform a content image to a style image.

During inference, the encoder takes a content image and a style image as input and predicts the parameters of the flow that will transform the content image to the style image. The flow is then used to transform the content image, resulting in a new image with the color distribution of the style image.

### Development Guidelines

*   All new code should be documented with Google Style Python Docstrings.
*   All new functions should have corresponding unit tests.
*   All code should be formatted with `black`.
*   All code should be linted with `pylint`.
