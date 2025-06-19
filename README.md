# Project Title

## Description
This project is focused on image processing and deep learning, specifically involving tasks such as object detection or image segmentation using PyTorch and torchvision.

## Project Structure

The project consists of the following main components:

- **data_preprocessing.py**: Contains functions for loading and preprocessing image data. Key functions include:
  - `load_labels`: Loads labels for the dataset.
  - `generate_mask`: Generates masks for images, likely for segmentation.
  - `show_image_with_mask`: Displays images with their corresponding masks.
  - `get_mean_pts_per_poly`: Computes average points per polygon in the dataset.
  - `normalize_rooftop`: Normalizes images of rooftops.

- **model.py**: Defines the architecture and utility functions for the deep learning model. Key functions include:
  - `get_model`: Initializes and configures the model based on specified parameters.
  - `check_trainable_params`: Checks the number of trainable parameters in the model.

- **configuration.py**: Centralizes global settings and configuration parameters, including paths to data and models, and device settings (CPU or GPU).

- **train.py**: Manages the training process. Key functions include:
  - `get_transform`: Defines transformations to be applied to the training images.
  - `train_one_epoch`: Trains the model for one epoch.
  - `evaluate`: Evaluates the model performance on validation data.
  - `progressive_training`: Implements a progressive training strategy.
  - `compute_iou`: Computes the Intersection over Union (IoU) metric.
  - `load_model`: Loads a pre-trained model.

- **utils.py**: Provides utility functions and classes for logging, distributed training, and more. Key components include:
  - Utility functions for metric calculation and process synchronization.
  - Classes like `SmoothedValue` for tracking metrics and `MetricLogger` for logging.

- **dataset.py**: Implements a custom PyTorch Dataset class for handling the dataset. It includes standard methods like:
  - `__init__`: Initializes the dataset.
  - `__len__`: Returns the number of items in the dataset.
  - `__getitem__`: Fetches a specific item from the dataset.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- PyTorch and torchvision
- OpenCV
- NumPy
- Matplotlib
- SciPy

## Installation

To install the necessary dependencies, you can use pip:

```bash
pip install torch torchvision opencv-python numpy matplotlib scipy
```

## Training Configuration

The training of the model is configured via a YAML file that outlines different stages of training, each with its specific settings for epochs, learning rate (lr), and step size for learning rate adjustment.

Here is an outline of the training stages:

- **Stage 1**:
  - Title: "STAGE 1: Training heads only + backbone frozen"
  - Epochs: 500
  - Learning Rate: 0.01
  - Step Size: 400

  This stage focuses on training only the head layers while the backbone is frozen to ensure stable initial learning.

- **Stage 2**:
  - Title: "STAGE 2: Training first backbone's layer"
  - Epochs: 100
  - Learning Rate: 0.005
  - Step Size: 70

  The first layer of the backbone is unfrozen and trained with a lower learning rate to fine-tune initial layers.

- **Stage 3**:
  - Title: "STAGE 3: Training everything"
  - Epochs: 200
  - Learning Rate: 0.005
  - Step Size: 150 
  
  All layers of the model are trained during this stage to fine-tune the entire model.
