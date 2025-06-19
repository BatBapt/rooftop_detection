# 🚀 Project Title: Image Segmentation with PyTorch

## 📝 Description
This project focuses on image processing and deep learning, specifically tackling tasks such as object detection and image segmentation using PyTorch and torchvision.

## 🛠️ Project Structure

The project consists of the following main components:

- **data_preprocessing.py**: 🖼️ Contains functions for loading and preprocessing image data. Key functions include:
  - `load_labels`: Loads labels for the dataset.
  - `generate_mask`: Generates masks for images, likely for segmentation.
  - `show_image_with_mask`: Displays images with their corresponding masks.
  - `get_mean_pts_per_poly`: Computes average points per polygon in the dataset.
  - `normalize_rooftop`: Normalizes images of rooftops.

- **model.py**: 🤖 Defines the architecture and utility functions for the deep learning model. Key functions include:
  - `get_model`: Initializes and configures the model based on specified parameters.
  - `check_trainable_params`: Checks the number of trainable parameters in the model.

- **configuration.py**: ⚙️ Centralizes global settings and configuration parameters, including paths to data and models, and device settings (CPU or GPU).

- **train.py**: 🏋️‍♂️ Manages the training process. Key functions include:
  - `get_transform`: Defines transformations to be applied to the training images.
  - `train_one_epoch`: Trains the model for one epoch.
  - `evaluate`: Evaluates the model performance on validation data.
  - `progressive_training`: Implements a progressive training strategy.
  - `compute_iou`: Computes the Intersection over Union (IoU) metric.
  - `load_model`: Loads a pre-trained model.

- **utils.py**: 🛠️ Provides utility functions and classes for logging, distributed training, and more. Key components include:
  - Utility functions for metric calculation and process synchronization.
  - Classes like `SmoothedValue` for tracking metrics and `MetricLogger` for logging.

- **dataset.py**: 📂 Implements a custom PyTorch Dataset class for handling the dataset. It includes standard methods like:
  - `__init__`: Initializes the dataset.
  - `__len__`: Returns the number of items in the dataset.
  - `__getitem__`: Fetches a specific item from the dataset.

## 📌 Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x
- PyTorch and torchvision
- OpenCV
- NumPy
- Matplotlib
- SciPy

## ↓ Installation

To install the necessary dependencies, you can use pip:

```bash
pip install opencv-python numpy matplotlib scipy
```
Please refere to the pytorch's website to install torch and torchvision: 
https://pytorch.org/get-started/locally/

## 🏁 Conclusion

Despite achieving a very low loss, the model's performance plateaus at a mean Average Precision (mAP) score of approximately 22%. 💔 Various approaches have been attempted to improve performance, but none have succeeded so far. This remains an area of active reflection and experimentation. 🧠🔍

## 🚀 Future Work

There are several TODOs scattered throughout the code that need attention. Future efforts will focus on addressing these open points and refining the model. 🛠️🔧

Additionally, there is a plan to develop a flexible architecture for MaskRCNN or FastRCNN that can adapt seamlessly to any backbone used. This would enhance the versatility and applicability of the model across different tasks and datasets. 🌟🛠️
