import os
import torch

BASE_PATH = "D:/Programmation/IA/datas/rooftop"  # your data_path

MODEL_HUB = "D:/models/torch/hub"  # where you want to download your models from pytorch

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')