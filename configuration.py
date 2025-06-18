import os
import torch

BASE_PATH = "D:/Programmation/IA/datas/rooftop"

MODEL_HUB = "D:/models/torch/hub"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')