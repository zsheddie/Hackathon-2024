import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from utils import create_mask, check_black_overlap, euclidean_distance, interactive_canny  ## import *
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import sys
from PIL import Image
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt



if __name__ == "__main__":
    img_path = 'D:/hackathon2024/Hackathon-2024/data/raw/part_1/mask_20241202-170222-294.png'
    interactive_canny(img_path)

# for i in range(100):
#     if i % 10 == 9:
#         print(i)