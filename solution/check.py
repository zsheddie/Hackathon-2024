import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from utils import *
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import os
import sys
from PIL import Image
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append('D:/hackathon2024/Hackathon-2024/solution')