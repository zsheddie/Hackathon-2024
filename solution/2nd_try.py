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

import cv2
import numpy as np

def check_overlap(image1, image2, x, y, rotate=None):
    """
    检查第二张图片是否与第一张图片的非透明区域重叠。

    :param image1: 第一张图片，黑白图像（numpy数组，单通道）
    :param image2: 第二张图片，带透明通道的图像（numpy数组，4通道）
    :param x: 第二张图的中心点 x 坐标（以第一张图的左上角为原点）
    :param y: 第二张图的中心点 y 坐标（以第一张图的左上角为原点）
    :param rotate: 旋转角度（顺时针方向，单位：度）
    :return: 是否重叠，1表示有重叠，0表示没有重叠
    """
    # 获取第二张图的高度和宽度
    h2, w2 = image2.shape[:2]
    center = (w2 // 2, h2 // 2)

    # 如果需要旋转，旋转图2
    if rotate is not None:
        rotation_matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
        rotated_image = cv2.warpAffine(
            image2, rotation_matrix, (w2, h2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)  # 使用透明背景
        )
        # 恢复alpha通道
        alpha_channel = cv2.warpAffine(
            image2[:, :, 3], rotation_matrix, (w2, h2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        rotated_image = np.dstack([rotated_image[:, :, :3], alpha_channel])
    else:
        rotated_image = image2

    # 获取第一张图的高度和宽度
    h1, w1 = image1.shape[:2]

    # 计算图2的左上角偏移量
    x_offset = x - w2 // 2
    y_offset = y - h2 // 2

    # 确定重叠区域的坐标范围
    overlap_x_start = max(x_offset, 0)
    overlap_y_start = max(y_offset, 0)
    overlap_x_end = min(x_offset + w2, w1)
    overlap_y_end = min(y_offset + h2, h1)

    # 如果没有重叠区域，直接返回0
    if overlap_x_start >= overlap_x_end or overlap_y_start >= overlap_y_end:
        return 0

    # 提取图1的对应重叠部分
    overlap_region_image1 = image1[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end]

    # 提取图2的对应重叠部分
    region_x_start = overlap_x_start - x_offset
    region_y_start = overlap_y_start - y_offset
    region_x_end = region_x_start + (overlap_x_end - overlap_x_start)
    region_y_end = region_y_start + (overlap_y_end - overlap_y_start)
    overlap_region_image2_alpha = rotated_image[region_y_start:region_y_end, region_x_start:region_x_end, 3]  # Alpha通道

    # 生成掩膜
    mask1 = (overlap_region_image1 != 0).astype(np.uint8)  # 图1的非黑色区域
    mask2 = (overlap_region_image2_alpha > 0).astype(np.uint8)  # 图2的非透明区域

    # 判断是否有重叠
    overlap = (mask1 & mask2).any()

    return 1 if overlap else 0

def find_valid_position_and_rotation(image1_path, image2_path):
    """
    寻找第二张图的位置和旋转角度，使得它的非透明部分不与第一张图的黑色部分重叠。
    
    :param image1_path: 第一张图片路径
    :param image2_path: 第二张图片路径
    :return: (x, y, angle) 表示找到的有效坐标和旋转角度
    """
    # 加载图片
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)  # 第一张图为黑白图
    image2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)  # 第二张图有透明通道
    
    # 获取第一张图的中心点
    h1, w1 = image1.shape[:2]
    center1_x, center1_y = w1 // 2, h1 // 2
    
    # 初始化旋转角度和搜索步长
    rotate_step = 5  # 旋转角度步长为 2 度
    max_distance = max(h1, w1) // 2  # 最大的搜索半径
    
    # 遍历从第一张图中心开始的所有可能坐标（包含外圈像素）
    for r in range(0, max_distance + 1):  # r 是从中心到外圈的半径
        
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for angle in tqdm(range(0, 180, rotate_step)):  # 遍历旋转角度
                    x = center1_x + dx
                    y = center1_y + dy
                    # print(x, y)
                    # 检查当前位置和旋转角度是否有效
                    # if x == 95 and dy == 0 and angle == 0:
                    #     visualize_image_transform(image1_path, image2_path, x, y, angle, angle)
                    if check_overlap(image1, image2, x, y, rotate=angle) == 0:
                        visualize_image_transform(image1_path, image2_path, x, y, angle, angle)
                        return x, y, angle
        
    # 如果没有找到合适的坐标和角度，返回 None
    return None, None, None

# image1_path = 'D:/hackathon2024/Hackathon-2024/data/evaluate/2/binary_mask_2.png'
# image2_path = 'D:/hackathon2024/Hackathon-2024/data/evaluate/2/gripper_1.png'

image1_path = 'D:/hackathon2024/Hackathon-2024/data/evaluate/4/binary_mask_4.png'
image2_path = 'D:/hackathon2024/Hackathon-2024/data/evaluate/4/gripper_5.png'

x, y, angle = find_valid_position_and_rotation(image1_path, image2_path)

if x is not None and y is not None and angle is not None:
    print(f"Found a valid position and rotation angle: coordinates = ({x}, {y}), rotation angle = {angle}°")
else:
    print("Failed to find a position and rotation angle that meet the conditions.")
