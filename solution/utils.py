import cv2
import numpy as np


def create_mask(image_path, output_path = None, is_element = True):
    """
    将背景为透明的图片中的透明背景设置为白色，其余部分设置为黑色。
    
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    """
    # 加载图片，保留 Alpha 通道
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 检查是否有 Alpha 通道
    # print(image.shape)
    if len(image.shape) == 3 and image.shape[2] == 4:
        # 提取 Alpha 通道
        alpha_channel = image[:, :, 3]
        #if is_element == False:
            #print('check 1', image.shape)
        
        # 创建一个全白图像（255 为白色）
        # white_background = np.ones_like(alpha_channel) * 255
        
        # 将透明背景设置为白色，其他区域为黑色
        if is_element == True:
            # mask = cv2.bitwise_not(alpha_channel)  # 反转 Alpha 通道
            mask = np.where(alpha_channel == 0, 255, 0).astype(np.uint8)
        else:
            #print('=======>', alpha_channel.shape)
            mask = np.where(alpha_channel == 0, 0, 255).astype(np.uint8)
        # result = cv2.merge((mask, mask, mask, white_background))
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # 假设三通道输入，透明区域定义为纯黑色（0,0,0）
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray_image == 0, 255 if is_element else 0, 0 if is_element else 255).astype(np.uint8)
    elif len(image.shape) == 2:
        mask = np.where(image == 0, 255 if is_element else 0, 0 if is_element else 255).astype(np.uint8)
        
    else:
        raise ValueError("Input image must be either 1 or 3 or 4 channels")
    # 保存结果
    # cv2.imwrite(output_path, mask)

    return mask


def check_black_overlap(image1, image2, rotate = None):
    """
    检测两张黑白图片的黑色部分是否有重叠。

    :param image1: numpy 数组格式的黑白图像1
    :param image2: numpy 数组格式的黑白图像2
    :return: 1 表示有重叠，0 表示无重叠
    """
    # 确保输入图像是单通道
    if len(image1.shape) != 2 or len(image2.shape) != 2:
        raise ValueError("input image must be black and white !")

    # 获取两张图片的大小
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # 找到最大尺寸
    max_height = max(h1, h2)
    max_width = max(w1, w2)

    # 扩展两张图片到相同大小，用白色填充（255）
    image1_padded = cv2.copyMakeBorder(image1, 0, max_height - h1, 0, max_width - w1, cv2.BORDER_CONSTANT, value=255)
    image2_padded = cv2.copyMakeBorder(image2, 0, max_height - h2, 0, max_width - w2, cv2.BORDER_CONSTANT, value=255)

    # 找到两张图片的黑色部分（像素值为 0）的重叠区域
    mask1 = (image1_padded == 0).astype(np.uint8)
    mask2 = (image2_padded == 0).astype(np.uint8)
    overlap = cv2.bitwise_and(mask1, mask2)

    # 判断重叠区域是否存在黑色像素
    if np.any(overlap):  # 如果有至少一个像素为 True（即重叠）
        return 1
    else:
        return 0

def euclidean_distance(coord1, coord2):
    """
    计算两个n维坐标的欧几里得距离

    参数:
    coord1 (ndarray): 第一个坐标 (n维数组)
    coord2 (ndarray): 第二个坐标 (n维数组)

    返回:
    float: 两个坐标之间的欧几里得距离
    """
    coord1 = np.array(coord1)
    coord2 = np.array(coord2)
    distance = np.linalg.norm(coord1 - coord2)  # 使用 NumPy 计算欧几里得距离
    return distance