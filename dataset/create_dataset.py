import cv2
import numpy as np

# 读取图像
image = cv2.imread('mask_20241202-170222-294.png')  # 加载图像

print("image data type:", type(image))  # 类型是 NumPy 数组
print("image shape:", image.shape)  # 打印形状 (height, width, channels)
# print("图片数据 (前5行):", image[:5])  # 打印前5行数据

### print a random coordinates
height, width = image.shape[0], image.shape[1]
y = np.random.randint(0, height)
x = np.random.randint(0, width)
alpha = np.random.randint(0, 360)

print("Random coordinates:", (x, y, alpha))

## batch process
num_points = 1000
height, width = image.shape[0], image.shape[1]
y = np.random.randint(0, height, num_points)
x = np.random.randint(0, width, num_points)
alpha = np.random.randint(0, 360, num_points)
coords = list(zip(x, y, alpha))

def detect_overlap(image1, image2):
    """
    检测两幅图像的图案部分是否有重叠
    :param image1: 第一幅图像（NumPy 数组）
    :param image2: 第二幅图像（NumPy 数组）
    :return: 是否有重叠 (True/False)
    """
    # 转为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # 二值化处理
    _, binary1 = cv2.threshold(gray1, 128, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 128, 255, cv2.THRESH_BINARY)

    cv2.imshow("Binary Image", binary1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Binary Image", binary2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 检测交集部分
    binary2_resized = cv2.resize(binary2, (binary1.shape[1], binary1.shape[0]))
    overlap = cv2.bitwise_and(binary1, binary2_resized)  # 计算交集
    has_overlap = np.any(overlap > 0)  # 检查是否存在交集
    
    return has_overlap

import cv2
import numpy as np

def detect_overlap(image1, image2):
    """
    使用图像分割和逻辑运算检测图案是否重叠
    """
    # 转为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 使用 Canny 边缘检测提取边缘
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)

    cv2.imshow("Binary Image", edges1)
    cv2.imshow("Binary Image", edges2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 调整图像大小一致
    if edges1.shape != edges2.shape:
        edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))

    # 检测重叠区域
    overlap = cv2.bitwise_and(edges1, edges2)

    # 判断是否有交集
    has_overlap = np.any(overlap > 0)
    return has_overlap

def create_mask(image_path, output_path):
    """
    将背景为透明的图片中的透明背景设置为白色，其余部分设置为黑色。
    
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径
    """
    # 加载图片，保留 Alpha 通道
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 检查是否有 Alpha 通道
    if image.shape[2] == 4:
        # 提取 Alpha 通道
        alpha_channel = image[:, :, 3]
        
        # 创建一个全白图像（255 为白色）
        white_background = np.ones_like(alpha_channel) * 255
        
        # 将透明背景设置为白色，其他区域为黑色
        mask = cv2.bitwise_not(alpha_channel)  # 反转 Alpha 通道
        # result = cv2.merge((mask, mask, mask, white_background))
    else:
        raise ValueError("input img no alpha channel !")
    
    # 保存结果
    # cv2.imwrite(output_path, mask)

    return mask

# 测试函数
element_path = "mask_20241202-170222-294.png"
gripper_path = '1.png'
output_path = "output_mask.png"

element_mask = create_mask(element_path, output_path)
gripper_mask = create_mask(gripper_path, output_path)
print(element_mask.shape)


# 显示结果
# cv2.imshow("Mask Image", cv2.imread(output_path, cv2.IMREAD_UNCHANGED))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''
# 读取两幅图像
image1 = cv2.imread('mask_20241202-170222-294.png')
image2 = cv2.imread('1.png')  # 替换为实际图像路径

# 检测重叠
overlap = segment_and_detect_overlap(image1, image2)
print("图案是否重叠:", overlap)
'''