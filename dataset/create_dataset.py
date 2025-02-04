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


def create_mask(image_path, output_path, is_element = True):
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
        if is_element == False:
            print('check 1', image.shape)
        
        # 创建一个全白图像（255 为白色）
        # white_background = np.ones_like(alpha_channel) * 255
        
        # 将透明背景设置为白色，其他区域为黑色
        if is_element == True:
            # mask = cv2.bitwise_not(alpha_channel)  # 反转 Alpha 通道
            mask = np.where(alpha_channel == 0, 255, 0).astype(np.uint8)
        else:
            print('=======>', alpha_channel.shape)
            mask = np.where(alpha_channel == 0, 0, 255).astype(np.uint8)
        # result = cv2.merge((mask, mask, mask, white_background))
    else:
        raise ValueError("input img no alpha channel !")
    
    # 保存结果
    # cv2.imwrite(output_path, mask)

    return mask


def check_black_overlap(image1, image2):
    """
    检测两张黑白图片的黑色部分是否有重叠。

    :param image1: numpy 数组格式的黑白图像1
    :param image2: numpy 数组格式的黑白图像2
    :return: 1 表示有重叠, 0 表示无重叠
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

if __name__ == "__main__":
    # 测试函数
    element_path = "mask_20241202-170222-294.png"
    gripper_path = '1.png'
    output_path = "output_mask.png"

    element_mask = create_mask(element_path, output_path, True)
    gripper_mask = create_mask(gripper_path, output_path, False)

    cv2.imshow("Mask element", element_mask)
    cv2.imshow("Mask Image", gripper_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()   

    image1 = element_mask
    image2 = gripper_mask

    result = check_black_overlap(image1, image2)
    print("是否有重叠:", result)