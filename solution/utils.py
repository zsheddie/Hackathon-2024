import cv2
import numpy as np
import os


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
    
    if rotate is not None:
        h2, w2 = image2.shape
        center = (w2 // 2, h2 // 2)  # 图片中心
        rotation_matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)  # 生成旋转矩阵
        image2 = cv2.warpAffine(image2, rotation_matrix, (w2, h2), borderValue=255)  # 旋转并填充白色

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

def interactive_canny(image_name):
    """
    Interactive window to adjust Canny edge detection parameters in real-time
    
    Parameters:
        image_name (str): Name of the image file in the current directory
    """
    def nothing(x):
        pass

    # Read image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_dir, image_name)
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_name}' not found in current directory")
        return
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Create window with trackbars
    cv2.namedWindow('Canny Edge Detection')
    cv2.createTrackbar('Low Threshold', 'Canny Edge Detection', 0, 255, nothing)
    cv2.createTrackbar('Hh Threshold', 'Canny Edge Detection', 0, 255, nothing)

    while True:
        # Get current positions of trackbars
        low = cv2.getTrackbarPos('Low Threshold', 'Canny Edge Detection')
        high = cv2.getTrackbarPos('Hh Threshold', 'Canny Edge Detection')

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low, high)
        
        # Show the image
        cv2.imshow('Canny Edge Detection', edges)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Save the final result
            output_name = 'canny_' + image_name
            output_path = os.path.join(current_dir, output_name)
            cv2.imwrite(output_path, edges)
            break

    cv2.destroyAllWindows()

def visualize_image_transform(image1_path, image2_path, x_offset, y_offset, rotation_angle, output_path):
    # 加载第一张图像
    image1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    if image1 is None:
        raise FileNotFoundError(f"无法加载图片：{image1_path}")

    # 加载第二张图像
    image2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)
    if image2 is None:
        raise FileNotFoundError(f"无法加载图片：{image2_path}")

    # 如果第一张图像是 3 通道，添加一个 alpha 通道并设置为不透明
    print(image1.shape)
    if image1.shape[2] == 3:
        alpha_channel = np.ones((image1.shape[0], image1.shape[1], 1), dtype=np.uint8) * 255
        image1 = np.concatenate((image1, alpha_channel), axis=2)

    # 确保第二张图像有 alpha 通道
    if image2.shape[2] != 4:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2BGRA)

    # 获取第一张图像的尺寸
    h1, w1 = image1.shape[:2]

    # 获取第二张图像的尺寸
    h2, w2 = image2.shape[:2]

    # 计算旋转中心（第二张图的中心）
    center = (w2 // 2, h2 // 2)

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 将第二张图像旋转
    rotated_image = cv2.warpAffine(image2, rotation_matrix, (w2, h2), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    # 计算最终图像的尺寸
    new_width = max(w1, w2 + x_offset)
    new_height = max(h1, h2 + y_offset)

    # 创建一个新的空白图像，背景填充为透明
    result_image = np.zeros((new_height, new_width, 4), dtype=np.uint8)

    # 将第一张图像粘贴到空白图像的左上角
    result_image[:h1, :w1] = image1

    # 计算第二张图放置的位置，考虑偏移量
    x_pos = max(0, x_offset)
    y_pos = max(0, y_offset)

    # 将旋转后的第二张图像粘贴到结果图像中
    for i in range(h2):
        for j in range(w2):
            if 0 <= i + y_pos < new_height and 0 <= j + x_pos < new_width:
                # 如果该位置在结果图像内，且不是透明像素，则将像素放置进去
                if rotated_image[i, j][3] > 0:  # 检查 alpha 通道
                    result_image[i + y_pos, j + x_pos] = rotated_image[i, j]

    # 保存结果图像
    output_full_path = f"output_visualize/test_visualization_{output_path}.png"
    cv2.imwrite(output_full_path, result_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"结果图像已保存到 {output_full_path}")