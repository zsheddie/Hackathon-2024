import cv2
import numpy as np

# 加载透明背景的大图 (RGBA 格式)
image = cv2.imread("mask_20241204-112748-242i + 1.png", cv2.IMREAD_UNCHANGED)  # 包含透明背景
mask = cv2.imread("mask_20241202-114427-169.png", cv2.IMREAD_GRAYSCALE)  # 加载mask图像 (灰度)

# 确保 mask 是二值化的
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# 查找轮廓
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 如果找到轮廓，确定位置
if contours:
    # 假设只存在一个主要的轮廓
    largest_contour = max(contours, key=cv2.contourArea)  # 找到最大轮廓

    # 计算外接矩形
    x, y, w, h = cv2.boundingRect(largest_contour)
    print(f"位置: 左上角({x}, {y}), 宽度: {w}, 高度: {h}")

    # 可选：在原始图像上可视化结果
    annotated_image = image.copy()
    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0, 255), 2)  # 绿色矩形框

    # 保存结果
    cv2.imwrite("annotated_image.png", annotated_image)
else:
    print("未检测到有效的轮廓！")
