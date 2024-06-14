import random

import cv2
import numpy as np
def generate_random_number():
    # 先随机选择在哪个范围内生成数值
    if random.choice([True, False]):
        # 在-90到-30之间生成随机数
        return random.uniform(-90, -30)
    else:
        # 在30到90之间生成随机数
        return random.uniform(30, 90)

def rotate_imgwbox(image, bbox):
    angle =generate_random_number()
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新图像的尺寸以适应旋转
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((image.shape[0] * sin) + (image.shape[1] * cos))
    new_h = int((image.shape[0] * cos) + (image.shape[1] * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 旋转图像
    rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

    # 包围框四个顶点
    points = np.array([
        [bbox[0], bbox[1]],
        [bbox[0] + bbox[2], bbox[1]],
        [bbox[0], bbox[1] + bbox[3]],
        [bbox[0] + bbox[2], bbox[1] + bbox[3]]
    ])

    # 变换顶点
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])

    transformed_points = M.dot(points_ones.T).T

    # 获取新的包围框
    x_min, y_min = np.min(transformed_points, axis=0)[:2]
    x_max, y_max = np.max(transformed_points, axis=0)[:2]

    rotated_bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

    return rotated_image, rotated_bbox
