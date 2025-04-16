import cv2
import numpy as np
import os


def convert_to_hsv(img_path):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print("无法加载图像，请检查路径是否正确。")
        return

    # 进行高斯滤波
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    # 进行腐蚀操作
    kernel = np.ones((5, 5), np.uint8)
    eroded_img = cv2.erode(blurred_img, kernel, iterations=3)

    # 进行膨胀操作
    dilated_img = cv2.dilate(eroded_img, kernel, iterations=3)

    # 转换到 HSV 颜色空间
    hsv_img = cv2.cvtColor(dilated_img, cv2.COLOR_BGR2HSV)

    # 定义红色的 HSV 范围（由于红色在 HSV 中分布在两个区域）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 创建掩码
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 应用掩码到原始图像
    red_img = cv2.bitwise_and(img, img, mask=mask)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 保存红色区域的坐标信息
    red_areas_coordinates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        red_areas_coordinates.append((x, y, x + w, y + h))

    # 输出红色区域的坐标值
    if red_areas_coordinates:
        print("红色区域的坐标值（左上角和右下角坐标）：")
        for coord in red_areas_coordinates:
            print(coord)
    else:
        print("未检测到红色区域。")

    # 在原图上绘制矩形框
    img_with_boxes = img.copy()
    for (x1, y1, x2, y2) in red_areas_coordinates:
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保存 HSV 图像、红色二值化图像和带框的原图
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    hsv_img_path = f"{name}_hsv{ext}"
    red_img_path = f"{name}_red{ext}"
    img_with_boxes_path = f"{name}_boxed{ext}"
    cv2.imwrite(hsv_img_path, hsv_img)
    cv2.imwrite(red_img_path, red_img)
    cv2.imwrite(img_with_boxes_path, img_with_boxes)

    # 显示保存信息
    print(f"HSV 图像已保存为: {hsv_img_path}")
    print(f"红色二值化图像已保存为: {red_img_path}")
    print(f"带框的原图已保存为: {img_with_boxes_path}")
    return hsv_img_path, red_img_path, img_with_boxes_path


if __name__ == '__main__':
    img_path = 'your_image_path.jpg'  # 替换为你的图像路径
    convert_to_hsv(img_path)
