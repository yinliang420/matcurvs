import cv2
import base64
import numpy as np
import itertools
import json
import string
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path

from ModalLLMResponse import ModalLLMResponse
from Axis_yolo_extraction import detect_axes_yolo
from ocr_compat import create_ocr, ocr_predict
from ultralytics import YOLO

# 加载YOLO模型
try:
    script_dir = Path(__file__).resolve().parent
    MODEL_PATH = script_dir / "yolo" / "yolov11n_lenged_clutter_detector" / "weights" / "best.pt"
    if Path(MODEL_PATH).exists():
        print(f"正在加载YOLO图例检测模型: {MODEL_PATH}")
        legend_model = YOLO(MODEL_PATH)
        print("模型加载成功。")
    else:
        legend_model = None
        print(f"警告：找不到模型文件 {MODEL_PATH}。图例检测将不可用。")
except Exception as e:
    legend_model = None
    print(f"加载YOLO模型时出错: {e}")


def encode_image(pat: str) -> str:
    """将图像编码为base64字符串"""
    with open(pat, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_letter_sequence(length):
    """
    根据长度生成 [A, B, C, D, ...] 的字母序列
    参数:
    - length: 所需序列的长度
    返回:
    - 字母序列列表
    """
    letters = []
    alphabet = string.ascii_uppercase  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    while len(letters) < length:
        # 添加单字母
        for letter in alphabet:
            if len(letters) >= length:
                break
            letters.append(letter)

        # 添加双字母（如 AA, AB, AC, ...）
        for first_letter in alphabet:
            for second_letter in alphabet:
                if len(letters) >= length:
                    break
                letters.append(first_letter + second_letter)

    return letters[:length]


def create_axis_mask(image_shape, axis_lines):
    """
    创建坐标轴区域掩膜，mask掉边框外的所有内容
    :param image_shape: 图像的形状 (height, width)
    :param axis_lines: 坐标轴线段列表 [(x1,y1,x2,y2), ...]，其中第一个为 x 轴，第二个为 y 轴
    :return: 边框外的掩膜（边框内为 0，边框外为 255）
    """
    # 获取图像的高度和宽度
    height, width = image_shape[:2]

    # 提取 x 轴和 y 轴的坐标
    x_axis = axis_lines[0]  # x 轴线段
    y_axis = axis_lines[1]  # y 轴线段

    # 计算边框的边界
    x1 = min(x_axis[0], x_axis[2], y_axis[0], y_axis[2]) + 4  # 左边界
    y1 = min(x_axis[1], x_axis[3], y_axis[1], y_axis[3]) + 4  # 上边界
    x2 = max(x_axis[0], x_axis[2], y_axis[0], y_axis[2]) - 4  # 右边界
    y2 = max(x_axis[1], x_axis[3], y_axis[1], y_axis[3]) - 4  # 下边界

    # 确保边界在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # 创建掩膜
    mask = np.ones((height, width), dtype=np.uint8) * 255  # 初始值为 255（边框外）
    cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)  # 边框内填充黑色（0）

    return mask


def create_text_mask(image, ocr, expand=1):
    """创建OCR文字区域掩膜"""
    result = ocr_predict(ocr, image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for line in result:
        if line is None:
            continue
        for word in line:
            box = word[0]
            x_coords = [int(p[0]) for p in box]
            y_coords = [int(p[1]) for p in box]
            x1, x2 = min(x_coords) - expand, max(x_coords) + expand
            y1, y2 = min(y_coords) - expand, max(y_coords) + expand
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    return mask


def detect_objects_with_yolo(model, image_path_str):
    """
    使用训练好的YOLO模型检测图像中的所有对象（例如图例和杂图）。

    :param model: 已加载的YOLO模型对象。
    :param image_path_str: 待检测图片的路径字符串。
    :return: 返回一个包含所有检测到的对象坐标字典的【列表】。
             列表中的每个字典格式为 {'top_left': [x1, y1], 'bottom_right': [x2, y2]}。
             如果没有检测到对象，则返回一个空列表 []。
    """
    if model is None:
        return []  # 返回空列表

    try:
        # 使用模型进行预测，verbose=False可以减少不必要的控制台输出
        results = model.predict(source=image_path_str, conf=0.5, verbose=False)
        result = results[0]
        detected_boxes = result.boxes

        if len(detected_boxes) == 0:
            return []  # 没有检测到对象，返回空列表

        # 创建一个列表来存储所有检测到的框
        all_detections = []
        
        # 遍历所有检测到的边界框
        for box in detected_boxes:
            # 提取边界框的左上角和右下角坐标
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)

            # 将坐标格式化为字典并添加到列表中
            detection_dict = {'top_left': [x1, y1], 'bottom_right': [x2, y2]}
            all_detections.append(detection_dict)
            
        return all_detections

    except Exception as e:
        print(f"使用YOLO模型检测对象时发生错误: {e}")
        return []  # 发生错误时也返回空列表


def create_legend_mask(img, legend_boxes):
    """
    生成图例区域掩膜，可以处理单个或多个图例。

    参数:
    img (numpy.ndarray): 输入图像 (用于获取尺寸)。
    legend_boxes (list of dict): 一个包含坐标字典的列表。
        每个字典代表一个图例框，格式为 {'top_left': [x1, y1], 'bottom_right': [x2, y2]}。
        例如: [{'top_left': [50, 50], 'bottom_right': [150, 100]}, ...]

    返回:
    mask (numpy.ndarray): 与输入图像尺寸相同的二值掩膜，
        所有图例区域的值为 255，其余区域为 0。
    """
    # 创建一个与图像等大的全零掩膜
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # 如果输入的列表为空，直接返回全零掩膜
    if not legend_boxes:
        return mask

    # 遍历列表中的每一个图例框
    for dict_legend in legend_boxes:
        # 确保字典格式正确
        if dict_legend and 'top_left' in dict_legend and 'bottom_right' in dict_legend:
            top_left = dict_legend['top_left']
            bottom_right = dict_legend['bottom_right']
            
            # 在掩膜上，将当前图例的矩形区域设置为 255
            mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255
            
    return mask


def combine_masks(masks):
    """合并多个掩膜"""
    combined = np.zeros_like(masks[0])
    for mask in masks:
        combined = cv2.bitwise_or(combined, mask)
    return combined


def ask_mm_model_for_line_colors_and_count(image_path, response_modal_llm: ModalLLMResponse):
    """
    使用多模态大模型识别线条数量和每条线条的颜色
    
    参数:
    - image_path: 原始图像路径
    - response_modal_llm: 多模态大模型响应对象
    
    返回:
    - line_info: 包含线条数量和颜色信息的字典
        {
            "total_lines_num": int,
            "lines": [
                {"line_id": 1, "color": "red", "rgb": [255, 0, 0]},
                {"line_id": 2, "color": "blue", "rgb": [0, 0, 255]},
                ...
            ]
        }
    """
    print("开始使用多模态大模型识别线条数量和颜色...")
    
    # 准备提示词
    prompt = """
**Line Count and Color Detection Protocol**

- **Task:**
  Analyze the chart image provided and identify:
  1. The total number of distinct data trend lines in the chart
  2. The color of each data line (use common color names like: red, blue, green, yellow, orange, purple, cyan, magenta, brown, pink, etc.)

- **Detection Rules:**
  1. Count only actual data trend lines (curves that represent data values).
  2. Exclude:
     - Grid lines (regular patterns)
     - Axis lines (border lines)
     - Text labels
     - Background areas
  3. For each data line, identify its primary color. If a line has multiple colors or gradients, identify the dominant color.
  4. Use standard color names (red, blue, green, yellow, orange, purple, cyan, magenta, brown, pink, black, gray, etc.)

- **Output Specification:**
  Strict JSON format:
  ```json
  {
    "total_lines_num": integer,
    "lines": [
      {"line_id": 1, "color": "red", "rgb": [255, 0, 0]},
      {"line_id": 2, "color": "blue", "rgb": [0, 0, 255]},
      ...
    ]
  }
  ```
  
  Example:
  {
    "total_lines_num": 3,
    "lines": [
      {"line_id": 1, "color": "red", "rgb": [255, 0, 0]},
      {"line_id": 2, "color": "blue", "rgb": [0, 0, 255]},
      {"line_id": 3, "color": "green", "rgb": [0, 255, 0]}
    ]
  }
  
  Note: RGB values should be in range [0, 255] for each channel.
"""
    
    # 调用多模态大模型
    image_base64 = encode_image(image_path)
    response = response_modal_llm.generate_response(image_base64, prompt)
    # print(response, type(response))
    # generate_response已经返回字典，无需再json.loads
    try:
        total_lines_num = response.get("total_lines_num", 0)
        lines = response.get("lines", [])
        print(f"多模态大模型识别到 {total_lines_num} 条线条")
        for line in lines:
            print(f"  线条 {line.get('line_id')}: {line.get('color')} (RGB: {line.get('rgb')})")
        return {
            "total_lines_num": total_lines_num,
            "lines": lines
        }
    except Exception as e:
        print(f"An error occurred while parsing response: {e}")
        print(f"Response: {response}")
        return None


def color_name_to_hsv_range(color_name):
    """
    将颜色名称转换为HSV颜色空间的范围
    
    参数:
    - color_name: 颜色名称字符串
    
    返回:
    - lower_bound: HSV下界 [H_min, S_min, V_min]
    - upper_bound: HSV上界 [H_max, S_max, V_max]
    """
    color_name_lower = color_name.lower()
    
    # HSV颜色范围定义 (OpenCV中H: 0-179, S: 0-255, V: 0-255)
    color_ranges = {
        "red": ([0, 50, 50], [10, 255, 255]),  # 红色范围1
        "green": ([40, 50, 50], [80, 255, 255]),  # 绿色范围
        "blue": ([100, 50, 50], [130, 255, 255]),  # 蓝色范围
        "yellow": ([20, 50, 50], [30, 255, 255]),  # 黄色范围
        "orange": ([10, 50, 50], [20, 255, 255]),  # 橙色范围
        "purple": ([130, 50, 50], [160, 255, 255]),  # 紫色范围
        "cyan": ([85, 50, 50], [100, 255, 255]),  # 青色范围
        "magenta": ([160, 50, 50], [180, 255, 255]),  # 品红色范围
        "pink": ([160, 50, 50], [180, 255, 255]),  # 粉色范围（接近品红）
        "brown": ([10, 50, 20], [20, 255, 200]),  # 棕色范围
        "black": ([0, 0, 0], [180, 255, 30]),  # 黑色范围（低亮度）
        "white": ([0, 0, 200], [180, 30, 255]),  # 白色范围（高亮度，低饱和度）
        "gray": ([0, 0, 50], [180, 30, 200]),  # 灰色范围
    }
    
    if color_name_lower in color_ranges:
        return color_ranges[color_name_lower]
    else:
        # 默认返回一个较宽的范围
        print(f"警告：未知颜色名称 '{color_name}'，使用默认范围")
        return ([0, 50, 50], [180, 255, 255])


def extract_pixels_by_color_name(image, color_name):
    """
    基于颜色名称从图像中提取属于该颜色的所有像素点
    
    参数:
    - image: 输入图像 (BGR格式)
    - color_name: 颜色名称字符串
    
    返回:
    - mask: 二值掩膜，属于该颜色的像素为255，其他为0
    """
    # 转换为HSV颜色空间（更适合颜色匹配）
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 获取颜色对应的HSV范围
    lower_bound, upper_bound = color_name_to_hsv_range(color_name)
    lower = np.array(lower_bound, dtype=np.uint8)
    upper = np.array(upper_bound, dtype=np.uint8)
    
    # 创建颜色掩膜
    mask = cv2.inRange(hsv_image, lower, upper)
    
    # 对于红色，需要处理HSV中的红色跨越0度的情况
    if color_name.lower() == "red":
        # 红色在HSV中跨越0度，需要两个范围
        lower1 = np.array([0, 50, 50], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 50, 50], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv_image, lower1, upper1)
        mask2 = cv2.inRange(hsv_image, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    
    return mask


def extract_lines_by_color_names(image, color_names, extract_extra=True):
    """
    基于颜色名称列表从图像中提取属于这些颜色的所有像素点
    
    参数:
    - image: 输入图像 (BGR格式)
    - color_names: 颜色名称列表，例如 ["red", "blue", "green"]
    - extract_extra: 是否提取额外的"其他颜色"类别（用于捕获遗漏的线条）
    
    返回:
    - color_masks: 每个颜色对应的掩膜列表（如果extract_extra=True，会多一个"其他颜色"掩膜）
    """
    color_masks = []
    
    # 提取指定颜色的像素点
    for color_name in color_names:
        print(f"  提取颜色 '{color_name}' 的像素点...")
        mask = extract_pixels_by_color_name(image, color_name)
        color_masks.append(mask)
    
    # 如果需要，提取额外的"其他颜色"类别
    if extract_extra and len(color_names) > 0:
        print(f"  提取额外的'其他颜色'类别（用于捕获遗漏的线条）...")
        # 合并所有已提取的颜色掩膜
        combined_mask = np.zeros_like(color_masks[0])
        for mask in color_masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 找到非白色背景的区域（排除已经被掩膜处理过的区域）
        # 白色背景的HSV范围：低饱和度，高亮度
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv_image, np.array([0, 0, 200]), np.array([180, 30, 255]))
        
        # "其他颜色" = 非白色背景 且 不在已提取的颜色中
        other_mask = cv2.bitwise_and(
            cv2.bitwise_not(white_mask),
            cv2.bitwise_not(combined_mask)
        )
        
        # 只保留有一定大小的连通区域（过滤噪声）
        kernel = np.ones((3, 3), np.uint8)
        other_mask = cv2.morphologyEx(other_mask, cv2.MORPH_OPEN, kernel)
        other_mask = cv2.morphologyEx(other_mask, cv2.MORPH_CLOSE, kernel)
        
        color_masks.append(other_mask)
    
    return color_masks


def compute_mask_features(mask):
    """
    计算掩码的特征值（用于选择最一致的掩码组合）
    
    参数:
    - mask: 二值掩码 (0为背景，255为前景)
    
    返回:
    - feature_value: 特征值（像素数量，用于一致性判断）
    """
    # 计算掩码的有效像素数量
    pixel_count = np.sum(mask > 0)
    
    # 计算连通区域数量
    num_labels, labels_img = cv2.connectedComponents(mask)
    num_components = num_labels - 1  # 减去背景
    
    # 计算连通区域的平均大小
    if num_components > 0:
        component_sizes = []
        for label_id in range(1, num_labels):
            component_size = np.sum(labels_img == label_id)
            component_sizes.append(component_size)
        avg_component_size = np.mean(component_sizes) if component_sizes else 0
    else:
        avg_component_size = 0
    
    # 使用像素数量作为主要特征（类似 filter_lines 中的宽度）
    # 也可以使用其他特征，如连通区域数量、平均连通区域大小等
    return pixel_count


def filter_masks(color_masks, goal_num):
    """
    从 n+1 个掩码中选择最一致的 n 个掩码（类似 filter_lines 函数）
    
    参数:
    - color_masks: n+1 个掩码列表
    - goal_num: 目标掩码数量 n
    
    返回:
    - best_indices: 选中的掩码索引列表
    """
    if len(color_masks) <= goal_num:
        # 如果掩码数量不超过目标数量，直接返回所有索引
        return list(range(len(color_masks)))
    
    # 计算每个掩码的特征值
    features = []
    for mask in color_masks:
        feature = compute_mask_features(mask)
        features.append(feature)
    
    print(f"掩码特征值: {features}")
    
    # 优先处理：如果有特征值为0的掩码，直接去除
    zero_indices = [i for i, f in enumerate(features) if f == 0]
    if zero_indices:
        # 去除特征值为0的掩码，保留非零的
        non_zero_indices = [i for i, f in enumerate(features) if f != 0]
        print(f"发现 {len(zero_indices)} 个空掩码（特征值为0），直接去除索引: {zero_indices}")
        
        # 如果去除0后的数量正好等于目标数量，直接返回
        if len(non_zero_indices) == goal_num:
            print(f"去除空掩码后正好 {goal_num} 个，选择索引: {non_zero_indices}")
            return non_zero_indices
        # 如果去除0后的数量小于目标数量，返回所有非零的
        elif len(non_zero_indices) < goal_num:
            print(f"去除空掩码后只剩 {len(non_zero_indices)} 个，全部保留")
            return non_zero_indices
        # 如果去除0后数量仍大于目标，则在非零掩码中继续筛选
        else:
            # 更新features和对应的索引映射
            remaining_features = [(i, features[i]) for i in non_zero_indices]
            print(f"去除空掩码后还有 {len(non_zero_indices)} 个，需要继续筛选到 {goal_num} 个")
    else:
        # 没有特征值为0的掩码，使用所有掩码
        remaining_features = [(i, f) for i, f in enumerate(features)]
    
    # 定义奇异系数：标准差（标准差越小，组合越一致）
    def compute_anomaly_score(subset):
        return np.std(subset)  # 标准差越小，组合越一致
    
    # 生成所有可能的 n 元素组合
    combinations = list(itertools.combinations(remaining_features, goal_num))  # 包含索引的组合
    
    # 初始化最小奇异系数和最佳组合
    min_score = float('inf')
    best_indices = None
    
    # 遍历所有组合，找到奇异系数最小的组合
    for combination in combinations:
        indices, values = zip(*combination)  # 分离索引和值
        score = compute_anomaly_score(values)
        if score < min_score:
            min_score = score
            best_indices = indices
    
    print(f"选择掩码索引: {best_indices} (标准差: {min_score:.2f})")
    
    return list(best_indices)


def ask_for_line_label(image_path, labels, response_modal_llm: ModalLLMResponse):
    """
    使用多模态大模型识别每条线条对应的 legend 标签
    
    参数:
    - image_path: 对比图路径（原图 vs 提取的数据图）
    - labels: 标签列表（A, B, C等）
    - response_modal_llm: 多模态大模型响应对象
    
    返回:
    - line_labels: legend 标签列表
    """
    print("开始查询线条legend")
    image_base64 = encode_image(image_path)
    text = """
    The image on the left is the original, while the image on the right represents the data I've extracted. I need your assistance in identifying which legend labels from the original image correspond to the different trends in the extracted data.
    If a legend is present, return the corresponding legend labels for each trend, denoted by letters (A, B, C, etc.). If no legend is present, return an empty value for each letter.
    Note: The answer format is strictly JSON and requires no additional explanation. 
    Example: { "A": "legend"} or {"A": ""}
"""

    response = response_modal_llm.generate_response(image_base64, text)
    # generate_response已经返回字典，无需再json.loads
    try:
        line_labels = []
        for label in labels:
            line_labels.append(response.get(label))
        print(f"识别到的legend标签: {line_labels}")
        return line_labels

    except Exception as e:
        print("An error occurred:", e)
        return None


def create_comparison_image_and_get_legends(original_image, color_masks, axis_lines, 
                                            response_modal_llm: ModalLLMResponse, 
                                            save_path=None):
    """
    创建对比图（原图 vs 提取的数据图）并使用多模态模型识别每条线条的 legend
    
    参数:
    - original_image: 原始图像
    - color_masks: 每个颜色对应的掩膜列表
    - axis_lines: 坐标轴线段列表
    - response_modal_llm: 多模态大模型响应对象
    - save_path: 保存路径（可选）
    
    返回:
    - line_legends: legend 标签列表
    """
    # 创建坐标轴掩码（用于限制绘制区域）
    axis_mask = create_axis_mask(original_image.shape, axis_lines)
    axis_mask_inv = cv2.bitwise_not(axis_mask)
    
    # 应用掩码
    masked_image_3 = original_image.copy()
    masked_image_3[axis_mask_inv > 0] = [255, 255, 255]  # 将掩膜区域设置为白色背景

    # 颜色列表（用于绘制不同线条）
    rainbow_colors = [
        [0, 0, 255],  # 红色 (BGR)
        [0, 127, 255],  # 橙色 (BGR)
        [0, 255, 255],  # 黄色 (BGR)
        [0, 255, 0],  # 绿色 (BGR)
        [255, 0, 0],  # 蓝色 (BGR)
        [130, 0, 75],  # 靛色 (BGR)
        [211, 0, 148]  # 紫色 (BGR)
    ]
    
    # 如果线条数量超过预定义颜色，循环使用颜色
    while len(rainbow_colors) < len(color_masks):
        rainbow_colors.extend(rainbow_colors)

    # 生成字母序列
    labels = generate_letter_sequence(len(color_masks))
    print(f"生成的标签序列: {labels}")

    # 应用颜色掩膜
    for mask, color in zip(color_masks, rainbow_colors):
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masked_image_3[binary_mask > 0] = color

    h, w = original_image.shape[:2]
    # 添加图例
    x_axis, y_axis = axis_lines
    legend_x_start = min(x_axis[0], x_axis[2]) + round(h / 30)  # 图例的起始 x 坐标
    legend_y_start = min(y_axis[1], y_axis[3]) + round(w / 30)  # 图例的起始 y 坐标
    legend_rect_size = round(h / 20)  # 图例中小矩形的大小
    legend_spacing = round(h / 8)  # 每个图例项之间的垂直间距

    # 绘制图例
    for i, (color, label) in enumerate(zip(rainbow_colors[:len(color_masks)], labels)):
        rect_x = legend_x_start + i * legend_spacing
        rect_y = legend_y_start

        # 绘制小矩形表示颜色
        cv2.rectangle(
            masked_image_3,
            (rect_x, rect_y),
            (rect_x + legend_rect_size, rect_y + legend_rect_size),
            color,
            -1  # 填充矩形
        )

        # 在矩形旁边绘制标签文本
        text_x = rect_x + legend_rect_size + 10  # 文本的 x 坐标
        text_y = rect_y + legend_rect_size // 2 + 5  # 文本的 y 坐标（居中对齐）
        cv2.putText(
            masked_image_3,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # 字体大小
            (0, 0, 0),  # 文本颜色（黑色）
            2  # 字体线条粗细
        )

    # 绘制对比图
    plt.ioff()
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(masked_image_3, cv2.COLOR_BGR2RGB))
    plt.title("Extracted_data Image")
    plt.axis('off')

    # 保存对比图
    if save_path:
        image_path_save = save_path.replace('.png', '_comparison.jpg') if save_path.endswith('.png') else save_path + '_comparison.jpg'
    else:
        image_path_save = "Raw vs. Data Extraction.jpg"
    plt.savefig(image_path_save)
    plt.close()

    # 使用多模态模型识别 legend
    line_legends = ask_for_line_label(image_path_save, labels, response_modal_llm)
    
    return line_legends


def visualize_color_based_line_results(original_image, color_masks, line_info, save_path=None):
    """
    可视化基于颜色的线条提取结果
    
    参数:
    - original_image: 原始图像
    - color_masks: 每个颜色对应的掩膜列表
    - line_info: 线条信息字典
    - save_path: 保存路径（可选）
    """
    h, w = original_image.shape[:2]
    n_lines = len(color_masks)
    
    # 创建颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_lines, 20)))[:n_lines, :3] * 255
    colors = colors.astype(np.uint8)
    
    # 创建四个子图：原图、所有颜色分类、选中的线条、对比图
    fig = plt.figure(figsize=(24, 16))
    
    # 子图1：原图
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image", fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # 子图2：所有颜色分类叠加
    ax2 = plt.subplot(2, 2, 2)
    combined_all = np.zeros_like(original_image)
    for idx, mask in enumerate(color_masks):
        color = colors[idx]
        color_bgr = [int(color[2]), int(color[1]), int(color[0])]
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = color_bgr
        combined_all = cv2.add(combined_all, colored_mask)
    ax2.imshow(cv2.cvtColor(combined_all, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"All {n_lines} Color Classifications", fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # 子图3：提取的线条
    ax3 = plt.subplot(2, 2, 3)
    combined_lines = np.zeros_like(original_image)
    line_labels_text = []
    for idx, mask in enumerate(color_masks):
        if idx < len(line_info.get('lines', [])):
            line_info_item = line_info['lines'][idx]
            color_name = line_info_item.get('color', f'line_{idx+1}')
            line_labels_text.append(f"Line {idx+1}: {color_name}")
        color = colors[idx]
        color_bgr = [int(color[2]), int(color[1]), int(color[0])]
        colored_mask = np.zeros_like(original_image)
        colored_mask[mask > 0] = color_bgr
        combined_lines = cv2.add(combined_lines, colored_mask)
    ax3.imshow(cv2.cvtColor(combined_lines, cv2.COLOR_BGR2RGB))
    title_text = f"Extracted Lines\nTotal: {n_lines} lines\n" + "\n".join(line_labels_text[:5])
    if len(line_labels_text) > 5:
        title_text += f"\n... and {len(line_labels_text) - 5} more"
    ax3.set_title(title_text, fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 子图4：原图和提取线条的对比
    ax4 = plt.subplot(2, 2, 4)
    comparison = np.hstack([original_image, combined_lines])
    ax4.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    ax4.set_title("Original vs Extracted Lines", fontsize=16, fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()
    
    # 额外创建一个详细的网格图，展示每个线条
    if color_masks:
        fig2 = plt.figure(figsize=(20, 6))
        cols = min(n_lines, 5)
        rows = (n_lines + cols - 1) // cols
        
        for i, mask in enumerate(color_masks):
            ax = plt.subplot(rows, cols, i + 1)
            color = colors[i]
            color_bgr = [int(color[2]), int(color[1]), int(color[0])]
            colored_mask = np.zeros_like(original_image)
            colored_mask[mask > 0] = color_bgr
            # 叠加在原图上
            overlay = cv2.addWeighted(original_image, 0.5, colored_mask, 0.5, 0)
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            if i < len(line_info.get('lines', [])):
                line_info_item = line_info['lines'][i]
                color_name = line_info_item.get('color', f'line_{i+1}')
                ax.set_title(f"Line {i+1}: {color_name}", fontsize=12, fontweight='bold')
            else:
                ax.set_title(f"Line {i+1}", fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(f"Extracted Lines Detail ({n_lines} lines)", 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            detail_save_path = save_path.replace('.png', '_detail.png')
            plt.savefig(detail_save_path, dpi=300, bbox_inches='tight')
            print(f"详细线条结果已保存到: {detail_save_path}")
        
        plt.show()


def grad_line_extract(image_path, axis_lines, response_modal_llm: ModalLLMResponse, visualize=True, save_path=None):
    """
    基于颜色聚类的线条提取主函数（使用多模态大模型验证）
    
    参数:
    - image_path: 图像路径
    - axis_lines: 坐标轴线段列表 [(x1,y1,x2,y2), ...]，其中第一个为 x 轴，第二个为 y 轴
    - response_modal_llm: 多模态大模型响应对象
    - visualize: 是否可视化结果
    - save_path: 保存路径（可选）
    
    返回:
    - color_masks: 选中的 n 个线条掩膜列表（从 n+1 个掩膜中通过多模态验证选择最合适的 n 个）
    - selected_labels: 对应的标签列表（颜色名称）
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图像，请检查路径是否正确！")
        return [], []
    
    print("步骤0: 创建掩膜，消除干扰信息...")
    # 创建坐标轴掩膜
    axis_mask = create_axis_mask(image.shape, axis_lines)
    
    # 初始化OCR并生成文本掩膜
    ocr = create_ocr(lang='en')
    text_mask = create_text_mask(image, ocr)
    
    # 使用YOLO检测图例
    dict_object = detect_objects_with_yolo(legend_model, image_path)
    print(f"YOLO检测结果: {dict_object}")
    
    if dict_object:
        print(f"YOLO成功检测到图例，坐标: {dict_object}")
        legend_mask = create_legend_mask(image, dict_object)
        # 合并所有干扰区域掩膜
        total_mask = combine_masks([axis_mask, text_mask, legend_mask])
    else:
        print("YOLO未检测到图例。")
        total_mask = combine_masks([axis_mask, text_mask])
    
    # 在原图上应用掩膜，移除坐标轴和文本区域
    masked_image = image.copy()
    masked_image[total_mask > 0] = [255, 255, 255]  # 将掩膜区域设置为白色背景
    
    if visualize:
        cv2.imshow("Masked Image (Before Color Clustering)", masked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"掩膜处理完成，已消除坐标轴、文本和图例区域")
    
    print("步骤1: 使用多模态大模型识别线条数量和颜色...")
    # 使用多模态大模型识别线条数量和颜色
    line_info = ask_mm_model_for_line_colors_and_count(image_path, response_modal_llm)
    
    if line_info is None or line_info.get('total_lines_num', 0) == 0:
        print("无法识别线条信息，返回空结果")
        return [], []
    
    # 提取颜色名称列表
    lines = line_info.get('lines', [])
    color_names = []
    for line in lines:
        color_name = line.get('color', '')
        if color_name:
            color_names.append(color_name)
        else:
            print(f"警告：线条 {line.get('line_id')} 没有颜色信息，跳过")
    
    if not color_names:
        print("没有有效的颜色信息，返回空结果")
        return [], []
    
    print(f"步骤2: 基于识别到的 {len(color_names)} 种颜色提取像素点（进行n+1分类）...")
    print(f"颜色列表: {color_names}")
    # 基于颜色名称从图像中提取像素点（n+1个分类：n个指定颜色 + 1个其他颜色）
    color_masks_all = extract_lines_by_color_names(masked_image, color_names, extract_extra=True)
    
    # 从 n+1 个掩膜中使用统计方法选择最一致的 n 个
    n_lines = len(color_names)
    print(f"步骤3: 从 {len(color_masks_all)} 个掩膜中选择最一致的 {n_lines} 个...")
    selected_indices = filter_masks(color_masks_all, n_lines)
    
    # 获取选中的掩膜和标签
    color_masks = [color_masks_all[i] for i in selected_indices]
    
    print(f"步骤4: 共选择了 {len(color_masks)} 个最佳掩膜")
    
    # 步骤5: 创建对比图并使用多模态模型识别每条线条的 legend
    print("步骤5: 创建对比图并识别每条线条的legend...")
    line_legends = create_comparison_image_and_get_legends(
        image, 
        color_masks, 
        axis_lines,
        response_modal_llm,
        save_path
    )
    
    # 如果识别失败，使用颜色名称作为后备
    if line_legends is None or len(line_legends) != len(color_masks):
        print("警告：legend识别失败，使用颜色名称作为后备")
        selected_labels = []
        lines = line_info.get('lines', [])
        for i in selected_indices:
            if i < len(lines):
                color_name = lines[i].get('color', f'line_{i+1}')
                selected_labels.append(color_name)
            else:
                selected_labels.append(f'line_{i+1}')
    else:
        # 使用识别到的 legend 作为标签
        selected_labels = line_legends
        # 如果某些 legend 为空，使用颜色名称作为后备
        lines = line_info.get('lines', [])
        for idx, legend in enumerate(selected_labels):
            if not legend or legend == "":
                i = selected_indices[idx]
                if i < len(lines):
                    color_name = lines[i].get('color', f'line_{idx+1}')
                    selected_labels[idx] = color_name
                else:
                    selected_labels[idx] = f'line_{idx+1}'
    
    print(f"最终标签（使用legend）: {selected_labels}")
    
    # 可视化结果（只展示选中的n个）
    if visualize:
        print("步骤6: 可视化结果（展示选中的n个分类）...")
        visualize_color_based_line_results(masked_image, color_masks, line_info, save_path)
    
    return color_masks, selected_labels


# # 测试代码
# if __name__ == "__main__":
#     # 示例用法
#     image_path = r"E:\MLLM_Selected\line_figures\0a9fc555e95fa2e70b8ad0104fdbe695282e3b17c796cc4b7dc67dcc0df99e9a_2.jpg"  # 替换为实际的图像路径
#     response_modal_llm = ModalLLMResponse()
    
#     x_axis, y_axis = detect_axes_yolo(image_path)
#     x_axis = [int(round(coord)) for coord in x_axis]
#     y_axis = [int(round(coord)) for coord in y_axis]
#     axis_lines = [x_axis, y_axis]

#     # 使用新的基于颜色的线条提取方法
#     color_masks, selected_labels = grad_line_extract(
#         image_path, 
#         axis_lines,
#         response_modal_llm, 
#         visualize=True,
#         save_path="color_based_line_extraction_result.png"
#     )
    
#     print(f"提取完成！提取了 {len(color_masks)} 条线条")
#     if selected_labels:
#         print(f"线条标签: {selected_labels}")
