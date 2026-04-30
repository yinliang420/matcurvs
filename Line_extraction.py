import string
import cv2
import json
import base64
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from ocr_compat import create_ocr, ocr_predict
import itertools
from ModalLLMResponse import ModalLLMResponse
from ultralytics import YOLO
from pathlib import Path


try:
    script_dir = Path(__file__).resolve().parent
    
    MODEL_PATH = script_dir / "yolo" / "runs" / "detect" / "yolov11n_lenged_clutter_detector" / "weights" / "best.pt"
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

# def detect_legend_with_yolo(model, image_path_str):
    """
    使用训练好的YOLO模型检测图像中的所有图例。

    :param model: 已加载的YOLO模型对象。
    :param image_path_str: 待检测图片的路径字符串。
    :return: 返回一个包含所有检测到的图例坐标字典的【列表】。
             列表中的每个字典格式为 {'top_left': [x1, y1], 'bottom_right': [x2, y2]}。
             如果没有检测到图例，则返回一个空列表 []。
    """
    if model is None:
        return [] # 返回空列表

    try:
        # 使用模型进行预测，verbose=False可以减少不必要的控制台输出
        results = model.predict(source=image_path_str, conf=0.5, verbose=False)
        result = results[0]
        detected_boxes = result.boxes

        if len(detected_boxes) == 0:
            return [] # 没有检测到图例，返回空列表

        # 【核心修改】创建一个列表来存储所有检测到的框
        all_legends = []
        
        # 遍历所有检测到的边界框
        for box in detected_boxes:
            # 提取边界框的左上角和右下角坐标
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)

            # 将坐标格式化为字典并添加到列表中
            dict_legend = {'top_left': [x1, y1], 'bottom_right': [x2, y2]}
            all_legends.append(dict_legend)
            
        return all_legends

    except Exception as e:
        print(f"使用YOLO模型检测图例时发生错误: {e}")
        return [] # 发生错误时也返回空列表

def detect_objects_with_yolo(model, image_path_str):
    """
    【已修改】使用训练好的YOLO模型检测图像中的所有对象（例如图例和杂图）。

    :param model: 已加载的YOLO模型对象。
    :param image_path_str: 待检测图片的路径字符串。
    :return: 返回一个包含所有检测到的对象坐标字典的【列表】。
             列表中的每个字典格式为 {'top_left': [x1, y1], 'bottom_right': [x2, y2]}。
             如果没有检测到对象，则返回一个空列表 []。
    """
    if model is None:
        return [] # 返回空列表

    try:
        # 使用模型进行预测，verbose=False可以减少不必要的控制台输出
        results = model.predict(source=image_path_str, conf=0.5, verbose=False)
        result = results[0]
        detected_boxes = result.boxes

        if len(detected_boxes) == 0:
            return [] # 没有检测到对象，返回空列表

        # 【核心修改】创建一个列表来存储所有检测到的框
        all_detections = []
        
        # 遍历所有检测到的边界框
        for box in detected_boxes:
            # 此时，您可以访问 box.cls[0].item() 来获取类别ID
            # 例如: class_id = int(box.cls[0].item())
            # 但根据您的要求，我们在这里统一处理，不区分具体类别

            # 提取边界框的左上角和右下角坐标
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)

            # 将坐标格式化为字典并添加到列表中
            detection_dict = {'top_left': [x1, y1], 'bottom_right': [x2, y2]}
            all_detections.append(detection_dict)
            
        return all_detections

    except Exception as e:
        print(f"使用YOLO模型检测对象时发生错误: {e}")
        return [] # 发生错误时也返回空列表

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
        所有图例区域的值为 1，其余区域为 0。
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
            
            # 在掩膜上，将当前图例的矩形区域设置为 1
            # 注意坐标顺序是 [y1:y2, x1:x2]
            mask[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 1
            
    return mask

def combine_masks(masks):
    """合并多个掩膜"""
    combined = np.zeros_like(masks[0])
    for mask in masks:
        combined = cv2.bitwise_or(combined, mask)
    return combined

# 利用多模态大模型查询图列中包含多少个颜色
def ask_mm_model_for_color_num(image_path, response_modal_llm: ModalLLMResponse):
    print("开始查询标签多少")
    image_base64 = encode_image(image_path)
    from prompt import get_total_lines_count
    response = response_modal_llm.generate_response(image_base64, get_total_lines_count())
    # generate_response已经返回字典，无需再json.loads
    try:
        label_num = response.get("total_lines_num")
        return label_num
    except Exception as e:
        print("An error occurred:", e)
        return None

# 根据平均线条宽度进行初步的过滤得到最终的线条
# 逻辑，遍历所有可能组合，计算奇异系数，因为合规的线条应该都是差不多宽度的
def filter_lines(widths, goal_num):
    """
    从 n+1 个线条中选择最一致的 n 个线条
    
    参数:
    - widths: n+1 个线条的宽度列表
    - goal_num: 目标线条数量 n
    
    返回:
    - best_indices: 选中的线条索引列表
    """
    if len(widths) <= goal_num:
        # 如果线条数量不超过目标数量，直接返回所有索引
        return list(range(len(widths)))
    
    print(f"线条宽度: {widths}")
    
    # 优先处理：如果有宽度为0的线条，直接去除
    zero_indices = [i for i, w in enumerate(widths) if w == 0]
    if zero_indices:
        # 去除宽度为0的线条，保留非零的
        non_zero_indices = [i for i, w in enumerate(widths) if w != 0]
        print(f"发现 {len(zero_indices)} 个空线条（宽度为0），直接去除索引: {zero_indices}")
        
        # 如果去除0后的数量正好等于目标数量，直接返回
        if len(non_zero_indices) == goal_num:
            print(f"去除空线条后正好 {goal_num} 个，选择索引: {non_zero_indices}")
            return non_zero_indices
        # 如果去除0后的数量小于目标数量，返回所有非零的
        elif len(non_zero_indices) < goal_num:
            print(f"去除空线条后只剩 {len(non_zero_indices)} 个，全部保留")
            return non_zero_indices
        # 如果去除0后数量仍大于目标，则在非零线条中继续筛选
        else:
            # 更新widths和对应的索引映射
            remaining_widths = [(i, widths[i]) for i in non_zero_indices]
            print(f"去除空线条后还有 {len(non_zero_indices)} 个，需要继续筛选到 {goal_num} 个")
    else:
        # 没有宽度为0的线条，使用所有线条
        remaining_widths = [(i, w) for i, w in enumerate(widths)]
    
    # 定义奇异系数：标准差（标准差越小，组合越一致）
    def compute_anomaly_score(subset):
        return np.std(subset)  # 标准差越小，组合越一致

    # 生成所有可能的 n 元素组合
    combinations = list(itertools.combinations(remaining_widths, goal_num))  # 包含索引的组合

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

    print(f"选择线条索引: {best_indices} (标准差: {min_score:.2f})")
    
    return list(best_indices)

def optimize_line_masks_with_contour_visualization(line_masks, labels_num, min_area=150, min_width=10,
                                                   visualise=False):
    """
    优化线条掩码，删除异常值（如过细或过短的线条），并可视化每一步的中间过程，包括轮廓

    参数:
    - line_masks: 包含多个二值图像（掩码）的列表
    - min_area: 最小面积阈值，低于此值的线条将被删除
    - min_width: 最小宽度阈值，低于此值的线条将被删除

    返回:
    - optimized_masks: 优化后的线条掩码列表
    """
    optimized_masks = []
    all_line_average_width = []
    # 遍历每个 mask
    for idx, mask in enumerate(line_masks):
        print(f"Processing mask {idx + 1}/{len(line_masks)}")

        # 可视化原始 mask
        if visualise:
            plt.figure(figsize=(15, 10))
            plt.subplot(2, 3, 1)
            plt.title("Original Mask")
            plt.imshow(mask, cmap='gray')
            plt.axis('off')

        # 形态学操作：先腐蚀后膨胀，去除小的噪声点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 可视化形态学操作后的 mask
        if visualise:
            plt.subplot(2, 3, 2)
            plt.title("After Morphological Operation")
            plt.imshow(mask_cleaned, cmap='gray')
            plt.axis('off')

        # 查找轮廓
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 在原始图像上绘制所有找到的轮廓（用于可视化）
        all_contours_image = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以便绘制彩色轮廓
        cv2.drawContours(all_contours_image, contours, -1, (0, 255, 0), 2)  # 绘制所有轮廓，绿色表示

        # 可视化所有找到的轮廓
        if visualise:
            plt.subplot(2, 3, 3)
            plt.title("All Contours Found")
            plt.imshow(cv2.cvtColor(all_contours_image, cv2.COLOR_BGR2RGB))  # 转换回 RGB 格式以正确显示颜色
            plt.axis('off')

        # 筛选符合条件的轮廓
        line_cnt_widths = []
        valid_contours = []
        for cnt in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(cnt)

            # 计算轮廓的最小外接矩形
            x, y, w, h = cv2.boundingRect(cnt)
            width = max(w, h)  # 取最大值作为宽度

            # 后续可以增加更多的筛选条件，比如平均宽度应该是相近的，删除平均宽度差异太大的

            # 筛选条件：面积大于 min_area 且宽度大于 min_width
            if area > min_area and width > min_width:
                valid_contours.append(cnt)
                line_cnt_widths.append(width)
        line_average_width = np.array(line_cnt_widths).mean()
        all_line_average_width.append(line_average_width)

        # 在原始图像上绘制筛选后的有效轮廓（用于可视化）
        valid_contours_image = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以便绘制彩色轮廓
        cv2.drawContours(valid_contours_image, valid_contours, -1, (255, 0, 0), 2)  # 绘制有效轮廓，蓝色表示

        # 可视化筛选后的有效轮廓
        if visualise:
            plt.subplot(2, 3, 4)
            plt.title("Valid Contours After Filtering")
            plt.imshow(cv2.cvtColor(valid_contours_image, cv2.COLOR_BGR2RGB))  # 转换回 RGB 格式以正确显示颜色
            plt.axis('off')

        # 创建一个新的掩码，只保留符合条件的轮廓
        optimized_mask = np.zeros_like(mask)
        cv2.drawContours(optimized_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

        # 可视化优化后的 mask
        if visualise:
            plt.subplot(2, 3, 5)
            plt.title("Optimized Mask")
            plt.imshow(optimized_mask, cmap='gray')
            plt.axis('off')

            # 显示当前 mask 的所有中间步骤
            plt.tight_layout()
            plt.show()

        # 将优化后的掩码添加到结果列表
        optimized_masks.append(optimized_mask)
    print(all_line_average_width)
    if labels_num == 1:
        if not all_line_average_width:
            return []
        best_idx = np.argmin(all_line_average_width)
        print(best_idx)
        # print(optimized_masks[best_idx])
        return [optimized_masks[best_idx]]
    else:
        optimized_masks_index = filter_lines(all_line_average_width, labels_num)
        print(optimized_masks_index)
        filter_optimized_masks = [optimized_masks[i] for i in optimized_masks_index]
        f_masked = (x for x in range(len(all_line_average_width)) if x != all_line_average_width.index(max(all_line_average_width)))
        if optimized_masks_index == f_masked:
            return filter_optimized_masks
        else:
            filter_optimized_masks = [optimized_masks[i] for i in f_masked]
            return filter_optimized_masks

import numba
@numba.njit(parallel=True)
def propagate_darkest_color(lab_image, kernel_size=3):
    """
    【核心函数】使用“暗色传播”逻辑处理图像。
    在每个局部窗口内，找到L值最小（最深）的像素，
    并将其完整的LAB颜色应用到窗口中心。

    参数:
    - lab_image: 输入的LAB格式图像 (numpy数组)。
    - kernel_size: 滑动窗口的大小，必须是奇数 (如 3, 5, 7)。

    返回:
    - processed_lab_image: 处理后的LAB格式图像。
    """
    h, w, c = lab_image.shape
    # 创建一个和原图一样大的输出图像
    processed_lab_image = np.copy(lab_image)

    # 计算内核半径
    radius = kernel_size // 2
    # 使用Numba的并行循环加速处理
    for y in numba.prange(radius, h - radius):
        for x in range(radius, w - radius):
            # 提取当前窗口
            window = lab_image[y - radius: y + radius + 1, x - radius: x + radius + 1]

            # 提取窗口内的L通道值
            l_channel = window[:, :, 0]

            # 找到L值最小的像素在窗口内的局部坐标
            min_l_val = 255  # L通道的值范围是0-255
            min_ky, min_kx = -1, -1

            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    if l_channel[ky, kx] < min_l_val:
                        min_l_val = l_channel[ky, kx]
                        min_ky, min_kx = ky, kx

            # 将找到的最深色的像素的完整LAB值，赋给输出图像的中心像素
            processed_lab_image[y, x] = window[min_ky, min_kx, :]

    return processed_lab_image

def extract_main_lines(image, labels_num, visualise=False):
    """提取主要数据线条"""
    # 后续利用多模态判断是否聚类准确
    # 颜色空间转换
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    kernel_size = 2
    processed_lab = propagate_darkest_color(lab, kernel_size)
    if labels_num <= 0:
        raise ValueError("labels_num must be at least 1")
    if visualise:
        processed_bgr = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
        comparison_image = np.hstack([image, processed_bgr])

        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB))
        plt.title("origin | darker-process")
        plt.axis('off')
        plt.show()
    # 准备数据
    pixels = lab.reshape(-1, 3)

    # 归一化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    # 聚类 如何分别太过于贴近的图片线条如何分离？
    kmeans = KMeans(
        n_clusters=labels_num + 1,
        init="k-means++",
        n_init=10,
        max_iter=1000,
        random_state=42
    )
    labels = kmeans.fit_predict(pixels_scaled)

    # 生成聚类掩膜
    clustered = labels.reshape(image.shape[:2])
    line_masks = []
    for i in range(kmeans.n_clusters):
        cluster_mask = np.uint8(clustered == i) * 255

        # 可选：形态学操作去噪
        kernel = np.ones((2, 2), np.uint8)
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel)

        line_masks.append(cluster_mask)
    if visualise:
        visualize_line_clusters(image, line_masks)

    valid_lines = optimize_line_masks_with_contour_visualization(line_masks, labels_num, visualise)
    # 用多模态评估聚类结果 用提示词去 for循环直接判断
    return valid_lines

def visualize_line_clusters(image, line_masks):
    """
    可视化线条聚类效果，为每个聚类分配不同颜色
    :param image: 输入图像
    :param line_masks: 聚类掩膜列表
    """
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255)]  # 定义不同颜色
    for i, mask in enumerate(line_masks):
        # 创建彩色图像
        color_image = np.zeros_like(image)
        color_image[:] = colors[i % len(colors)]  # 设置背景颜色
        # 将掩膜应用到彩色图像上
        color_image[mask == 0] = [0, 0, 0]  # 掩膜外的内容设置为黑色
        # 显示结果
        cv2.imshow(f"Cluster {i + 1}", color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_results(original, cleaned, lines):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    plt.title("Cleaned Image")

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(lines, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines")

    plt.show()

def show_line_and_label(final_lines, axis_mask, img, axis_lines, response_modal_llm: ModalLLMResponse,
                        visualise=False):
    # 坐标轴掩码反转
    axis_mask_inv = cv2.bitwise_not(axis_mask)
    # 应用掩码
    masked_image_3 = img.copy()
    masked_image_3[axis_mask_inv > 0] = [255, 255, 255]  # 将掩膜区域设置为白色背景

    # 颜色列表
    rainbow_colors = [
        [0, 0, 255],  # 红色 (BGR)
        [0, 127, 255],  # 橙色 (BGR)
        [0, 255, 255],  # 黄色 (BGR)
        [0, 255, 0],  # 绿色 (BGR)
        [255, 0, 0],  # 蓝色 (BGR)
        [130, 0, 75],  # 靛色 (BGR)
        [211, 0, 148]  # 紫色 (BGR)
    ]

    # 生成字母序列
    labels = generate_letter_sequence(len(final_lines))
    # print(f"final_lines_labels = {labels}")

    # 应用颜色掩膜
    for mask, color in zip(final_lines, rainbow_colors):
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        masked_image_3[binary_mask > 0] = color

    if visualise:
        cv2.imshow("Black Image", masked_image_3)
        cv2.waitKey(0)

    h, w = img.shape[:2]
    # 添加图例
    x_axis, y_axis = axis_lines
    legend_x_start = min(x_axis[0], x_axis[2]) + round(h / 30)  # 图例的起始 x 坐标
    legend_y_start = min(y_axis[1], y_axis[3]) + round(w / 30)  # 图例的起始 y 坐标
    legend_rect_size = round(h / 20)  # 图例中小矩形的大小
    legend_spacing = round(h / 8)  # 每个图例项之间的垂直间距

    # 绘制图例
    for i, (color, label) in enumerate(zip(rainbow_colors, labels)):
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

    if visualise:
        cv2.imshow("Image with Legend", masked_image_3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 绘制对比图
    plt.ioff()
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(masked_image_3, cv2.COLOR_BGR2RGB))
    plt.title("Extracted_data Image")
    plt.axis('off')

    image_path_save = "Raw vs. Data Extraction.jpg"
    plt.savefig(image_path_save)

    if visualise:
        plt.show()  # 手动触发显示
    else:
        plt.close()  # 直接关闭

    line_labels = ask_for_line_label(image_path_save, labels, response_modal_llm)

    return line_labels

def ask_for_line_label(image_path, labels, response_modal_llm: ModalLLMResponse):
    print("开始查询线条label")
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
        print(line_labels)
        return line_labels

    except Exception as e:
        print("An error occurred:", e)
        return None

def line_extract(image_path, axis_lines, response_modal_llm: ModalLLMResponse, visualise=False):
    # 读取图像
    img = cv2.imread(image_path)  # 替换为你的图像路径
    if img is None:
        print("无法加载图像，请检查路径是否正确！")
        exit()
    labels_num = ask_mm_model_for_color_num(image_path, response_modal_llm)

    axis_mask = create_axis_mask(img.shape, axis_lines)
    
    # 初始化OCR并生成掩膜
    ocr = create_ocr(lang='en')
    text_mask = create_text_mask(img, ocr)
    dict_object = detect_objects_with_yolo(legend_model, image_path)
    print(dict_object)
    if dict_object:
        print(f"YOLO成功检测到图例，坐标: {dict_object}")
        legend_mask = create_legend_mask(img, dict_object)
        # 合并所有干扰区域掩膜
        total_mask = combine_masks([axis_mask, text_mask, legend_mask])
    else:
        print("YOLO未检测到图例。")
        total_mask = combine_masks([axis_mask, text_mask])
    # 获取有效区域
    valid_region = cv2.bitwise_not(total_mask)

    # 在原图上应用掩膜，移除坐标轴和文本区域
    masked_image = img.copy()
    masked_image[total_mask > 0] = [255, 255, 255]  # 将掩膜区域设置为白色背景
    if visualise:
        cv2.imshow("Masked Image", masked_image)
        cv2.waitKey(0)
    # 提取最终线条
    final_lines = extract_main_lines(masked_image, labels_num, visualise)
    # 创建一个副本以避免修改原始图像
    masked_image_2 = img.copy()
    for mask in final_lines:
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # 确保掩码是二值图像
        masked_image_2[binary_mask > 0] = (255, 255, 255)  # 将掩码区域设置为目标颜色
        # 显示结果
    if visualise:
        cv2.imshow("Masked Image with final_lines", masked_image_2)
    if visualise:
        visualize_results(img, masked_image, masked_image_2)
    # 多模态返回线条图注
    line_labels = show_line_and_label(final_lines, axis_mask, img, axis_lines, response_modal_llm, visualise=visualise)

    return final_lines, line_labels  # 返回的是一个列表，里面是不同颜色的掩码

# image_path = r"E:\MLLM_Selected\line_figures\\0a9b326cfc213ff4e980c087e2edf80e53df372532cf1555a194a8dbd0f46b95_4.jpg"
# # image_path = "C:\\Users\Yin Liang\Desktop\split_images\\1_110.jpg"
# # image_path = "C:\\Users\Yin Liang\Desktop\split_images\\1.jpg"
# # responded = ModalLLMResponse()
# from Axis_yolo_extraction import detect_axes_yolo
# x, y = detect_axes_yolo(image_path)
# x = [int(round(coord)) for coord in x]
# y = [int(round(coord)) for coord in y]
# axis_lines = [x, y]
# responded = ModalLLMResponse()
# line_labels = line_extract(image_path, axis_lines, responded, visualise=True)
# line_extract(image_path, axis_lines, responded, visualise=True)
# # # x_mapping, y_mapping = axis_pixel_mapping(image_path, x, y, responsed, True)
# # x = (73, 256, 897, 256)
# # y = (73, 256, 73, 30)
# # x = (59, 293, 59, 15)
# # y = (59, 293, 407,293)
# axis_lines = [x, y]
# line_extract(image_path, axis_lines, responded, Visualise=True)
