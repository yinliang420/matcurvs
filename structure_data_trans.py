import os
import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from Axis_scale_extraction import axis_pixel_mapping
from grad_line_extraction import grad_line_extract
from ModalLLMResponse import ModalLLMResponse
from Axis_yolo_extraction import detect_axes_yolo
from Axis_extraction import detect_axes
from Line_extraction import line_extract

class DataExporter:
    def __init__(self, x_mapping: callable, y_mapping: callable):
        """
        :param x_mapping: 像素X坐标到实际值的映射函数
        :param y_mapping: 像素Y坐标到实际值的映射函数
        """
        self.x_mapping = x_mapping
        self.y_mapping = y_mapping
        self.series_data = {}  # 存储各系列数据 {series_name: {'x':[], 'y':[]}}

    def process_mask(self, mask: np.ndarray, series_name: str = "Series",
                     sampling_interval: int = 5, max_points: int = 1000):
        """
        处理单个颜色掩码，提取数据点
        :param mask: 二值掩码图像（0为背景，255为前景）
        :param series_name: 数据系列名称
        :param sampling_interval: 采样间隔（像素）
        :param max_points: 最大采样点数（防止内存溢出）
        """
        # 验证输入有效性
        if mask.ndim != 2:
            raise ValueError("掩码必须为单通道二值图像")

        # 提取有效像素坐标
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            print(f"警告：掩码 '{series_name}' 未检测到有效像素")
            return

        # 下采样处理
        step = max(1, len(x_coords) // max_points)
        sampled_x = x_coords[::step]
        sampled_y = y_coords[::step]

        # 坐标转换
        try:
            x_values = [self.x_mapping(x) for x in sampled_x]
            y_values = [self.y_mapping(y) for y in sampled_y]
        except Exception as e:
            raise RuntimeError(f"坐标映射失败: {str(e)}")

        # 按X值排序
        sorted_data = sorted(zip(x_values, y_values), key=lambda p: p[0])
        x_sorted, y_sorted = zip(*sorted_data) if sorted_data else ([], [])


        # 存储数据
        self.series_data[series_name] = {
            'x': list(x_sorted),
            'y': list(y_sorted)
        }

    def export_to_excel(self, output_path: str, sheet_name: str = "Chart Data"):
        """
        导出所有系列数据到Excel文件
        :param output_path: 输出文件路径
        :param sheet_name: Excel工作表名称
        """
        if not self.series_data:
            raise ValueError("没有可导出的数据，请先处理掩码")

        # 创建DataFrame
        max_len = max(len(v['x']) for v in self.series_data.values())
        data_dict = {}

        for name, series in self.series_data.items():
            # 填充缺失值为空字符串
            x_data = series['x'] + [''] * (max_len - len(series['x']))
            y_data = series['y'] + [''] * (max_len - len(series['y']))

            data_dict[f"{name}_X"] = x_data
            data_dict[f"{name}_Y"] = y_data

        df = pd.DataFrame(data_dict)

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

                # 自动调整列宽
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = max(len(str(cell.value)) for cell in column)
                    adjusted_width = (max_length + 2) * 1.2
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        except Exception as e:
            raise IOError(f"Excel文件写入失败: {str(e)}")

    @staticmethod
    def visualize_series(series_data: Dict, figsize=(8, 6)):
        """
        数据可视化（散点图，自动5秒后关闭）
        注意：此函数仅用于临时可视化，不保存图像。
        如需保存原图与提取线条的对比图，请使用 batch_lines_extraction.py 中的 create_comparison_image 函数。
        
        Args:
            series_data: 数据字典 {series_name: {'x':[], 'y':[]}}
            figsize: 图像大小
        """
        import matplotlib.pyplot as plt
        try:
            plt.rcParams['font.family'] = 'Times New Roman'
        except:
            print("Warning: Times New Roman not found, using serif.")
            plt.rcParams['font.family'] = 'serif'
        # 设置matplotlib为非阻塞模式
        plt.ion()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制散点图
        for name, data in series_data.items():
            ax.scatter(data['x'], data['y'], label=name, s=20, alpha=0.7)

        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title("Extracted Data Visualization (Scatter Plot)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 显示图像
        plt.show()
        
        # 5秒后自动关闭
        plt.pause(5)
        plt.close()
        
        # 关闭交互模式
        plt.ioff()

async def fig_to_data(image_path, Visualisable = False, export_excel=True):
    responses_llm = ModalLLMResponse()
    
    # 尝试使用YOLO检测坐标轴，如果失败则使用备用方法
    try:
        x_axis, y_axis = detect_axes_yolo(image_path)
        x_axis = [int(round(coord)) for coord in x_axis]
        y_axis = [int(round(coord)) for coord in y_axis]
    except Exception as e:
        print(f"YOLO坐标轴检测失败: {e}，尝试使用备用方法 detect_axes...")
        try:
            x_axis, y_axis = await detect_axes(image_path, responses_llm, visualise=Visualisable, enable_feedback=False)
            if x_axis is not None:
                x_axis = [int(round(coord)) for coord in x_axis]
            else:
                raise ValueError("detect_axes 返回的 x_axis 为 None")
            if y_axis is not None:
                y_axis = [int(round(coord)) for coord in y_axis]
            else:
                raise ValueError("detect_axes 返回的 y_axis 为 None")
        except Exception as e2:
            print(f"备用坐标轴检测也失败: {e2}")
            raise RuntimeError(f"无法检测坐标轴: YOLO失败({e}), 备用方法失败({e2})")
    
    x_mapping, y_mapping = axis_pixel_mapping(image_path, x_axis, y_axis, responses_llm, Visualisable)
    axis_lines = [x_axis, y_axis]
    
    # 尝试使用grad_line_extract提取线条，如果失败则使用备用方法
    try:
        lines_mask, line_labels = grad_line_extract(image_path, axis_lines, responses_llm, Visualisable)
    except Exception as e:
        print(f"grad_line_extract 线条提取失败: {e}，尝试使用备用方法 line_extract...")
        try:
            lines_mask, line_labels = line_extract(image_path, axis_lines, responses_llm, Visualisable)
        except Exception as e2:
            print(f"备用线条提取也失败: {e2}")
            raise RuntimeError(f"无法提取线条: grad_line_extract失败({e}), line_extract失败({e2})")

    # 检查是否有提取到掩码
    if not lines_mask or len(lines_mask) == 0:
        print(f"警告: 未提取到任何线条掩码，跳过处理")
        return {}

    # 1. 初始化导出器
    exporter = DataExporter(x_mapping=x_mapping,
                            y_mapping=y_mapping)
    #生成图标名称
    num = len(lines_mask)
    name = line_labels if line_labels else [f"Series_{i+1}" for i in range(num)]
    bb = {k:v for k,v in zip(name,lines_mask)}
    # 2. 处理每个颜色掩码
    for name, mask in bb.items():
        if name:
            exporter.process_mask(mask, series_name=name)
        else:
            exporter.process_mask(mask)
    out_path = os.path.splitext(image_path)[0] + ".xlsx"
    # 3. 导出Excel
    if export_excel:
        try:
            exporter.export_to_excel(out_path)
            print(f"数据导出成功！保存到: {out_path}")
        except Exception as e:
            print(f"导出失败: {str(e)}")
            raise
    # 4. 可视化验证（可选）
    if Visualisable:
        exporter.visualize_series(exporter.series_data)
    
    # 返回提取的数据
    return exporter.series_data

async def batch_process_linecharts_from_json(json_file_path, Visualisable=False):
    """
    批量处理 JSON 文件中所有 linechart 类型的图片
    
    参数:
    - json_file_path: JSON 文件路径
    - Visualisable: 是否可视化结果
    
    返回:
    - 处理结果统计字典
    """
    import json
    
    # 读取 JSON 文件
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON 文件不存在: {json_file_path}")
    
    print(f"正在读取 JSON 文件: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计信息
    total_linecharts = 0
    processed_count = 0
    success_count = 0
    error_count = 0
    error_details = []
    
    # 遍历所有条目
    print(f"\n开始处理 JSON 文件中的 linechart 图片...")
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        
        # 检查是否为 line chart
        chart_type = value.get("chart_type", "")
        if chart_type.lower() != "line chart":
            continue
        
        total_linecharts += 1
        file_path = value.get("file_path", "")
        
        if not file_path:
            print(f"  警告: 条目 '{key}' 没有 file_path，跳过")
            continue
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"  警告: 文件不存在，跳过: {file_path}")
            error_count += 1
            error_details.append({
                'key': key,
                'file_path': file_path,
                'error': '文件不存在'
            })
            continue
        
        processed_count += 1
        print(f"\n[{processed_count}/{total_linecharts}] 处理图片: {key}")
        print(f"  文件路径: {file_path}")
        
        try:
            # 处理图片
            await fig_to_data(file_path, Visualisable)
            success_count += 1
            print(f"  ✓ 处理成功")
        except Exception as e:
            error_count += 1
            error_msg = str(e)
            print(f"  ✗ 处理失败: {error_msg}")
            error_details.append({
                'key': key,
                'file_path': file_path,
                'error': error_msg
            })
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"  总 linechart 数量: {total_linecharts}")
    print(f"  成功处理: {success_count}")
    print(f"  处理失败: {error_count}")
    print(f"{'='*60}")
    
    if error_details:
        print(f"\n失败详情:")
        for detail in error_details[:10]:  # 只显示前10个错误
            print(f"  - {detail['key']}: {detail['error']}")
        if len(error_details) > 10:
            print(f"  ... 还有 {len(error_details) - 10} 个错误未显示")
    
    return {
        'total_linecharts': total_linecharts,
        'success_count': success_count,
        'error_count': error_count,
        'error_details': error_details
    }

# 示例用法
# if __name__ == "__main__":
    # 单个图片处理示例
    # image_path = r"E:\MLLM_Selected\label_line_figure\00dd3ec5cade8e553b93da26aa5fd1592b8f4c84bbfef355e28b4373f1a09ea7_3.jpg"
    # asyncio.run(fig_to_data(image_path, True))
# image_path = "C:\\Users\Yin Liang\Desktop\split_images\A_Monolayer_High_Entropy_Layered_Hydroxide_Frame_for_Efficient_Oxygen_Evolution_Reaction\\figure2_i.png"
# image_path = "C:\\Users\Yin Liang\Desktop\split_images\\test\\Cheng2025Interlayerexpandedcarbonanodeswithexceptional\\figure4_G.png"
# image_path = "C:\\Users\Yin Liang\Desktop\split_images\\8a8b025bf05e3f04fab8f4f4d6569953.png"
# image_path = "C:\\Users\Yin Liang\Desktop\split_images\\test\AdvancedScience2017WangUtilizingCo2\\figure6_d.png"
    # path = r"E:\batterylife/battery_10_69443caabaa41946/c_rate/c_rate_a2fce9dddf65d07ed10ae4a1f8d464cc05fde429875d8c74de80c15cdf9b329d_3.jpg"
    # asyncio.run(fig_to_data(path, True))
    # # import glob
    # for file in glob.glob(os.path.join(path, "*.png")) + glob.glob(os.path.join(path, "*.jpg")):
    #     asyncio.run(fig_to_data(file, False))


    # 批量处理 JSON 文件示例
    # json_file_path = r"C:\path\to\class-battery1.json"
    # asyncio.run(batch_process_linecharts_from_json(json_file_path, Visualisable=False))
    # pass