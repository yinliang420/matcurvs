import base64
import json
import os
from pathlib import Path
from ModalLLMResponse import ModalLLMResponse  # 确保这个模块在您的项目中
import asyncio
import aiofiles  # 用于异步文件操作
from datetime import datetime
import re
import sys

# --- 并发配置 ---
# 调整此值以控制同时发送的API请求数量，避免触发API速率限制
MAX_CONCURRENT_TASKS = 30

# --- 异步核心函数 ---

async def encode_image_async(image_path: str) -> str:
    """异步地将图片编码为base64格式"""
    async with aiofiles.open(image_path, "rb") as image_file:
        content = await image_file.read()
        return base64.b64encode(content).decode("utf-8")

async def run_llm_in_thread(response_handler, encoded_image, prompt_text):
    """
    将同步的LLM调用包装在一个独立的线程中，以防止其阻塞asyncio事件循环。
    这是处理同步库与异步代码集成的关键。
    """
    loop = asyncio.get_running_loop()
    # 使用loop.run_in_executor将阻塞函数在线程池中运行
    result = await loop.run_in_executor(
        None,  # 使用默认的线程池执行器
        response_handler.generate_response,
        encoded_image,
        prompt_text
    )
    return result

async def figure_legend_axis_name_async(path: str) -> str:
    """异步识别图例和坐标轴名称，并将结果保存到.txt文件中。"""
    response_handler = ModalLLMResponse()
    
    def get_sub_figure_axes_titles():
        return '''
     **Axis Title Extraction Protocol**

     - **Task:** Analyze the provided single chart image and identify the titles for the x-axis and the y-axis.

     - **Output Specification:**
     Return the result in a strict JSON format. Do not include any explanations.
     ```json
     {
         "x": "the complete text of the x-axis title",
         "y": "the complete text of the y-axis title"
     }
     '''
    
    try:
        encoded_image = await encode_image_async(path)
        result = await run_llm_in_thread(response_handler, encoded_image, get_sub_figure_axes_titles())
        
        print(f">>> [AXIS] Model returned for: {Path(path).name}")
        
        image_path = Path(path)
        output_txt_path = image_path.with_suffix('.txt')
        
        async with aiofiles.open(output_txt_path, 'w', encoding='utf-8') as f:
            formatted_result = json.dumps(result, indent=4)
            await f.write(formatted_result)
            
        print(f">>> [AXIS] Saved to: {output_txt_path}")
        return str(output_txt_path)

    except Exception as e:
        print(f"Error in figure_legend_axis_name_async for {path}: {e}")
        return None

async def figure_chart_type_async(path: str) -> str:
    """异步识别图表类型"""
    response_handler = ModalLLMResponse()
    text = """
     **Chart Type Classification with Chain-of-Thought Analysis**

     Please analyze the following image step by step to identify its chart type. Use the following systematic approach:

     **Step 1: Visual Element Detection**
     - Look for coordinate axes (x-axis and y-axis)
     - Identify data representation methods (lines, bars, pie slices, points, etc.)
     - Check for legends, axis labels, and data markers

     **Step 2: Chart Type Analysis**
     Examine these specific characteristics:

     **Line Chart Indicators:**
     - Continuous lines connecting data points
     - Data points plotted on x-y coordinates
     - Multiple series may be shown with different line styles/colors
     - Common in time series, trends, relationships between variables
     - Examples: XRD patterns, electrochemical curves, performance over time

     **Bar Chart Indicators:**
     - Rectangular bars with heights/lengths representing values
     - Bars can be vertical or horizontal
     - Discrete categories on one axis, values on the other
     - Common for comparing quantities across categories

     **Pie Chart Indicators:**
     - Circular chart divided into sectors/slices
     - Each slice represents a proportion of the whole
     - Usually includes percentages or labels
     - Common for showing composition or distribution

     **Scatter Plot Indicators:**
     - Individual data points plotted on x-y coordinates
     - Points may or may not be connected
     - Shows correlation or relationship between two variables

     **Heatmap Indicators:**
     - Color-coded matrix or grid
     - Colors represent values or intensities
     - Often includes color scale/legend

     **Other Types:**
     - Microscopy images (SEM, TEM, etc.)
     - Spectroscopy data with unique patterns
     - Maps, diagrams, or structural illustrations
     - Images without clear chart structure

     **Step 3: Decision Logic**
     Based on the analysis above, determine the most appropriate category:

     1. If there are clear x-y axes with continuous lines or curves → "line chart"
     2. If there are rectangular bars representing data → "bar chart"  
     3. If there's a circular division into sectors → "pie chart"
     4. If there are scattered data points without connecting lines → "scatter plot"
     5. If there's a color-coded matrix or grid → "heatmap"
     6. If it's microscopy, spectroscopy without standard chart format, or other visualization → "other"

     **Step 4: Confidence Check**
     Verify your classification by checking:
     - Are the identified visual elements consistent with the chosen chart type?
     - Does the data presentation method match the classification?

     **Output Specification:**
     Provide your analysis in strict JSON format:
     ```json
     {
       "step1_elements": "description of visual elements found",
       "step2_analysis": "detailed analysis of chart characteristics", 
       "step3_decision": "reasoning for the chosen chart type",
       "chart_type": "line chart" // Must be exactly one of: "line chart", "bar chart", "pie chart", "scatter plot", "heatmap", "other"
     }
     ```

     **Important Notes:**
     - XRD patterns, electrochemical curves, and performance plots are typically "line chart"
     - Microscopy images (SEM, TEM, AFM) are "other"
     - Elemental mapping images are "other"
     - Be very careful to distinguish between line charts and other visualizations
     """
    try:
        encoded_image = await encode_image_async(path)
        result = await run_llm_in_thread(response_handler, encoded_image, text)
        return result.get("chart_type", "other")
    except Exception as e:
        print(f"Error analyzing chart type for {path}: {e}")
        return "other"

async def figure_data_origin_class_async(path: str) -> dict:
    """异步根据legend和图片信息进行数据来源分类"""
    response_handler = ModalLLMResponse()
    text = """
     Please analyze the following image and classify the data origin based on the legend and chart information. 
     Classify according to the following hierarchy:

     **First Level - Data Origin:**
     - Experimental Data
     - Computational Data  
     - Schematics / Illustrations

     **Second Level - Detailed Classification:**
     
     For "Experimental Data":
     - Microscopy: [SEM, TEM, AFM, etc.]
     - Structural Characterization: [XRD, SAXS, Raman, FTIR, etc.]
     - Compositional Analysis: [XPS, EDS, Auger, etc.]
     - Performance: [Mechanical, Electrochemical, Optical, Thermal, etc.]

     For "Computational Data":
     - DFT: [Band Structure, DOS, Charge Density Difference, etc.]
     - MD: [MSD, RDF, Energy Force, etc.]
     - FEM: [Stress/Strain, etc.]

     For "Schematics / Illustrations":
     - Structural diagrams
     - Process flow diagrams
     - Conceptual illustrations

     Analyze the chart content, legend, axis labels, and any text to determine the most appropriate classification.
     
     - **Output Specification:**
       Strict JSON format:
       ```json
     {
       "first_level": "Experimental Data", // or "Computational Data", "Schematics / Illustrations"
       "second_level": "Microscopy", // specific category from second level
       "confidence": 0.8, // confidence score from 0.0 to 1.0
       "reasoning": "brief explanation of classification"
     }
     ```
     """
    try:
        encoded_image = await encode_image_async(path)
        result = await run_llm_in_thread(response_handler, encoded_image, text)
        return {
            "first_level": result.get("first_level", "Unknown"),
            "second_level": result.get("second_level", "Unknown"),
            "confidence": result.get("confidence", 0.0),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"Error analyzing data origin for {path}: {e}")
        return {
            "first_level": "Unknown", "second_level": "Unknown", 
            "confidence": 0.0, "reasoning": f"Error: {str(e)}"
        }

async def analyze_single_image_async(image_path: str) -> dict:
    """异步分析单个图片，返回完整的分类信息"""
    
    # 为了更高的效率，对同一张图片的两项分析也可以并发进行
    chart_type_task = asyncio.create_task(figure_chart_type_async(image_path))
    data_origin_task = asyncio.create_task(figure_data_origin_class_async(image_path))

    # 等待两个核心分析任务完成
    chart_type, data_origin = await asyncio.gather(chart_type_task, data_origin_task)

    # 如果是线图，再额外执行坐标轴识别任务
    if chart_type == "line chart":
        await figure_legend_axis_name_async(image_path)
    
    path_obj = Path(image_path)
    result = {
        "file_path": image_path,
        "file_name": path_obj.name,
        "parent_folder": path_obj.parent.name,
        "chart_type": chart_type,
        "data_origin": data_origin,
        "timestamp": datetime.now().isoformat()
    }
    return result

async def batch_analyze_images_async(folder_path: str, output_file: str = "class_new.json"):
    """使用asyncio并发批量分析文件夹中的所有图片并保存结果"""
    
    # 查找所有符合命名规则的图片文件
    pattern = re.compile(r'^(.*)_\d+\.(jpg|png|jpeg|bmp|tif|tiff)$', re.IGNORECASE)
    image_files = [fp for fp in Path(folder_path).rglob("*") if fp.is_file() and pattern.match(fp.name)]

    if not image_files:
        print(f"No matching image files found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to analyze...")
    
    output_path = Path(output_file)

    results = load_existing_results(output_path)
    print(f"Loaded {len(results)} existing results")
    
    # 使用Semaphore控制并发任务数量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    # 创建一个包装函数来使用semaphore
    async def process_with_semaphore(image_path_str):
        async with semaphore:
            return await analyze_single_image_async(image_path_str)

    # 创建所有需要处理的任务列表
    tasks_to_run = []
    for image_path in image_files:
        unique_key = f"{image_path.parent.parent.name}_{image_path.name}"
        if unique_key not in results:
            # 将任务和其唯一标识符关联起来
            task = asyncio.create_task(process_with_semaphore(str(image_path)))
            tasks_to_run.append((unique_key, task))
        else:
            print(f"Skipping: {unique_key} (already processed)")

    if not tasks_to_run:
        print("No new images to process.")
        print_statistics(results)
        return
        
    print(f"Processing {len(tasks_to_run)} new images with up to {MAX_CONCURRENT_TASKS} concurrent tasks...")

    # 使用asyncio.as_completed来在任务完成时立刻处理结果
    processed_count = 0
    total_tasks = len(tasks_to_run)
    for i, (unique_key, task) in enumerate(tasks_to_run, 1):
        try:
            # 等待下一个完成的任务
            result = await task
            results[unique_key] = result
            processed_count += 1
            print(f"({i}/{total_tasks}) COMPLETED: {unique_key}")
        except Exception as e:
            # 记录错误信息
            print(f"({i}/{total_tasks}) FAILED: {unique_key} with error: {e}")
            # 从任务中恢复image_path可能会复杂，直接记录错误
            results[unique_key] = {"error": str(e), "unique_key": unique_key}
        finally:
            # 每次有任务完成（无论成功还是失败）都保存一次
            save_results_incrementally(results, output_path)

    print("\nProcessing complete!")
    print(f"New images processed: {processed_count}")
    print(f"Total results: {len(results)}")
    print(f"Final results saved to: {output_path}")
    
    print_statistics(results)

# --- 同步辅助函数 (无需修改) ---

def load_existing_results(output_path: Path) -> dict:
    """加载已有的JSON结果文件"""
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load or parse existing results file: {e}. Starting fresh.")
    return {}

def save_results_incrementally(results: dict, output_path: Path):
    """增量保存结果到JSON文件"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")

def print_statistics(results: dict):
    """打印分析结果的统计信息"""
    total_images = len(results)
    chart_types, data_origins_first, data_origins_second = {}, {}, {}
    
    for result in results.values():
        if isinstance(result, dict):
            chart_type = result.get('chart_type', 'unknown')
            chart_types[chart_type] = chart_types.get(chart_type, 0) + 1
            
            data_origin = result.get('data_origin', {})
            if isinstance(data_origin, dict):
                first_level = data_origin.get('first_level', 'unknown')
                second_level = data_origin.get('second_level', 'unknown')
                data_origins_first[first_level] = data_origins_first.get(first_level, 0) + 1
                data_origins_second[second_level] = data_origins_second.get(second_level, 0) + 1
    
    print("\n" + "="*50)
    print("ANALYSIS STATISTICS")
    print("="*50)
    print(f"Total images in result file: {total_images}")
    print("\nChart Types:")
    for chart_type, count in sorted(chart_types.items()): print(f"  {chart_type}: {count}")
    print("\nData Origin (First Level):")
    for origin, count in sorted(data_origins_first.items()): print(f"  {origin}: {count}")
    print("\nData Origin (Second Level):")
    for origin, count in sorted(data_origins_second.items()): print(f"  {origin}: {count}")

# --- 主程序入口 ---

async def main_async():
    """主异步函数 - 自动化识别指定文件夹中的所有图片"""
    for i in range(36,38):
        target_folder = Path(rf"E:\split_battery_sorted\battery_{i}")
        print("Starting automatic image classification (Concurrent Mode)...")
        print(f"Target folder: {target_folder}")
        jsonpath = Path(rf"E:\split_battery_sorted\classbattery_{i}.json")
        await batch_analyze_images_async(str(target_folder), jsonpath)

# 这里保留了您原来的reprocess逻辑，但请注意它是同步的
def main_reprocess():
    # ... 您原来的 reprocess_chart_types 和 main_reprocess 函数代码可以放在这里 ...
    print("Reprocess function is not implemented in this async script yet.")
    print("Please run it as a separate synchronous script if needed.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reprocess":
        main_reprocess()
    else:
        # 使用 asyncio.run() 来启动异步主函数
        asyncio.run(main_async())
