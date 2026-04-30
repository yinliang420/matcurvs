
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import asyncio
from ModalLLMResponse import ModalLLMResponse
from prompt import get_axis_colors, get_axis_candidates, get_axis_verification_prompt, optimize_detection_params, verify_axis_alignment


def image_to_base64(image):
    """
    Converts an OpenCV image object to a Base64 encoded string.

    This function is designed to take an image represented as a NumPy array
    (the standard format for OpenCV) and convert it into a Base64 string.
    This is commonly required for sending image data via text-based protocols
    like JSON or for embedding images directly into HTML or other documents
    using data URIs.

    The process involves:
    1. Converting the image's color space from OpenCV's default BGR to RGB,
       which is expected by the Pillow library.
    2. Creating a Pillow Image object from the array.
    3. Saving the image to an in-memory binary buffer in PNG format. Using an
       in-memory buffer is efficient as it avoids writing temporary files to disk.
    4. Encoding the binary data from the buffer into Base64.
    5. Decoding the result into a standard UTF-8 string.

    Args:
        image (numpy.ndarray): The input image loaded by OpenCV. It is expected
            to be in the BGR color format.

    Returns:
        str: A UTF-8 encoded string containing the Base64 representation of
             the image, formatted as a PNG.
    """
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def draw_candidates(image, candidates):
    """
    Draws the identified candidate axes onto an image for visualization.

    This function takes an image and a dictionary of candidate axes (as returned
    by `filter_candidate_axes`) and draws them onto a copy of the image. It is
    useful for debugging and visually verifying the results of the axis detection
    process. Different colors can be used to distinguish between primary,
    secondary, horizontal, and vertical axes.

    Args:
        image (numpy.ndarray): The source image on which to draw the lines.
            This should be an OpenCV-compatible image format (e.g., a NumPy array).
        candidates (dict): A dictionary containing the axes to be drawn. It is
            expected to have "horizontal" and "vertical" keys. Each key should
            contain a list of axes, where each axis is a NumPy array with the
            shape (1, 4) containing the coordinates [x1, y1, x2, y2].

    Returns:
        tuple: A tuple containing:
            - marked_image (numpy.ndarray): A new image (a copy of the original) with the candidate
              lines drawn on it.
            - labels (list): A list of tuples containing orientation, coordinates, and color name
              for each drawn axis.
    """
    marked_image = image.copy()
    labels = []
    colors = {
        "horizontal": [(0, 0, 255), (0, 165, 255)],  # 红、橙
        "vertical": [(0, 255, 0), (255, 0, 0)]  # 绿、蓝
    }
    color_names = {
        "horizontal": ["red", "orange"],
        "vertical": ["green", "blue"]
    }

    for i, line in enumerate(candidates["horizontal"]):
        x1, y1, x2, y2 = line[0]
        cv2.line(marked_image, (x1, y1), (x2, y2), colors["horizontal"][i], 5)
        labels.append(("horizontal", (x1, y1, x2, y2), color_names["horizontal"][i]))

    for i, line in enumerate(candidates["vertical"]):
        x1, y1, x2, y2 = line[0]
        cv2.line(marked_image, (x1, y1), (x2, y2), colors["vertical"][i], 5)
        labels.append(("vertical", (x1, y1, x2, y2), color_names["vertical"][i]))

    return marked_image, labels

def are_axes_consistent(axis1, axis2, dimension_size, orientation, threshold=0.01):
    """
    Checks if two axes are consistent within a given tolerance.
    
    This function compares two detected axes to determine if they represent the same
    line within a specified tolerance threshold. It calculates the position of each
    axis based on its orientation and checks if the deviation is within acceptable limits.
    
    Args:
        axis1 (list): The first axis coordinates [x1, y1, x2, y2].
        axis2 (list): The second axis coordinates [x1, y1, x2, y2].
        dimension_size (int): The image width for vertical axes or height for horizontal axes.
        orientation (str): Axis orientation, either "vertical" or "horizontal".
        threshold (float, optional): The allowed deviation percentage (e.g., 0.01 for 1%).
            Defaults to 0.01.

    Returns:
        bool: True if the axes are consistent within the threshold, False otherwise.
    """
    # If either axis is missing, they are not consistent
    if axis1 is None or axis2 is None:
        return False

    # For a vertical line, its position is defined by its x-coordinate.
    if orientation == "vertical":
        pos1 = (axis1[0] + axis1[2]) / 2
        pos2 = (axis2[0] + axis2[2]) / 2
    # For a horizontal line, its position is defined by its y-coordinate.
    elif orientation == "horizontal":
        pos1 = (axis1[1] + axis1[3]) / 2
        pos2 = (axis2[1] + axis2[3]) / 2
    else:
        return False # Should not happen

    # Calculate the deviation and check if it's within the tolerance
    deviation = abs(pos1 - pos2)
    return deviation < (dimension_size * threshold)

async def filter_candidate_axes(lines, image_shape, base64_no_color, response_modal_llm):
    """
    Identifies and filters candidate axes from detected lines using multimodal LLM analysis.

    This function takes a list of lines detected in an image and filters them into
    horizontal and vertical candidates based on their angle and length. It then
    leverages a multimodal Large Language Model (LLM) to analyze the image and
    the candidate lines to intelligently select the most likely primary (e.g., Y-axis)
    and secondary (e.g., Y2-axis) axes. To ensure reliability, it includes a retry
    mechanism that makes multiple calls to the LLM, accepting a result only when
    the identified 'primary_axis' is consistent between two consecutive attempts,
    allowing for a small deviation.

    Args:
        lines (list or None): A list of lines detected by a line detection algorithm
            (like OpenCV's HoughLinesP). Each element should be a numpy array in the
            format [[x1, y1, x2, y2]]. If no lines are detected, this can be None.
        image_shape (tuple): A tuple containing the dimensions of the source image,
            typically in (height, width) or (height, width, channels) format.
        base64_no_color (str): A Base64 encoded string of the source image,
            preferably in grayscale or with color information removed, to be
            sent to the LLM for analysis.
        response_modal_llm (ModalLLMResponse): An instance of a client class used to
            interact with a multimodal Large Language Model. This object must
            have a `generate_response` method that accepts an image string
            and a text prompt.

    Returns:
        tuple: A tuple containing three elements:
            - new_candidates (dict): A dictionary with "vertical" and "horizontal" keys.
              Each key holds a list containing two numpy arrays: the primary axis and
              the secondary axis as identified by the LLM.
            - horizontal (list): A list of all raw horizontal line coordinates
              `[x1, y1, x2, y2]` that were considered candidates.
            - vertical (list): A list of all raw vertical line coordinates
              `[x1, y1, x2, y2]` that were considered candidates.
    """
    height, width = image_shape[:2]
    candidates = {"horizontal": [], "vertical": []}

    # --- NEW: Constants for the consistency check ---
    MAX_RETRIES = 1  # Maximum number of attempts to get a consistent result
    DEVIATION_THRESHOLD = 0.01  # 1% tolerance

    if lines is None:
        return candidates, [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx)) % 180

        # 筛选水平线
        if abs(angle) < 5 or abs(180 - angle) < 5:
            length = np.sqrt(dx**2 + dy**2)
            if length > 100:
                candidates["horizontal"].append(line)
        # 筛选垂直线
        elif abs(90 - angle) < 5:
            length = np.sqrt(dx**2 + dy**2)
            if length > 100:
                candidates["vertical"].append(line)

    horizontal = [arr.tolist()[0] for arr in candidates["horizontal"]]
    vertical = [arr.tolist()[0] for arr in candidates["vertical"]]
    print(f"Found {len(horizontal)} horizontal and {len(vertical)} vertical candidate lines.")

    # --- NEW: Retry loop for vertical axis consistency ---
    ver_dict = {}
    prev_ver_dict = None
    print("\n--- Checking for consistent Vertical axis ---")
    loop = asyncio.get_running_loop()
    for i in range(MAX_RETRIES):
        print(f"Attempt {i+1}/{MAX_RETRIES}...")
        prompt = get_axis_candidates(vertical, "vertical", height, width)
        ver_dict = await loop.run_in_executor(
            None, response_modal_llm.generate_response, base64_no_color, prompt
        )
        
        # On the first successful run, there's nothing to compare to yet.
        if prev_ver_dict is not None:
            if are_axes_consistent(
                ver_dict.get("primary_axis"),
                prev_ver_dict.get("primary_axis"),
                dimension_size=width,
                orientation="vertical",
                threshold=DEVIATION_THRESHOLD
            ):
                print("Vertical 'primary_axis' is consistent. Proceeding.")
                break  # Exit loop if consistent
        
        prev_ver_dict = ver_dict
        if i == MAX_RETRIES - 1:
            print("Max retries reached for vertical axis. Using last result.")

    # --- NEW: Retry loop for horizontal axis consistency ---
    hor_dict = {}
    prev_hor_dict = None
    print("\n--- Checking for consistent Horizontal axis ---")
    for i in range(MAX_RETRIES):
        print(f"Attempt {i+1}/{MAX_RETRIES}...")
        prompt = get_axis_candidates(horizontal, "horizontal", height, width)
        hor_dict = await loop.run_in_executor(
            None, response_modal_llm.generate_response, base64_no_color, prompt
        )

        if prev_hor_dict is not None:
            if are_axes_consistent(
                hor_dict.get("primary_axis"),
                prev_hor_dict.get("primary_axis"),
                dimension_size=height,
                orientation="horizontal",
                threshold=DEVIATION_THRESHOLD
            ):
                print("Horizontal 'primary_axis' is consistent. Proceeding.")
                break # Exit loop if consistent
        
        prev_hor_dict = hor_dict
        if i == MAX_RETRIES - 1:
            print("Max retries reached for horizontal axis. Using last result.")

    print("\nFinal Dictionaries (pre-cleanup):", ver_dict, hor_dict)

    # --- The rest of your logic remains the same ---

    # Fill with default values if an axis is None
    if ver_dict.get("primary_axis") is None:
        ver_dict["primary_axis"] = [width, 0, width, height] # Default to right edge
    if ver_dict.get("secondary_axis") is None:
        ver_dict["secondary_axis"] = [0, 0, 0, height] # Default to left edge
        
    if hor_dict.get("primary_axis") is None:
        hor_dict["primary_axis"] = [0, 0, width, 0] # Default to top edge
    if hor_dict.get("secondary_axis") is None:
        hor_dict["secondary_axis"] = [0, height, width, height] # Default to bottom edge

    print("Final Dictionaries (post-cleanup):", ver_dict, hor_dict)
    
    new_candidates = {
        "vertical": [np.array([ver_dict["primary_axis"]]), np.array([ver_dict["secondary_axis"]])],
        "horizontal": [np.array([hor_dict["primary_axis"]]), np.array([hor_dict["secondary_axis"]])]
    }
    
    return new_candidates, horizontal, vertical

def _detect_edges(image, thresholds=None):
    """
    Enhanced edge detection optimized for detecting long, straight axis lines.
    
    This function performs Canny edge detection with configurable thresholds, optimized
    for detecting long, straight axis lines in chart images. It focuses on preserving
    long linear structures while suppressing short edges and curves.
    
    Args:
        image (numpy.ndarray): Input image in grayscale or color format (BGR).
            If color, it will be automatically converted to grayscale.
        thresholds (list, optional): List containing [threshold1, threshold2] for Canny
            edge detection. threshold1 is the lower threshold for edge linking,
            threshold2 is the upper threshold for edge detection.
            If None, uses default values [10, 150].
    
    Returns:
        numpy.ndarray: Binary edge map where edges are represented as white pixels
            (255) on a black background (0).
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if thresholds is None:
        threshold1, threshold2 = 10, 150
    else:
        threshold1, threshold2 = thresholds[0], thresholds[1]
    
    edges = cv2.Canny(image, threshold1=threshold1, threshold2=threshold2)
    return edges

def _hough_lines(edges, params=None):
    """
    Performs an enhanced Hough line detection using multiple parameter sets and a robust merging logic.

    This function improves upon the standard Hough Transform by adopting a two-stage approach.
    It is specifically designed to handle difficult images where lines may be faint, dotted,
    or fragmented.

    **Stage 1: Multi-Pass Detection**
    The function first runs the Probabilistic Hough Transform (`cv2.HoughLinesP`) multiple
    times. Each run uses a different set of parameters, from very lenient (to detect
    weak and dashed lines) to more strict. This ensures that a comprehensive set of
    line candidates is collected.

    **Stage 2: Line Merging and Filtering**
    The raw list of lines from the first stage is often redundant and fragmented. This stage
    cleans up the data by iterating through all detected lines and merging those that are
    similar in angle and close to each other. This process combines collinear fragments
    into single, longer lines, resulting in a much cleaner and more accurate final output.

    Args:
        edges (numpy.ndarray): The input binary edge map, typically the output from a
            Canny edge detector.
        params (list, optional): A list of dictionaries, where each dictionary defines a
            set of parameters for a `cv2.HoughLinesP` call. If not provided (None),
            it defaults to a list of pre-defined parameter sets tuned for both
            strong and weak line detection.

    Returns:
        numpy.ndarray or None: A NumPy array containing the final, merged lines in the
            format `[[[x1, y1, x2, y2]]]`. Returns `None` if no lines are ultimately
            detected after all steps.
    """
    height, width = edges.shape

    if params is None:
        # 优化：参数组合更偏向长直线、宽松gap，特别检测横/竖线
        params = [
            # 标准模式
            {'rho': 1, 'theta': np.pi/180, 'threshold': 70,
             'minLineLength': max(100, width//4), 'maxLineGap': 5},
            # 严格模式：过滤噪声
            {'rho': 1, 'theta': np.pi/180, 'threshold': 100,
             'minLineLength': max(150, width//3), 'maxLineGap': 2},
        ]

    all_lines = []
    for param_set in params:
        lines = cv2.HoughLinesP(
            edges,
            rho=param_set.get('rho', 1),
            theta=param_set.get('theta', np.pi / 180),
            threshold=param_set.get('threshold', 50),
            minLineLength=param_set.get('minLineLength', 100),
            maxLineGap=param_set.get('maxLineGap', 5)
        )
        if lines is not None:
            all_lines.extend([line[0] for line in lines.tolist()])

    if not all_lines:
        return None

    # Improved line merging and filtering
    merged_lines = []
    angle_threshold = 2 * np.pi / 180  # 5 degrees
    distance_threshold = 10  # pixels

    for line in all_lines:
        x1, y1, x2, y2 = line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.arctan2(y2 - y1, x2 - x1)

        # Filter very short lines
        if length < 100:
            continue

        merged = False
        for i, (ml_x1, ml_y1, ml_x2, ml_y2) in enumerate(merged_lines):
            ml_angle = np.arctan2(ml_y2 - ml_y1, ml_x2 - ml_x1)
            angle_diff = abs(angle - ml_angle)

            # Check if lines are similar in angle
            if angle_diff < angle_threshold or \
                    abs(angle_diff - np.pi) < angle_threshold:

                # Check distance between lines
                dist1 = np.sqrt((x1 - ml_x1) ** 2 + (y1 - ml_y1) ** 2)
                dist2 = np.sqrt((x2 - ml_x2) ** 2 + (y2 - ml_y2) ** 2)

                if dist1 < distance_threshold and dist2 < distance_threshold:
                    # Merge lines by taking the longest combination
                    new_line = [
                        min(x1, ml_x1, x2, ml_x2),
                        min(y1, ml_y1, y2, ml_y2),
                        max(x1, ml_x1, x2, ml_x2),
                        max(y1, ml_y1, y2, ml_y2)
                    ]
                    merged_lines[i] = new_line
                    merged = True
                    break

        if not merged:
            merged_lines.append(line)

    # Convert back to numpy array format
    return np.array([[[x1, y1, x2, y2]] for x1, y1, x2, y2 in merged_lines])

def preprocess_image(image_path, preprocess_params=None):
    """
    Loads and preprocesses an image to enhance its quality for axis detection analysis.

    This function implements a comprehensive image preprocessing workflow designed to
    improve the clarity and contrast of an image, making subsequent tasks like
    edge detection or feature extraction more effective. The pipeline consists
    of the following steps:
    1.  **Loading**: The image is read from the specified file path.
    2.  **Grayscale Conversion**: The image is converted from color to grayscale,
        as many feature detection algorithms operate on single-channel images.
    3.  **Initial Denoising**: Applies Gaussian blur for initial noise reduction.
    4.  **Contrast Enhancement**: Uses Contrast Limited Adaptive Histogram
        Equalization (CLAHE) to improve local contrast. Unlike global histogram
        equalization, CLAHE enhances detail in both dark and bright regions of
        the image without over-saturating them.
    5.  **Final Denoising**: Applies a second Gaussian blur to smooth the enhanced image.

    Args:
        image_path (str): The file path to the input image file.
        preprocess_params (dict, optional): Dictionary containing preprocessing parameters:
            - 'gaussian_blur_kernel' (list): List of kernel sizes [k1, k2] for two blur
              operations. Both values must be odd integers. If None, uses default [3, 3].
            - 'clahe_tile_grid_size' (list): List [width, height] for CLAHE tile grid size.
              Controls the local contrast enhancement granularity. If None, uses default [16, 16].

    Returns:
        tuple: A tuple containing two NumPy arrays:
            - image (numpy.ndarray): The original, unmodified color image in BGR format.
            - enhanced (numpy.ndarray): The final grayscale image after all
              preprocessing steps have been applied.

    Raises:
        FileNotFoundError: If the image cannot be found or read from the
            specified `image_path`.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Default parameters
    if preprocess_params is None:
        blur_kernel1, blur_kernel2 = 3, 3
        clahe_tile_size = (16, 16)
    else:
        blur_kernels = preprocess_params.get('gaussian_blur_kernel', [3, 3])
        blur_kernel1, blur_kernel2 = blur_kernels[0], blur_kernels[1]
        clahe_tile_size = tuple(preprocess_params.get('clahe_tile_grid_size', [16, 16]))
    
    # Ensure kernel sizes are odd
    blur_kernel1 = blur_kernel1 if blur_kernel1 % 2 == 1 else blur_kernel1 + 1
    blur_kernel2 = blur_kernel2 if blur_kernel2 % 2 == 1 else blur_kernel2 + 1
    
    denoised = cv2.GaussianBlur(gray, (blur_kernel1, blur_kernel1), 0)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=clahe_tile_size)
    enhanced = clahe.apply(denoised)
    enhanced = cv2.GaussianBlur(enhanced, (blur_kernel2, blur_kernel2), 0)
    return image, enhanced

def plot_images(images, titles, figsize=(15, 10), save_path=None):
    """
    Displays multiple images side by side in a single matplotlib figure.
    
    This utility function creates a horizontal subplot layout to visualize
    multiple images together, which is useful for comparing original images,
    processed images, edge maps, and detection results.
    
    Args:
        images (list): List of numpy arrays representing images to display.
            Images can be grayscale (2D array) or color (3D array in BGR format).
        titles (list): List of strings, one title for each image.
            Must have the same length as the images list.
        figsize (tuple, optional): Figure size in inches (width, height).
            Defaults to (15, 10).
        save_path (str, optional): If provided, saves the figure to this path instead
            of displaying it. The path should include the filename and extension.
            Defaults to None (displays the plot).
    
    Returns:
        None: This function displays the plot or saves it but does not return a value.
    """
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        plt.rcParams['font.family'] = 'serif'
    fontsize=18
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=fontsize, pad=4)
        plt.axis('off')
    
    if save_path is not None:
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Image saved to: {save_path}")
    else:
        plt.show()

def show_result_for_feedback(original_image, x_axis, y_axis, edges_image=None, marked_image=None):
    """
    Displays axis detection results and prompts user for feedback confirmation.
    
    This function creates a visualization showing the original image, detected axes,
    and optionally all candidate lines. It then prompts the user to confirm or reject
    the detection results. The function blocks until the user provides input.
    
    Args:
        original_image (numpy.ndarray): The original input image in BGR format.
        x_axis (list or None): Detected X-axis coordinates as [x1, y1, x2, y2],
            or None if no X-axis was detected.
        y_axis (list or None): Detected Y-axis coordinates as [x1, y1, x2, y2],
            or None if no Y-axis was detected.
        edges_image (numpy.ndarray, optional): Binary edge map for visualization.
            If provided, will be displayed in the visualization. Defaults to None.
        marked_image (numpy.ndarray, optional): Image with all candidate axes marked
            in different colors. If provided, will be displayed. Defaults to None.
    
    Returns:
        bool: True if the user confirms the result is correct, False if the user
            rejects the result and requests parameter optimization.
    """
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        plt.rcParams['font.family'] = 'serif'
    fig, axes = plt.subplots(1, 2 if marked_image is None else 3, figsize=(15, 6))
    
    # Original image with detected axes
    final_image = original_image.copy()
    if x_axis:
        cv2.line(final_image, (int(x_axis[0]), int(x_axis[1])),
                 (int(x_axis[2]), int(x_axis[3])), (0, 255, 0), 3)
        cv2.putText(final_image, 'X-axis', (int(x_axis[0]) + 10, int(x_axis[1]) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if y_axis:
        cv2.line(final_image, (int(y_axis[0]), int(y_axis[1])),
                 (int(y_axis[2]), int(y_axis[3])), (0, 0, 255), 3)
        cv2.putText(final_image, 'Y-axis', (int(y_axis[0]) + 10, int(y_axis[1]) + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=18)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Detected Axes (Green=X-axis, Red=Y-axis)', fontsize=18)
    axes[1].axis('off')
    
    if marked_image is not None:
        axes[2].imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title('All Candidates', fontsize=18)
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Axis Detection Results - Please Review', fontsize=18, y=0.08)
    
    print("\n" + "="*60)
    print("检测结果已显示。请查看图像并确认：")
    print("="*60)
    print("X轴坐标:", x_axis)
    print("Y轴坐标:", y_axis)
    print("="*60)
    print("\n请选择：")
    print("1. 输入 'y' 或 'yes' 确认结果正确")
    print("2. 输入 'n' 或 'no' 拒绝结果，将使用多模态大模型优化参数")
    print("="*60)
    
    plt.show(block=False)
    plt.pause(0.1)  # Give time for the plot to render
    
    # Get user input
    while True:
        user_input = input("\n您的选择 (y/n): ").strip().lower()
        if user_input in ['y', 'yes', '是', '确认']:
            plt.close()
            return True
        elif user_input in ['n', 'no', '否', '拒绝']:
            plt.close()
            return False
        else:
            print("无效输入，请输入 'y' 或 'n'")

def encode_image(image_path: str) -> str:
    """
    Encodes an image file to Base64 string format.
    
    This function reads an image file from disk and encodes it as a Base64 string,
    which is commonly used for transmitting image data over text-based protocols
    or embedding images in JSON/API requests.
    
    Args:
        image_path (str): The file path to the image file to be encoded.
            Supports common image formats (PNG, JPEG, etc.).
    
    Returns:
        str: A UTF-8 encoded Base64 string representation of the image file.
    
    Raises:
        FileNotFoundError: If the image file cannot be found at the specified path.
        IOError: If the file cannot be read.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

async def detect_axes(image_path, response_modal_llm: ModalLLMResponse, visualise=False, enable_feedback=True, max_iterations=3, save_path=None):
    """
    Executes the main pipeline to detect the X and Y axes in a chart image with interactive feedback.

    This function combines traditional computer vision techniques with a multimodal
    Large Language Model (LLM) to accurately identify the primary axes of a chart.
    The workflow includes an interactive feedback mechanism:

    1.  **Initial Detection with Default Parameters**:
        Uses default parameters for preprocessing, edge detection, and line detection.

    2.  **User Feedback**:
        Displays the detection results and asks the user to confirm or reject.

    3.  **Parameter Optimization (if rejected)**:
        If the user rejects the result, uses a multimodal LLM to analyze the image
        and optimize detection parameters (preprocessing, Canny thresholds, Hough parameters).

    4.  **Re-detection with Optimized Parameters**:
        Repeats the detection process with optimized parameters.

    5.  **Iteration**:
        Repeats steps 2-4 until user confirms or maximum iterations reached.

    Args:
        image_path (str): The file path to the chart image to be analyzed.
        response_modal_llm (ModalLLMResponse): An instantiated client object for the
            multimodal LLM, capable of processing image and text prompts.
        visualise (bool, optional): If True, displays a plot showing the intermediate
            and final results of the detection process. Defaults to False.
        enable_feedback (bool, optional): If True, enables interactive user feedback.
            Defaults to True.
        max_iterations (int, optional): Maximum number of optimization iterations.
            Defaults to 3.
        save_path (str, optional): If provided and visualise is True, saves the
            visualization plot to this path instead of displaying it. The path should
            include the filename and extension. Defaults to None.

    Returns:
        tuple: A tuple `(x_axis, y_axis)` where each element is either a list of
            coordinates `[x1, y1, x2, y2]` for the identified axis, or `None` if
            the axis could not be determined.
    """
    # Default parameters
    preprocess_params = None
    edge_params = None
    hough_params = None
    
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"第 {iteration} 次检测尝试")
        print(f"{'='*60}")
        
        # Step 1: Preprocessing
        original, enhanced = preprocess_image(image_path, preprocess_params)
        
        # Step 2: Edge detection
        edges = _detect_edges(enhanced, edge_params)
        
        # Step 3: Line detection
        lines = _hough_lines(edges, hough_params)
        
        # Step 4: Candidate filtering with LLM
        base64_no_color = encode_image(image_path)
        candidates, h, v = await filter_candidate_axes(lines, original.shape, base64_no_color, response_modal_llm)
        
        # Step 5: Draw candidates
        marked_image, labels = draw_candidates(original, candidates)
        
        # Step 6: Final identification with LLM
        img_base64 = image_to_base64(marked_image)
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, response_modal_llm.generate_response, img_base64, get_axis_colors()
        )
        
        # Step 7: Parse results
        x_color = response.get('x_axis')
        y_color = response.get('y_axis')
        
        x_axis = None
        y_axis = None
        for label in labels:
            if label[2] == x_color:
                x_axis = label[1]
            if label[2] == y_color:
                y_axis = label[1]
        
        # Step 8: User feedback (if enabled)
        if enable_feedback:
            #user_confirmed = show_result_for_feedback(original, x_axis, y_axis, edges, marked_image)
            user_confirmed = True
            if user_confirmed:
                print("\n✓ 用户已确认检测结果")
                break
            else:
                print(f"\n✗ 用户拒绝了检测结果，准备进行参数优化...")
                
                # Use LLM to verify alignment
                verification_img = original.copy()
                if x_axis:
                    cv2.line(verification_img, (int(x_axis[0]), int(x_axis[1])),
                            (int(x_axis[2]), int(x_axis[3])), (0, 255, 0), 3)
                if y_axis:
                    cv2.line(verification_img, (int(y_axis[0]), int(y_axis[1])),
                            (int(y_axis[2]), int(y_axis[3])), (0, 0, 255), 3)
                
                verification_base64 = image_to_base64(verification_img)
                verification_prompt = verify_axis_alignment(x_axis, y_axis)
                loop = asyncio.get_running_loop()
                verification_result = await loop.run_in_executor(
                    None, response_modal_llm.generate_response, verification_base64, verification_prompt
                )
                
                # Optimize parameters using LLM
                print("正在使用多模态大模型优化参数...")
                current_preprocess = preprocess_params or {
                    'gaussian_blur_kernel': [3, 3],
                    'clahe_tile_grid_size': [16, 16]
                }
                current_edge = edge_params or [10, 150]
                current_hough = hough_params or [
                    {'rho': 1, 'theta': np.pi/180, 'threshold': 70,
                     'minLineLength': max(100, original.shape[1]//4), 'maxLineGap': 5},
                    {'rho': 1, 'theta': np.pi/180, 'threshold': 100,
                     'minLineLength': max(150, original.shape[1]//3), 'maxLineGap': 2}
                ]
                
                optimization_prompt = optimize_detection_params(
                    current_edge, current_hough, current_preprocess, verification_result
                )
                optimized_params = await loop.run_in_executor(
                    None, response_modal_llm.generate_response, base64_no_color, optimization_prompt
                )
                
                # Extract optimized parameters
                if 'preprocess_params' in optimized_params:
                    preprocess_params = optimized_params['preprocess_params']
                    print(f"优化后的预处理参数: {preprocess_params}")
                
                if 'edge_params' in optimized_params:
                    edge_params = optimized_params['edge_params']
                    print(f"优化后的边缘检测参数: {edge_params}")
                
                if 'hough_params' in optimized_params:
                    hough_params = optimized_params['hough_params']
                    print(f"优化后的霍夫变换参数: {hough_params}")
                
                if iteration >= max_iterations:
                    print(f"\n已达到最大迭代次数 ({max_iterations})，返回最后一次检测结果")
        else:
            # If feedback is disabled, just return the result
            break
    
    # Final visualization (if requested)
    if visualise:
        final_image = original.copy()
        if x_axis:
            cv2.line(final_image, (int(x_axis[0]), int(x_axis[1])),
                     (int(x_axis[2]), int(x_axis[3])), (0, 255, 0), 5)
        if y_axis:
            cv2.line(final_image, (int(y_axis[0]), int(y_axis[1])),
                     (int(y_axis[2]), int(y_axis[3])), (0, 0, 255), 5)
        
        images = [original, edges, marked_image, final_image]
        titles = ["Original", "Edges", "Candidates", "Final Axes"]
        plot_images(images, titles, figsize=(16, 6), save_path=save_path)
    
    return x_axis, y_axis


# # # 示例图像路径
# image_path = "C:\\Users\Yin Liang\Desktop\split_images\\figure2_a.png"
# # image_path = "C:\\Users\Yin Liang\Desktop\split_images\\test\AdvancedScience2017WangUtilizingCo2\\figure6_d.png"
# # # 创建多模态响应对象
# responsed = ModalLLMResponse()  # model_name="qwen-vl-max"
# #
# # # 使用默认参数检测
# # print("使用默认参数检测:")
# x, y = asyncio.run(detect_axes(image_path, responsed, visualise=True, enable_feedback=True, max_iterations=3))
# print(x, y)
