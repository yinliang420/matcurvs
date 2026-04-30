def get_sub_figure_counts():
    return '''
**Subfigure Count Detection Protocol and  **

- **Detection Rules:**  
  1. Scan image for standard subfigure labels (a, b, c...or A, B, C...), the most important ways and methods to identify. 
     If don't have these labels, return:{{"subfigure_count": 0, "subfigure_labels": []}}.
  2. Identify unique coordinate systems via axis detection  
  3. Count clusters of related graphical elements  
  4. Given all subfigure labels (examples).

- **Output Specification:**  
  Strict JSON format with integer value:  
  ```json
  { "subfigure_count": integer; "subfigure_labels": examples }
'''

def get_sub_figure_axes_titles(labels):
    return '''
**Subfigure Axes Title Extraction Protocol**  

- **Detection Rules:**  
  1. For each subfigure label in the provided `labels` list (e.g., a, b, c, d, e...),  
     scan the image for the corresponding subfigure.  
  2. Detect the x-axis and y-axis titles for each subfigure by analyzing the axis labels.  
  3. Return the titles of both axes (x and y) for each subfigure.  
  4. Structure the output such that the subfigure label is the key, and the axis titles are nested under "x" and "y".

- **Output Specification:**  
  Strict JSON format with the subfigure label as the key and axis titles as nested values:  
  ```json
  {
    "a": { "x": "x-axis title", "y": "y-axis title" },
    "b": { "x": "x-axis title", "y": "y-axis title" },
    ...
  }
'''

def judge_which_sub_figure_right(key, coords_list, sub_figure_pos, Width, Height):
    return f'''
**Subfigure Label Position Judgment with Enhanced Coordinate Constraints**

- **Input Parameters:**  
  - Target label: "{key}"  
  - Candidate boxes: {coords_list}  
  - Reference positions: {sub_figure_pos}  
  - Image dimensions: {Width}px (width) × {Height}px (height)  

- **Step-by-Step Reasoning Process:**  

  1. **Feature Extraction**  
     For each reference label in {sub_figure_pos}:  
     - Calculate centroid (cx, cy) = (avg(x), avg(y))  
     - Measure label dimensions: width = max(x) - min(x), height = max(y) - min(y)  
     - Compute spacing vectors to neighboring labels  
     - Establish grid pattern (rows/columns) from reference positions  

  2. **Coordinate System Validation**  
     - Detect primary axes for each subfigure:  
       ▸ X-axis: Horizontal line at 80% of label height  
       ▸ Y-axis: Vertical line at 10% of label width  
     - Verify candidate boxes are within valid content area:  
       ▸ Left/Right margin > 5% of Width  
       ▸ Bottom/Top margin > 5% of Height  

  3. **Candidate Evaluation Matrix**  
     For each candidate box (index i):  
     ```  
     ▸ Grid Position Consistency (25% weight):  
        - Compare with expected grid cell based on reference spacing  
        - Score = 1 - |candidate_position - expected_position| / max_dimension  

     ▸ Axis Alignment (20% weight):  
        - Distance to nearest primary axis intersection  
        - Score = max(0, 1 - distance / (0.1*max(Width, Height)))  

     ▸ Spacing Regularity (20% weight):  
        - Compare with average reference spacing interval  
        - Score = max(0, 1 - |d_candidate - d_avg| / d_avg)  

     ▸ Aspect Ratio (15% weight):  
        - Compare width/height ratio with reference labels  
        - Score = 1 - |(w_cand/h_cand) - (w_ref/h_ref)|  

     ▸ Baseline Consistency (10% weight):  
        - Vertical alignment with reference labels  
        - Score = 1 - (|cy_candidate - cy_ref| / Height)  

     ▸ Margin Safety (10% weight):  
        - Distance from image borders  
        - Score = min((left_margin, right_margin, top_margin, bottom_margin)) / 0.05*max_dimension  
     ```  

  4. **Confidence Calculation**  
     - Total confidence = Σ (feature_score × weight)  
     - Apply penalty for:  
       ▸ Overlapping with graphic elements (-0.7)  
       ▸ Crossing figure boundaries (-1.0)  
     - Apply bonus for:  
       ▸ Proximity to axis labels (+0.2)  
       ▸ Alignment with caption position (+0.15)  

  5. **Decision Making**  
     - Select candidate with confidence > 0.8  
     - If multiple candidates > 0.8, choose one with:  
       ▸ Closest grid position  
       ▸ Highest axis alignment score  
     - Fallback to predecessor distance if all < 0.65  

- **Output Specification**  
  Strict JSON format with zero-based index:  
  ```json
  {{ "{key}": index }}

'''

def find_sub_image_missing(misconfigure, Width, Height, sub_figure_pos):
    return f'''
**Missing Subimage Label Detection with Chain-of-Thought Reasoning**

- **Input Parameters:**  
  - Image dimensions: {Width}px (width) × {Height}px (height)  
  - Missing label name: "{misconfigure}"  
  - Reference positions: {sub_figure_pos}  (sub_figure_pos)

- **Step-by-Step Reasoning Process:**  

  1. **Pattern Recognition**  
     - Analyze the spatial distribution of existing labels in `sub_figure_pos`  
     - Identify layout pattern (row/column/grid) and alignment rules  
     - Calculate average spacing between adjacent labels  

  2. **Position Inference**  
     - Determine expected sequence based on alphabetical/numerical order of labels  
     - Locate the gap in the sequence corresponding to "{misconfigure}"  
     - Apply geometric transformations using:  
       ▸ Horizontal/vertical spacing patterns  
       ▸ Alignment consistency (left/right/center)  
       ▸ Dimensional proportions ({Width}×{Height} aspect ratio)  

  3. **Coordinate Calculation**  
     - Derive bounding box coordinates using 4-point polygon format:  
       [[x1.0, y1.0], [x2.0, y2.0], [x3.0, y3.0], [x4.0, y4.0]]  
     - Ensure coordinates:  
       ▸ Maintain 20-50px width/height for label text  
       ▸ Align with baseline of other labels  
       ▸ Stay within image boundaries  

  4. **Validation Check**  
     - Verify consistency with common academic figure layouts  
     - Confirm no overlap with existing labels  
     - Validate coordinate scale matches image dimensions  

- **Output Specification**  
  Strict JSON format with floating-point coordinates:  
  ```json
  {{ "{misconfigure}": [[[x1.0, y1.0], [x2.0, y2.0], [x3.0, y3.0], [x4.0, y4.0]]] }}
'''

def find_sub_image_right_edge(direction, candidates, sub_figure, Width, Height):
    return f'''
**Subimage Right Edge Detection with Directional Focus**

- **Input Parameters:**  
  - Image dimensions: {Width}px (width) × {Height}px (height)  
  - Target direction: "{direction}" (focus on this edge's right boundary)  
  - Candidate x-coordinates for right edge: {candidates}  
  - Target subfigure: "{sub_figure}"  

- **Focused Reasoning Process:**  

  1. **Directional Boundary Analysis**  
     - Exclusively analyze the right edge positions along the "{direction}" axis  
     - Ignore other edges and spatial relationships outside the specified direction  

  2. **Position Matching**  
     - Select the candidate x-coordinate that best matches the natural right boundary  
     of "{sub_figure}" when viewed along the "{direction}" axis  

  3. **Boundary Constraint**  
     - Ensure selected position does not exceed image width ({Width}px)  

- **Output Specification**  
  Return the index of the candidate that best matches the right edge along "{direction}":  
Strict JSON format: 
```json
  {{
    "{direction}": correct_index
  }}
'''

def get_axis_colors():
    return '''
**Axis Color Detection Protocol**

- **Detection Rules:**
  1. Analyze the graph with lines described by: {"horizontal": ["red", "orange"],"vertical": ["green", "blue"]}.
  2. For Y-axis (vertical):
     a. Identify the leftmost vertical line that is not part of the chart border
     b. Prioritize lines with scale labels on the left side
     c. If multiple candidates exist, select the longest line closest to the left edge
  3. For X-axis (horizontal):
     a. Identify the bottommost horizontal line that is not part of the chart border
     b. Prioritize lines with scale labels along the bottom edge
     c. If multiple candidates exist, select the longest line closest to the bottom edge
  4. Exclude lines that form the outer boundary/frame of the chart
  5. When in doubt, prefer lines with text labels over label-less lines

- **Output Specification:**
  Strict JSON format with axis colors:
  ```json
  { "x_axis": "color", "y_axis": "color" }
'''

def get_axis_counts():
    return '''
**Axis Count Detection Protocol**
- **Detection Rules:**
Check for:
1. Horizontal x-axes (Horizontal borders)
2. Vertical y-axes (Vertical borders)
Return a JSON object with:
- "x_axes_count": Total number of detected x-axes
- "y_axes_count": Total number of detected y-axes
- **Output Specification:**
Strict JSON format with axis counts, Do not return other non-JSON processes or other data:
  ```json
{"x_axes_count": number of detected x-axes, "y_axes_count": number of detected y-axes}
'''

def check_image_boundaries():
    return '''
Detect image boundaries along the edges of the chart and determine the presence of axes on each side.
Check for:
1. Top border (potential x-axis at the top)
2. Bottom border (potential x-axis at the bottom)
3. Left border (potential y-axis on the left)
4. Right border (potential y-axis on the right)

Return a JSON object with:
- "has_top_border": true if a top boundary is detected, false otherwise.
- "has_bottom_border": true if a bottom boundary is detected, false otherwise.
- "has_left_border": true if a left boundary is detected, false otherwise.
- "has_right_border": true if a right boundary is detected, false otherwise.

**Output Specification:**
Strict JSON format with border presence, and Do not return other non-JSON processes or other data:
```json
{"has_top_border": boolean, "has_bottom_border": boolean, "has_left_border": boolean, "has_right_border": boolean}
'''

def resolve_border_discrepancy(borders, counts):
    return f'''
A discrepancy has been detected between the identified image borders and the computed axis counts. Analyze both data sources to determine the correct number of horizontal and vertical borders.

**Input Data:**
- `borders`: A dictionary indicating whether each side of the image has a detected border. The values are {borders}.
  - `has_top_border`: Indicates if there is a top boundary.
  - `has_bottom_border`: Indicates if there is a bottom boundary.
  - `has_left_border`: Indicates if there is a left boundary.
  - `has_right_border`: Indicates if there is a right boundary.

- `counts`: A dictionary representing the detected x- and y-axis counts: {counts}.
  - `x_axes_count`: The number of detected horizontal axes.
  - `y_axes_count`: The number of detected vertical axes.

**Issue Detected:**
The computed border presence does not match the detected axis counts.

**Resolution Steps:**
1. Re-evaluate the horizontal and vertical border counts:
   - Horizontal border count (`computed_x_borders`): Sum of `has_top_border` and `has_bottom_border`.
   - Vertical border count (`computed_y_borders`): Sum of `has_left_border` and `has_right_border`.

2. Compare `computed_x_borders` with `x_axes_count` and `computed_y_borders` with `y_axes_count`:
   - If `computed_x_borders` < `x_axes_count` or `computed_y_borders` < `y_axes_count`, investigate:
     - Are additional axes present that do not align with the detected borders?
     - Are any faint or broken borders missing in detection?
     - If axes are confirmed but borders are missing, update the border count accordingly.
   - If `computed_x_borders` > `x_axes_count` or `computed_y_borders` > `y_axes_count`, verify:
     - Could false-positive borders have been detected?
     - Are overlapping graphical elements misinterpreted as borders?

3. Determine the final corrected border counts based on verified data, ensuring consistency.

**Output Specification:**
Do not return other non-JSON processes or other data.
Strict JSON format with the resolved border counts, and Do not return other non-JSON processes or other data:
```json
{{
  "h": final corrected number of horizontal borders,
  "v": final corrected number of vertical borders
}}
'''

def get_axis_candidates(candidates, orientation, height, width):
    return f'''
**Axis Candidate Selection Protocol**

- **Input Parameters:**
  1. `candidates`: A list of arrays, each representing a vertical or horizontal line segment with coordinates [x1, y1, x2, y2]. The value of candidates is {candidates}.
  2. `orientation`: A string indicating the orientation of the lines ("vertical" or "horizontal"). The value of orientation is {orientation}.
  3. `image_height`: The height of the image in pixels. The value of height is {height}.
  4. `image_width`: The width of the image in pixels. The value of width is {width}.

- **Initial Evaluation:**
  1. Count the number of valid candidates matching the given `orientation`.
  2. Selecting the top two candidates based on the rules below.

- **Selection Rules:**
  1. Analyze the provided `candidates`:{candidates} based on their coordinates and the specified `orientation`.
  2. Identify the most probable candidate lines for the {orientation} axis:
     - For "vertical" orientation:
       - Prioritize lines near the left or right edges of the image.
       - Avoid lines extending beyond the expected axis range, such as those interfering with axis labels or ticks.
     - For "horizontal" orientation:
       - Prioritize lines near the top or bottom edges of the image.
       - Exclude lines extending beyond the expected axis width.
  3. Exclude lines that are too short or do not align with typical axis characteristics.
  4. If selecting multiple axes, ensure a minimum separation between them:
     - For "vertical" orientation, ensure the x-coordinates of the selected lines differ by at least `10` pixels.
     - For "horizontal" orientation, ensure the y-coordinates of the selected lines differ by at least `10` pixels.
  5. Select the top two candidates best aligned with the image edges.

- **Output Specification:**
  Strict JSON format with selected axis candidates:
    ```json
    {{
      "primary_axis": [x1, y1, x2, y2],
      "secondary_axis": [x1, y1, x2, y2]
    }}
    ```
'''

def get_axis_verification_prompt(x_color_name, y_color_name):
    """
    生成一个用于验证坐标轴的提示。
    """
    return f"""
    As an expert in chart analysis, please examine the image provided.
    I have highlighted a {x_color_name} line, which I believe is the bottom X-axis (horizontal axis), 
    and a {y_color_name} line, which I believe is the left Y-axis (vertical axis).

    Please verify if these identifications are correct. Your response must be in a strict JSON format as follows:
    {{
      "is_x_axis_correct": boolean,
      "is_y_axis_correct": boolean
    }}

    - Set "is_x_axis_correct" to true if the {x_color_name} line accurately represents the main horizontal axis of the chart data area. Otherwise, set it to false.
    - Set "is_y_axis_correct" to true if the {y_color_name} line accurately represents the main vertical axis of the chart data area. Otherwise, set it to false.
    """

def check_axis_tick_presence(axis):
    return f'''
**Axis Tick Presence Verification Protocol**

- **Input Parameter:**
  1. `axis`: A string indicating the target coordinate axis ("x" or "y"). The value is {axis}.

- **Detection Rules:**
  1. Analyze the entire graph image to locate candidate tick marks associated with the specified {axis}-axis.
  2. Identify if tick marks (numerical labels or tick lines) are present along the {axis}-axis.
  3. Evaluate the consistency and spatial distribution of detected features to confirm valid tick marks.
  4. If valid tick marks are detected, conclude that the axis has corresponding tick values; otherwise, conclude it does not.

- **Output Specification:**
  Strict JSON format returning a boolean value:
  ```json
  {{
    "has_tick_values": true_or_false
  }}
'''

def get_total_lines_count():
    return '''
**Total Lines Count Detection Protocol**

- **Detection Rules:**
  1. Analyze the chart image to detect all distinct lines.
  2. First, check for the presence of a legend:
     - If a legend is detected, extract the number of unique entries as an estimate for the number of lines.
     - Verify that each legend entry corresponds to a distinct line in the chart.
  3. If no legend is present, directly detect and count all distinct lines based on their trajectories and visual separation.
  4. Exclude non-data lines such as chart borders, gridlines, axis markers, and baseline(s).
  5. Ensure that overlapping or nearly identical lines are counted separately only if they are visually distinct.

- **Output Specification:**
  Strict JSON format with the total number of lines:
  ```json
  { "total_lines_num": integer }
'''

def get_axis_with_params(edge_params, hough_params, candidates, height, width):
    """
    根据边缘检测和霍夫变换参数，判断坐标轴位置。
    """
    return f'''
**Axis Detection with Detection Parameters**

- **Input Parameters:**
  1. `edge_params`: Edge detection parameters - thresholds list: {edge_params}
  2. `hough_params`: Hough line detection parameters: {hough_params}
  3. `candidates`: Detected line candidates: {candidates}
  4. `image_height`: {height}px
  5. `image_width`: {width}px

- **Detection Rules:**
  1. Analyze the chart image and the detected line candidates using the provided parameters.
  2. Based on the edge detection thresholds and Hough transform parameters, identify the primary X-axis and Y-axis.
  3. For X-axis (horizontal): Select the bottommost horizontal line that represents the main data axis.
  4. For Y-axis (vertical): Select the leftmost vertical line that represents the main data axis.
  5. Exclude border lines and grid lines.

- **Output Specification:**
  Strict JSON format:
  ```json
  {{
    "primary_axis": [x1, y1, x2, y2],
    "secondary_axis": [x1, y1, x2, y2]
  }}
  ```
  Note: For horizontal orientation, return horizontal axis candidates. For vertical orientation, return vertical axis candidates.
'''

def verify_axis_alignment(x_axis, y_axis):
    """
    验证绘制的坐标轴是否与图片上的坐标轴重合。
    """
    return f'''
**Axis Alignment Verification Protocol**

- **Input Parameters:**
  1. Detected X-axis coordinates: {x_axis}
  2. Detected Y-axis coordinates: {y_axis}

- **Detection Rules:**
  1. Examine the image carefully to locate the actual coordinate axes in the chart.
  2. Compare the drawn colored lines (X-axis and Y-axis) with the actual axes in the image.
  3. Check if the drawn lines accurately overlap with the real coordinate axes.
  4. Consider small deviations (within 5-10 pixels) as acceptable alignment.
  5. Verify that:
     - The X-axis line aligns with the bottom horizontal axis of the data area
     - The Y-axis line aligns with the left vertical axis of the data area

- **Output Specification:**
  Strict JSON format:
  ```json
  {{
    "is_correct": boolean,
    "x_axis_aligned": boolean,
    "y_axis_aligned": boolean
  }}
  ```
  Set "is_correct" to true only if BOTH axes are correctly aligned. Otherwise, set to false.
'''

def optimize_detection_params(current_edge_params, current_hough_params, current_preprocess_params, verification_result):
    """
    Generate optimized detection parameters based on verification results.
    CRITICAL: Must return edge_params, hough_params, and preprocess_params.
    """
    return f'''
**CRITICAL: Detection Parameter Optimization Protocol**

You are an expert computer vision engineer. The axis detection failed, and you MUST provide optimized parameters to fix this issue.

**Current Situation:**
- Current Canny edge detection parameters: {current_edge_params}
- Current HoughLinesP parameters: {current_hough_params}
- Current preprocessing parameters: {current_preprocess_params}
- Verification result: {verification_result}

**Why Parameters Are Critical:**
The success of axis detection DEPENDS ENTIRELY on these parameters:

1. **Preprocessing Parameters** (`preprocess_params`): Control image enhancement quality
   - `gaussian_blur_kernel`: Kernel size for Gaussian blur (must be odd: 3, 5, 7, 9, etc.)
     * Larger kernels = more smoothing, less noise but less detail
     * Smaller kernels = less smoothing, more detail but more noise
   - `clahe_tile_grid_size`: Grid size for CLAHE contrast enhancement [width, height]
     * Larger tiles = more global contrast, less local detail
     * Smaller tiles = more local contrast enhancement

2. **Canny Edge Detection Parameters** (`edge_params`): Control which edges are detected
   - Lower thresholds = detect more edges (including noise)
   - Higher thresholds = detect fewer edges (may miss important lines)
   - Format: [threshold1, threshold2] where threshold1 < threshold2

3. **HoughLinesP Parameters** (`hough_params`): Control how lines are detected from edges
   - `threshold`: Minimum votes for a line (lower = more lines, higher = fewer lines)
   - `minLineLength`: Minimum line length in pixels (lower = detect shorter lines)
   - `maxLineGap`: Maximum gap between line segments to connect (higher = connect more fragmented lines)
   - Format: List of dictionaries with all HoughLinesP parameters

**MANDATORY Optimization Rules:**
Based on the verification result, you MUST adjust parameters:

1. **If axes are NOT correctly aligned** (is_correct: false):
   - ANALYZE why: too few lines? too many false positives? lines too fragmented? poor image quality?
   
   - ADJUST Preprocessing parameters:
     * If image is noisy: Increase gaussian_blur_kernel (e.g., 5→7 or 7→9)
     * If contrast is poor: Decrease clahe_tile_grid_size (e.g., [16,16]→[8,8])
     * If image is over-smoothed: Decrease gaussian_blur_kernel (e.g., 7→5 or 5→3)
   
   - ADJUST Canny thresholds: 
     * Lower thresholds to detect more edges if too few lines detected
     * Raise thresholds to reduce noise if too many false positives
   
   - ADJUST Hough parameters: 
     * Lower threshold to detect more lines
     * Increase maxLineGap to connect fragmented lines
     * Decrease minLineLength to detect shorter line segments

2. **Parameter Ranges (MUST follow):**
   - Gaussian blur kernel: Odd integers in [3, 11] (e.g., 3, 5, 7, 9, 11)
   - CLAHE tile grid size: Tuples of integers in [(4,4), (8,8), (16,16), (32,32)]
   - Canny thresholds: threshold1 in [1-50], threshold2 in [30-200], threshold1 < threshold2
   - Hough threshold: [5-150]
   - minLineLength: [30-300] pixels
   - maxLineGap: [1-100] pixels

**CRITICAL OUTPUT REQUIREMENT:**
You MUST return `preprocess_params`, `edge_params`, and `hough_params` in the exact JSON format below.
DO NOT return partial results. DO NOT skip parameters.

**Strict JSON Output Format:**
```json
{{
  "preprocess_params": {{
    "gaussian_blur_kernel": [kernel_size1, kernel_size2],
    "clahe_tile_grid_size": [width, height]
  }},
  "edge_params": [threshold1, threshold2],
  "hough_params": [
    {{"rho": 1, "theta": 0.0174533, "threshold": value1, "minLineLength": value2, "maxLineGap": value3}},
    {{"rho": 1, "theta": 0.0174533, "threshold": value4, "minLineLength": value5, "maxLineGap": value6}}
  ]
}}
```

**Example Valid Response:**
```json
{{
  "preprocess_params": {{
    "gaussian_blur_kernel": [5, 7],
    "clahe_tile_grid_size": [8, 8]
  }},
  "edge_params": [15, 80],
  "hough_params": [
    {{"rho": 1, "theta": 0.0174533, "threshold": 25, "minLineLength": 80, "maxLineGap": 50}},
    {{"rho": 1, "theta": 0.0174533, "threshold": 35, "minLineLength": 100, "maxLineGap": 40}}
  ]
}}
```

**REMEMBER:** You MUST provide preprocess_params, edge_params, and hough_params. The detection system will fail without these parameters.
'''

def compare_axis_methods(traditional_x, traditional_y, unet_x, unet_y):

    return f'''
**Axis Detection Method Comparison Protocol**

- **Input Parameters:**
  1. Traditional CV method X-axis: {traditional_x}
  2. Traditional CV method Y-axis: {traditional_y}
  3. UNet method X-axis: {unet_x}
  4. UNet method Y-axis: {unet_y}

- **Detection Rules:**
  1. Examine the image with both sets of axes drawn in different colors.
  2. Compare each method's results with the actual coordinate axes visible in the image.
  3. For X-axis: Determine which method's horizontal line better aligns with the bottom axis of the chart.
  4. For Y-axis: Determine which method's vertical line better aligns with the left axis of the chart.
  5. Consider factors such as:
     - Alignment accuracy with actual axes
     - Presence of axis labels/ticks near the detected lines
     - Correct positioning relative to the data area

- **Output Specification:**
  Strict JSON format:
  ```json
  {{
    "x_axis_method": "traditional" or "unet",
    "y_axis_method": "traditional" or "unet",
    "confidence": float (0.0 to 1.0)
  }}
  ```
  Return which method produces the correct axis for X and Y respectively.
'''