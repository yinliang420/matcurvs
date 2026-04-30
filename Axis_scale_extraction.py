import json,re
import cv2
from ocr_compat import create_ocr, ocr_predict
from sklearn.linear_model import LinearRegression
from PIL import Image
import io
from sklearn.pipeline import make_pipeline
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import FunctionTransformer
import base64
from ModalLLMResponse import OCRModalLLMResponse
from Axis_extraction import encode_image
from prompt import check_axis_tick_presence

# Relationship between pixels and scale values for reading curve coordinates later

class RobustAxisFitter:
    def __init__(self, max_trials=100, residual_threshold=5.0):
        self.max_trials = max_trials  # Maximum number of RANSAC iterations
        self.residual_threshold = residual_threshold  # Residual threshold (in pixels)

    def _linear_model(self):
        """Linear model RANSAC"""
        return RANSACRegressor(
            min_samples=2,
            max_trials=self.max_trials,
            residual_threshold=self.residual_threshold,
            random_state=42
        )

    def _log_model(self):
        """Logarithmic model RANSAC (requires x > 0)"""
        log_transformer = FunctionTransformer(
            func=np.log, inverse_func=np.exp,
            check_inverse=False
        )
        return RANSACRegressor(
            estimator=make_pipeline(log_transformer, LinearRegression()),
            min_samples=2,
            max_trials=self.max_trials,
            residual_threshold=self.residual_threshold,
            random_state=42
        )

    def detect_outliers(self, pixels, values):
        """Robust outlier detection"""
        # Convert input format
        X = pixels.reshape(-1, 1)
        y = values

        # Try linear model
        linear_ransac = self._linear_model()
        try:
            linear_ransac.fit(X, y)
            linear_inliers = linear_ransac.inlier_mask_
        except:
            linear_inliers = np.zeros_like(y, dtype=bool)

        # Try logarithmic model (requires all values > 0)
        log_inliers = np.zeros_like(y, dtype=bool)
        if np.all(y > 0):
            try:
                log_ransac = self._log_model()
                log_ransac.fit(X, y)
                log_inliers = log_ransac.inlier_mask_
            except:
                pass

        # Select best model (most inliers)
        linear_score = sum(linear_inliers)
        log_score = sum(log_inliers)

        if max(linear_score, log_score) == 0:
            return np.arange(len(y))  # All points are outliers

        if linear_score >= log_score:
            best_inliers = linear_inliers
            model_type = 'linear'
        else:
            best_inliers = log_inliers
            model_type = 'log'

        # Record model information
        self.model_type = model_type
        self.inlier_mask = best_inliers

        return np.where(~best_inliers)[0]


class AxisValidator:
    def __init__(self):
        self.ocr = create_ocr(use_angle_cls=True, lang='en')
        self.fitter = RobustAxisFitter()

    def _is_numeric(self, text):
        """Primary validation: check if text is a valid number"""
        try:
            float(text)
            return True
        except:
            return False

    def distance_filter(self,texts,axis_type,x1,y1,x_max_distance,y_max_distance):
        # Filter points that are too far from the axis line
        valid_ticks = []
        for value, (x, y) in texts:
            # Calculate shortest distance to the axis line
            if axis_type == 'x':
                distance = abs(y - y1)
                if distance < y_max_distance:
                    valid_ticks.append((value, (x, y)))
            else:
                distance = abs(x - x1)
                if distance < x_max_distance:
                    valid_ticks.append((value, (x, y)))

        return valid_ticks

    def _multimodal_verify(self, roi, axis_type, response_modal_llm: OCRModalLLMResponse):
        """Multimodal verification"""
        _, buffer = cv2.imencode('.jpg', roi)
        image_base64= base64.b64encode(buffer).decode('utf-8')


        text = f"""The graph is part of an axis and I want you to accurately identify the scale labels of {axis_type} axis.
        Note:   The answer format is strictly JSON and requires no additional explanation. \n
                Example:{{"{axis_type}": [0,1,2,3]}}\n"""

        response = response_modal_llm.generate_response(image_base64, text)
        try:
            if response:
                print(response)
                # generate_response已经返回字典，无需再json.loads
            else:
                print("No valid JSON found in the output.")

            if axis_type =="x":
                label = response.get("x")
            else:
                label = response.get("y")

            return label
        except Exception as e:
            print("An error occurred:", e)
            return None

    def _remove_duplicates(self, ticks):
        """Remove duplicate values"""
        seen = set()
        return [t for t in ticks if not (t[0] in seen or seen.add(t[0]))]

    def validate_ticks(self, result, image, axis_line, offset_y, offset_x, roi, axis_type,
                       response_modal_llm: OCRModalLLMResponse, mm_verify=False):
        """Comprehensive validation process"""
        # Ensure coordinates are integers (YOLO may return floats)
        x1, y1, x2, y2 = [int(round(coord)) for coord in axis_line]
        h,w = image.shape[:2]
        y_max_distance = round(h / 15)
        x_max_distance = round(w / 15)
        # result is OCR result
        # Start validation
        # Step 1: Validate if numbers are floats
        texts = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                # print(line)
                text = line[1][0]
                if self._is_numeric(text):
                    value = float(text)
                    box = line[0]
                    # Convert to original image coordinates (critical correction)
                    if axis_type == 'x':
                        # ROI's X direction has full width, Y direction has offset
                        center_x = (box[0][0] + box[2][0]) / 2  # top-left, top-right, bottom-right, bottom-left
                        center_y = (box[0][1] + box[2][1]) / 2 + offset_y  # Use actual offset
                    else:
                        # ROI's Y direction has full height, X direction has offset
                        center_x = (box[0][0] + box[2][0]) / 2 + offset_x
                        center_y = (box[0][1] + box[2][1]) / 2
                    texts.append((value, (center_x, center_y)))
        # Step 2: Distance validation, remove numbers too far from the axis
        print(texts)
        primary_ticks = self.distance_filter(texts,axis_type,x1,y1,x_max_distance,y_max_distance)
        print(primary_ticks)
        # Step 3: Fitting validation
        if len(primary_ticks) < 3:  # At least 2 points needed for fitting
            return primary_ticks

        # Extract pixel positions and values
        pixels = np.array([t[1][0] if axis_type=='x' else t[1][1] for t in primary_ticks])
        values = np.array([t[0] for t in primary_ticks])
        # Sort processing
        sort_idx = np.argsort(pixels)
        sorted_pixels = pixels[sort_idx]
        sorted_values = values[sort_idx]
        # RANSAC outlier detection
        outlier_indices = self.fitter.detect_outliers(sorted_pixels, sorted_values)
        valid_mask = np.ones(len(sorted_pixels), dtype=bool)
        valid_mask[outlier_indices] = False

        # Separate normal and suspect points
        valid_ticks = [primary_ticks[sort_idx[i]] for i in np.where(valid_mask)[0]]
        suspect_ticks = [primary_ticks[sort_idx[i]] for i in outlier_indices]

        # Multimodal verification of suspect points
        final_ticks = valid_ticks.copy()
        # Multimodal outlier identification: draw bounding boxes, analyze, use multiple models for voting
        print(final_ticks)
        print(suspect_ticks)
        if suspect_ticks and mm_verify:
            mm_llm_result = self._multimodal_verify(roi, axis_type, response_modal_llm)
            print("Multimodal recognition")
            print(mm_llm_result)

        return self._remove_duplicates(final_ticks)


class AxisCalibrator:
    def __init__(self):
        self.ocr = create_ocr(use_angle_cls=True, lang='en')
        self.validator = AxisValidator()

    def _linear_func(self, x, a, b):
        return a * x + b

    def _log_func(self, x, a, b):
        return a * np.log(x) + b

    def generate_equidistant_ticks(self, axis_line, axis_type):
        """Generate equidistant reference coordinate system"""
        # Select longest axis line segment
        # line = max(axis_line, key=lambda l: np.hypot(l[2] - l[0], l[3] - l[1]))
        # Ensure coordinates are integers (YOLO may return floats)
        x1, y1, x2, y2 = [int(round(coord)) for coord in axis_line]  # Parse start and end points of axis line

        # Generate pixel coordinates
        if axis_type == 'y':
            # Y axis: pixel values increase from top to bottom
            start, end = min(y1, y2), max(y1, y2)
            pixel_coords = np.linspace(start, end, 6).astype(int)
            pixel_coords = [(x1, y) for y in pixel_coords]  # Direct mapping, no inversion needed

            # Value mapping: use Y axis pixel values as numeric values
            values = np.linspace(end, start, 6)  # Directly use y values
            # print(list(zip(values, pixel_coords)))
        else:
            # X axis: pixel values increase from left to right
            start, end = min(x1, x2), max(x1, x2)
            pixel_coords = np.linspace(start, end, 6).astype(int)
            pixel_coords = [(x, y1) for x in pixel_coords]

            # Value mapping: use X axis pixel values as numeric values
            values = np.linspace(start, end, 6)  # Directly use x values

        return list(zip(values, pixel_coords))

    def safe_crop(self, image, center, axis='x'):
        """
        Safely crop ROI region and return actual offset
        :param center: Axis center point coordinates (x, y)
        :param axis: 'x' for horizontal axis (expand vertically), 'y' for vertical axis (expand horizontally)
        :return: (roi, offset_x, offset_y)
        """
        h, w = image.shape[:2]
        x, y = center
        
        # Ensure coordinates are integers (YOLO may return floats)
        x = int(round(x))
        y = int(round(y))

        y_max_distance = round(h / 10)
        x_max_distance = round(w / 10)
        if axis == 'x':
            # Expand vertically
            y_start = int(max(0, y - y_max_distance))
            y_end = int(min(h, y + y_max_distance))
            roi = image[y_start:y_end, :]
            return roi, 0, y_start  # offset_x, offset_y
        else:
            # Expand horizontally
            x_start = int(max(0, x - x_max_distance))
            x_end = int(min(w, x + x_max_distance))
            roi = image[:, x_start:x_end]
            return roi, x_start, 0

    def extract_axis_ticks(self, image, axis_line, axis_type, response_modal_llm: OCRModalLLMResponse, base_image, max_distance=20):
        """
        Extract axis tick labels
        :param base_image:
        :param response_modal_llm:
        :param image: Original image (BGR format)
        :param axis_line: Axis line segment (x1,y1,x2,y2)
        :param axis_type: 'x' or 'y'
        :param max_distance: Maximum allowed distance from label to axis line
        """

        # Get axis line midpoint as ROI center
        # Ensure coordinates are integers (YOLO may return floats)
        x1, y1, x2, y2 = [int(round(coord)) for coord in axis_line]
        center = ((x1 + x2) // 2, (y1 + y2) // 2) if axis_type == 'x' else (x1, y1)
        # Safely crop ROI and get actual offset
        roi, offset_x, offset_y = self.safe_crop(image, center, axis_type)

        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        io.BytesIO()
        result = ocr_predict(self.ocr, roi)
        
        # 检查OCR结果是否为None或空
        if result is None or len(result) == 0 or result[0] is None:
            print(f"⚠️ {axis_type.upper()} axis: OCR returned no results, generating reference coordinate system...")
            return self.generate_equidistant_ticks(axis_line, axis_type)

        valid_ticks = self.validator.validate_ticks(result, image, axis_line, offset_y, offset_x, roi, axis_type, response_modal_llm)
        ####  Here we need multimodal to check again if this axis has no ticks, return a bool for if condition
        # have_ticks = response_modal_llm.generate_response(base_image, check_axis_tick_presence(axis_type)).get("has_tick_values")
        # print(have_ticks)
        if len(valid_ticks) < 2:
            print(f"⚠️ {axis_type.upper()} axis: no ticks detected, generating reference coordinate system...")
            valid_ticks = self.generate_equidistant_ticks(axis_line, axis_type)

        return valid_ticks

    def calibrate_axis(self, axis_line, ticks, axis_type):
        pixels, values = [], []
        for val, (x, y) in ticks:
            pixels.append(x if axis_type == 'x' else y)
            values.append(val)

        if axis_type == 'y':
            pixels = pixels[::-1]  # **Ensure Y axis direction is correct**
            values = values[::-1]

        pixels, values = np.array(pixels).ravel(), np.array(values).ravel()

        # **Prevent np.interp length error**
        if len(pixels) < 2 or len(values) < 2 or len(pixels) != len(values):
            print(f"⚠️ {axis_type.upper()} axis: insufficient data, using default mapping...")
            return lambda p: float(p), 'default'

        try:
            lin_model = LinearRegression()
            lin_model.fit(pixels.reshape(-1, 1), values)
            lin_score = lin_model.score(pixels.reshape(-1, 1), values)

            if np.all(values > 0):
                try:
                    log_popt, _ = curve_fit(self._log_func, pixels, values)
                    log_pred = self._log_func(pixels, *log_popt)
                    log_rmse = np.sqrt(np.mean((log_pred - values) ** 2))
                except:
                    log_rmse = np.inf  # **Ensure failure doesn't affect the process**

            else:
                log_rmse = np.inf

            if log_rmse < 0.1 and log_rmse < lin_score:
                def mapping_func(pixel):
                    return self._log_func(float(pixel), *log_popt)

                scale_type = 'log'
            else:
                def mapping_func(pixel):
                    return lin_model.predict([[float(pixel)]])[0]

                scale_type = 'linear'

        except:
            print(f"⚠️ {axis_type.upper()} axis: fitting failed, falling back to linear interpolation...")

            def mapping_func(pixel):
                if len(pixels) < 2:
                    return float(pixel)
                return np.interp(float(pixel), pixels, values)

            scale_type = 'linear'

        return mapping_func, scale_type

def axis_pixel_mapping(image_path, x_axis, y_axis, response_modal_llm: OCRModalLLMResponse, Visualisable=False):
    # Assume x_axis and y_axis are obtained from previous steps
    # x_axis, y_axis = [96,392,523,392],[95,391,95,46]

    original = cv2.imread(image_path)
    base_image = encode_image(image_path)
    calibrator = AxisCalibrator()
    # Process X axis
    print("Starting X axis extraction")
    x_axis_line = x_axis  # (x1,y1,x2,y2)
    x_ticks = calibrator.extract_axis_ticks(original, x_axis_line, 'x', response_modal_llm, base_image)
    x_mapping, x_scale_type = calibrator.calibrate_axis(x_axis_line, x_ticks, 'x')

    # Process Y axis
    print("Starting Y axis extraction")
    y_axis_line = y_axis
    y_ticks = calibrator.extract_axis_ticks(original, y_axis_line, 'y', response_modal_llm, base_image)
    y_mapping, y_scale_type = calibrator.calibrate_axis(y_axis_line, y_ticks, 'y')

    return x_mapping, y_mapping


# from Axis_extraction import detect_axes, encode_image
# # image_path = "C:\\Users\Yin Liang\Desktop\split_images\A_Monolayer_High_Entropy_Layered_Hydroxide_Frame_for_Efficient_Oxygen_Evolution_Reaction\\figure4_e.png"
# image_path = "C:\\Users\Yin Liang\Desktop\split_images\\figure1.png"
# responsed = OCRModalLLMResponse()
#
# x, y = detect_axes(image_path, responsed, visualise=True)
# x_mapping, y_mapping = axis_pixel_mapping(image_path, x, y, responsed, True)
# print(x_mapping, y_mapping)
