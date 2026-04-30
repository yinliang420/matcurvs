import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import numpy as np
from pathlib import Path
import cv2

# --- 1. Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256  # Must match the training dimensions
IMAGE_WIDTH = 256   # Must match the training dimensions

# --- Path to your trained model checkpoint ---
try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()
CHECKPOINT_PATH = script_dir / "Unet++" / "axis_lins_detect" /  "unet_plusplus_axes_checkpoint.pth.tar"


# --- 2. U-Net++ Model Definition ---
# This class must be identical to the one used for training to load the weights correctly.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNET_plusplus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET_plusplus, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = DoubleConv(in_channels, features[0])
        self.conv1_0 = DoubleConv(features[0], features[1])
        self.conv2_0 = DoubleConv(features[1], features[2])
        self.conv3_0 = DoubleConv(features[2], features[3])
        self.conv4_0 = DoubleConv(features[3], features[3] * 2)
        self.conv0_1 = DoubleConv(features[0] + features[1], features[0])
        self.conv1_1 = DoubleConv(features[1] + features[2], features[1])
        self.conv2_1 = DoubleConv(features[2] + features[3], features[2])
        self.conv3_1 = DoubleConv(features[3] + features[3] * 2, features[3])
        self.conv0_2 = DoubleConv(features[0]*2 + features[1], features[0])
        self.conv1_2 = DoubleConv(features[1]*2 + features[2], features[1])
        self.conv2_2 = DoubleConv(features[2]*2 + features[3], features[2])
        self.conv0_3 = DoubleConv(features[0]*3 + features[1], features[0])
        self.conv1_3 = DoubleConv(features[1]*3 + features[2], features[1])
        self.conv0_4 = DoubleConv(features[0]*4 + features[1], features[0])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        output = self.final_conv(x0_4)
        return output

# --- 3. Core Detection Logic Function ---
# This function is kept separate for clarity and focus on the CV logic.
def _perform_detection_on_image(image_path, model, visualise=False):
    """
    Internal function that runs the full detection pipeline on a single image.
    """
    try:
        # Use a robust method to read the image, handling special characters
        image_bytes = np.fromfile(image_path, dtype=np.uint8)
        original_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if original_image is None:
            raise IOError("Could not decode image.")
        
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None

    # Preprocessing must match the validation transform from training
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    with torch.no_grad():
        transformed = transform(image=image_rgb)
        x = transformed["image"].to(DEVICE).unsqueeze(0)
        
        # Get model prediction
        preds = torch.sigmoid(model(x))
        preds = (preds > 0.5).float()
        
        # Post-process the mask
        prediction_mask = preds.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        prediction_mask_resized = cv2.resize(
            prediction_mask,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Find contours (lines) in the resized mask
    contours, _ = cv2.findContours(prediction_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_axes = []
    vertical_axes = []

    # Filter contours to find plausible axes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out very small noise
        if w < 20 and h < 20:
            continue
        
        # Classify as horizontal or vertical based on aspect ratio
        if w > h * 2: # Likely a horizontal line
            horizontal_axes.append((x, y, w, h))
        elif h > w * 2: # Likely a vertical line
            vertical_axes.append((x, y, w, h))

    x_axis, y_axis = None, None

    # Find the bottom-most horizontal axis
    if horizontal_axes:
        bottom_most_axis = max(horizontal_axes, key=lambda item: item[1])
        x, y, w, h = bottom_most_axis
        x_axis = (x, y + h // 2, x + w, y + h // 2)

    # Find the left-most vertical axis
    if vertical_axes:
        left_most_axis = min(vertical_axes, key=lambda item: item[0])
        x, y, w, h = left_most_axis
        y_axis = (x + w // 2, y, x + w // 2, y + h)

    if visualise:
        final_image = original_image.copy()
        if x_axis:
            cv2.line(final_image, (x_axis[0], x_axis[1]), (x_axis[2], x_axis[3]), (0, 0, 255), 3) # Red X-axis
        if y_axis:
            cv2.line(final_image, (y_axis[0], y_axis[1]), (y_axis[2], y_axis[3]), (0, 255, 0), 3) # Green Y-axis
        
        cv2.imshow("Original Image", original_image)
        cv2.imshow("Prediction Mask", prediction_mask_resized * 255)
        cv2.imshow("Final Detected Axes", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x_axis, y_axis

# --- 4. Callable Class for Easy Integration ---
class AxisDetector:
    """
    A convenient wrapper class to load the U-Net++ model once and perform
    axis detection on multiple images.
    """
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        """
        Initializes the detector and loads the trained model.
        
        Args:
            checkpoint_path (str or Path): Path to the .pth.tar model checkpoint file.
        """
        self.model = None
        if not Path(checkpoint_path).exists():
            print(f"Error: Model checkpoint not found at {checkpoint_path}")
            print("The detector will not be able to make predictions.")
        else:
            try:
                self.model = UNET_plusplus(in_channels=3, out_channels=1).to(DEVICE)
                print(f"Loading model from {checkpoint_path}...")
                checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.eval()
                print("Model loaded successfully and is ready for prediction.")
            except Exception as e:
                print(f"An error occurred while loading the model: {e}")
                self.model = None

    def detect(self, image_path, visualise=False):
        """
        Detects the primary X and Y axes in a single chart image.

        Args:
            image_path (str): The file path to the chart image.
            visualise (bool, optional): If True, displays intermediate and
                final results. Defaults to False.

        Returns:
            tuple: A tuple `(x_axis, y_axis)` where each element is either a tuple
                of coordinates `(x1, y1, x2, y2)` or `None`.
        """
        if self.model is None:
            print("Error: Model is not loaded. Cannot perform detection.")
            return None, None
            
        return _perform_detection_on_image(image_path, self.model, visualise)


### Example usage:
# # 导入您刚刚创建的类
# from axis_detector import AxisDetector

# # 1. 在您的程序开始时，初始化一次检测器
# #    （模型会在这里被加载到GPU）
# detector = AxisDetector()

# # 2. 准备一个图片路径列表
# image_paths = ["path/to/image1.png", "path/to/image2.jpg"]

# # 3. 循环调用 detect 方法处理每一张图片
# for path in image_paths:
#     if detector.model: # 最好检查一下模型是否加载成功
#         x_axis, y_axis = detector.detect(path, visualise=False)
#         print(f"Results for {path}:")
#         print(f"  X-Axis: {x_axis}")
#         print(f"  Y-Axis: {y_axis}")