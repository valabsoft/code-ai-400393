import cv2
import numpy as np
import os
import random
from enum import Enum
from typing import List, Tuple, Dict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Define augmentation methods enum
class AugmentationMethod(Enum):
    NONE = 0
    FLIP_HORIZONTAL = 1
    FLIP_VERTICAL = 2
    FLIP_HORIZONTAL_AND_VERTICAL = 3
    ROTATE_IMAGE_90 = 4
    ROTATE_IMAGE_45 = 5
    ROTATE_IMAGE_270 = 6
    ROTATE_IMAGE_315 = 7
    BRIGHTNESS_CONTRAST_ADJUST = 8
    GAUSSIAN_NOISE = 9
    COLOR_JITTER = 10
    GAUSSIAN_BLUR = 11
    RANDOM_CROP = 12
    PERSPECTIVE_WARP = 13
    TEST = 14


class BatchAugmentationConfig:
    def __init__(self):
        self.keep_original = False
        self.total_output_count = 0
        self.random_seed = 0
        self.method_weights = {}  # Dict[AugmentationMethod, float]


def rotate_image(image_input: np.ndarray, angle: float) -> Tuple[int, np.ndarray]:
    """
    Rotate an image by a specified angle.

    Args:
        image_input: Input image (numpy array)
        angle: Rotation angle in degrees

    Returns:
        Tuple of (status code, output image)
        Status: 0=Success, -1=Error
    """
    try:
        if image_input.size == 0:
            return 1, None

        height, width = image_input.shape[:2]
        center = (width / 2.0, height / 2.0)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image_output = cv2.warpAffine(image_input, rotation_matrix, (width, height))

        return 0, image_output
    except Exception as e:
        logger.error(f"Rotation error: {str(e)}")
        return -1, None


def flip_image(image_input: np.ndarray, flip_code: int) -> Tuple[int, np.ndarray]:
    """
    Flip an image horizontally, vertically, or both.

    Args:
        image_input: Input image
        flip_code: 0=vertical, 1=horizontal, -1=both

    Returns:
        Tuple of (status code, output image)
    """
    try:
        if image_input.size == 0:
            return 1, None

        image_output = cv2.flip(image_input, flip_code)
        return 0, image_output
    except Exception as e:
        logger.error(f"Flip error: {str(e)}")
        return -1, None


def adjust_brightness_contrast(image_input: np.ndarray, alpha: float, beta: float) -> Tuple[int, np.ndarray]:
    """
    Adjust image brightness and contrast.

    Args:
        image_input: Input image
        alpha: Contrast control (0.0-3.0)
        beta: Brightness control (-100 to 100)

    Returns:
        Tuple of (status code, output image)
    """
    try:
        if image_input.size == 0:
            return 1, None

        image_output = cv2.convertScaleAbs(image_input, alpha=alpha, beta=beta)
        return 0, image_output
    except Exception as e:
        logger.error(f"Brightness/contrast error: {str(e)}")
        return -1, None


def add_noise(image_input: np.ndarray, strength: float) -> Tuple[int, np.ndarray]:
    """
    Add Gaussian noise to image.

    Args:
        image_input: Input image
        strength: Noise strength

    Returns:
        Tuple of (status code, output image)
    """
    try:
        if image_input.size == 0:
            return 1, None

        noise = np.random.normal(0, strength * 255, image_input.shape).astype(np.float32)
        image_output = image_input.astype(np.float32) + noise
        image_output = cv2.normalize(image_output, None, 0, 255, cv2.NORM_MINMAX)
        image_output = image_output.astype(np.uint8)

        return 0, image_output
    except Exception as e:
        logger.error(f"Noise addition error: {str(e)}")
        return -1, None


def adjust_color_balance(image_input: np.ndarray, factors: List[float]) -> Tuple[int, np.ndarray]:
    """
    Adjust color balance of image.

    Args:
        image_input: Input image
        factors: RGB scaling factors (3 values)

    Returns:
        Tuple of (status code, output image)
    """
    try:
        if image_input.size == 0:
            return 1, None
        if len(factors) != 3:
            return -1, None

        channels = cv2.split(image_input)
        for i in range(3):
            channels[i] = cv2.convertScaleAbs(channels[i], alpha=factors[i])

        image_output = cv2.merge(channels)
        return 0, image_output
    except Exception as e:
        logger.error(f"Color balance error: {str(e)}")
        return -1, None


def apply_gaussian_blur(image_input: np.ndarray, kernel_size: int) -> Tuple[int, np.ndarray]:
    """
    Apply Gaussian blur to image.

    Args:
        image_input: Input image
        kernel_size: Kernel size (must be odd)

    Returns:
        Tuple of (status code, output image)
    """
    try:
        if image_input.size == 0:
            return 1, None

        if kernel_size % 2 == 0:
            kernel_size += 1

        image_output = cv2.GaussianBlur(image_input, (kernel_size, kernel_size), 0)
        return 0, image_output
    except Exception as e:
        logger.error(f"Gaussian blur error: {str(e)}")
        return -1, None


def random_crop(image_input: np.ndarray, crop_ratio: float) -> Tuple[int, np.ndarray]:
    """
    Randomly crop image.

    Args:
        image_input: Input image
        crop_ratio: Ratio of original size to crop (0.0-1.0)

    Returns:
        Tuple of (status code, output image)
    """
    try:
        if image_input.size == 0:
            return 1, None

        height, width = image_input.shape[:2]
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        x = random.randint(0, width - crop_width)
        y = random.randint(0, height - crop_height)

        image_output = image_input[y:y + crop_height, x:x + crop_width].copy()
        return 0, image_output
    except Exception as e:
        logger.error(f"Random crop error: {str(e)}")
        return -1, None


def perspective_transform(image_input: np.ndarray, strength: float) -> Tuple[int, np.ndarray]:
    """
    Apply perspective transformation to image.

    Args:
        image_input: Input image
        strength: Transformation strength

    Returns:
        Tuple of (status code, output image)
    """
    try:
        if image_input.size == 0:
            return 1, None

        height, width = image_input.shape[:2]

        src = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])

        dst = src.copy()
        for i in range(4):
            dst[i] += np.float32([
                random.uniform(-width * strength / 2, width * strength / 2),
                random.uniform(-height * strength / 2, height * strength / 2)
            ])

        M = cv2.getPerspectiveTransform(src, dst)
        image_output = cv2.warpPerspective(image_input, M, (width, height))

        return 0, image_output
    except Exception as e:
        logger.error(f"Perspective transform error: {str(e)}")
        return -1, None


def augmentation(input_images: List[np.ndarray],
                 augmentation_methods: List[AugmentationMethod]) -> Tuple[int, List[np.ndarray]]:
    """
    Perform augmentation on input images.

    Args:
        input_images: List of input images
        augmentation_methods: List of augmentation methods to apply

    Returns:
        Tuple of (status code, list of output images)
    """
    output_images = []
    saved_files_count = 0
    methods_used = set()

    try:
        output_folder = os.path.join("files", "augmented_images")
        os.makedirs(output_folder, exist_ok=True)

        random.seed(0)

        for method in augmentation_methods:
            for i, image in enumerate(input_images):
                if image.size == 0:
                    continue

                result_image = None
                method_name = ""
                status = 0

                if method == AugmentationMethod.FLIP_HORIZONTAL:
                    status, result_image = flip_image(image, 1)
                    method_name = "flipHorizontal"
                elif method == AugmentationMethod.FLIP_VERTICAL:
                    status, result_image = flip_image(image, 0)
                    method_name = "flipVertical"
                elif method == AugmentationMethod.FLIP_HORIZONTAL_AND_VERTICAL:
                    status, result_image = flip_image(image, -1)
                    method_name = "flipHorizontalandVertical"
                elif method == AugmentationMethod.ROTATE_IMAGE_90:
                    status, result_image = rotate_image(image, 90)
                    method_name = "rotate90"
                elif method == AugmentationMethod.ROTATE_IMAGE_45:
                    status, result_image = rotate_image(image, 45)
                    method_name = "rotate45"
                elif method == AugmentationMethod.ROTATE_IMAGE_315:
                    status, result_image = rotate_image(image, 315)
                    method_name = "rotate315"
                elif method == AugmentationMethod.ROTATE_IMAGE_270:
                    status, result_image = rotate_image(image, 270)
                    method_name = "rotate270"
                elif method == AugmentationMethod.BRIGHTNESS_CONTRAST_ADJUST:
                    alpha = 0.7 + 0.6 * random.random()
                    beta = -30 + 60 * random.random()
                    status, result_image = adjust_brightness_contrast(image, alpha, beta)
                    method_name = "brightnessContrast"
                elif method == AugmentationMethod.GAUSSIAN_NOISE:
                    status, result_image = add_noise(image, 0.05)
                    method_name = "gaussianNoise"
                elif method == AugmentationMethod.COLOR_JITTER:
                    factors = [0.7 + 0.6 * random.random() for _ in range(3)]
                    status, result_image = adjust_color_balance(image, factors)
                    method_name = "colorJitter"
                elif method == AugmentationMethod.GAUSSIAN_BLUR:
                    kernel_size = 3 + 2 * random.randint(0, 2)
                    status, result_image = apply_gaussian_blur(image, kernel_size)
                    method_name = "gaussianBlur"
                elif method == AugmentationMethod.RANDOM_CROP:
                    ratio = 0.7 + 0.2 * random.random()
                    status, result_image = random_crop(image, ratio)
                    method_name = "randomCrop"
                elif method == AugmentationMethod.PERSPECTIVE_WARP:
                    status, result_image = perspective_transform(image, 0.1)
                    method_name = "perspectiveWarp"
                else:
                    result_image = image.copy()
                    method_name = "none"

                if result_image is None or result_image.size == 0:
                    continue

                output_images.append(result_image)

                filename = os.path.join(output_folder, f"augmented_{i}_{method_name}.bmp")
                is_saved = cv2.imwrite(filename, result_image)
                if is_saved:
                    saved_files_count += 1
                    methods_used.add(method_name)

        methods_string = ", ".join(methods_used) if methods_used else "none"
        logger.info(
            f"Augmentation completed successfully. Methods used: {methods_string}. Files saved: {saved_files_count}")

        return 0, output_images
    except cv2.error as e:
        logger.error(f"Augmentation failed: {str(e)}")
        return -1, []
    except OSError as e:
        logger.error(f"Filesystem error: {str(e)}")
        return -1, []
    except Exception as e:
        logger.error(f"Unhandled exception during augmentation: {str(e)}")
        return -1, []


def batch_augmentation(inputs: List[np.ndarray],
                       config: BatchAugmentationConfig,
                       output_dir: str) -> Tuple[int, List[np.ndarray]]:
    """
    Perform batch augmentation on input images.

    Args:
        inputs: List of input images
        config: Batch augmentation configuration
        output_dir: Output directory for saving images

    Returns:
        Tuple of (status code, list of output images)
    """
    if not inputs:
        return 1, []

    outputs = []
    random.seed(config.random_seed)

    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if config.keep_original:
            for i, img in enumerate(inputs):
                if img.size == 0:
                    continue
                outputs.append(img.copy())
                if output_dir:
                    filename = os.path.join(output_dir, f"original_{len(outputs)}.png")
                    cv2.imwrite(filename, img)

        total_weight = sum(weight for _, weight in config.method_weights.items())
        if total_weight <= 0.0:
            return 3, []

        remaining = (config.total_output_count - len(outputs) if config.keep_original
                     else len(inputs) * len(config.method_weights))

        method_counts = {}
        for method, weight in config.method_weights.items():
            if weight < 0.0:
                return 2, []
            method_counts[method] = int((weight / total_weight) * remaining)

        successful_augmentations = 0
        method_stats = {augmentation_method_to_string(m): 0 for m in config.method_weights.keys()}

        for method, count in method_counts.items():
            method_name = augmentation_method_to_string(method)
            for _ in range(count):
                input_idx = random.randint(0, len(inputs) - 1)
                input_img = inputs[input_idx]
                if input_img.size == 0:
                    continue

                status, single_output = augmentation([input_img], [method])
                if status != 0 or not single_output:
                    continue

                outputs.append(single_output[0])
                method_stats[method_name] += 1
                successful_augmentations += 1

                if output_dir:
                    filename = os.path.join(output_dir, f"{method_name}_{len(outputs)}.png")
                    cv2.imwrite(filename, single_output[0])

        print(f"Batch augmentation completed. Total: {successful_augmentations} images.")
        for method, count in method_stats.items():
            percentage = (count * 100.0 / successful_augmentations) if successful_augmentations > 0 else 0
            print(f"  {method}: {count} images ({percentage:.1f}%)")

        return 0, outputs
    except Exception as e:
        logger.error(f"Batch augmentation error: {str(e)}")
        return -1, []


def augmentation_method_to_string(method: AugmentationMethod) -> str:
    """
    Convert augmentation method enum to string.

    Args:
        method: Augmentation method enum

    Returns:
        String representation of the method
    """
    method_map = {
        AugmentationMethod.NONE: "none",
        AugmentationMethod.FLIP_HORIZONTAL: "flip_h",
        AugmentationMethod.FLIP_VERTICAL: "flip_v",
        AugmentationMethod.FLIP_HORIZONTAL_AND_VERTICAL: "flip_both",
        AugmentationMethod.ROTATE_IMAGE_90: "rotate_90",
        AugmentationMethod.ROTATE_IMAGE_45: "rotate_45",
        AugmentationMethod.ROTATE_IMAGE_270: "rotate_270",
        AugmentationMethod.ROTATE_IMAGE_315: "rotate_315",
        AugmentationMethod.BRIGHTNESS_CONTRAST_ADJUST: "brightness",
        AugmentationMethod.GAUSSIAN_NOISE: "noise",
        AugmentationMethod.COLOR_JITTER: "color_jitter",
        AugmentationMethod.GAUSSIAN_BLUR: "blur",
        AugmentationMethod.RANDOM_CROP: "crop",
        AugmentationMethod.PERSPECTIVE_WARP: "perspective",
        AugmentationMethod.TEST: "test"
    }
    return method_map.get(method, "unknown")

