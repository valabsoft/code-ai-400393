import os
import cv2
from mrcv import AugmentationMethod, BatchAugmentationConfig, augmentation, batch_augmentation

input_images = []
for i in range(10):
    img_path = os.path.join("files", f"img{i}.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
    input_images.append(img)

# Create a copy of input images
input_images_copy = [img.copy() for img in input_images]

# Define augmentation methods
augmentation_methods = [
    AugmentationMethod.ROTATE_IMAGE_90,
    AugmentationMethod.FLIP_HORIZONTAL,
    AugmentationMethod.FLIP_VERTICAL,
    AugmentationMethod.ROTATE_IMAGE_45,
    AugmentationMethod.ROTATE_IMAGE_315,
    AugmentationMethod.ROTATE_IMAGE_270,
    AugmentationMethod.FLIP_HORIZONTAL_AND_VERTICAL,
]

# Perform augmentation
state, output_images = augmentation(input_images, augmentation_methods)
if state != 0:
    print(f"Error: Augmentation failed with code {state}")

# Configure batch augmentation
config = BatchAugmentationConfig()
config.keep_original = True
config.total_output_count = 100
config.random_seed = 42
config.method_weights = {
    AugmentationMethod.FLIP_HORIZONTAL: 0.2,
    AugmentationMethod.ROTATE_IMAGE_90: 0.2,
    AugmentationMethod.BRIGHTNESS_CONTRAST_ADJUST: 0.3,
    AugmentationMethod.PERSPECTIVE_WARP: 0.2,
    AugmentationMethod.COLOR_JITTER: 0.1,
}

# Perform batch augmentation
state, batch_output = batch_augmentation(
    input_images_copy,
    config,
    os.path.join("files", "batch_output")
)
if state != 0:
    print(f"Error: Batch augmentation failed with code: {state}")
