import cv2
import os
import numpy as np
from enum import Enum
from mrcv import disparity_map

# Определение перечисления для типов диспаратности
class DisparityType(Enum):
    BASIC_DISPARITY = 0
    BASIC_HEATMAP = 1
    FILTERED_DISPARITY = 2
    FILTERED_HEATMAP = 3
    ALL = 4
# Загрузка тестовых изображений
file_image_left = os.path.join("../../../examples/disparitymap/files", "example_left.jpg")
file_image_right = os.path.join("../../../examples/disparitymap/files", "example_right.jpg")

current_path = os.getcwd()
path_image_left = os.path.join(current_path, file_image_left)
path_image_right = os.path.join(current_path, file_image_right)

image_left = cv2.imread(path_image_left, cv2.IMREAD_COLOR)
image_right = cv2.imread(path_image_right, cv2.IMREAD_COLOR)

# Параметры функции
min_disparity = 16
num_disparities = 16 * 10
block_size = 15
lambda_val = 5000.0
sigma = 3
color_map = cv2.COLORMAP_TURBO
disparity_type = DisparityType.ALL

# Проверка загрузки изображений
if image_left is None or image_right is None:
    print("Ошибка: Не удалось загрузить одно или оба изображения")
    exit(1)

# Создание пустого массива для результата
disparity_map_output = np.zeros_like(image_left)

# Построение карты диспаратности
disparity_map(disparity_map_output, image_left, image_right, min_disparity, num_disparities,
              block_size, lambda_val, sigma, disparity_type, color_map, True, True)

# Отображение результата
cv2.namedWindow("MRCV Disparity Map", cv2.WINDOW_AUTOSIZE)
cv2.imshow("MRCV Disparity Map", disparity_map_output)

cv2.waitKey(0)
cv2.destroyAllWindows()