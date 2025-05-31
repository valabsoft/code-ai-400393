import cv2
import os
import numpy as np
from enum import Enum

class DisparityType(Enum):
    BASIC_DISPARITY = 0
    BASIC_HEATMAP = 1
    FILTERED_DISPARITY = 2
    FILTERED_HEATMAP = 3
    ALL = 4

def disparity_map(map_output, image_left, image_right, min_disparity, num_disparities,
                 block_size, lambda_val, sigma, disparity_type, color_map, save_to_file, show_images):
    # Преобразование в GrayScale
    image_left_gs = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gs = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    # Построение карты диспаратности (базовый метод)
    left_matcher = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    image_disparity_left = left_matcher.compute(image_left_gs, image_right_gs)
    image_disparity_right = right_matcher.compute(image_right_gs, image_left_gs)

    # Конвертация типов
    disparity_left = image_disparity_left.astype(np.float32)
    disparity_right = image_disparity_right.astype(np.float32)

    disparity_left = (disparity_left / 16.0 - min_disparity) / num_disparities
    disparity_right = (disparity_right / 16.0 - min_disparity) / num_disparities

    if show_images:
        cv2.namedWindow("Image Left", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Image Left", image_left)

        cv2.namedWindow("Image Right", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Image Right", image_right)

        cv2.namedWindow("Disparity", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Disparity", disparity_left)

    # Расцвечивание карты диспаратности
    disparity_tmp = cv2.convertScaleAbs(disparity_left, alpha=255.0)
    disparity = disparity_tmp.copy()

    heatmap = cv2.applyColorMap(disparity_tmp, color_map)
    if show_images:
        cv2.namedWindow("Heatmap", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Heatmap", heatmap)

    # Построение карты диспаратности (фильтрация)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(lambda_val)
    wls_filter.setSigmaColor(sigma)

    filtered_disparity_t = wls_filter.filter(image_disparity_left, image_left, None, image_disparity_right)
    filtered_disparity_t = filtered_disparity_t.astype(np.float32) / 16.0  # Масштабирование
    filtered_disparity = cv2.ximgproc.getDisparityVis(filtered_disparity_t)

    if show_images:
        cv2.namedWindow("Disparity Filtered", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Disparity Filtered", filtered_disparity)

    filtered_heatmap = cv2.applyColorMap(filtered_disparity, color_map)
    if show_images:
        cv2.namedWindow("Heatmap Filtered", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Heatmap Filtered", filtered_heatmap)

    if save_to_file:
        current_path = os.getcwd()
        disparity_path = os.path.join(current_path, "disparity")

        if os.path.exists(disparity_path):
            import shutil
            shutil.rmtree(disparity_path)
        os.makedirs(disparity_path)

        file_disparity = os.path.join("disparity", "disparity-basic.jpg")
        file_heatmap = os.path.join("disparity", "heatmap-basic.jpg")
        file_disparity_filtered = os.path.join("disparity", "disparity-filtered.jpg")
        file_heatmap_filtered = os.path.join("disparity", "heatmap-filtered.jpg")

        cv2.imwrite(file_disparity, disparity)
        cv2.imwrite(file_heatmap, heatmap)
        cv2.imwrite(file_disparity_filtered, filtered_disparity)
        cv2.imwrite(file_heatmap_filtered, filtered_heatmap)

    # Выбор выходного изображения в зависимости от disparity_type
    if disparity_type == DisparityType.BASIC_DISPARITY:
        map_output[:] = disparity
    elif disparity_type == DisparityType.BASIC_HEATMAP:
        map_output[:] = heatmap
    elif disparity_type == DisparityType.FILTERED_DISPARITY:
        map_output[:] = filtered_disparity
    elif disparity_type == DisparityType.FILTERED_HEATMAP or disparity_type == DisparityType.ALL:
        map_output[:] = filtered_heatmap
    else:
        map_output[:] = filtered_heatmap

    if show_images:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0