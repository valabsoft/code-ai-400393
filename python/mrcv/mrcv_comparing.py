import cv2
import numpy as np

def compare_images(img1, img2, method_compare):
    """
    Функция сравнения изображений.
    :param img1: Исходное фото 1
    :param img2: Исходное фото 2
    :param method_compare: Метод сравнения (True - гистограммное, False - L2-норма)
    :return: Различия фотографий в процентном соотношении
    """
    if method_compare:
        # Преобразование изображений в пространство HSV
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

        # Параметры для гистограммы
        h_bins = 50
        s_bins = 60
        hist_size = [h_bins, s_bins]
        ranges = [0, 180, 0, 256]  # Исправлено: одномерный список для диапазонов H и S
        channels = [0, 1]

        # Вычисление гистограмм
        hist1 = cv2.calcHist([hsv1], channels, None, hist_size, ranges, accumulate=False)
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist2 = cv2.calcHist([hsv2], channels, None, hist_size, ranges, accumulate=False)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Сравнение гистограмм
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    else:
        # Вычисление L2-нормы
        error_l2 = np.linalg.norm(img1.astype(float) - img2.astype(float))
        return 1 - error_l2 / (img1.shape[0] * img1.shape[1])

    return 0.0