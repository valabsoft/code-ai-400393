import cv2
import numpy as np
from enum import Enum
from pathlib import Path

class METOD_MORF(Enum):
    OPEN = 1
    CLOSE = 2
    DILAT = 3
    ERODE = 4
    GRADIENT = 5


def opening_morphological(image: np.ndarray, out: str, element: np.ndarray) -> int:
    """
    @brief Функция морфологического открытия.
    @param image Исходное изображение (в градациях серого).
    @param out Путь для сохранения обработанного изображения.
    @param element Структурирующий элемент для морфологической операции.
    @return Результат работы функции (0 при успехе).
    """
    output = cv2.morphologyEx(image, cv2.MORPH_OPEN, element, iterations=2)
    cv2.imwrite(out, output)
    return 0


def closing_morphological(image: np.ndarray, out: str, element: np.ndarray) -> int:
    """
    @brief Функция морфологического закрытия.
    @param image Исходное изображение (в градациях серого).
    @param out Путь для сохранения обработанного изображения.
    @param element Структурирующий элемент для морфологической операции.
    @return Результат работы функции (0 при успехе).
    """
    output = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element, iterations=2)
    cv2.imwrite(out, output)
    return 0


def gradient_morphological(image: np.ndarray, out: str, element: np.ndarray) -> int:
    """
    @brief Функция морфологического градиента.
    @param image Исходное изображение (в градациях серого).
    @param out Путь для сохранения обработанного изображения.
    @param element Структурирующий элемент для морфологической операции.
    @return Результат работы функции (0 при успехе).
    """
    output = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, element, iterations=1)
    cv2.imwrite(out, output)
    return 0


def erode_morphological(image: np.ndarray, out: str, element: np.ndarray) -> int:
    """
    @brief Функция морфологической эрозии.
    @param image Исходное изображение (в градациях серого).
    @param out Путь для сохранения обработанного изображения.
    @param element Структурирующий элемент для морфологической операции.
    @return Результат работы функции (0 при успехе).
    """
    output = cv2.erode(image, element, iterations=1)
    cv2.imwrite(out, output)
    return 0


def dilation_morphological(image: np.ndarray, out: str, element: np.ndarray) -> int:
    """
    @brief Функция морфологического расширения.
    @param image Исходное изображение (в градациях серого).
    @param out Путь для сохранения обработанного изображения.
    @param element Структурирующий элемент для морфологической операции.
    @return Результат работы функции (0 при успехе).
    """
    output = cv2.dilate(image, element, iterations=1)
    cv2.imwrite(out, output)
    return 0


def morphology_image(image: np.ndarray, out: str, method: METOD_MORF, morph_size: int) -> int:
    """
    @brief Главная функция для применения морфологических операций к изображению.
    @param image Исходное изображение (в градациях серого).
    @param out Путь для сохранения обработанного изображения.
    @param method Метод морфологической операции (OPEN, CLOSE, DILAT, ERODE, GRADIENT).
    @param morph_size Размер структурирующего элемента.
    @return Результат работы функции (0 при успехе).
    """
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * morph_size + 1, 2 * morph_size + 1),
        (morph_size, morph_size)
    )

    if method == METOD_MORF.OPEN:
        return opening_morphological(image, out, element)
    elif method == METOD_MORF.CLOSE:
        return closing_morphological(image, out, element)
    elif method == METOD_MORF.DILAT:
        return dilation_morphological(image, out, element)
    elif method == METOD_MORF.ERODE:
        return erode_morphological(image, out, element)
    elif method == METOD_MORF.GRADIENT:
        return gradient_morphological(image, out, element)

    return 0