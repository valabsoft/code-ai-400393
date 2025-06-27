import cv2
import numpy as np
import os
from mrcv import MRCV, METOD_IMAGE_PERPROCESSIN, LOGTYPE

# Create output directory
os.makedirs("./files/outImages", exist_ok=True)

# Logging is already configured in mrcv.py
MRCV.writeLog(" ")
MRCV.writeLog(" === НОВЫЙ ЗАПУСК === ")

# Load image
imageInputFilePath = "./files/img02.jfif"
imageIn = cv2.imread(imageInputFilePath, cv2.IMREAD_COLOR)
imageOut = imageIn.copy() if imageIn is not None else None

if imageIn is not None:
    MRCV.writeLog(f"    загружено изображение: {imageInputFilePath} :: "
                  f"{imageIn.shape[1]}x{imageIn.shape[0]}x{imageIn.shape[2]}")
else:
    MRCV.writeLog(f"    не удалось загрузить изображение: {imageInputFilePath}", LOGTYPE.ERROR)
    exit(1)

# Define preprocessing methods (removed CORRECTION_GEOMETRIC_DEFORMATION)
methodImagePreProcessingBrightnessContrast = [
    METOD_IMAGE_PERPROCESSIN.NOISE_FILTERING_01_MEDIAN_FILTER,
    METOD_IMAGE_PERPROCESSIN.BALANCE_CONTRAST_10_LAB_CLAHE,
    METOD_IMAGE_PERPROCESSIN.SHARPENING_02,
    METOD_IMAGE_PERPROCESSIN.BRIGHTNESS_LEVEL_DOWN,
    METOD_IMAGE_PERPROCESSIN.NONE,
]

# Preprocess image
state, imageOut = MRCV.preprocessingImage(imageOut, methodImagePreProcessingBrightnessContrast, "./files/camera-parameters.xml")
if state == 0:
    MRCV.writeLog(" Предобработка изображения завершена (успешно)")
else:
    MRCV.writeLog(f" preprocessingImage, state = {state}", LOGTYPE.ERROR)

# Save output image
imageOutputFilePath = "./files/outImages/test.png"
cv2.imwrite(imageOutputFilePath, imageOut)
MRCV.writeLog(f"\t результат предобработки сохранён: {imageOutputFilePath}")

# Display images
CoefShowWindow = 0.5
imageIn = cv2.resize(imageIn, None, fx=CoefShowWindow, fy=CoefShowWindow, interpolation=cv2.INTER_LINEAR)
imageOut = cv2.resize(imageOut, None, fx=CoefShowWindow, fy=CoefShowWindow, interpolation=cv2.INTER_LINEAR)

cv2.namedWindow("imageIn", cv2.WINDOW_AUTOSIZE)
cv2.imshow("imageIn", imageIn)
cv2.namedWindow("imageOut", cv2.WINDOW_AUTOSIZE)
cv2.imshow("imageOut", imageOut)
cv2.waitKey(0)
cv2.destroyAllWindows()