import os
import cv2
from mrcv import compare_images

# Путь к папке с изображениями
image_dir = "files"
current_path = os.getcwd()
image_path = os.path.join(current_path, image_dir)

# Загрузка изображений
img1 = cv2.imread(os.path.join(image_path, "1.png"))
img2 = cv2.imread(os.path.join(image_path, "2.png"))

# Проверка, что изображения загружены
if img1 is None or img2 is None:
    print("Ошибка: Не удалось загрузить одно или оба изображения.")

# Сравнение изображений
similarity = compare_images(img1, img2, True)
print(f"Сходство: {similarity}")