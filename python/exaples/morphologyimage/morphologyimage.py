import cv2
from mrcv import morphology_image, METOD_MORF
from pathlib import Path

image_file = Path("files")
current_path = Path.cwd()
image_path = current_path / image_file
input_image = str(image_path / "opening.png")
output_image = str(image_path / "out.png")

morph_size = 1

# Чтение изображения в градациях серого
image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)

# Проверка успешности загрузки изображения
if image is None:
    print("Не удалось открыть или найти изображение")
else:
    result = morphology_image(image, output_image, METOD_MORF.OPEN, morph_size)