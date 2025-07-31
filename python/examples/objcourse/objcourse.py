import cv2
from pathlib import Path
import logging
from mrcv import ObjCourse  # Предполагается, что класс ObjCourse находится в файле objcourse.py

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# Пути к файлам модели
model_file = Path("../../../examples/objcourse/files/ship.onnx")
class_file = Path("../../../examples/objcourse/files/ship.names")
ship_file = Path("../../../examples/objcourse/files/ship.bmp")

current_path = Path.cwd()

model_path = current_path / model_file
class_path = current_path / class_file
ship_path = current_path / ship_file

# Экземпляр класса детектора
objcourse = ObjCourse(str(model_path), str(class_path))

# Загрузка изображения
frame_ship = cv2.imread(str(ship_path), cv2.IMREAD_COLOR)
if frame_ship is None:
    logger.error(f"Failed to load image: {ship_path}")

# Подсчет объектов
obj_count = objcourse.get_object_count(frame_ship)

# Расчет курса
obj_angle = objcourse.get_object_course(frame_ship, 640, 80)

# Логирование результатов
logger.info("")  # Пустая строка для разделения логов
logger.info(f"Файл модели: {model_path}")
logger.info(f"Файл классов: {class_path}")
logger.info(f"Входное изображение: {ship_path}")
logger.info(f"Обнаружено объектов: {obj_count}")
logger.info(f"Курс на цель в градусах: {obj_angle}")
