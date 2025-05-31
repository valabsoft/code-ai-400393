import cv2
import numpy as np
import torch
import onnxruntime as ort
import logging
import os
from datetime import datetime
import time
from typing import List, Tuple
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
OBJCOURSE_FONT_SCALE = 0.5
OBJCOURSE_THICKNESS = 2
OBJCOURSE_BLACK = (0, 0, 0)
OBJCOURSE_YELLOW = (0, 255, 255)
OBJCOURSE_GREEN = (0, 255, 0)
OBJCOURSE_DRAW_LABEL = True
IS_DEBUG_LOG_ENABLED = True

class ObjCourse:
    def __init__(self, path_to_model: str, path_to_classes: str, width: int = None, height: int = None,
                 score_threshold: float = None, nms_threshold: float = None,
                 confidence_threshold: float = None, camera_angle: float = None):
        self._input_width = width if width is not None else 640
        self._input_height = height if height is not None else 640
        self._score_threshold = score_threshold if score_threshold is not None else 0.5
        self._nms_threshold = nms_threshold if nms_threshold is not None else 0.45
        self._confidence_threshold = confidence_threshold if confidence_threshold is not None else 0.25
        self._camera_angle = camera_angle if camera_angle is not None else 60.0
        self._classes = []
        self._classes_id_set = []
        self._confidences_set = []
        self._boxes_set = []
        self._classes_set = []
        self._inference_time = 0.0

        # Инициализация нейронной сети
        if not self.init_nn(path_to_model, path_to_classes):
            logger.info("The neural network has been initiated successfully!")
            logger.info(f"Input width: {self._input_width}")
            logger.info(f"Input height: {self._input_height}")
        else:
            logger.error("The neural network initialization ERROR!")

    def init_nn(self, path_to_model: str, path_to_classes: str) -> int:
        # Чтение классов
        err = self.read_classes(path_to_classes)
        if err == 0:
            try:
                # Загрузка ONNX модели через onnxruntime
                self._session = ort.InferenceSession(path_to_model)
                self._input_name = self._session.get_inputs()[0].name
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                return 101  # Аналог ENETDOWN
        return err

    def read_classes(self, path_to_classes: str) -> int:
        try:
            with open(path_to_classes, 'r') as f:
                self._classes = [line.strip() for line in f if line.strip()]
            return 0
        except FileNotFoundError:
            logger.error("Failed to open classes file!")
            return 2  # Аналог ENOENT

    def draw_label(self, img: np.ndarray, label: str, left: int, top: int) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_size, baseline = cv2.getTextSize(label, font, OBJCOURSE_FONT_SCALE, OBJCOURSE_THICKNESS)
        top = max(top, label_size[1])
        top_left = (left, top)
        bottom_right = (left + label_size[0], top + label_size[1] + baseline)
        cv2.rectangle(img, top_left, bottom_right, OBJCOURSE_BLACK, cv2.FILLED)
        cv2.putText(img, label, (left, top + label_size[1]), font, OBJCOURSE_FONT_SCALE,
                    OBJCOURSE_YELLOW, OBJCOURSE_THICKNESS)

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        # Преобразование изображения в формат, подходящий для модели
        img_resized = cv2.resize(img, (self._input_width, self._input_height))
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Перевод в формат [N, C, H, W] для PyTorch/ONNX
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def post_process(self, img: np.ndarray, outputs: np.ndarray, class_names: List[str]) -> np.ndarray:
        processed_image = img.copy()
        self._classes_id_set.clear()
        self._confidences_set.clear()
        self._boxes_set.clear()
        self._classes_set.clear()

        class_ids = []
        confidences = []
        boxes = []

        x_factor = img.shape[1] / self._input_width
        y_factor = img.shape[0] / self._input_height

        # Предполагается, что модель возвращает [batch, rows, dimensions]
        data = outputs[0]  # Первый выход модели
        rows = data.shape[0]
        dimensions = len(class_names) + 5

        for i in range(rows):
            confidence = data[i, 4]
            if confidence >= self._confidence_threshold:
                classes_scores = data[i, 5:]
                class_id = np.argmax(classes_scores)
                max_class_score = classes_scores[class_id]
                if max_class_score > self._score_threshold:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = data[i, :4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    boxes.append((left, top, width, height))

        # Применение NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self._score_threshold, self._nms_threshold)

        logger.info("POST PROCESS ==>")
        logger.info(f"boxes.size(): {len(boxes)}")
        logger.info(f"confidences.size(): {len(confidences)}")

        biggest_area = -float('inf')
        biggest_index = -1
        box_index = -1

        for i in indices.flatten():
            box_index += 1
            box = boxes[i]
            area = box[2] * box[3]
            if area > biggest_area:
                biggest_index = box_index
                biggest_area = area

        if biggest_index > -1:
            idx = indices[biggest_index]
            box = boxes[idx]
            self._boxes_set.append(box)
            self._confidences_set.append(confidences[idx])
            self._classes_id_set.append(class_ids[idx])
            self._classes_set.append(class_names[class_ids[idx]])

            left, top, width, height = box
            cv2.rectangle(processed_image, (left, top), (left + width, top + height),
                         OBJCOURSE_GREEN, 3 * OBJCOURSE_THICKNESS)

            if OBJCOURSE_DRAW_LABEL:
                label = f"{class_names[class_ids[idx]]}: {confidences[idx]:.2f}"
                self.draw_label(processed_image, label, left, top)

        return processed_image

    def get_info(self) -> str:
        return "\n".join(f"{cls}: {conf:.2f}" for cls, conf in zip(self._classes_set, self._confidences_set))

    def main_process(self, img: np.ndarray) -> np.ndarray:
        start_time = time.time()
        inputs = self.pre_process(img)
        outputs = self._session.run(None, {self._input_name: inputs})[0]
        processed_img = self.post_process(img, outputs, self._classes)
        self._inference_time = time.time() - start_time
        return processed_img

    def find_angle(self, resolution: float, camera_angle: float, cx: float) -> int:
        self._camera_angle = camera_angle
        return int((cx * camera_angle / resolution) - camera_angle / 2)

    def get_timestamp(self) -> str:
        now = datetime.now()
        return now.strftime("%d-%m-%Y %H:%M:%S.%f")[:-3]

    def get_object_count(self, frame: np.ndarray) -> int:
        img = self.main_process(frame)

        if IS_DEBUG_LOG_ENABLED:
            output_path = Path("files") / "output.bmp"
            output_path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(output_path), img)

        ids = self.get_class_ids()
        confidences = self.get_confidences()
        boxes = self.get_boxes()
        classes = self.get_classes()

        logger.info(f"IDs: {';'.join(map(str, ids))}")
        logger.info(f"Confidence: {';'.join(map(str, confidences))}")
        logger.info(f"Boxes: {len(boxes)}")

        return len(boxes)

    def get_object_course(self, frame: np.ndarray, frame_width: float, camera_angle: float) -> float:
        img = self.main_process(frame)
        timestamp = self.get_timestamp()

        ids = self.get_class_ids()
        confidences = self.get_confidences()
        boxes = self.get_boxes()
        classes = self.get_classes()

        biggest_area = -float('inf')
        biggest_index = -1
        box_index = -1

        for box in boxes:
            box_index += 1
            area = box[2] * box[3]
            if area > biggest_area:
                biggest_index = box_index
                biggest_area = area

        if boxes:
            box = boxes[biggest_index]
            left, top, width, height = box
            center = (left + width / 2, top + height / 2)
            angle = self.find_angle(frame_width, camera_angle, center[0])
            direction = "RIGHT" if center[0] > frame_width / 2 else "LEFT"

            sight_width = 50
            board_box_pt1 = (int(frame_width / 2 - sight_width), int(frame.shape[0] / 2 - sight_width))
            board_box_pt2 = (int(frame_width / 2 + sight_width), int(frame.shape[0] / 2 + sight_width))

            if board_box_pt1[0] <= center[0] <= board_box_pt2[0]:
                direction = "HOLD"

            diagnostic_info = f"CMD: ({direction}:{angle}) TIME: {self._inference_time:.2f} {timestamp}"
            logger.info(diagnostic_info)
            return angle

        logger.info(f"TIME: {timestamp} NOT FOUND...")
        return 0.0

    # Геттеры
    def get_class_ids(self) -> List[int]:
        return self._classes_id_set

    def get_confidences(self) -> List[float]:
        return self._confidences_set

    def get_boxes(self) -> List[Tuple[int, int, int, int]]:
        return self._boxes_set

    def get_classes(self) -> List[str]:
        return self._classes_set

    def get_inference(self) -> float:
        return self._inference_time