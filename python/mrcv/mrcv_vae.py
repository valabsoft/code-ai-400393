import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import logging
import random
import time
import onnxruntime as ort
from typing import List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, h_dim: int, z_dim: int):
        super(VariationalAutoEncoder, self).__init__()

        self.encoder1 = nn.Linear(input_dim, h_dim)
        self.encoder2 = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, z_dim)
        self.logvar = nn.Linear(h_dim, z_dim)
        self.decoder1 = nn.Linear(z_dim, h_dim)
        self.decoder2 = nn.Linear(h_dim, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.encoder1(x))
        x = self.relu(self.encoder2(x))
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.relu(self.decoder1(z))
        z = torch.sigmoid(self.decoder2(z))
        return z

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x = self.relu(self.decoder1(z))
        x = self.decoder2(x)
        return x, mu, logvar


class LoadImageDataset(Dataset):
    def __init__(self, root: str, height: int, width: int, num_color: int):
        self.num_color = num_color
        self.height = height
        self.width = width
        self.images = [str(p) for p in Path(root).iterdir() if p.is_file()]
        logger.info(f"Loaded Images: {len(self.images)}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.images[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if image is None:
            logger.error(f"Failed to load image: {path}")
            return torch.tensor([]), torch.tensor(0)

        # Convert color based on num_color
        if self.num_color == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif self.num_color == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, (self.width, self.height))

        # Convert to tensor and normalize
        tensor = torch.from_numpy(image).float()
        if self.num_color == 1:
            tensor = tensor.unsqueeze(2)  # Add channel dimension for grayscale
        tensor = tensor.permute(2, 0, 1) / 255.0  # Change to CxHxW
        label = torch.tensor(0, dtype=torch.long)

        return tensor.clone(), label.clone()

def neural_network_augmentation_as_tensor(
        root: str,
        height: int,
        width: int,
        h_dim: int,
        z_dim: int,
        num_epoch: int,
        batch_size: int,
        lr_rate: float
) -> torch.Tensor:
    num_color = 1
    input_dim = num_color * height * width
    device = torch.device("cpu")

    # Load dataset
    dataset = LoadImageDataset(root, height, width, num_color)
    if len(dataset) == 0:
        logger.error("No images loaded!")
        return torch.tensor([])

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Initialize model
    model = VariationalAutoEncoder(input_dim, h_dim, z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    logger.info("Training...")
    for epoch in range(num_epoch):
        model.train()
        for batch, _ in data_loader:
            data = batch.to(device).view(-1, input_dim)

            if torch.isnan(data).any() or torch.isinf(data).any():
                logger.error("Data tensor contains NaN or Inf values!")
                continue

            optimizer.zero_grad()
            output, mu, logvar = model(data)
            output = torch.sigmoid(output)

            reconstruction_loss = nn.functional.binary_cross_entropy(
                output, data, reduction='sum'
            )

            reconstruction_loss.backward()
            optimizer.step()

    logger.info("Training is DONE!")

    # Generate output
    num_examples = 1
    random.seed(time.time())
    rand_idx = random.randrange(len(dataset))
    image, _ = dataset[rand_idx]
    images = [image.to(device)]

    logger.info("Encoding...")
    encodings_digit = []
    with torch.no_grad():
        for d in range(num_examples):
            flattened_image = images[d].view(1, input_dim)
            mu, logvar = model.encode(flattened_image)
            encodings_digit.append((mu, logvar))

    logger.info("Decoding...")
    for _ in range(num_examples):
        mu, logvar = encodings_digit[0]
        sample = model.reparameterize(mu, logvar)
        tensor = model.decode(sample)
        tensor = torch.sigmoid(tensor).view(num_color, width, height)
        logger.info("Generated tensor is DONE!")
        return tensor.clone()

    return torch.tensor([])

def neural_network_augmentation_as_mat(
        root: str,
        height: int,
        width: int,
        h_dim: int,
        z_dim: int,
        num_epoch: int,
        batch_size: int,
        lr_rate: float
) -> np.ndarray:
    num_color = 1
    tensor = neural_network_augmentation_as_tensor(
        root, height, width, h_dim, z_dim, num_epoch, batch_size, lr_rate
    )

    if not tensor.numel():
        logger.error("Generated tensor empty!")
        return np.array([])

    # Convert tensor to OpenCV Mat
    tensor = (tensor * 255).clamp_(0, 255).to(torch.uint8).cpu()
    tensor = tensor.view(num_color, height, width).permute(1, 2, 0)

    image = tensor.numpy()
    if image.size == 0:
        logger.error("Generated image empty!")
        return np.array([])

    # Adjust contrast and brightness
    image = cv2.convertScaleAbs(image, alpha=2.3, beta=-300)

    logger.info("Generated image is DONE!")
    return image.copy()

########################################################################################################################

# Константы (переносим из C++)
SCORE_THRESHOLD = 0.20
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.15

# Параметры шрифтов
FONT_SCALE = 0.7
THICKNESS = 1

# Цветовые константы (в формате BGR для OpenCV)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)


class NNPreLabeler:
    def __init__(self, model: str, classes: str, width: int, height: int):
        # Ширина и высота входного изображения
        self.input_width = width
        self.input_height = height
        # Размер исходного изображения (будет установлен в process)
        self.source_size = None
        # Структура нейросети (используем onnxruntime вместо cv::dnn::Net)
        self.session = None
        # Вектор распознаваемых классов
        self.classes: List[str] = []
        # Структуры для хранения результатов обработки
        self.classes_id_set: List[int] = []
        self.boxes_set: List[np.ndarray] = []
        self.confidences_set: List[float] = []
        self.classes_set: List[str] = []
        # Время обработки
        self.inference_time = 0.0

        # Инициализация сети
        err = self.init_network(model, classes)
        if err:
            logger.error("Failed to init neural network!")
            raise RuntimeError("Failed to init neural network")

    def read_classes(self, file_path: str) -> int:
        """Чтение списка классов из файла."""
        try:
            with open(file_path, 'r') as f:
                self.classes = [line.strip() for line in f if line.strip()]
            return 0
        except FileNotFoundError:
            logger.error("Failed to open classes names!")
            return 2  # Аналог ENOENT

    def init_network(self, model_path: str, classes_path: str) -> int:
        """Инициализация нейронной сети."""
        err = self.read_classes(classes_path)
        if err == 0:
            try:
                # Загрузка ONNX модели через onnxruntime
                self.session = ort.InferenceSession(model_path)
                # Установка предпочтительного бэкенда (аналог setPreferableBackend)
                # В onnxruntime это настраивается при выборе провайдера
                # Здесь оставляем по умолчанию (CPU)
            except Exception as e:
                logger.error(f"Failed to load ONNX model: {e}")
                return 101  # Аналог ENETDOWN
        return err

    def draw_label(self, img: np.ndarray, label: str, left: int, top: int) -> None:
        """Отрисовка метки над ограничивающим прямоугольником."""
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS
        )
        top = max(top, label_height)
        tlc = (left, top)  # Верхний левый угол
        brc = (left + label_width, top + label_height + baseline)  # Нижний правый угол

        # Рисуем черный прямоугольник
        cv2.rectangle(img, tlc, brc, BLACK, cv2.FILLED)
        # Пишем текст желтым цветом
        cv2.putText(
            img, label, (left, top + label_height),
            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS
        )

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Предобработка изображения перед подачей в сеть."""
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / 255.0, (self.input_width, self.input_height),
            swapRB=True, crop=False
        )
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        return outputs

    def postprocess(self, img: np.ndarray, outputs: List[np.ndarray], class_names: List[str]) -> np.ndarray:
        """Обработка выхода сети и отрисовка результатов."""
        ret = img.copy()
        self.classes_id_set.clear()
        self.confidences_set.clear()
        self.boxes_set.clear()
        self.classes_set.clear()
        class_ids = []
        confidences = []
        boxes = []

        # Факторы масштабирования
        x_factor = img.shape[1] / self.input_width
        y_factor = img.shape[0] / self.input_height

        # Выход сети (предполагаем формат YOLO: [batch, num_boxes, num_classes + 5])
        output = outputs[0]  # (1, num_boxes, num_classes + 5)
        rows = output.shape[1]
        data = output[0]  # (num_boxes, num_classes + 5)

        for i in range(rows):
            row = data[i]
            confidence = row[4]  # Уверенность
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]  # Оценки классов
                class_id = np.argmax(classes_scores)
                max_class_score = classes_scores[class_id]

                if max_class_score > SCORE_THRESHOLD:
                    confidences.append(confidence)
                    class_ids.append(class_id)

                    # Координаты центра и размеры бокса
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])
                    boxes.append(box)

        # Применяем NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)

        for i in indices:
            idx = i if isinstance(i, int) else i[0]
            box = boxes[idx]
            self.boxes_set.append(box)
            self.confidences_set.append(confidences[idx])
            self.classes_id_set.append(class_ids[idx])
            self.classes_set.append(class_names[class_ids[idx]])

            left, top, width, height = box
            # Рисуем прямоугольник
            cv2.rectangle(
                ret, (left, top), (left + width, top + height),
                GREEN, 3 * THICKNESS
            )
            # Формируем метку
            label = f"{class_names[class_ids[idx]]}: {confidences[idx]:.2f}"
            self.draw_label(ret, label, left, top)

        return ret

    def process(self, img: np.ndarray) -> np.ndarray:
        """Обработка изображения."""
        self.source_size = img.shape[:2]  # (height, width)
        detections = self.preprocess(img)

        start_time = time.time()
        res = self.postprocess(img, detections, self.classes)
        self.inference_time = time.time() - start_time

        return res

    def write_labels(self, filename: str) -> None:
        """Запись YOLOv5 лейблов в текстовый файл."""
        try:
            with open(filename, 'w') as f:
                for i in range(len(self.classes_id_set)):
                    class_id = self.classes_id_set[i]
                    box = self.boxes_set[i]
                    left, top, width, height = box

                    # Преобразуем в формат YOLO (нормализованные координаты)
                    x_center = (left + width / 2.0) / self.source_size[1]
                    y_center = (top + height / 2.0) / self.source_size[0]
                    norm_width = width / self.source_size[1]
                    norm_height = height / self.source_size[0]

                    f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")
        except Exception as e:
            logger.error(f"Failed to open file {filename}: {e}")

    # Геттеры
    def get_class_ids(self) -> List[int]:
        return self.classes_id_set

    def get_confidences(self) -> List[float]:
        return self.confidences_set

    def get_boxes(self) -> List[np.ndarray]:
        return self.boxes_set

    def get_classes(self) -> List[str]:
        return self.classes_set

    def get_inference(self) -> float:
        return self.inference_time


def semi_automatic_labeler_image(input_image: np.ndarray, height: int, width: int,
                                 output_path: str, model_path: str, classes_path: str) -> int:
    """Полуавтоматическая разметка для изображения (np.ndarray)."""
    labeler = NNPreLabeler(model_path, classes_path, 640, 640)
    input_image = cv2.resize(input_image, (640, 640))
    img = labeler.process(input_image)

    class_ids = labeler.get_class_ids()
    confidences = labeler.get_confidences()
    boxes = labeler.get_boxes()
    classes = labeler.get_classes()

    for conf in confidences:
        logger.info(f"confidences: {conf}")

    logger.info(f"inference time: {labeler.get_inference()}")

    # Сохранение результата
    filename = os.path.join(output_path, "result.jpg")
    labels_file = os.path.join(output_path, "result.txt")

    cv2.imwrite(filename, img)
    labeler.write_labels(labels_file)
    logger.info("Labeling image is DONE!")
    return 0


def semi_automatic_labeler_file(root: str, height: int, width: int,
                                output_path: str, model_path: str, classes_path: str) -> int:
    """Полуавтоматическая разметка для файла изображения."""
    labeler = NNPreLabeler(model_path, classes_path, 640, 640)
    input_image = cv2.imread(root)
    if input_image is None:
        logger.error(f"Failed to load image from {root}")
        return 1

    input_image = cv2.resize(input_image, (640, 640))
    img = labeler.process(input_image)

    class_ids = labeler.get_class_ids()
    confidences = labeler.get_confidences()
    boxes = labeler.get_boxes()
    classes = labeler.get_classes()

    for conf in confidences:
        logger.info(f"confidences: {conf}")

    logger.info(f"inference time: {labeler.get_inference()}")

    # Сохранение результата
    filename = os.path.join(output_path, "result.jpg")
    labels_file = os.path.join(output_path, "result.txt")

    cv2.imwrite(filename, img)
    labeler.write_labels(labels_file)
    logger.info("Labeling image is DONE!")
    return 0