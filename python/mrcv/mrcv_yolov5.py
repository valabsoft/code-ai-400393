import cv2
import numpy as np
import random
from pathlib import Path
import yaml
from enum import Enum
from typing import List, Dict, Tuple

# Глобальные переменные для хранения состояния
boxes: List[Dict[str, any]] = []
current_class_id: int = 0
drawing: bool = False
current_box: cv2.typing.Rect = (0, 0, 0, 0)
start_point: Tuple[int, int] = (0, 0)
is_editing: bool = False
selected_box_index: int = -1
resizing: bool = False
resize_start_point: Tuple[int, int] = (0, 0)
class_colors: Dict[int, Tuple[int, int, int, int]] = {}

def get_random_color() -> Tuple[int, int, int, int]:
    """Генерация случайного цвета с альфа-каналом."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 128)

def draw_class_info(info_panel: np.ndarray, class_counts: Dict[int, int]) -> None:
    """Отображение информации о классах и количестве объектов."""
    info_panel[:] = (50, 50, 50)  # Очистка панели
    y_offset = 30

    # Краткая инструкция
    instructions = [
        "Instructions:",
        "1. Draw: Left-click & drag",
        "2. Edit: Press 'e'",
        "3. Change class: 0-9 keys",
        "4. Delete: Right-click on box",
        "5. Exit: ESC"
    ]

    for line in instructions:
        cv2.putText(info_panel, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 20

    y_offset += 20

    # Информация о классах
    for class_id, count in class_counts.items():
        text = f"Class {class_id}: {count} objects"
        cv2.putText(info_panel, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[class_id], 2)
        y_offset += 30

    # Отображение текущего класса и режима
    status_text = f"Class: {current_class_id} | Mode: {'Edit' if is_editing else 'Draw'}"
    text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(info_panel, status_text, (10, info_panel.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def is_near_edge(pt: Tuple[int, int], box: cv2.typing.Rect, threshold: int = 10) -> bool:
    """Проверка, находится ли точка рядом с углом или краем рамки."""
    x, y = pt
    bx, by, bw, bh = box

    # Проверка углов
    if (abs(x - bx) < threshold and abs(y - by) < threshold or
        abs(x - (bx + bw)) < threshold and abs(y - by) < threshold or
        abs(x - bx) < threshold and abs(y - (by + bh)) < threshold or
        abs(x - (bx + bw)) < threshold and abs(y - (by + bh)) < threshold):
        return True

    # Проверка краев
    if (abs(x - bx) < threshold and by <= y <= by + bh or
        abs(x - (bx + bw)) < threshold and by <= y <= by + bh or
        abs(y - by) < threshold and bx <= x <= bx + bw or
        abs(y - (by + bh)) < threshold and bx <= x <= bx + bw):
        return True

    return False

def on_mouse(event: int, x: int, y: int, flags: int, userdata: np.ndarray) -> None:
    """Обработка событий мыши."""
    global drawing, current_box, start_point, is_editing, selected_box_index, resizing, resize_start_point, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        if is_editing:
            for i, box in enumerate(boxes):
                if (cv2.pointPolygonTest(np.array([(box['box'][0], box['box'][1]),
                                                  (box['box'][0] + box['box'][2], box['box'][1]),
                                                  (box['box'][0] + box['box'][2], box['box'][1] + box['box'][3]),
                                                  (box['box'][0], box['box'][1] + box['box'][3])]), (x, y), False) >= 0 or
                    is_near_edge((x, y), box['box'])):
                    selected_box_index = i
                    box['is_selected'] = True
                    resize_start_point = (x, y)
                    resizing = is_near_edge((x, y), box['box'])
                    break
        else:
            drawing = True
            start_point = (x, y)
            current_box = (x, y, 0, 0)

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, box in enumerate(boxes):
            if cv2.pointPolygonTest(np.array([(box['box'][0], box['box'][1]),
                                             (box['box'][0] + box['box'][2], box['box'][1]),
                                             (box['box'][0] + box['box'][2], box['box'][1] + box['box'][3]),
                                             (box['box'][0], box['box'][1] + box['box'][3])]), (x, y), False) >= 0:
                boxes.pop(i)
                break

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box = (min(start_point[0], x), min(start_point[1], y),
                          abs(x - start_point[0]), abs(y - start_point[1]))
        elif is_editing and selected_box_index != -1:
            if resizing:
                box = boxes[selected_box_index]['box']
                dx = x - resize_start_point[0]
                dy = y - resize_start_point[1]

                if abs(x - box[0]) < 10:
                    box = (box[0] + dx, box[1], box[2] - dx, box[3])
                if abs(y - box[1]) < 10:
                    box = (box[0], box[1] + dy, box[2], box[3] - dy)
                if abs(x - (box[0] + box[2])) < 10:
                    box = (box[0], box[1], box[2] + dx, box[3])
                if abs(y - (box[1] + box[3])) < 10:
                    box = (box[0], box[1], box[2], box[3] + dy)

                boxes[selected_box_index]['box'] = box
                resize_start_point = (x, y)
            else:
                box = boxes[selected_box_index]['box']
                boxes[selected_box_index]['box'] = (x - box[2] // 2, y - box[3] // 2, box[2], box[3])

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            boxes.append({'class_id': current_class_id, 'box': current_box, 'is_selected': False})
            if current_class_id not in class_colors:
                class_colors[current_class_id] = get_random_color()
        elif is_editing and selected_box_index != -1:
            boxes[selected_box_index]['is_selected'] = False
            selected_box_index = -1
            resizing = False

def interactive_marking(image: np.ndarray) -> None:
    """Интерактивная разметка изображения."""
    global boxes, class_colors, current_class_id, is_editing

    cv2.namedWindow("Marking", cv2.WINDOW_AUTOSIZE)
    info_panel = np.zeros((image.shape[0], 400, 3), dtype=np.uint8)
    info_panel[:] = (50, 50, 50)

    cv2.setMouseCallback("Marking", on_mouse, image)

    while True:
        display_image = image.copy()

        # Подсчет количества объектов для каждого класса
        class_counts = {}
        for box in boxes:
            class_counts[box['class_id']] = class_counts.get(box['class_id'], 0) + 1

        draw_class_info(info_panel, class_counts)

        # Отображение всех размеченных объектов
        for box in boxes:
            color = class_colors[box['class_id']]
            overlay = display_image.copy()
            cv2.rectangle(overlay, (box['box'][0], box['box'][1]),
                         (box['box'][0] + box['box'][2], box['box'][1] + box['box'][3]), color, -1)
            cv2.addWeighted(overlay, 0.3, display_image, 0.7, 0, display_image)
            cv2.rectangle(display_image, (box['box'][0], box['box'][1]),
                         (box['box'][0] + box['box'][2], box['box'][1] + box['box'][3]), color, 2)
            cv2.putText(display_image, str(box['class_id']),
                        (box['box'][0], box['box'][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if drawing:
            cv2.rectangle(display_image, (current_box[0], current_box[1]),
                         (current_box[0] + current_box[2], current_box[1] + current_box[3]), (0, 0, 255), 2)

        combined_image = np.hstack((display_image, info_panel))
        cv2.imshow("Marking", combined_image)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key in range(ord('0'), ord('9') + 1):
            current_class_id = key - ord('0')
        elif key == ord('e'):
            is_editing = not is_editing

    cv2.destroyWindow("Marking")

def save_yolo_format(filename: str, boxes: List[Dict[str, any]], image_size: Tuple[int, int]) -> None:
    """Сохранение разметки в формате YOLO."""
    with open(filename, 'w') as file:
        for box in boxes:
            x_center = (box['box'][0] + box['box'][2] / 2.0) / image_size[1]
            y_center = (box['box'][1] + box['box'][3] / 2.0) / image_size[0]
            width = box['box'][2] / image_size[1]
            height = box['box'][3] / image_size[0]
            file.write(f"{box['class_id']} {x_center} {y_center} {width} {height}\n")

def yolov5_labeler_processing(input_dir: str, output_dir: str) -> None:
    """Обработка изображений для разметки в формате YOLOv5."""
    global boxes, class_colors

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for entry in Path(input_dir).iterdir():
        if entry.is_file() and entry.suffix.lower() in ('.jpg', '.png'):
            image = cv2.imread(str(entry))
            if image is None:
                print(f"Ошибка загрузки изображения: {entry}")
                continue

            boxes.clear()
            class_colors.clear()
            interactive_marking(image)

            output_filename = output_path / f"{entry.stem}.txt"
            save_yolo_format(str(output_filename), boxes, image.shape[:2])

class YOLOv5Model(Enum):
    YOLOv5n = "YOLOv5n"
    YOLOv5s = "YOLOv5s"
    YOLOv5m = "YOLOv5m"
    YOLOv5l = "YOLOv5l"
    YOLOv5x = "YOLOv5x"

def yolov5_generate_config(model: YOLOv5Model, output_file: str, nc: int) -> None:
    """Генерация конфигурационного файла YOLOv5."""
    config = {
        "nc": nc,
        "anchors": [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ],
        "backbone": [
            [-1, 1, "Conv", [64, 6, 2, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 3, "C3", [128]],
            [-1, 1, "Conv", [256, 3, 2]],
            [-1, 6, "C3", [256]],
            [-1, 1, "Conv", [512, 3, 2]],
            [-1, 9, "C3", [512]],
            [-1, 1, "Conv", [1024, 3, 2]],
            [-1, 3, "C3", [1024]],
            [-1, 1, "SPPF", [1024, 5]]
        ],
        "head": [
            [-1, 1, "Conv", [512, 1, 1]],
            [-1, 1, "nn.Upsample", [None, 2, 'nearest']],
            [[-1, 6], 1, "Concat", [1]],
            [-1, 3, "C3", [512, False]],
            [-1, 1, "Conv", [256, 1, 1]],
            [-1, 1, "nn.Upsample", [None, 2, 'nearest']],
            [[-1, 4], 1, "Concat", [1]],
            [-1, 3, "C3", [256, False]],
            [-1, 1, "Conv", [256, 3, 2]],
            [[-1, 14], 1, "Concat", [1]],
            [-1, 3, "C3", [512, False]],
            [-1, 1, "Conv", [512, 3, 2]],
            [[-1, 10], 1, "Concat", [1]],
            [-1, 3, "C3", [1024, False]],
            [[17, 20, 23], 1, "Detect", ["nc", "anchors"]]
        ]
    }

    model_params = {
        YOLOv5Model.YOLOv5n: {"depth_multiple": 0.33, "width_multiple": 0.25},
        YOLOv5Model.YOLOv5s: {"depth_multiple": 0.33, "width_multiple": 0.50},
        YOLOv5Model.YOLOv5m: {"depth_multiple": 0.67, "width_multiple": 0.75},
        YOLOv5Model.YOLOv5l: {"depth_multiple": 1.0, "width_multiple": 1.0},
        YOLOv5Model.YOLOv5x: {"depth_multiple": 1.33, "width_multiple": 1.25}
    }

    if model not in model_params:
        raise ValueError("Unsupported YOLOv5 model type!")

    config.update(model_params[model])

    with open(output_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

def yolov5_generate_hyperparameters(model: YOLOv5Model, img_width: int, img_height: int, output_file: str, nc: int) -> None:
    """Генерация гиперпараметров для YOLOv5."""
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive!")

    weight_decay = 0.0005
    box_gain = 0.05
    cls_gain = 0.5
    cls_pw = 1.0
    obj_gain = 1.0
    obj_pw = 1.0
    anchor_threshold = 4.0
    fl_gamma = 0.0

    model_params = {
        YOLOv5Model.YOLOv5n: {"weight_decay": 1.0, "box_gain": 1.0, "cls_gain": 0.9, "fl_gamma": 0.1},
        YOLOv5Model.YOLOv5s: {"weight_decay": 1.1, "box_gain": 1.1, "cls_gain": 1.0, "fl_gamma": 0.2},
        YOLOv5Model.YOLOv5m: {"weight_decay": 1.2, "box_gain": 1.2, "cls_gain": 1.1, "fl_gamma": 0.3},
        YOLOv5Model.YOLOv5l: {"weight_decay": 1.3, "box_gain": 1.3, "cls_gain": 1.2, "fl_gamma": 0.4},
        YOLOv5Model.YOLOv5x: {"weight_decay": 1.4, "box_gain": 1.4, "cls_gain": 1.3, "fl_gamma": 0.5}
    }

    if model not in model_params:
        raise ValueError("Unsupported YOLOv5 model type!")

    weight_decay *= model_params[model]["weight_decay"]
    box_gain *= model_params[model]["box_gain"]
    cls_gain *= model_params[model]["cls_gain"]
    fl_gamma += model_params[model]["fl_gamma"]

    if nc > 80:
        weight_decay += 0.0001 * (nc - 80)

    resolution_scale = (img_width * img_height) / (640 * 640)
    box_gain *= np.sqrt(resolution_scale)
    cls_gain = 0.5 + 0.005 * nc
    fl_gamma += 0.1 * np.log2(resolution_scale + 1)

    config = {
        "weight_decay": weight_decay,
        "box": box_gain,
        "cls": cls_gain,
        "cls_pw": cls_pw,
        "obj": obj_gain,
        "obj_pw": obj_pw,
        "anchor_t": anchor_threshold,
        "fl_gamma": fl_gamma
    }

    with open(output_file, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
