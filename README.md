# Marine Robotics Computer Vision
Открытая библиотека компьютерного зрения для морских робототехнических систем

## Оглавление
1. [Структура каталогов](#Структура-каталогов)
2. [Предобученные нейросети](#Предобученные-нейросети)

## Структура каталогов

```
.
├── cmake # Утилиты для сборки
│   ├── mrcv-config.cmake.in
│   ├── silent_copy.cmake
│   └── utils.cmake
│
├── examples # Примеры использования
│   ├── mrcv-example
│   │	├── main.cpp
│   │	└── CMakeLists.txt
│   └── CMakeLists.txt
│
├── include # Публичные заголовки
│   └── mrcv
│   	├── export.h
│   	├── mrcv-common.h
│   	└── mrcv.h
│
├── python # Версия библиотеки на Python
│		├── examples
│		└── src
│
├── src # Исходники функций библиотеки
│	├── mrcv-augmentation.cpp
│	├── mrcv-calibration.cpp
│	├── ...
│	└── mrcv.cpp
│
├── tests # Тесты
│	├── add_test.cpp
│	└── CMakeLists.txt
│
├── CMakeLists.txt
├── CMakePresets.json
└── README.md
```

Датасет изображений для работы с библиотекой доступен по ссылке [code-ai-400393-image-dataset.7z](https://disk.yandex.ru/d/TxReQ9J6PAo9Nw).

## Предобученные нейросети
Ниже представлены предобученные модели YOLOv5 для различных задач подводного обнаружения объектов. Каждая модель обучалась на специализированных датасетах.

Детали обучения:
- Эпохи: 500
- Фреймворк: YOLOv5 (PyTorch)
- Разрешение: 640×640 (по умолчанию)

| Назначение | Классы | Архитектуры |
|------------|--------|-------------|
| Подводная археология | Амфоры, пушечные ядра, посуда, фрагменты кораблей, корабельные орудия                 | [yolov5n](https://disk.yandex.ru/d/v7zyKX-ggxNm-g), [yolov5s](https://disk.yandex.ru/d/B1xEyi3OhfJIcw) |
| Поисковые операции   | Затонувшие корабли, самолёты                                                          | [yolov5n](https://disk.yandex.ru/d/QYB4u4gkHHIWoQ), [yolov5s](https://disk.yandex.ru/d/QYB4u4gkHHIWoQ) |
| Классификация судов  | Авианосцы, грузовые суда, круизные лайнеры, рыболовные суда, танкеры, военные корабли | [yolov5n](https://disk.yandex.ru/d/BvOk5oChQ67Vnw), [yolov5s](https://disk.yandex.ru/d/cqkz5-npK2RQaA) |

____
[:arrow_up:Оглавление](#Оглавление)
____