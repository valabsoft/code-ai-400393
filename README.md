# Marine Robotics Computer Vision
Открытая библиотека компьютерного зрения для морских робототехнических систем

## Оглавление
1. [Структура каталогов](#Структура-каталогов)

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

Датасет изображений для работы с библиотекой доступен по ссылке [Датасет](https://disk.yandex.ru/d/TxReQ9J6PAo9Nw).
____
[:arrow_up:Оглавление](#Оглавление)
____