# Marine Robotics Computer Vision
Открытая библиотека компьютерного зрения для морских робототехнических систем

## Оглавление
1. [Структура каталогов](#Структура-каталогов)

```
.
├── cmake # Утилиты для сборки
│   ├── mrcv-config.cmake.in
│   ├── silent_copy.cmake
│   └── utils.cmake
│
├── examples # Примеры использования
│   ├── add
│   │	├── main.cpp
│   │	└── CMakeLists.txt
│   └── CMakeLists.txt
│
├── include # Публичные заголовки
│   └── mrcv
│   	├── export.h
│   	└── mrcv.h
│
├── python # Версия библиотеки на Python
│		├── examples
│		└── src
│
├── src # Исходники функций библиотеки
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

При разработке структуры каталогов библиотеки и настройки автоматизированной сборки библиотеки использованы материалы: https://habr.com/ru/articles/683204/
____
[:arrow_up:Оглавление](#Оглавление)
____