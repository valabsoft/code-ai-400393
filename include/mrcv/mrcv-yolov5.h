#pragma once

#include <string>

namespace mrcv
{
    /** @brief Перечисление поддерживаемых типов моделей YOLOv5. */
    enum class YOLOv5Model
    {
        YOLOv5n,
        YOLOv5s,
        YOLOv5m,
        YOLOv5l,
        YOLOv5x
    };

    /**
     * @brief Функция создания файла конфигурации YOLOv5 с указанными параметрами.
     * @param model - тип модели YOLOv5 (например, YOLOv5n, YOLOv5s, YOLOv5m,
     * YOLOv5l, YOLOv5x).
     * @param outputFile - путь к выходному конфигурационному файлу YAML.
     * @param nc - количество классов (по умолчанию 1).
     * @throws - std::runtime_error если файл не может быть записан.
     */
    void YOLOv5GenerateConfig(YOLOv5Model model,
                              const std::string &outputFile,
                              unsigned int nc = 1);

    /**
     * @brief Функция создания файла конфигурации гиперпараметров YOLOv5 для
     * заданных параметров.
     * @param model - тип модели YOLOv5 (например, YOLOv5n, YOLOv5s, YOLOv5m,
     * YOLOv5l, YOLOv5x).
     * @param imgWidth - средняя ширина изображений датасета.
     * @param imgHeight - средняя высота изображений датасета.
     * @param outputFile - путь к выходному конфигурационному файлу YAML.
     * @param nc - количество классов (по умолчанию 1).
     * @throws - std::runtime_error если файл не может быть записан.
     */
    void YOLOv5GenerateHyperparameters(YOLOv5Model model,
                                       unsigned int imgWidth,
                                       unsigned int imgHeight,
                                       const std::string& outputFile,
                                       unsigned int nc = 1);
}
