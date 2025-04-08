#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <map>
#include <filesystem>
#include <random>
#ifndef YAML_CPP_STATIC_DEFINE
#define YAML_CPP_STATIC_DEFINE
#endif
#include <yaml-cpp/yaml.h>

#include "mrcv/mrcv-yolov5.h"

namespace mrcv
{
    // Структура для хранения информации о разметке
    struct BoundingBox
    {
        int classId;
        cv::Rect box;
        bool isSelected = false; // флаг выделения рамки
    };

    // Глобальные переменные для хранения состояния
    std::vector<BoundingBox> boxes;
    int currentClassId = 0;
    bool drawing = false;
    cv::Rect currentBox;
    cv::Point startPoint;
    bool isEditing = false;     // флаг для режима редактирования
    int selectedBoxIndex = -1;  // индекс выбранной рамки
    bool resizing = false;      // флаг для изменения размера рамки
    cv::Point resizeStartPoint; // начальная точка для изменения размера

    // Генератор случайных цветов для каждого класса
    std::map<int, cv::Scalar> classColors;

    // Функция для генерации случайного цвета
    cv::Scalar getRandomColor()
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<int> dis(0, 255);
        return cv::Scalar(dis(gen), dis(gen), dis(gen), 128);
    }

    // Функция для отображения информации о классах и количестве объектов
    void drawClassInfo(cv::Mat &infoPanel, const std::map<int, int> &classCounts)
    {
        infoPanel.setTo(cv::Scalar(50, 50, 50)); // очищаем панель
        int yOffset = 30;

        // Краткая инструкция
        std::vector<std::string> instructions = {"Instructions:",
                                                "1. Draw: Left-click & drag",
                                                "2. Edit: Press 'e'",
                                                "3. Change class: 0-9 keys",
                                                "4. Delete: Right-click on box",
                                                "5. Exit: ESC"};

        // Отображаем инструкцию с переносом строк
        for (const auto &line : instructions)
        {
            putText(infoPanel, line, cv::Point(10, yOffset),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            yOffset += 20;
        }

        yOffset += 20;

        // Информация о классах
        for (const auto &[classId, count] : classCounts)
        {
            std::string text = "Class " + std::to_string(classId) + ": " +
                               std::to_string(count) + " objects";
            putText(infoPanel, text, cv::Point(10, yOffset),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, classColors[classId], 2);
            yOffset += 30;
        }

        // Отображаем текущий classId и режим внизу панели
        std::string statusText = "Class: " + std::to_string(currentClassId) +
                                 " | Mode: " + (isEditing ? "Edit" : "Draw");
        int baseline = 0;
        cv::Size textSize =
            getTextSize(statusText, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        cv::Point textOrg(10, infoPanel.rows - 10);
        putText(infoPanel, statusText, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
    }

    // Функция для проверки, находится ли точка рядом с углом или краем рамки
    bool isNearEdge(const cv::Point &pt, const cv::Rect &box, int threshold = 10)
    {
        // Проверяем углы
        if (abs(pt.x - box.x) < threshold && abs(pt.y - box.y) < threshold)
            return true; // Левый верхний
        if (abs(pt.x - (box.x + box.width)) < threshold &&
            abs(pt.y - box.y) < threshold)
            return true; // Правый верхний
        if (abs(pt.x - box.x) < threshold &&
            abs(pt.y - (box.y + box.height)) < threshold)
            return true; // Левый нижний
        if (abs(pt.x - (box.x + box.width)) < threshold &&
            abs(pt.y - (box.y + box.height)) < threshold)
            return true; // Правый нижний

        // Проверяем края
        if (abs(pt.x - box.x) < threshold && pt.y >= box.y &&
            pt.y <= box.y + box.height)
            return true; // Левый край
        if (abs(pt.x - (box.x + box.width)) < threshold && pt.y >= box.y &&
            pt.y <= box.y + box.height)
            return true; // Правый край
        if (abs(pt.y - box.y) < threshold && pt.x >= box.x &&
            pt.x <= box.x + box.width)
            return true; // Верхний край
        if (abs(pt.y - (box.y + box.height)) < threshold && pt.x >= box.x &&
            pt.x <= box.x + box.width)
            return true; // Нижний край

        return false;
    }

    // Функция для обработки событий мыши
    void onMouse(int event, int x, int y, int flags, void *userdata)
    {
        cv::Mat &image = *((cv::Mat *)userdata);

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            if (isEditing)
            {
                // Проверяем, была ли выбрана существующая рамка
                for (size_t i = 0; i < boxes.size(); ++i)
                {
                    if (boxes[i].box.contains(cv::Point(x, y)) ||
                        isNearEdge(cv::Point(x, y), boxes[i].box))
                    {
                        selectedBoxIndex = i;
                        boxes[i].isSelected = true;
                        resizeStartPoint = cv::Point(x, y);
                        resizing = isNearEdge(cv::Point(x, y), boxes[i].box);
                        break;
                    }
                }
            }
            else
            {
                // Начинаем рисовать новую рамку
                drawing = true;
                startPoint = cv::Point(x, y);
                currentBox = cv::Rect(x, y, 0, 0);
            }
        }
        else if (event == cv::EVENT_RBUTTONDOWN)
        {
            // Удаление рамки правой кнопкой мыши
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                if (boxes[i].box.contains(cv::Point(x, y)))
                {
                    boxes.erase(boxes.begin() + i);
                    break;
                }
            }
        }
        else if (event == cv::EVENT_MOUSEMOVE)
        {
            if (drawing)
            {
                // Обновляем размер текущей рамки (в любом направлении)
                currentBox.x = std::min(startPoint.x, x);
                currentBox.y = std::min(startPoint.y, y);
                currentBox.width = abs(x - startPoint.x);
                currentBox.height = abs(y - startPoint.y);
            }
            else if (isEditing && selectedBoxIndex != -1)
            {
                if (resizing)
                {
                    // Изменяем размер рамки
                    cv::Rect &box = boxes[selectedBoxIndex].box;
                    int dx = x - resizeStartPoint.x;
                    int dy = y - resizeStartPoint.y;

                    // Изменяем размер в зависимости от того, за какой край тянем
                    if (abs(x - box.x) < 10)
                        box.x += dx; // Левый край
                    if (abs(y - box.y) < 10)
                        box.y += dy; // Верхний край
                    if (abs(x - (box.x + box.width)) < 10)
                        box.width += dx; // Правый край
                    if (abs(y - (box.y + box.height)) < 10)
                        box.height += dy; // Нижний край

                    resizeStartPoint = cv::Point(x, y);
                }
                else
                {
                    // Перемещаем выбранную рамку
                    boxes[selectedBoxIndex].box.x =
                        x - boxes[selectedBoxIndex].box.width / 2;
                    boxes[selectedBoxIndex].box.y =
                        y - boxes[selectedBoxIndex].box.height / 2;
                }
            }
        }
        else if (event == cv::EVENT_LBUTTONUP)
        {
            if (drawing)
            {
                // Завершаем рисование новой рамки
                drawing = false;
                boxes.push_back({currentClassId, currentBox});
                if (classColors.find(currentClassId) == classColors.end())
                {
                    classColors[currentClassId] = getRandomColor();
                }
            }
            else if (isEditing && (selectedBoxIndex != -1))
            {
                // Снимаем выделение с рамки
                boxes[selectedBoxIndex].isSelected = false;
                selectedBoxIndex = -1;
                resizing = false;
            }
        }
    }

    // Функция для интерактивной разметки
    void interactiveMarking(cv::Mat &image)
    {
        namedWindow("Marking", cv::WINDOW_AUTOSIZE);

        // Создаем панель для текстовой информации
        cv::Mat infoPanel = cv::Mat::zeros(cv::Size(400, image.rows), CV_8UC3);
        infoPanel.setTo(cv::Scalar(50, 50, 50));

        setMouseCallback("Marking", onMouse, &image);

        while (true)
        {
            cv::Mat displayImage = image.clone();

            // Подсчитываем количество объектов для каждого класса
            std::map<int, int> classCounts;
            for (const auto &box : boxes)
            {
                classCounts[box.classId]++;
            }

            // Отображаем информацию о классах и количестве объектов
            drawClassInfo(infoPanel, classCounts);

            // Отображаем все размеченные объекты
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                cv::Scalar color = classColors[boxes[i].classId];
                cv::Mat overlay = displayImage.clone();
                cv::rectangle(overlay, boxes[i].box, color, -1);
                cv::addWeighted(overlay, 0.3, displayImage, 0.7, 0,
                                displayImage);
                cv::rectangle(displayImage, boxes[i].box, color, 2);
                cv::putText(displayImage, std::to_string(boxes[i].classId),
                            cv::Point(boxes[i].box.x, boxes[i].box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }

            // Отображаем текущий прямоугольник, если он рисуется
            if (drawing)
            {
                cv::rectangle(displayImage, currentBox, cv::Scalar(0, 0, 255), 2);
            }

            // Объединяем изображение и панель информации
            cv::Mat combinedImage;
            hconcat(displayImage, infoPanel, combinedImage);
            imshow("Marking", combinedImage);

            char key = cv::waitKey(1);
            if (key == 27)
            { // ESC для завершения разметки
                break;
            }
            else if (key >= '0' && key <= '9')
            { // Изменение classId через цифровые клавиши
                currentClassId = key - '0';
            }
            else if (key == 'e')
            { // Переключение режима редактирования
                isEditing = !isEditing;
            }
        }

        cv::destroyWindow("Marking");
    }

    // Функция для сохранения разметки в YOLO-формате
    void saveYoloFormat(const std::string &filename,
                        const std::vector<BoundingBox> &boxes,
                        const cv::Size &imageSize)
    {
        std::ofstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Ошибка открытия файла для записи!" << std::endl;
            return;
        }

        for (const auto &box : boxes)
        {
            float x_center = (box.box.x + box.box.width / 2.0) / imageSize.width;
            float y_center = (box.box.y + box.box.height / 2.0) / imageSize.height;
            float width = box.box.width / (float)imageSize.width;
            float height = box.box.height / (float)imageSize.height;

            file << box.classId << " " << x_center << " " << y_center << " "
                 << width << " " << height << std::endl;
        }

        file.close();
    }

    void YOLOv5LabelerProcessing(const std::string &inputDir,
                                 const std::string &outputDir)
    {
        if (!std::filesystem::exists(outputDir))
        {
            std::filesystem::create_directory(outputDir);
        }

        for (const auto &entry : std::filesystem::directory_iterator(inputDir))
        {
            if (entry.is_regular_file() && (entry.path().extension() == ".jpg" ||
                                            entry.path().extension() == ".png"))
            {
                cv::Mat image = cv::imread(entry.path().string());
                if (image.empty())
                {
                    std::cerr << "Ошибка загрузки изображения: " << entry.path()
                              << std::endl;
                    continue;
                }

                boxes.clear();
                classColors.clear();
                interactiveMarking(image);

                std::string outputFilename =
                    (std::filesystem::path(outputDir) / entry.path().stem()).string() +
                    ".txt";
                saveYoloFormat(outputFilename, boxes, image.size());
            }
        }
    }
}

namespace mrcv
{
    void YOLOv5GenerateConfig(YOLOv5Model model,
                              const std::string &outputFile,
                              unsigned int nc)
    {
        YAML::Node config;

        config["nc"] = nc;

        switch (model)
        {
            case YOLOv5Model::YOLOv5n:
                config["depth_multiple"] = 0.33;
                config["width_multiple"] = 0.25;
                break;

            case YOLOv5Model::YOLOv5s:
                config["depth_multiple"] = 0.33;
                config["width_multiple"] = 0.50;
                break;

            case YOLOv5Model::YOLOv5m:
                config["depth_multiple"] = 0.67;
                config["width_multiple"] = 0.75;
                break;

            case YOLOv5Model::YOLOv5l:
                config["depth_multiple"] = 1.0;
                config["width_multiple"] = 1.0;
                break;

            case YOLOv5Model::YOLOv5x:
                config["depth_multiple"] = 1.33;
                config["width_multiple"] = 1.25;
                break;

            default:
                throw std::invalid_argument("Unsupported YOLOv5 model type!");
        }

        config["anchors"] = YAML::Node(YAML::NodeType::Sequence);
        config["anchors"].push_back(YAML::Load("[10, 13, 16, 30, 33, 23]"));
        config["anchors"].push_back(YAML::Load("[30, 61, 62, 45, 59, 119]"));
        config["anchors"].push_back(YAML::Load("[116, 90, 156, 198, 373, 326]"));

        std::string backbone;
        backbone.append("[");
        backbone.append("[-1, 1, Conv, [64, 6, 2, 2]],");
        backbone.append("[-1, 1, Conv, [128, 3, 2]],");
        backbone.append("[-1, 3, C3, [128]],");
        backbone.append("[-1, 1, Conv, [256, 3, 2]],");
        backbone.append("[-1, 6, C3, [256]],");
        backbone.append("[-1, 1, Conv, [512, 3, 2]],");
        backbone.append("[-1, 9, C3, [512]],");
        backbone.append("[-1, 1, Conv, [1024, 3, 2]],");
        backbone.append("[-1, 3, C3, [1024]],");
        backbone.append("[-1, 1, SPPF, [1024, 5]],");
        backbone.append("]");
        config["backbone"] = YAML::Load(backbone);

        std::string head;
        head.append("[");
        head.append("[-1, 1, Conv, [512, 1, 1]],");
        head.append("[-1, 1, nn.Upsample, [None, 2, 'nearest']],");
        head.append("[[-1, 6], 1, Concat, [1]],");
        head.append("[-1, 3, C3, [512, False]],");

        head.append("[-1, 1, Conv, [256, 1, 1]],");
        head.append("[-1, 1, nn.Upsample, [None, 2, 'nearest']],");
        head.append("[[-1, 4], 1, Concat, [1]],");
        head.append("[-1, 3, C3, [256, False]],");

        head.append("[-1, 1, Conv, [256, 3, 2]],");
        head.append("[[-1, 14], 1, Concat, [1]],");
        head.append("[-1, 3, C3, [512, False]],");

        head.append("[-1, 1, Conv, [512, 3, 2]],");
        head.append("[[-1, 10], 1, Concat, [1]],");
        head.append("[-1, 3, C3, [1024, False]],");

        head.append("[[17, 20, 23], 1, Detect, [nc, anchors]],");
        head.append("]");
        config["head"] = YAML::Load(head);

        std::ofstream fout(outputFile);
        if (!fout.is_open())
        {
            throw std::runtime_error("Unable to open file for writing: " +
                                     outputFile);
        }

        fout << config;
        fout.close();
    }

    void YOLOv5GenerateHyperparameters(YOLOv5Model model,
                                       unsigned int imgWidth,
                                       unsigned int imgHeight,
                                       const std::string &outputFile,
                                       unsigned int nc)
    {
        if ((imgWidth <= 0) || (imgHeight <= 0))
        {
            throw std::invalid_argument("Image dimensions must be positive!");
        }

        YAML::Node config;

        // Базовые гиперпараметры
        double weight_decay = 0.0005;
        double box_gain = 0.05;
        double cls_gain = 0.5;
        double cls_pw = 1.0;
        double obj_gain = 1.0;
        double obj_pw = 1.0;
        double anchor_threshold = 4.0;
        double fl_gamma = 0.0;

        // Настройка гиперпараметров на основе архитектуры
        switch (model)
        {
            case YOLOv5Model::YOLOv5n:
                weight_decay *= 1.0;
                box_gain *= 1.0;
                cls_gain *= 0.9;
                fl_gamma = 0.1;
                break;

            case YOLOv5Model::YOLOv5s:
                weight_decay *= 1.1;
                box_gain *= 1.1;
                cls_gain *= 1.0;
                fl_gamma = 0.2;
                break;

            case YOLOv5Model::YOLOv5m:
                weight_decay *= 1.2;
                box_gain *= 1.2;
                cls_gain *= 1.1;
                fl_gamma = 0.3;
                break;

            case YOLOv5Model::YOLOv5l:
                weight_decay *= 1.3;
                box_gain *= 1.3;
                cls_gain *= 1.2;
                fl_gamma = 0.4;
                break;

            case YOLOv5Model::YOLOv5x:
                weight_decay *= 1.4;
                box_gain *= 1.4;
                cls_gain *= 1.3;
                fl_gamma = 0.5;
                break;

            default:
                throw std::invalid_argument("Unsupported YOLOv5 model type!");
        }

        // Дополнительные настройки на основе количества классов
        if (nc > 80)
        {
            weight_decay += 0.0001 * (nc - 80);
        }

        // Настройка box_gain на основе разрешения изображения
        double resolution_scale =
            static_cast<double>(imgWidth * imgHeight) / (640 * 640);
        box_gain *= std::sqrt(resolution_scale);

        // Настройка cls_gain на основе количества классов
        cls_gain = 0.5 + 0.005 * nc;

        // Настройка fl_gamma на основе разрешения изображения
        // Добавляем 1, чтобы избежать log2(0)
        fl_gamma += 0.1 * std::log2(resolution_scale + 1);

        config["weight_decay"] = weight_decay;
        config["box"] = box_gain;
        config["cls"] = cls_gain;
        config["cls_pw"] = cls_pw;
        config["obj"] = obj_gain;
        config["obj_pw"] = obj_pw;
        config["anchor_t"] = anchor_threshold;
        config["fl_gamma"] = fl_gamma;

        std::ofstream fout(outputFile);
        if (!fout.is_open())
        {
            throw std::runtime_error("Unable to open file for writing: " +
                                     outputFile);
        }
        fout << config;
        fout.close();
    }
}
