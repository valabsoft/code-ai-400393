#include <fstream>
#include <stdexcept>
#ifndef YAML_CPP_STATIC_DEFINE
#define YAML_CPP_STATIC_DEFINE
#endif
#include <yaml-cpp/yaml.h>

#include "mrcv/mrcv-yolov5.h"

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
