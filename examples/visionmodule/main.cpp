#include "mrcv/mrcv-visionmodule.h"
#include <iostream>

int main() {
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "files";
    
    // Пути к файлам 
    std::filesystem::path model_path = path / "ship.onnx";
       if (!std::filesystem::exists(model_path)) {
           std::cerr << "Model file does not exist: " << model_path << std::endl;
           return -1;
       }

       std::filesystem::path class_path =  path / "ship.names";
       if (!std::filesystem::exists(class_path)) {
           std::cerr << "Class file does not exist: " << class_path << std::endl;
           return -1;
       }

       std::filesystem::path segmentor_weights =  path / "weights" / "segmentor.pt";
       if (!std::filesystem::exists(segmentor_weights)) {
           std::cerr << "Segmentor weights file does not exist: " << segmentor_weights << std::endl;
           return -1;
       }

       std::filesystem::path weightsFile =  path / "weights" / "resnet34.pt";
       if (!std::filesystem::exists(weightsFile)) {
           std::cerr << "Segmentor weights file does not exist: " << segmentor_weights << std::endl;
           return -1;
       }

       std::filesystem::path camera_params =  path / "camCalibrarion.xml";
       if (!std::filesystem::exists(camera_params)) {
           std::cerr << "Camera parameters file does not exist: " << camera_params << std::endl;
           return -1;
       }
    // Инициализация модуля
    mrcv::VisionModule vision_module(model_path.u8string(), class_path.u8string(),
        weightsFile.u8string(), segmentor_weights.u8string(), camera_params.u8string());
    // Инициализация камеры
    if (!vision_module.initializeCamera(0)) {
        std::cerr << "Failed to initialize camera" << std::endl;
        return -1;
    }

    // Создание окон для отображения
    cv::namedWindow("Raw Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Preprocessed Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Segmentation Mask", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Detected Objects", cv::WINDOW_AUTOSIZE);

    // Основной цикл обработки
    while (true) {
        auto result = vision_module.processFrame();
        std::cout << "Objects detected: " << result.object_count
            << ", Course: " << result.object_course << " degrees" << std::endl;

        // Отображение всех этапов обработки
        cv::Mat raw_frame = vision_module.getRawFrame();
        cv::Mat preprocessed_frame = vision_module.getPreprocessedFrame();
        cv::Mat segmentation_mask = vision_module.getSegmentationMask();
        cv::Mat detection_frame = vision_module.getDetectionFrame();

        if (!raw_frame.empty()) {
            cv::imshow("Raw Frame", raw_frame);
        }
        if (!preprocessed_frame.empty()) {
            cv::imshow("Preprocessed Frame", preprocessed_frame);
        }
        if (!segmentation_mask.empty()) {
            cv::imshow("Segmentation Mask", segmentation_mask);
        }
        if (!detection_frame.empty()) {
            cv::imshow("Detected Objects", detection_frame);
        }

        // Выход по клавише ESC
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // Закрытие окон
    cv::destroyAllWindows();

    return 0;
}
