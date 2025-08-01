#include "mrcv/mrcv-sterevisionmodule.h"
#include <iostream>

int main() {
    // Пути к файлам (настрой под свои)
    // Пути к файлам 
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "files";
    
    std::filesystem::path model_path = path / "ship.onnx";
    if (!std::filesystem::exists(model_path)) {
        std::cerr << "Model file does not exist: " << model_path << std::endl;
        return -1;
    }

    std::filesystem::path class_path = path / "ship.names";
    if (!std::filesystem::exists(class_path)) {
        std::cerr << "Class file does not exist: " << class_path << std::endl;
        return -1;
    }

    std::filesystem::path segmentor_weights = path / "weights/segmentor.pt";
    if (!std::filesystem::exists(segmentor_weights)) {
        std::cerr << "Segmentor weights file does not exist: " << segmentor_weights << std::endl;
        return -1;
    }

    std::filesystem::path weightsFile = path / "weights" / "resnet34.pt";
    if (!std::filesystem::exists(weightsFile)) {
        std::cerr << "Segmentor weights file does not exist: " << segmentor_weights << std::endl;
        return -1;
    }

    std::filesystem::path stereo_params = path / "camCalibrarion.xml";
    if (!std::filesystem::exists(stereo_params)) {
        std::cerr << "Camera parameters file does not exist: " << stereo_params << std::endl;
        return -1;
    }

    // Инициализация модуля
    mrcv::StereoVisionModule vision_module(model_path.u8string(), class_path.u8string(),
        weightsFile.u8string(), segmentor_weights.u8string(), stereo_params.u8string());

    // Инициализация стереокамеры
    if (!vision_module.initializeStereoCamera(1, 0)) {
        std::cerr << "Failed to initialize stereo camera" << std::endl;
        return -1;
    }

    // Создание окон для отображения
    cv::namedWindow("Raw Left Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Raw Right Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Preprocessed Left Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Preprocessed Right Frame", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Disparity Map", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Segmentation Mask", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Detected Objects", cv::WINDOW_AUTOSIZE);

    // Основной цикл обработки
    while (true) {
        auto result = vision_module.processFrame();
        std::cout << "Objects detected: " << result.object_count
            << ", Course: " << result.object_course << " degrees" << std::endl;

        // Отображение всех этапов обработки
        cv::Mat raw_left = vision_module.getRawLeftFrame();
        cv::Mat raw_right = vision_module.getRawRightFrame();
        cv::Mat preprocessed_left = vision_module.getPreprocessedLeftFrame();
        cv::Mat preprocessed_right = vision_module.getPreprocessedRightFrame();
        cv::Mat disparity_map = vision_module.getDisparityMap();
        cv::Mat segmentation_mask = vision_module.getSegmentationMask();
        cv::Mat detection_frame = vision_module.getDetectionFrame();

        if (!raw_left.empty()) {
            cv::imshow("Raw Left Frame", raw_left);
        }
        if (!raw_right.empty()) {
            cv::imshow("Raw Right Frame", raw_right);
        }
        if (!preprocessed_left.empty()) {
            cv::imshow("Preprocessed Left Frame", preprocessed_left);
        }
        if (!preprocessed_right.empty()) {
            cv::imshow("Preprocessed Right Frame", preprocessed_right);
        }
        if (!disparity_map.empty()) {
            cv::imshow("Disparity Map", disparity_map);
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
