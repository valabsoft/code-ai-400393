#pragma once
#include <mrcv/mrcv.h>
#include <filesystem>

namespace mrcv {

    struct StereoVisionResult {
        int object_count;               // Количество обнаруженных объектов
        float object_course;            // Курс на объект (градусы)
        cv::Mat segmentation_mask;      // Маска сегментации (для левого кадра)
        cv::Mat raw_left_frame;         // Исходный левый кадр
        cv::Mat raw_right_frame;        // Исходный правый кадр
        cv::Mat preprocessed_left_frame;// Предобработанный левый кадр
        cv::Mat preprocessed_right_frame;// Предобработанный правый кадр
        cv::Mat disparity_map;          // Карта диспаратности
    };

    class StereoVisionModule {
    public:
        StereoVisionModule(const std::string& model_path, const std::string& class_path,
            const std::string& weightsFile, const std::string& segmentor_weights, const std::string& stereo_params);
        ~StereoVisionModule();

        // Инициализация стереокамеры
        bool initializeStereoCamera(int left_camera_id, int right_camera_id);

        // Обработка одного кадра
        StereoVisionResult processFrame();

        // Геттеры для получения изображений
        cv::Mat getRawLeftFrame() const { return last_result_.raw_left_frame; }
        cv::Mat getRawRightFrame() const { return last_result_.raw_right_frame; }
        cv::Mat getPreprocessedLeftFrame() const { return last_result_.preprocessed_left_frame; }
        cv::Mat getPreprocessedRightFrame() const { return last_result_.preprocessed_right_frame; }
        cv::Mat getSegmentationMask() const { return last_result_.segmentation_mask; }
        cv::Mat getDisparityMap() const { return last_result_.disparity_map; }
        cv::Mat getDetectionFrame() const;

    private:
        cv::VideoCapture cap_left_;     // Левая камера
        cv::VideoCapture cap_right_;    // Правая камера
        mrcv::ObjCourse* obj_detector_; // Детектор объектов
        mrcv::Segmentor segmentor_;     // Сегментатор
        std::string stereo_params_path_; // Путь к параметрам стереокалибровки
        cv::Size frame_size_;           // Размер кадра
        StereoVisionResult last_result_; // Последний результат обработки
    };

} // namespace mrcv_vision