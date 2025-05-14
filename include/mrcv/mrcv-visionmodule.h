#pragma once
#include <mrcv/mrcv.h>
#include <filesystem>

namespace mrcv {

    struct VisionResult {
        int object_count;               // Количество обнаруженных объектов
        float object_course;            // Курс на объект (градусы)
        cv::Mat segmentation_mask;      // Маска сегментации
        std::vector<cv::Rect> bboxes;   // Bounding boxes объектов
        cv::Mat raw_frame;              // Исходный кадр
        cv::Mat preprocessed_frame;     // Кадр после предобработки
    };

    class VisionModule {
    public:
        VisionModule(const std::string& model_path, const std::string& class_path,
            const std::string& weightsFile, const std::string& segmentor_weights, const std::string& camera_params);
        ~VisionModule();

        // Инициализация камеры
        bool initializeCamera(int camera_id);

        // Обработка одного кадра
        VisionResult processFrame();

        // Геттеры для получения изображений
        cv::Mat getRawFrame() const { return last_result_.raw_frame; }
        cv::Mat getPreprocessedFrame() const { return last_result_.preprocessed_frame; }
        cv::Mat getSegmentationMask() const { return last_result_.segmentation_mask; }
        cv::Mat getDetectionFrame() const;

    private:
        cv::VideoCapture cap_;          // Камера
        mrcv::ObjCourse* obj_detector_; // Детектор объектов
        mrcv::Segmentor segmentor_;     // Сегментатор
        std::string camera_params_path_; // Путь к параметрам камеры
        cv::Size frame_size_;           // Размер кадра
        VisionResult last_result_;      // Последний результат обработки
    };

} // namespace mrcv_vision