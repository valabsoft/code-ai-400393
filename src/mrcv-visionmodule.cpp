#include "mrcv/mrcv-visionmodule.h"
#include <mrcv/mrcv.h>

namespace mrcv {    
    VisionModule::VisionModule(const std::string& model_path, const std::string& class_path,
        const std::string& weightsFile, const std::string& segmentor_weights, const std::string& camera_params)
        : obj_detector_(new mrcv::ObjCourse(model_path, class_path)),
        camera_params_path_(camera_params){
        // Инициализация сегментатора
        segmentor_.Initialize(512, 320, { "background", "ship" }, "resnet34", weightsFile);
        segmentor_.LoadWeight(segmentor_weights);
    }

    VisionModule::~VisionModule() {
        delete obj_detector_;
        if (cap_.isOpened()) {
            cap_.release();
        }
    }

    bool VisionModule::initializeCamera(int camera_id) {
        if (!cap_.open(camera_id, cv::CAP_ANY)) {
            mrcv::writeLog("Can't open camera ID = " + std::to_string(camera_id), mrcv::LOGTYPE::ERROR);
            return false;
        }
        frame_size_ = cv::Size(640, 480);
        return true;
    }

    VisionResult VisionModule::processFrame() {
        VisionResult result;
        cv::Mat frame;
        cap_ >> frame;

        if (frame.empty()) {
            mrcv::writeLog("Empty frame received", mrcv::LOGTYPE::ERROR);
            return result;
        }

        // Сохранение исходного кадра
        result.raw_frame = frame.clone();

        // Предобработка изображения
        cv::Mat processed_frame = frame.clone();
        std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> preprocess_methods = {
            mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_01_MEDIAN_FILTER,
            mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_10_LAB_CLAHE,
            mrcv::METOD_IMAGE_PERPROCESSIN::SHARPENING_02,
            mrcv::METOD_IMAGE_PERPROCESSIN::CORRECTION_GEOMETRIC_DEFORMATION
        };
        int state = mrcv::preprocessingImage(processed_frame, preprocess_methods, camera_params_path_);
        if (state != 0) {
            mrcv::writeLog("Preprocessing failed, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
        }
        result.preprocessed_frame = processed_frame.clone();

        // Детекция объектов
        result.object_count = obj_detector_->getObjectCount(processed_frame);
        result.object_course = obj_detector_->getObjectCourse(processed_frame, frame_size_.width, frame_size_.height);

        // Сегментация
        cv::Mat segmentation_frame = processed_frame.clone();
        segmentor_.Predict(segmentation_frame, "ship");
        result.segmentation_mask = segmentation_frame; // Предполагается, что Predict модифицирует кадр

        // Сохранение результата
        last_result_ = result;

        return result;
    }

    cv::Mat VisionModule::getDetectionFrame() const {
        cv::Mat detection_frame = last_result_.preprocessed_frame.clone();

        // Если обнаружен объект, применяем маску сегментации
        if (last_result_.object_count > 0 && !last_result_.segmentation_mask.empty()) {
            // Предполагаем, что segmentation_mask — бинарная (или одноканальная) маска
            cv::Mat mask;
            if (last_result_.segmentation_mask.channels() > 1) {
                // Если маска цветная, преобразуем в grayscale
                cv::cvtColor(last_result_.segmentation_mask, mask, cv::COLOR_BGR2GRAY);
            }
            else {
                mask = last_result_.segmentation_mask;
            }

            // Убедимся, что маска бинарная
            cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

            // Применяем маску к кадру (оставляем только область объекта)
            detection_frame.setTo(cv::Scalar(0, 0, 0)); // Затемняем весь кадр
            last_result_.preprocessed_frame.copyTo(detection_frame, mask);
        }

        // Рисуем текст с количеством объектов и курсом
        std::string text = "Objects: " + std::to_string(last_result_.object_count) +
            ", Course: " + std::to_string(last_result_.object_course) + " deg";
        cv::putText(detection_frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(0, 255, 0), 2);

        return detection_frame;
    }

} // namespace mrcv