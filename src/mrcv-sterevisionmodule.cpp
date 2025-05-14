#include "mrcv/mrcv-sterevisionmodule.h"
#include <mrcv/mrcv.h>
#include <opencv2/calib3d.hpp>

namespace mrcv {

    StereoVisionModule::StereoVisionModule(const std::string& model_path, const std::string& class_path,
        const std::string& weightsFile, const std::string& segmentor_weights, const std::string& stereo_params)
        : obj_detector_(new mrcv::ObjCourse(model_path, class_path)),
        stereo_params_path_(stereo_params) {
        // ������������� ������������
        segmentor_.Initialize(512, 320, { "background", "ship" }, "resnet34", weightsFile);
        segmentor_.LoadWeight(segmentor_weights);
    }

    StereoVisionModule::~StereoVisionModule() {
        delete obj_detector_;
        if (cap_left_.isOpened()) {
            cap_left_.release();
        }
        if (cap_right_.isOpened()) {
            cap_right_.release();
        }
    }

    bool StereoVisionModule::initializeStereoCamera(int left_camera_id, int right_camera_id) {
        if (!cap_left_.open(left_camera_id, cv::CAP_DSHOW)) {
            mrcv::writeLog("Can't open left camera ID = " + std::to_string(left_camera_id), mrcv::LOGTYPE::ERROR);
            return false;
        }
        if (!cap_right_.open(right_camera_id, cv::CAP_DSHOW)) {
            mrcv::writeLog("Can't open right camera ID = " + std::to_string(right_camera_id), mrcv::LOGTYPE::ERROR);
            return false;
        }
        frame_size_ = cv::Size(640, 480); // ����� ��������� ��� ������
        return true;
    }

    StereoVisionResult StereoVisionModule::processFrame() {
        StereoVisionResult result;
        cv::Mat left_frame, right_frame;
        cap_left_ >> left_frame;
        cap_right_ >> right_frame;

        if (left_frame.empty() || right_frame.empty()) {
            mrcv::writeLog("Empty frame received from stereo camera", mrcv::LOGTYPE::ERROR);
            return result;
        }

        // ���������� �������� ������
        result.raw_left_frame = left_frame.clone();
        result.raw_right_frame = right_frame.clone();

        // ������������� �����������
        cv::Mat processed_left = left_frame.clone();
        cv::Mat processed_right = right_frame.clone();
        std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> preprocess_methods = {
            mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_01_MEDIAN_FILTER,
            mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_10_LAB_CLAHE,
            mrcv::METOD_IMAGE_PERPROCESSIN::SHARPENING_02,
            mrcv::METOD_IMAGE_PERPROCESSIN::CORRECTION_GEOMETRIC_DEFORMATION
        };
        int state_left = mrcv::preprocessingImage(processed_left, preprocess_methods, stereo_params_path_);
        int state_right = mrcv::preprocessingImage(processed_right, preprocess_methods, stereo_params_path_);
        if (state_left != 0 || state_right != 0) {
            mrcv::writeLog("Preprocessing failed, state_left = " + std::to_string(state_left) +
                ", state_right = " + std::to_string(state_right), mrcv::LOGTYPE::ERROR);
        }
        result.preprocessed_left_frame = processed_left.clone();
        result.preprocessed_right_frame = processed_right.clone();

        // ��������� ����� �������������
        int minDisparity = 16;
        int numDisparities = 16 * 10;
        int blockSize = 15;
        double lambda = 5000.0;
        double sigma = 3;
        int colorMap = cv::COLORMAP_TURBO;
        mrcv::DISPARITY_TYPE disparityType = mrcv::DISPARITY_TYPE::ALL;

        mrcv::disparityMap(result.disparity_map, processed_left, processed_right,
            minDisparity, numDisparities, blockSize, lambda, sigma,
            disparityType, colorMap, true, true);
        if (result.disparity_map.empty()) {
            mrcv::writeLog("Failed to generate disparity map", mrcv::LOGTYPE::ERROR);
        }

        // �������� �������� (�� ����� �����)
        result.object_count = obj_detector_->getObjectCount(processed_left);
        result.object_course = obj_detector_->getObjectCourse(processed_left, frame_size_.width, frame_size_.height);

        // ����������� (�� ����� �����)
        cv::Mat segmentation_frame = processed_left.clone();
        segmentor_.Predict(segmentation_frame, "ship");
        result.segmentation_mask = segmentation_frame; // ��������������, ��� Predict ������������ ����

        // ���������� ����������
        last_result_ = result;

        return result;
    }

    cv::Mat StereoVisionModule::getDetectionFrame() const {
        cv::Mat detection_frame = last_result_.preprocessed_left_frame.clone();

        // ���� ��������� ������, ��������� ����� �����������
        if (last_result_.object_count > 0 && !last_result_.segmentation_mask.empty()) {
            // ������������, ��� segmentation_mask � �������� (��� �������������) �����
            cv::Mat mask;
            if (last_result_.segmentation_mask.channels() > 1) {
                // ���� ����� �������, ����������� � grayscale
                cv::cvtColor(last_result_.segmentation_mask, mask, cv::COLOR_BGR2GRAY);
            }
            else {
                mask = last_result_.segmentation_mask;
            }

            // ��������, ��� ����� ��������
            cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

            // ��������� ����� � ����� (��������� ������ ������� �������)
            detection_frame.setTo(cv::Scalar(0, 0, 0)); // ��������� ���� ����
            last_result_.preprocessed_left_frame.copyTo(detection_frame, mask);
        }

        // ������ ����� � ����������� �������� � ������
        std::string text = "Objects: " + std::to_string(last_result_.object_count) +
            ", Course: " + std::to_string(last_result_.object_course) + " deg";
        cv::putText(detection_frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(0, 255, 0), 2);

        return detection_frame;
    }

} // namespace mrcv