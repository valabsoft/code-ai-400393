#pragma once
#include <mrcv/mrcv.h>
#include <filesystem>

namespace mrcv {

    struct StereoVisionResult {
        int object_count;               // ���������� ������������ ��������
        float object_course;            // ���� �� ������ (�������)
        cv::Mat segmentation_mask;      // ����� ����������� (��� ������ �����)
        cv::Mat raw_left_frame;         // �������� ����� ����
        cv::Mat raw_right_frame;        // �������� ������ ����
        cv::Mat preprocessed_left_frame;// ���������������� ����� ����
        cv::Mat preprocessed_right_frame;// ���������������� ������ ����
        cv::Mat disparity_map;          // ����� �������������
    };

    class StereoVisionModule {
    public:
        StereoVisionModule(const std::string& model_path, const std::string& class_path,
            const std::string& weightsFile, const std::string& segmentor_weights, const std::string& stereo_params);
        ~StereoVisionModule();

        // ������������� ������������
        bool initializeStereoCamera(int left_camera_id, int right_camera_id);

        // ��������� ������ �����
        StereoVisionResult processFrame();

        // ������� ��� ��������� �����������
        cv::Mat getRawLeftFrame() const { return last_result_.raw_left_frame; }
        cv::Mat getRawRightFrame() const { return last_result_.raw_right_frame; }
        cv::Mat getPreprocessedLeftFrame() const { return last_result_.preprocessed_left_frame; }
        cv::Mat getPreprocessedRightFrame() const { return last_result_.preprocessed_right_frame; }
        cv::Mat getSegmentationMask() const { return last_result_.segmentation_mask; }
        cv::Mat getDisparityMap() const { return last_result_.disparity_map; }
        cv::Mat getDetectionFrame() const;

    private:
        cv::VideoCapture cap_left_;     // ����� ������
        cv::VideoCapture cap_right_;    // ������ ������
        mrcv::ObjCourse* obj_detector_; // �������� ��������
        mrcv::Segmentor segmentor_;     // �����������
        std::string stereo_params_path_; // ���� � ���������� ����������������
        cv::Size frame_size_;           // ������ �����
        StereoVisionResult last_result_; // ��������� ��������� ���������
    };

} // namespace mrcv_vision