#pragma once
#include <mrcv/mrcv.h>
#include <filesystem>

namespace mrcv {

    struct VisionResult {
        int object_count;               // ���������� ������������ ��������
        float object_course;            // ���� �� ������ (�������)
        cv::Mat segmentation_mask;      // ����� �����������
        std::vector<cv::Rect> bboxes;   // Bounding boxes ��������
        cv::Mat raw_frame;              // �������� ����
        cv::Mat preprocessed_frame;     // ���� ����� �������������
    };

    class VisionModule {
    public:
        VisionModule(const std::string& model_path, const std::string& class_path,
            const std::string& weightsFile, const std::string& segmentor_weights, const std::string& camera_params);
        ~VisionModule();

        // ������������� ������
        bool initializeCamera(int camera_id);

        // ��������� ������ �����
        VisionResult processFrame();

        // ������� ��� ��������� �����������
        cv::Mat getRawFrame() const { return last_result_.raw_frame; }
        cv::Mat getPreprocessedFrame() const { return last_result_.preprocessed_frame; }
        cv::Mat getSegmentationMask() const { return last_result_.segmentation_mask; }
        cv::Mat getDetectionFrame() const;

    private:
        cv::VideoCapture cap_;          // ������
        mrcv::ObjCourse* obj_detector_; // �������� ��������
        mrcv::Segmentor segmentor_;     // �����������
        std::string camera_params_path_; // ���� � ���������� ������
        cv::Size frame_size_;           // ������ �����
        VisionResult last_result_;      // ��������� ��������� ���������
    };

} // namespace mrcv_vision