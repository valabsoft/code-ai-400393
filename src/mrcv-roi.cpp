#include "mrcv/mrcv.h"

namespace mrcv
{
    cv::Mat extractROI(const cv::Mat& image, const cv::Point& center, const cv::Size& roiSize) 
    {
        // ����� ������� ���� ROI
        int leftUpperCornerX = center.x - roiSize.width / 2;
        int leftUpperCornerY = center.y - roiSize.height / 2;
        // ���������, ����� ���������� ���� �� �������� �� ������� �����������
        leftUpperCornerX = std::max(0, leftUpperCornerX);
        leftUpperCornerY = std::max(0, leftUpperCornerY);
        // ������������ ������ � ������ ROI, ���� ROI ������� �� ������� �����������
        int width = roiSize.width;
        int height = roiSize.height;
        if (leftUpperCornerX + width > image.cols) {
            width = image.cols - leftUpperCornerX;
        }
        if (leftUpperCornerY + height > image.rows) {
            height = image.rows - leftUpperCornerY;
        }
        // ������� ������������� ROI
        cv::Rect roiRect(leftUpperCornerX, leftUpperCornerY, width, height);
        // �������� ROI �� ��������� �����������
        cv::Mat roi = image(roiRect).clone();
        return roi;
    }
}