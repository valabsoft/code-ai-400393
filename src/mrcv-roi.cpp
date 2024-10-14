#include "mrcv/mrcv.h"

namespace mrcv
{
    cv::Mat extractROI(const cv::Mat& image, const cv::Point& center, const cv::Size& roiSize) 
    {
        // Левый верхний угол ROI
        int leftUpperCornerX = center.x - roiSize.width / 2;
        int leftUpperCornerY = center.y - roiSize.height / 2;
        // Проверяем, чтобы координаты угла не выходили за границы изображения
        leftUpperCornerX = std::max(0, leftUpperCornerX);
        leftUpperCornerY = std::max(0, leftUpperCornerY);
        // Корректируем ширину и высоту ROI, если ROI выходит за пределы изображения
        int width = roiSize.width;
        int height = roiSize.height;
        if (leftUpperCornerX + width > image.cols) {
            width = image.cols - leftUpperCornerX;
        }
        if (leftUpperCornerY + height > image.rows) {
            height = image.rows - leftUpperCornerY;
        }
        // Создаем прямоугольник ROI
        cv::Rect roiRect(leftUpperCornerX, leftUpperCornerY, width, height);
        // Вырезаем ROI из исходного изображения
        cv::Mat roi = image(roiRect).clone();
        return roi;
    }
}