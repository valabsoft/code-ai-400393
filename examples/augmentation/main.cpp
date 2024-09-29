#include "mrcv/mrcv.h"

int main()
{
    //=========================================
    // Загрузка изображений
    //=========================================
    std::vector<cv::Mat> inputImagesAugmetation(10);

    inputImagesAugmetation[0] = cv::imread("files\\img0.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[1] = cv::imread("files\\img1.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[2] = cv::imread("files\\img2.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[3] = cv::imread("files\\img3.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[4] = cv::imread("files\\img4.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[5] = cv::imread("files\\img5.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[6] = cv::imread("files\\img6.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[7] = cv::imread("files\\img7.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[8] = cv::imread("files\\img8.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[9] = cv::imread("files\\img9.jpg", cv::IMREAD_COLOR);

    // Проверка на успешную загрузку изображений
    for (size_t i = 0; i < inputImagesAugmetation.size(); i++) {
        if (inputImagesAugmetation[i].empty()) {
            std::cerr << "Error: Could not load image at index " << i << std::endl;
            return -1;
        }
    }

    // Создаем копию входных изображений, чтобы сохранить оригиналы нетронутыми
    std::vector<cv::Mat> inputImagesCopy = inputImagesAugmetation;

    // Вектор для хранения выходных изображений
    std::vector<cv::Mat> outputImagesAugmetation;

    // Методы аугментации
    std::vector<mrcv::augmetationMethodFunctions> augmetationMethod = {
        mrcv::augmetationMethodFunctions::rotateImage90,
        mrcv::augmetationMethodFunctions::flipHorizontal,
        mrcv::augmetationMethodFunctions::flipVertical,
        mrcv::augmetationMethodFunctions::rotateImage45,
        mrcv::augmetationMethodFunctions::rotateImage315,
        mrcv::augmetationMethodFunctions::rotateImage270,
        mrcv::augmetationMethodFunctions::flipHorizontalandVertical,
    };

    // Вызов функции аугментации
    int state = mrcv::augmetation(inputImagesAugmetation, outputImagesAugmetation, augmetationMethod);
    if (state != 0) {
        std::cerr << "Error: Augmentation failed with code " << state << std::endl;
        return -1;
    }

    return 0;
}
