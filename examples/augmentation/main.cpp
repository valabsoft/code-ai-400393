#include "mrcv/mrcv.h"

#ifdef _WIN32
    #define FILE_SEPARATOR "\\"
#else
    #define FILE_SEPARATOR "/"
#endif

int main()
{
    //=========================================
    // Загрузка изображений
    //=========================================
    std::vector<cv::Mat> inputImagesAugmetation(10);

    inputImagesAugmetation[0] = cv::imread("files" FILE_SEPARATOR "img0.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[1] = cv::imread("files" FILE_SEPARATOR "img1.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[2] = cv::imread("files" FILE_SEPARATOR "img2.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[3] = cv::imread("files" FILE_SEPARATOR "img3.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[4] = cv::imread("files" FILE_SEPARATOR "img4.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[5] = cv::imread("files" FILE_SEPARATOR "img5.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[6] = cv::imread("files" FILE_SEPARATOR "img6.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[7] = cv::imread("files" FILE_SEPARATOR "img7.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[8] = cv::imread("files" FILE_SEPARATOR "img8.jpg", cv::IMREAD_COLOR);
    inputImagesAugmetation[9] = cv::imread("files" FILE_SEPARATOR "img9.jpg", cv::IMREAD_COLOR);

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
    std::vector<mrcv::AUGMENTATION_METHOD> augmetationMethod = {
        mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_90,
        mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL,
        mrcv::AUGMENTATION_METHOD::FLIP_VERTICAL,
        mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_45,
        mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_315,
        mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_270,
        mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL_AND_VERTICAL,
    };

    // Вызов функции аугментации
    int state = mrcv::augmetation(inputImagesAugmetation, outputImagesAugmetation, augmetationMethod);
    if (state != 0) {
        std::cerr << "Error: Augmentation failed with code " << state << std::endl;
        return -1;
    }

    // Тест пакетной аугментации
    mrcv::BatchAugmentationConfig config;
    config.keep_original = true;
    config.total_output_count = 100;
    config.random_seed = 42;

    config.method_weights = {
        {mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL, 0.2},
        {mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_90, 0.2},
        {mrcv::AUGMENTATION_METHOD::BRIGHTNESS_CONTRAST_ADJUST, 0.3},
        {mrcv::AUGMENTATION_METHOD::PERSPECTIVE_WARP, 0.2},
        {mrcv::AUGMENTATION_METHOD::COLOR_JITTER, 0.1},
    };

    std::vector<cv::Mat> batchOutput;
    state = mrcv::batchAugmentation(inputImagesCopy, batchOutput, config, "files" FILE_SEPARATOR "batch_output");
    if (state != 0) {
        std::cerr << "Error: Batch augmentation failed with code: " << state << std::endl;
        return -1;
    }

    return 0;
}
