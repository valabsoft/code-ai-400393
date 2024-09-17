#include <mrcv/mrcv.h>

int main()
{
    // Буфер для изображений
    cv::Mat imageIn;
    cv::Mat imageOut;

    std::filesystem::path imgFile("files\\example01.bmp");
    std::filesystem::path cameraFile("files\\camera-parameters.xml");

    auto currentPath = std::filesystem::current_path();
    auto imgPath = currentPath / imgFile;
    auto cameraPath = currentPath / cameraFile;

    imageIn = imread(imgPath.u8string(), cv::IMREAD_COLOR);
    imageOut = imageIn.clone();

    // Набор методов предобработки    
    std::vector<mrcv::IMG_PREPROCESSING_METHOD> methods =
    {
        mrcv::IMG_PREPROCESSING_METHOD::NOISEFILTERINGMEDIANFILTER,
        mrcv::IMG_PREPROCESSING_METHOD::COLORLABCLAHE,
        mrcv::IMG_PREPROCESSING_METHOD::SHARPENING02,
        mrcv::IMG_PREPROCESSING_METHOD::BRIGHTNESSLEVELDOWN,
        mrcv::IMG_PREPROCESSING_METHOD::CORRECTIONGEOMETRICDEFORMATION,
        mrcv::IMG_PREPROCESSING_METHOD::NONE
    };
    
    // Функция предварительной обработки изображений (автоматическая коррекция контраста и яркости, резкости)
    int state = mrcv::preprocessingImage(imageOut, methods, cameraPath.u8string());

    // Вывод результатов
    double scale = 0.5;
    cv::resize(imageIn, imageIn, cv::Size(double(imageIn.cols * scale), double(imageIn.rows * scale)), 0, 0, cv::INTER_LINEAR);
    cv::resize(imageOut, imageOut, cv::Size(double(imageOut.cols * scale), double(imageOut.rows * scale)), 0, 0, cv::INTER_LINEAR);
    
    cv::namedWindow("Before", cv::WINDOW_AUTOSIZE);    
    cv::namedWindow("After", cv::WINDOW_AUTOSIZE);

    imshow("Before", imageIn);    
    imshow("After", imageOut);

    cv::waitKey(0);
    
    return EXIT_SUCCESS;
} 
