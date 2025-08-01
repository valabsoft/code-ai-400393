#include <mrcv/mrcv.h>

int main()
{
    // Загрузка тестовых изображений
    //std::filesystem::path fileImageLeft("files\\example_left.jpg");
    //std::filesystem::path fileImageRight("files\\example_right.jpg");

    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "files";
    std::filesystem::path pathImageLeft = path / "example_left.jpg";
    std::filesystem::path pathImageRight = path / "example_right.jpg";

    //auto pathImageLeft = currentPath / fileImageLeft;
    //auto pathImageRight = currentPath / fileImageRight;

    cv::Mat imageLeft = cv::imread(pathImageLeft.u8string(), cv::IMREAD_COLOR);
    cv::Mat imageRight = cv::imread(pathImageRight.u8string(), cv::IMREAD_COLOR);

    ///////////////////////////////////////////////////////////////////////////
    // Параметры функции    
    int minDisparity = 16;
    int numDisparities = 16 * 10;
    int blockSize = 15;

    // Настройки для примеров с сайта OpenCV: lambda = 8000.0; sigma = 1.5;
    // Настройки для примеров с амфорами: lambda = 5000.0; sigma = 3;
    double lambda = 5000.0;
    double sigma = 3;

    // COLORMAP_JET
    // COLORMAP_VIRIDIS
    // COLORMAP_TURBO
    // COLORMAP_HOT
    int colorMap = cv::COLORMAP_TURBO;
    mrcv::DISPARITY_TYPE disparityType = mrcv::DISPARITY_TYPE::ALL;

    ///////////////////////////////////////////////////////////////////////////
    // Построение карты диспаратности
    cv::Mat disparitymap;
    mrcv::disparityMap(disparitymap, imageLeft, imageRight, minDisparity, numDisparities, blockSize, lambda, sigma, disparityType, colorMap, true, true);

    cv::namedWindow("MRCV Disparity Map", cv::WINDOW_AUTOSIZE);
    cv::imshow("MRCV Disparity Map", disparitymap);

    cv::waitKey(0);

    return EXIT_SUCCESS;
}
