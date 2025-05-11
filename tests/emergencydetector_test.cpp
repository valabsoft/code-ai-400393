#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(emergencydetector_test, emergencydetector)
{
    auto currentPath = std::filesystem::current_path();

    std::filesystem::path path = currentPath / "data" / "emergencydetector";
    std::filesystem::path imagePath = path / "pipes" / "test" / "images" / "burst_augment_56_blur45_jpg.rf.714ba15fef8cf1f9ebcc19bbdb07dd2a.jpg";
    std::filesystem::path vocClassesPath = path / "pipes" / "voc_classes.txt";
    std::filesystem::path weightPath = path / "emergency_detector.pt";

    bool imagePathExists = std::filesystem::exists(imagePath);
    bool classesPathExists = std::filesystem::exists(vocClassesPath);
    bool weightPathExists = std::filesystem::exists(weightPath);

    auto imageExists = imagePathExists ? "-- OK" : "-- FAILURE";
    auto classesExists = classesPathExists ? "-- OK" : "-- FAILURE";
    auto weightExists = weightPathExists ? "-- OK" : "-- FAILURE";

    std::cout << "Image path: " << imagePath.u8string() << imageExists << std::endl;
    std::cout << "Classes path: " << vocClassesPath.u8string() << classesExists << std::endl;
    std::cout << "Weight path: " << weightPath.u8string() << weightExists << std::endl;

    cv::Mat image = cv::imread(imagePath.u8string());

    int exitcode = EXIT_FAILURE;

    try
    {
        mrcv::Detector detector;
        // Инициализация структуры модели
        detector.Initialize(0, 416, 416, vocClassesPath.string());
        // Загрузка весов обученной модели
        detector.LoadWeight(weightPath.u8string());
        // Детекция объектов на изображении
        exitcode = detector.Predict(image, false, 0.1);
    }
    catch (...)
    {
        exitcode = EXIT_FAILURE;
    }

    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}