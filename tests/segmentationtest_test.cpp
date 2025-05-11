#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(segmentationtest_test, segmentationtest)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "segmentationtest";

    auto weightsPath = (path / "weights" / "resnet34.pt");
    auto segmentorPath = (path / "weights" / "segmentor.pt");
    auto imagePath = (path / "images" / "43.jpg");

    bool weightsPathExists = std::filesystem::exists(weightsPath);
    bool segmentorPathExists = std::filesystem::exists(segmentorPath);
    bool imagePathExists = std::filesystem::exists(imagePath);

    auto weightsExists = weightsPathExists ? "-- OK" : "-- FAILURE";
    auto segmentorExists = segmentorPathExists ? "-- OK" : "-- FAILURE";
    auto imageExists = imagePathExists ? "-- OK" : "-- FAILURE";

    std::cout << "Weights path: " << weightsPath.u8string() << weightsExists << std::endl;
    std::cout << "Segmentor path: " << segmentorPath.u8string() << segmentorExists << std::endl;
    std::cout << "Image path: " << imagePath.u8string() << imageExists << std::endl;
        
    cv::Mat image = cv::imread(imagePath.u8string());

    mrcv::Segmentor segmentor;

    segmentor.Initialize(512, 320, { "background","ship" }, "resnet34", weightsPath.u8string());
    segmentor.LoadWeight(segmentorPath.u8string());
    int exitcode = segmentor.Predict(image, "ship");

    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}