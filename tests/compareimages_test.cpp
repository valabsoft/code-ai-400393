#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(compareimages_test, compareimages)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "compareimages";

    cv::Mat img1 = cv::imread((path / "img1.png").u8string());
    cv::Mat img2 = cv::imread((path / "img2.png").u8string());

    std::cout << "Similarity: " << mrcv::compareImages(img1, img2, 1) << std::endl;
    double result = mrcv::compareImages(img1, img2, 1);

    EXPECT_NE(result, 0);
}