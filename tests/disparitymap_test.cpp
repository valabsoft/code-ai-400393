#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(disparity_test, disparity)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "disparitymap";

    cv::Mat imageLeft = cv::imread((path / "example_left.jpg").u8string(), cv::IMREAD_COLOR);
    cv::Mat imageRight = cv::imread((path / "example_right.jpg").u8string(), cv::IMREAD_COLOR);

    ///////////////////////////////////////////////////////////////////////////
    // Построение карты диспаратности
    cv::Mat disparitymap;
    int exitcode = mrcv::disparityMap(
        disparitymap,
        imageLeft,
        imageRight,
        16,
        160,
        15,
        5000,
        3,
        mrcv::DISPARITY_TYPE::ALL,
        cv::COLORMAP_TURBO,
        true,
        true);

    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}