#include <mrcv/mrcv.h>
#include <gtest/gtest.h>
#include <mrcv/mrcv-yolov5.h>

TEST(configuration_test, generateconfig)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "configuration";

    int exitcode = mrcv::YOLOv5GenerateConfig(mrcv::YOLOv5Model::YOLOv5s, (path / "yolov5s-coco.yaml").u8string(), 80);
    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}
TEST(configuration_test, generatehyperparameters)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "configuration";

    int exitcode = mrcv::YOLOv5GenerateHyperparameters(mrcv::YOLOv5Model::YOLOv5s, 640, 640, (path / "yolov5s-hyp.yaml").u8string(), 80);
    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}