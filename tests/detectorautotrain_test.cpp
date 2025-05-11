#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(detectorautotrain_test, detectorautotrain)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "detectorautotrain";

    auto datasetPath = path / "onwater";
    auto vocClassesPath = datasetPath / "voc_classes.txt";
    auto modelSavePath = path / "onwater_autodetector.pt";
    auto pretrainedModelPath = path / "yolo4_tiny.pt";

    mrcv::Detector detector;
    detector.Initialize(0, 416, 416, vocClassesPath.string());
    int exitcode = detector.AutoTrain
    (
        datasetPath.u8string(),
        ".jpg",
        { 2 }, // { 10, 15, 30 },
        { 2 }, // { 4, 8 },
        { 0.1, 0.01 }, // { 0.001, 1.0E-4F },
        pretrainedModelPath.u8string(),
        modelSavePath.u8string()
    );

    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}