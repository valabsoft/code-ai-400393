#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(segmentationtrain_test, segmentationtrain)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "segmentationtrain";

    auto weightsPath = (path / "weights" / "resnet34.pt");
    auto dataPath = (path / "images");
    auto savePath = (path / "weights" / "segmentor.pt");

    bool weightsPathExists = std::filesystem::exists(weightsPath);
    bool savePathExists = std::filesystem::exists(savePath);

    auto weightsExists = weightsPathExists ? "-- OK" : "-- FAILURE";
    auto saveExists = savePathExists ? "-- OK" : "-- FAILURE";

    std::cout << "Weights path: " << weightsPath.u8string() << weightsExists << std::endl;
    std::cout << "Data path: " << dataPath.u8string() << "-- OK" << std::endl;
    std::cout << "Save path: " << savePath.u8string() << saveExists << std::endl;

    int exitcode = -1;

    try
    {
        mrcv::Segmentor segmentor;
        segmentor.Initialize(512, 320, { "background", "ship" }, "resnet34", weightsPath.u8string());
        exitcode = segmentor.Train(0.1, 2, 4, dataPath.u8string(), ".jpg", savePath.u8string());
    }
    catch (...)
    {
        exitcode = EXIT_FAILURE;
    }

    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}