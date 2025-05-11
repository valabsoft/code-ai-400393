#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(extractfeature_test, extractfeature)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "extractfeature";

    std::filesystem::path imagePath = path / "Images";
    std::filesystem::path fusePath = path / "fuseData.yaml";
    std::filesystem::path featuresPath = path / "extractedData.yaml";

    bool imagePathExists = std::filesystem::exists(imagePath);
    bool fusePathExists = std::filesystem::exists(fusePath);

    auto imageExists = imagePathExists ? "-- OK" : "-- FAILURE";
    auto fuseExists = fusePathExists ? "-- OK" : "-- FAILURE";

    std::cout << "Image path: " << imagePath.u8string() << imageExists << std::endl;
    std::cout << "Fuse path: " << fusePath.u8string() << fuseExists << std::endl;

    int exitcode = mrcv::extractFeatureVector(fusePath.u8string(), imagePath.u8string(), featuresPath.u8string());
    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}