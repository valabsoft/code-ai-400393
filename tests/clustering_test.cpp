#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(clustering_test, makeClustering)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "clustering";
    
    mrcv::DenseStereo denseStereo;
    denseStereo.loadDataFromFile((path / "claster.dat").u8string());
    int exitcode = denseStereo.makeClustering();

    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}