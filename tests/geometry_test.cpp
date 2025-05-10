#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(geometry_test, geometry)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "geometry";

    // Загрузка исходных данных
    std::vector<mrcv::Cloud3DItem> cloud3D = mrcv::geometryLoadData((path / "points3D.dat").u8string(), 5);
    
    // Рассчет геометрии точек
    double L, W, H, Length, Width, Distance;
    int exitcode = mrcv::geometryCalculateSize(cloud3D, &L, &W, &H, &Length, &Width, &Distance);

    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}