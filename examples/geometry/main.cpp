#include <mrcv/mrcv.h>
#include <iostream>

int main()
{
    // Путь к файлу с данными
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path fileName = "points3D.dat";
    std::filesystem::path pathToFile = currentPath / "data" / fileName;

    // Загрузка исходных данных
    std::vector<mrcv::Cloud3DItem> cloud3D = mrcv::geometryLoadData(pathToFile.string(), 5);
    // Рассчет геометрии точек
    double L, W, H, Length, Width, Distance;
    mrcv::geometryCalculateSize(cloud3D, &L, &W, &H, &Length, &Width, &Distance);

    std::cout << "L: " << L << std::endl;
    std::cout << "W: " << W << std::endl;
    std::cout << "H: " << H << std::endl;
    std::cout << "Length: " << Length << std::endl;
    std::cout << "Width: " << Width << std::endl;
    std::cout << "Distance: " << Distance << std::endl;
}