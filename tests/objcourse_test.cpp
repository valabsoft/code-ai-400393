#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(objcourse_test, objcourse)
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "objcourse";

    auto modelPath = (path / "ship.onnx").u8string();
    auto classPath = (path / "ship.names").u8string();
    auto shipPath = (path / "ship.bmp").u8string();

    // Экземпляр класса детектора
    mrcv::ObjCourse* objcourse = new mrcv::ObjCourse(modelPath, classPath);
    cv::Mat frameShip = cv::imread(shipPath, cv::IMREAD_COLOR);

    // Подсчет объектов
    int objCount = objcourse->getObjectCount(frameShip);

    // Расчет курса
    float objAngle = objcourse->getObjectCourse(frameShip, 640, 80);

    std::cout << "Model file: " + modelPath << std::endl;
    std::cout << "Classes file: " + classPath << std::endl;
    std::cout << "Input image: " + shipPath << std::endl;
    std::cout << "Object detected: " + std::to_string(objCount) << std::endl;
    std::cout << "Calculated course: " + std::to_string(objAngle) << std::endl;

    bool exitcode = (objCount > 0) && (objAngle != 0);
    EXPECT_EQ(exitcode, true);
    delete objcourse;
}