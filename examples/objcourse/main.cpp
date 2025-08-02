#include <mrcv/mrcv.h>

int main()
{
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "files";
    
    // Пути к файлам модели
    std::filesystem::path modelPath = path / "ship.onnx";
    std::filesystem::path classPath = path / "ship.names";
    std::filesystem::path shipPath = path / "ship.bmp";
    

    // Экземпляр класса детектора
    mrcv::ObjCourse *objcourse= new mrcv::ObjCourse(modelPath.u8string(), classPath.u8string());
    cv::Mat frameShip = cv::imread(shipPath.u8string(), cv::IMREAD_COLOR);
    
    // Подсчет объектов
    int objCount = objcourse->getObjectCount(frameShip);
    
    // Расчет курса
    float objAngle = objcourse->getObjectCourse(frameShip, 640, 80);
    
    mrcv::writeLog();
    mrcv::writeLog("Файл модели: " + modelPath.u8string());
    mrcv::writeLog("Файл классов: " + classPath.u8string());
    mrcv::writeLog("Входное изображение: " + shipPath.u8string());
    mrcv::writeLog("Обнаружено объектов: " + std::to_string(objCount));
    mrcv::writeLog("Курс на цель в градусах: " + std::to_string(objAngle));
    
    delete objcourse;
    
    return EXIT_SUCCESS;
} 
