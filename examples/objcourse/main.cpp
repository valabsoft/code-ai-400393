#include <mrcv/mrcv.h>

int main()
{
    // Пути к файлам модели
    std::filesystem::path modelFile("files\\ship.onnx");
    std::filesystem::path classFile("files\\ship.names");
    std::filesystem::path shipFile("files\\ship.bmp");
    
    auto currentPath = std::filesystem::current_path();
    
    auto modelPath = currentPath / modelFile;
    auto classPath = currentPath / classFile;
    auto shipPath = currentPath / shipFile;
    
    // Экземпляр класса детектора
    mrcv::ObjCourse *objcourse= new mrcv::ObjCourse(modelPath.u8string(), classPath.u8string(), 640, 640);
    cv::Mat frameShip = cv::imread(shipPath.u8string(), cv::IMREAD_COLOR);
    
    // Подсчет объектов
    int objCount = objcourse->getObjectCount(frameShip);
    
    // Расчет курса
    float objAngle = objcourse->getObjectCourse(frameShip, 640, 640);
    
    mrcv::writeLog();
    mrcv::writeLog("Файл модели: " + modelPath.u8string());
    mrcv::writeLog("Файл классов: " + classPath.u8string());
    mrcv::writeLog("Входное изображение: " + shipPath.u8string());
    mrcv::writeLog("Обнаружено: " + std::to_string(objCount) + " объектов");
    mrcv::writeLog("Курс на цель: " + std::to_string(objAngle) + " градусов");
    
    delete objcourse;
    
    return EXIT_SUCCESS;
} 
