#ifndef SENSORSFUSION_H
#define SENSORSFUSION_H
#pragma once

#include"mrcv/mrcv-common.h"
#include <yaml-cpp/yaml.h>


namespace mrcv 
{
    struct IMUData 
    {
        double accel[3];   // Линейные скорости локального устройтсва
        double gyro[3];    // Угловые скорости локального устройтсва
        std::chrono::system_clock::time_point timestamp;
    };

    struct USBLData 
    {
        std::string datetime;   // Временная метка
        double x, y, z;         // Относительные координаты удалённого устройства
        double azimuth;         // Азимут угла прихода сигнала от удалённого устройства (в градусах)
        double localDepth;      // Глубина локального устройства
        double remoteDepth;     // Глубина удалённого устройства
        double propagationTime; // Время распространения сигнала
        double rs, rh;          // Наклонная дальность до удалённого устройства и её проекция
    };

    struct CameraFrame {
        std::string filename;   // Имя файла изображения
        cv::Mat image;          // Контейнер для сохранения изображения
        std::chrono::system_clock::time_point timestamp;
    };

    struct FusedData 
    {
        std::chrono::system_clock::time_point timestamp;
        double accel[3];                        // Линейные скорости локального устройства
        double gyro[3];                         // Угловые скорости локального устройства
        double position[3];                     // Координаты локального устройства
        double relativeCoords[3];               // Относительные координаты удалённого устройства
        std::vector<cv::Point2f> featurePoints; // Ключевые точки на изображении
        std::string imageFilename;              // Имя файла изображения
        float azimuth;                          // Азимут в градусах
        float localDepth;                       // Глубина локального устройства
        float remoteDepth;                      // Глубина удалённого устройства
        float propagationTime;                  // Время распространения сигнала
        float rs;                               // Наклонная дальность до удалённого устройства
        float rh;                               // Проекция наклонной дальности на плоскости морской поверхности
    };
}
#endif // !SENSORSFUSION_H