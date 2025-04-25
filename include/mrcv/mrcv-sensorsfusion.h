#ifndef SENSORSFUSION_H
#define SENSORSFUSION_H
#pragma once

#include"mrcv/mrcv-common.h"
#include <yaml-cpp/yaml.h>


namespace mrcv {
    struct IMUData {
        double accel[3];   // x, y, z
        double gyro[3];    // угловая скорость
        std::chrono::system_clock::time_point timestamp;
    };

    struct USBLData {
        std::string datetime;
        double x, y, z;
        double azimuth;
        double localDepth;
        double remoteDepth;
        double propagationTime;
        double rs, rh;
    };

    struct CameraFrame {
        std::string filename;
        cv::Mat image;
        std::chrono::system_clock::time_point timestamp;
    };

    struct FusedData {
        std::chrono::system_clock::time_point timestamp;
        std::string imageFilename;
        double accel[3];
        double gyro[3];
        double position[3];
        double relativeCoords[3];
        std::vector<std::vector<float>> featurePoints;
    };

    // Загрузка данных с IMU
    std::vector<IMUData> loadIMUData(const std::string& csvFile);
    // Загрузка USBL-лога 
    std::vector<USBLData> loadAcousticCSV(const std::string& filename);
    // Загрузка изображений
    std::vector<CameraFrame> loadCameraFrames(const std::string& folder);

    // Функция комплексирования
    int fuseSensorData(const std::string& usblPath, const std::string& imuPath,
        const std::string& camFolder, const std::string& outYAML, bool visFlag);

    // Визуализация результата
    void visualizeResult(const std::string& fileName, const std::string& frameFolder);
}
#endif // !SENSORSFUSION_H