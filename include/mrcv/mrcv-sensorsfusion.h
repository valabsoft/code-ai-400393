#ifndef SENSORSFUSION_H
#define SENSORSFUSION_H
#pragma once

#include"mrcv/mrcv-common.h"
#include <yaml-cpp/yaml.h>


namespace mrcv {
    struct IMUData {
        double accel[3];   // �������� �������� ���������� ����������
        double gyro[3];    // ������� �������� ���������� ����������
        std::chrono::system_clock::time_point timestamp;
    };

    struct USBLData {
        std::string datetime;   // ��������� �����
        double x, y, z;         // ������������� ���������� ��������� ����������
        double azimuth;         // ������ ���� ������� ������� �� ��������� ���������� (� ��������)
        double localDepth;      // ������� ���������� ����������
        double remoteDepth;     // ������� ��������� ����������
        double propagationTime; // ����� ��������������� �������
        double rs, rh;          // ��������� ��������� �� ��������� ���������� � � ��������
    };

    struct CameraFrame {
        std::string filename;   // ��� ����� �����������
        cv::Mat image;          // ��������� ��� ���������� �����������
        std::chrono::system_clock::time_point timestamp;
    };

    struct FusedData {
        std::chrono::system_clock::time_point timestamp;
        double accel[3];                        // �������� �������� ���������� ����������
        double gyro[3];                         // ������� �������� ���������� ����������
        double position[3];                     // ���������� ���������� ����������
        double relativeCoords[3];               // ������������� ���������� ��������� ����������
        std::vector<cv::Point2f> featurePoints; // �������� ����� �� �����������
        std::string imageFilename;              // ��� ����� �����������
        float azimuth;                          // ������ � ��������
        float localDepth;                       // ������� ���������� ����������
        float remoteDepth;                      // ������� ��������� ����������
    };
}
#endif // !SENSORSFUSION_H