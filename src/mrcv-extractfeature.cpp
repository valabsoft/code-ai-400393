#include "mrcv/mrcv.h"
#include "mrcv/mrcv-sensorsfusion.h"
#include <mrcv/mrcv-common.h>


namespace mrcv 
{
    // Вспомогательная функция для нормализации данных в диапазон [0, 1]
    float normalize(float value, float minVal, float maxVal) 
    {
        if (maxVal == minVal) return 0.0f;
        return (value - minVal) / (maxVal - minVal);
    }

    // Вспомогательная функция для вычисления среднего значения вектора
    float computeMean(const std::vector<float>& values) 
    {
        if (values.empty()) return 0.0f;
        float sum = std::accumulate(values.begin(), values.end(), 0.0f);
        return sum / values.size();
    }

    // Вспомогательная функция для вычисления дисперсии вектора
    float computeVariance(const std::vector<float>& values, float mean) 
    {
        if (values.size() <= 1) return 0.0f;
        float sum = 0.0f;
        for (const auto& val : values) {
            sum += (val - mean) * (val - mean);
        }
        return sum / (values.size() - 1);
    }

    // Функция для извлечения признаков из изображений
    void extractImageFeatures(const cv::Mat& image, std::vector<float>& features) 
    {
        // Преобработка изображения
        cv::Mat gray, blurred, thresh;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
        cv::adaptiveThreshold(blurred, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV, 11, 2);

        // Поиск контуров
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Вычисление центроидов и площадей
        float centroidX = 0.0f, centroidY = 0.0f;
        float numObjects = static_cast<float>(contours.size());
        std::vector<float> areas;

        for (const auto& contour : contours) 
        {
            cv::Moments moments = cv::moments(contour);
            if (moments.m00 > 0) 
            {
                centroidX += moments.m10 / moments.m00;
                centroidY += moments.m01 / moments.m00;
                areas.push_back(static_cast<float>(cv::contourArea(contour)));
            }
        }

        if (numObjects > 0) 
        {
            centroidX /= numObjects;
            centroidY /= numObjects;
        }

        // Вычисление средней площади и дисперсии
        float meanArea = computeMean(areas);
        float varArea = computeVariance(areas, meanArea);

        // Нормализация центроидов относительно размеров изображения
        centroidX = normalize(centroidX, 0.0f, image.cols);
        centroidY = normalize(centroidY, 0.0f, image.rows);

        // Добавление признаков
        features.push_back(centroidX);
        features.push_back(centroidY);
        features.push_back(numObjects);
        features.push_back(normalize(meanArea, 0.0f, image.cols * image.rows));
        features.push_back(normalize(varArea, 0.0f, image.cols * image.rows));
    }

    std::vector<FusedData> loadFusedDataFromYAML(const std::string& yamlFile) 
    {
        std::vector<FusedData> fusedData;

        try 
        {
            YAML::Node root = YAML::LoadFile(yamlFile);
            if (!root.IsSequence()) 
            {
                std::cerr << "Error: YAML file does not contain a sequence" << std::endl;
                writeLog("Error: YAML file does not contain a sequence", mrcv::LOGTYPE::ERROR);
                return fusedData;
            }

            for (const auto& entry : root) 
            {
                try 
                {
                    FusedData data;

                    auto timestampMs = entry["timestamp"].as<long long>();
                    data.timestamp = std::chrono::system_clock::time_point(
                        std::chrono::milliseconds(timestampMs));

                    auto accel = entry["accel"].as<std::vector<double>>();
                    if (accel.size() >= 3) 
                    {
                        std::copy(accel.begin(), accel.begin() + 3, data.accel);
                    }
                    else 
                    {
                        std::cerr << "Error: Invalid accel data size" << std::endl;
                        continue;
                    }

                    auto gyro = entry["gyro"].as<std::vector<double>>();
                    if (gyro.size() >= 3) 
                    {
                        std::copy(gyro.begin(), gyro.begin() + 3, data.gyro);
                    }
                    else 
                    {
                        std::cerr << "Error: Invalid gyro data size" << std::endl;
                        continue;
                    }

                    auto position = entry["position"].as<std::vector<double>>();
                    if (position.size() >= 3) 
                    {
                        std::copy(position.begin(), position.begin() + 3, data.position);
                    }
                    else 
                    {
                        std::cerr << "Error: Invalid position data size" << std::endl;
                        continue;
                    }

                    auto relativeCoords = entry["relativeCoords"].as<std::vector<double>>();
                    if (relativeCoords.size() >= 3) 
                    {
                        std::copy(relativeCoords.begin(), relativeCoords.begin() + 3, data.relativeCoords);
                    }
                    else 
                    {
                        std::cerr << "Error: Invalid relativeCoords data size" << std::endl;
                        continue;
                    }

                    data.azimuth = entry["azimuth"].as<float>();
                    data.localDepth = entry["localDepth"].as<float>();
                    data.remoteDepth = entry["remoteDepth"].as<float>();
                    data.propagationTime = entry["propagationTime"].as<float>();
                    data.rs = entry["rs"].as<float>();
                    data.rh = entry["rh"].as<float>();

                    data.imageFilename = entry["image"].as<std::string>();

                    fusedData.push_back(data);
                }
                catch (const std::exception& e) 
                {
                    std::cerr << "Error parsing YAML entry: " << e.what() << std::endl;
                    writeLog("Error parsing YAML entry: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
                    continue;
                }
            }
        }
        catch (const std::exception& e) 
        {
            std::cerr << "Error loading YAML file: " << e.what() << std::endl;
            writeLog("Error loading YAML file: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            return fusedData;
        }

        writeLog("Loaded " + std::to_string(fusedData.size()) + " fused data entries from YAML", mrcv::LOGTYPE::INFO);
        return fusedData;
    }

    // Основная функция для формирования вектора признаков
    int extractFeatureVector(const std::string& fusedDataPath, const std::string& camFolder, const std::string& extractedDataPath) 
    {
        // Загрузка данных из YAML
        std::vector<FusedData> fusedData = loadFusedDataFromYAML(fusedDataPath);
        if (fusedData.empty()) 
        {
            std::cerr << "Error: No data loaded from YAML file " << fusedDataPath << std::endl;
            writeLog("Error: No data loaded from YAML file " + fusedDataPath, mrcv::LOGTYPE::ERROR);
            return EXIT_FAILURE;
        }

        // Извлечение признаков
        std::vector<std::vector<float>> featureVectors;
        const size_t windowSize = 10;

        for (size_t i = 0; i < fusedData.size(); ++i) 
        {
            std::vector<float> features;

            // Проверка корректности данных
            if (fusedData[i].relativeCoords[0] == 0.0 && fusedData[i].relativeCoords[1] == 0.0 &&
                fusedData[i].relativeCoords[2] == 0.0 && fusedData[i].imageFilename.empty()) 
            {
                writeLog("Skipping invalid data at index " + std::to_string(i), mrcv::LOGTYPE::WARNING);
                continue;
            }

            // USBL признаки
            features.push_back(normalize(fusedData[i].relativeCoords[0], -1000.0f, 1000.0f));
            features.push_back(normalize(fusedData[i].relativeCoords[1], -1000.0f, 1000.0f));
            features.push_back(normalize(fusedData[i].relativeCoords[2], -1000.0f, 1000.0f));
            features.push_back(normalize(fusedData[i].azimuth, 0.0f, 360.0f));
            features.push_back(normalize(fusedData[i].localDepth, 0.0f, 1000.0f));
            features.push_back(normalize(fusedData[i].remoteDepth, 0.0f, 1000.0f));
            features.push_back(normalize(fusedData[i].propagationTime, 0.0f, 1.0f));

            // IMU признаки
            std::vector<float> accelX, accelY, accelZ, gyroX, gyroY, gyroZ;
            size_t start = (i >= windowSize) ? i - windowSize : 0;
            size_t end = i + 1;

            for (size_t j = start; j < end; ++j) 
            {
                accelX.push_back(static_cast<float>(fusedData[j].accel[0]));
                accelY.push_back(static_cast<float>(fusedData[j].accel[1]));
                accelZ.push_back(static_cast<float>(fusedData[j].accel[2]));
                gyroX.push_back(static_cast<float>(fusedData[j].gyro[0]));
                gyroY.push_back(static_cast<float>(fusedData[j].gyro[1]));
                gyroZ.push_back(static_cast<float>(fusedData[j].gyro[2]));
            }

            float meanAx = computeMean(accelX);
            float meanAy = computeMean(accelY);
            float meanAz = computeMean(accelZ);
            float meanGx = computeMean(gyroX);
            float meanGy = computeMean(gyroY);
            float meanGz = computeMean(gyroZ);

            float varAx = computeVariance(accelX, meanAx);
            float varAy = computeVariance(accelY, meanAy);
            float varAz = computeVariance(accelZ, meanAz);
            float varGx = computeVariance(gyroX, meanGx);
            float varGy = computeVariance(gyroY, meanGy);
            float varGz = computeVariance(gyroZ, meanGz);

            features.push_back(normalize(meanAx, -10.0f, 10.0f));
            features.push_back(normalize(meanAy, -10.0f, 10.0f));
            features.push_back(normalize(meanAz, -10.0f, 10.0f));
            features.push_back(normalize(varAx, 0.0f, 100.0f));
            features.push_back(normalize(varAy, 0.0f, 100.0f));
            features.push_back(normalize(varAz, 0.0f, 100.0f));
            features.push_back(normalize(meanGx, -3.14159f, 3.14159f));
            features.push_back(normalize(meanGy, -3.14159f, 3.14159f));
            features.push_back(normalize(meanGz, -3.14159f, 3.14159f));
            features.push_back(normalize(varGx, 0.0f, 10.0f));
            features.push_back(normalize(varGy, 0.0f, 10.0f));
            features.push_back(normalize(varGz, 0.0f, 10.0f));

            // Признаки камеры
            std::filesystem::path imagePath = std::filesystem::path(camFolder) / fusedData[i].imageFilename;
            cv::Mat image = cv::imread(imagePath.string());
            if (image.empty()) 
            {
                std::cerr << "Failed to load image: " << imagePath << std::endl;
                writeLog("Failed to load image: " + imagePath.string(), mrcv::LOGTYPE::ERROR);
                features.push_back(0.0f);
                features.push_back(0.0f);
                features.push_back(0.0f);
                features.push_back(0.0f);
                features.push_back(0.0f);
            }
            else 
            {
                cv::Mat gray, blurred, thresh;
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
                cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
                cv::adaptiveThreshold(blurred, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv::THRESH_BINARY_INV, 11, 2);

                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                float centroidX = 0.0f, centroidY = 0.0f;
                float numObjects = static_cast<float>(contours.size());
                std::vector<float> areas;

                for (const auto& contour : contours) 
                {
                    cv::Moments moments = cv::moments(contour);
                    if (moments.m00 > 0) 
                    {
                        centroidX += moments.m10 / moments.m00;
                        centroidY += moments.m01 / moments.m00;
                        areas.push_back(static_cast<float>(cv::contourArea(contour)));
                    }
                }

                if (numObjects > 0) 
                {
                    centroidX /= numObjects;
                    centroidY /= numObjects;
                }

                float meanArea = computeMean(areas);
                float varArea = computeVariance(areas, meanArea);

                centroidX = normalize(centroidX, 0.0f, image.cols);
                centroidY = normalize(centroidY, 0.0f, image.rows);

                features.push_back(centroidX);
                features.push_back(centroidY);
                features.push_back(numObjects);
                features.push_back(normalize(meanArea, 0.0f, image.cols * image.rows));
                features.push_back(normalize(varArea, 0.0f, image.cols * image.rows));
            }

            featureVectors.push_back(features);
        }

        // Сохранение признаков в YAML
        try 
        {
            YAML::Emitter out;
            out << YAML::BeginSeq;
            for (const auto& features : featureVectors) 
            {
                out << YAML::BeginMap;
                out << YAML::Key << "features" << YAML::Value << YAML::Flow << std::vector<float>(features);
                out << YAML::EndMap;
            }
            out << YAML::EndSeq;

            std::ofstream fout(extractedDataPath);
            fout << out.c_str();
            fout.close();

            writeLog("Saved " + std::to_string(featureVectors.size()) + " feature vectors to " + extractedDataPath, mrcv::LOGTYPE::INFO);
        }
        catch (const std::exception& e) 
        {
            std::cerr << "Error writing features YAML: " << e.what() << std::endl;
            writeLog("Error writing features YAML: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
        }
        
        return EXIT_SUCCESS;
    }
}