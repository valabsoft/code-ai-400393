#include "mrcv/mrcv-sensorsfusion.h"


namespace mrcv
{
    std::vector<USBLData> loadAcousticCSV(const std::string& filename)
    {
        if (!std::filesystem::exists(filename)) {
            std::cerr << "File is not excist: " << filename << std::endl;
            return {};
        }

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "error while file opening" << filename << std::endl;

            return {};
        }
        else {
            std::cout << "File was open: " << filename << std::endl;
        }


        std::string line;
        std::vector<USBLData> entries;

        std::getline(file, line); // пропуск заголовка

        while (std::getline(file, line))
        {
            if (line.empty()) continue;

            std::cout << "OK 2: " << line << std::endl;
            std::regex delim(R"([ ]+)");
            std::sregex_token_iterator it(line.begin(), line.end(), delim, -1);
            std::sregex_token_iterator end;
            std::vector<std::string> tokens(it, end);
            if (tokens.size() < 10) continue;

            USBLData entry;
            entry.datetime = tokens[0];
            entry.x = std::stod(tokens[1]);
            entry.y = std::stod(tokens[2]);
            entry.z = std::stod(tokens[3]);
            entry.azimuth = std::stod(tokens[4]);
            entry.localDepth = std::stod(tokens[5]);
            entry.remoteDepth = std::stod(tokens[6]);
            entry.propagationTime = std::stod(tokens[7]);
            entry.rs = std::stod(tokens[8]);
            entry.rh = std::stod(tokens[9]);

            entries.push_back(entry);
        }
        return entries;
    }

    std::vector<CameraFrame> loadCameraFrames(const std::string& folder)
    {
        std::vector<CameraFrame> frames;
        for (const auto& entry : std::filesystem::directory_iterator(folder)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                cv::Mat img = cv::imread(path);
                if (!img.empty()) {
                    // Предположим, что имя файла: YYYY-MM-DD_HH-MM-SS.png
                    std::string filename = entry.path().stem().string();
                    std::tm tm = {};
                    std::istringstream ss(filename);
                    ss >> std::get_time(&tm, "%Y-%m-%d_%H-%M-%S");
                    auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));

                    frames.push_back({ path, img, tp });
                }
            }
        }

        std::sort(frames.begin(), frames.end(), [](const auto& a, const auto& b) {
            return a.timestamp < b.timestamp;
            });

        return frames;
    }

    std::vector<IMUData> loadIMUData(const std::string& csvFile)
    {
        std::vector<IMUData> imuData;
        std::ifstream file(csvFile);
        std::string line;

        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string timeStr;

            std::getline(ss, timeStr, ',');

            std::tm tm = {};
            std::istringstream tss(timeStr);
            tss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
            auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));

            double ax, ay, az, gx, gy, gz;
            char delim;

            ss >> ax >> delim >> ay >> delim >> az >> delim;
            ss >> gx >> delim >> gy >> delim >> gz;

            IMUData data;
            data.timestamp = tp;
            data.accel[0] = ax;
            data.accel[1] = ay;
            data.accel[2] = az;
            data.gyro[0] = gx;
            data.gyro[1] = gy;
            data.gyro[2] = gz;

            imuData.push_back(data);
        }

        return imuData;
    }

    // Вспомогательная функция для парсинга времени
    std::chrono::system_clock::time_point parseTimestamp(const std::string& datetime) {
        std::tm tm = {};
        std::istringstream ss(datetime);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        return std::chrono::system_clock::from_time_t(std::mktime(&tm));
    }

    const IMUData* findNearestIMU(std::chrono::system_clock::time_point& ts,
        const std::vector<IMUData>& imuData)
    {
        const IMUData* nearest = nullptr;
        auto minDiff = std::chrono::milliseconds(100000); // произвольно большое значение

        for (const auto& imu : imuData) {
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(ts - imu.timestamp);
            if (std::abs(diff.count()) < std::abs(minDiff.count())) {
                minDiff = diff;
                nearest = &imu;
            }
        }
        return nearest;
    }

    const CameraFrame* findNearestFrame(std::chrono::system_clock::time_point& ts,
        const std::vector<CameraFrame>& frames)
    {
        const CameraFrame* nearest = nullptr;
        auto minDiff = std::chrono::milliseconds(100000); // произвольно большое значение

        for (const auto& frame : frames) {
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(ts - frame.timestamp);
            if (std::abs(diff.count()) < std::abs(minDiff.count())) {
                minDiff = diff;
                nearest = &frame;
            }
        }
        return nearest;
    }

    const USBLData* findNearestUSBL(std::chrono::system_clock::time_point& ts,
        const std::vector<USBLData>& usblData)
    {
        const USBLData* nearest = nullptr;
        auto minDiff = std::chrono::milliseconds(100000); // достаточно большое значение

        for (const auto& usbl : usblData) {
            auto usblTs = parseTimestamp(usbl.datetime);
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(ts - usblTs);
            if (std::abs(diff.count()) < std::abs(minDiff.count())) {
                minDiff = diff;
                nearest = &usbl;
            }
        }
        return nearest;
    }


    int fuseSensorData(const std::string& usblPath,
        const std::string& imuPath,
        const std::string& camFolder,
        const std::string& outYAML, bool visFlag)
    {
        std::vector<USBLData> usblData = loadAcousticCSV(usblPath);
        if (usblData.empty()) return 1; // Ошибка загрузки данных с USBL

        std::vector<IMUData> imuData = loadIMUData(imuPath);
        if (imuData.empty()) return 2; // Ошибка загрузки данных с IMU

        std::vector<CameraFrame> frames = loadCameraFrames(camFolder);
        if (frames.empty()) return 3; // Ошибка загрузки изображений

        std::vector<FusedData> fused;

        for (const auto& usbl : usblData) {
            auto ts = parseTimestamp(usbl.datetime);

            const IMUData* imuNearest = findNearestIMU(ts, imuData);
            const CameraFrame* camNearest = findNearestFrame(ts, frames);


            if (imuNearest && camNearest) {
                FusedData data;
                data.timestamp = ts;

                std::copy(std::begin(imuNearest->accel), std::end(imuNearest->accel), data.accel);
                std::copy(std::begin(imuNearest->gyro), std::end(imuNearest->gyro), data.gyro);

                // Простая заглушка для положения аппарата
                data.position[0] = 0;
                data.position[1] = 0;
                data.position[2] = 0;

                data.relativeCoords[0] = usbl.x;
                data.relativeCoords[1] = usbl.y;
                data.relativeCoords[2] = usbl.z;

                data.featurePoints = {};  // Пока пусто

                // Отображение
                float yaw = imuNearest->gyro[2]; // yaw — вращение по Z
                cv::Point2f relativePoint(usbl.x * std::cos(-yaw) - usbl.y * std::sin(-yaw),
                    usbl.x * std::sin(-yaw) + usbl.y * std::cos(-yaw));

                int cx = camNearest->image.cols / 2;
                int cy = camNearest->image.rows / 2;

                cv::Point2f pointer(cx + relativePoint.x * 10, cy - relativePoint.y * 10);

                data.imageFilename = camNearest->filename;

                fused.push_back(data);
            }
        }

        // Сохраняем в YAML
        try {
            YAML::Emitter out;
            out << YAML::BeginSeq;
            for (const auto& d : fused) {
                out << YAML::BeginMap;
                out << YAML::Key << "timestamp" << YAML::Value << std::chrono::duration_cast<std::chrono::milliseconds>(d.timestamp.time_since_epoch()).count();
                out << YAML::Key << "accel" << YAML::Value << YAML::Flow << std::vector<double>(d.accel, d.accel + 3);
                out << YAML::Key << "gyro" << YAML::Value << YAML::Flow << std::vector<double>(d.gyro, d.gyro + 3);
                out << YAML::Key << "position" << YAML::Value << YAML::Flow << std::vector<double>(d.position, d.position + 3);
                out << YAML::Key << "relativeCoords" << YAML::Value << YAML::Flow << std::vector<double>(d.relativeCoords, d.relativeCoords + 3);
                out << YAML::Key << "featurePoints" << YAML::Value << YAML::Flow << d.featurePoints;
                out << YAML::Key << "image" << YAML::Value << d.imageFilename;
                out << YAML::EndMap;
            }
            out << YAML::EndSeq;

            std::ofstream fout(outYAML);
            fout << out.c_str();
            fout.close();

            if (visFlag) visualizeResult(outYAML, camFolder);
        }
        catch (const std::exception& e) {
            std::cerr << "YAML write error: " << e.what() << std::endl;
            return 4; // Ошибка сохранения файла с результатами
        }

        return 0;
    }

    void visualizeResult(const std::string& yamlFile, const std::string& frameFolder)
    {
        YAML::Node root;
        try {
            root = YAML::LoadFile(yamlFile);
        }
        catch (const std::exception& e) {
            std::cerr << "YAML Load error: " << e.what() << std::endl;
            return;
        }

        for (const auto& entry : root) {
            try {
                auto timestamp_ms = entry["timestamp"].as<long long>();
                auto accel = entry["accel"].as<std::vector<double>>();
                auto gyro = entry["gyro"].as<std::vector<double>>();
                auto relativeCoords = entry["relativeCoords"].as<std::vector<double>>();
                auto imageName = entry["image"].as<std::string>();

                // yaw = вращение по оси Z
                float yaw = static_cast<float>(gyro[2]);

                // Расчёт относительного положения на изображении с учётом yaw
                cv::Point2f relativePoint(
                    relativeCoords[0] * std::cos(-yaw) - relativeCoords[1] * std::sin(-yaw),
                    relativeCoords[0] * std::sin(-yaw) + relativeCoords[1] * std::cos(-yaw)
                );

                // Загрузка изображения
                //std::filesystem::path imagePath = std::filesystem::path(frameFolder) / imageName;
                cv::Mat image = cv::imread(/*imagePath.string()*/ imageName);
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << imageName << std::endl;
                    continue;
                }

                // Центр изображения
                int cx = image.cols / 2;
                int cy = image.rows / 2;

                cv::Point2f pointer(cx + relativePoint.x * 10, cy - relativePoint.y * 10);

                // Рисуем маркер
                cv::circle(image, pointer, 8, cv::Scalar(0, 0, 255), -1);

                // Подпись
                std::string label = "t: " + std::to_string(timestamp_ms);
                cv::putText(image, label, pointer + cv::Point2f(10, -10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                // Показываем изображение
                cv::imshow("Visualized Frame", image);
                cv::waitKey(0);  // Ждём нажатия клавиши

            }
            catch (const std::exception& e) {
                std::cerr << "Error while parsing entry: " << e.what() << std::endl;
                continue;
            }
        }
        cv::destroyAllWindows();
    }
}