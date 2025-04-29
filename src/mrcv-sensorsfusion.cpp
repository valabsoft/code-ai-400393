#include "mrcv/mrcv-sensorsfusion.h"
#include <mrcv/mrcv.h>


namespace mrcv
{
    // Функция синтаксического анализа временных меток 
    std::chrono::system_clock::time_point parseTimestamp(const std::string& input) {
        std::cout << "Parsing input: " << input << std::endl;

        // Регулярное выражение для формата кадров: L_ГГ-ММ-ДД_ЧЧ-ММ-СС или R_ГГ-ММ-ДД_ЧЧ-ММ-СС
        std::regex frame_re(R"((L|R)_(\d{2})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})(?:\.\w+)?)");
        // Регулярное выражение для формата USBL: ГГГГ-ММ-ДД ЧЧ:ММ:СС
        std::regex usbl_re(R"((\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2}))");

        std::smatch match;
        int year, month, day, hour, minute, second;

        try {
            // Проверяем формат кадров
            if (std::regex_match(input, match, frame_re)) {
                if (match.size() != 8) {
                    throw std::runtime_error("Unexpected number of matches for frame format in: " + input);

                }

                year = std::stoi(match[2]) + 2000; // ГГ -> ГГГГ (например, 25 -> 2025)
                month = std::stoi(match[3]);
                day = std::stoi(match[4]);
                hour = std::stoi(match[5]);
                minute = std::stoi(match[6]);
                second = std::stoi(match[7]);
            }
            // Проверяем формат USBL
            else if (std::regex_match(input, match, usbl_re)) {
                if (match.size() != 7) {
                    throw std::runtime_error("Unexpected number of matches for USBL format in: " + input);
                }

                year = std::stoi(match[1]); // ГГГГ
                month = std::stoi(match[2]);
                day = std::stoi(match[3]);
                hour = std::stoi(match[4]);
                minute = std::stoi(match[5]);
                second = std::stoi(match[6]);
            }
            else {
                throw std::runtime_error("Input does not match any supported timestamp format: " + input);
            }

            // Проверка диапазонов
            if (month < 1 || month > 12 || day < 1 || day > 31 || hour < 0 || hour > 23 ||
                minute < 0 || minute > 59 || second < 0 || second > 59) {
                throw std::runtime_error("Invalid date/time values in: " + input);
            }

            std::tm tm = {};
            tm.tm_year = year - 1900;
            tm.tm_mon = month - 1;
            tm.tm_mday = day;
            tm.tm_hour = hour;
            tm.tm_min = minute;
            tm.tm_sec = second;

            std::time_t time = std::mktime(&tm);
            if (time == -1) {
                throw std::runtime_error("Failed to convert timestamp to time_t: " + input);
            }

            return std::chrono::system_clock::from_time_t(time);
        }
        catch (const std::invalid_argument& e) {
            throw std::runtime_error("Invalid stoi argument for input: " + input + ", error: " + e.what());
        }
        catch (const std::out_of_range& e) {
            throw std::runtime_error("Out of range in stoi for input: " + input + ", error: " + e.what());
        }
    }

    // Загрузка данных с USBL системы
    std::vector<USBLData> loadAcousticCSV(const std::string& filename)
    {
        if (!std::filesystem::exists(filename)) {
            std::cerr << "File does not exist: " << filename << std::endl;
            return {};
        }

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error while opening file: " << filename << std::endl;
            return {};
        }
        else {
            std::cout << "File was opened: " << filename << std::endl;
        }

        std::string line;
        std::vector<USBLData> entries;

        std::getline(file, line); // Пропуск заголовка

        while (std::getline(file, line))
        {
            if (line.empty()) continue;

            //std::cout << "Raw line: [" << line << "]" << std::endl;

            // Удаляем кавычки из строки, если они есть
            if (!line.empty() && line.front() == '"' && line.back() == '"') {
                line = line.substr(1, line.length() - 2);
            }

            // Извлекаем временную метку (первые 19 символов: ГГГГ-ММ-ДД ЧЧ:ММ:СС)
            if (line.length() < 19) {
                std::cerr << "Skipping invalid line (too short for timestamp): " << line << std::endl;
                continue;
            }
            std::string datetime = line.substr(0, 19); // "ГГГГ-ММ-ДД ЧЧ:ММ:СС" = 19 символов
            std::string rest = line.substr(20); // Пропускаем пробел после временной метки

            // Разбиваем остальную часть строки на токены по пробелам
            std::regex delim(R"([ ]+)");
            std::sregex_token_iterator it(rest.begin(), rest.end(), delim, -1);
            std::sregex_token_iterator end;
            std::vector<std::string> tokens(it, end);

            if (tokens.size() < 9) {
                std::cerr << "Skipping invalid line (not enough tokens): " << line << std::endl;
                continue;
            }

            try {
                USBLData entry;
                entry.datetime = datetime;
                entry.x = std::stod(tokens[0]);
                entry.y = std::stod(tokens[1]);
                entry.z = std::stod(tokens[2]);
                entry.azimuth = std::stod(tokens[3]);
                entry.localDepth = std::stod(tokens[4]);
                entry.remoteDepth = std::stod(tokens[5]);
                entry.propagationTime = std::stod(tokens[6]);
                entry.rs = std::stod(tokens[7]);
                entry.rh = std::stod(tokens[8]);

                entries.push_back(entry);
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing line: " << line << ", error: " << e.what() << std::endl;
                continue;
            }
        }

        if (entries.empty()) {
            std::cerr << "No valid USBL data loaded from " << filename << std::endl;
        }
        else {
            std::cout << "Loaded " << entries.size() << " USBL entries" << std::endl;
        }

        return entries;
    }

    // Загрузка данных с СТЗ (Системы Технического Зрения)
    std::vector<CameraFrame> loadCameraFrames(const std::filesystem::path& folderPath) {
        std::vector<CameraFrame> frames;

        writeLog("Loading frames from folder:" + folderPath.u8string(), LOGTYPE::INFO);
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();

                if (ext != ".png" && ext != ".jpg") {
                    std::cerr << "Skipping non-image file: " << filename << std::endl;
                    writeLog("Skipping non-image file: " + filename, LOGTYPE::ERROR);
                    continue;
                }
                if (filename.substr(0, 2) != "L_" && filename.substr(0, 2) != "R_") {
                    std::cerr << "Skipping file with invalid prefix: " << filename << std::endl;
                    writeLog("Skipping file with invalid prefix: " + filename, LOGTYPE::ERROR);
                    continue;
                }

                try {
                    CameraFrame frame;
                    frame.image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
                    if (frame.image.empty()) {
                        std::cerr << "Failed to load image: " << filename << std::endl;
                        writeLog("Failed to load image: " + filename, LOGTYPE::ERROR);
                        continue;
                    }
                    frame.timestamp = parseTimestamp(filename);
                    frame.filename = filename;
                    frames.push_back(frame);
                }
                catch (const std::exception& e) {
                    std::cerr << "Skipping frame " << filename << ": " << e.what() << std::endl;
                    writeLog("Skipping frame " + filename + ": " + std::string(e.what()),mrcv::LOGTYPE::EXCEPTION);
                }
            }
        }

        if (frames.empty()) {
            std::cerr << "Error: No valid frames loaded from folder: " << folderPath << std::endl;
            writeLog("Error: No valid frames loaded from folder: " + folderPath.u8string(), LOGTYPE::ERROR);
        }
        else {
            std::cout << "Loaded " << frames.size() << " valid frames" << std::endl;
            writeLog("Loaded valid frames: " + frames.size(), mrcv::LOGTYPE::INFO);
        }

        return frames;
    }

    // Загрузка данных с инерциальной системы аппарата
    std::vector<IMUData> loadIMUData(const std::string& csvFile)
    {
        std::vector<IMUData> imuData;
        std::ifstream file(csvFile);
        std::string line;

        writeLog("Loading IMU data from file:" + csvFile, LOGTYPE::INFO);
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
        writeLog("IMU data download is complete", LOGTYPE::INFO);
        return imuData;
    }

    // Поиск ближайшего к временной метке значения от IMU
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

    // Поиск ближайшего к временной метке значения от Камеры
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

    // Поиск ближайшего к временной метке значения от USBL
    const USBLData* findNearestUSBL(std::chrono::system_clock::time_point& ts,
        const std::vector<USBLData>& usblData)
    {
        const USBLData* nearest = nullptr;
        auto minDiff = std::chrono::milliseconds(100000);

        for (const auto& usbl : usblData) {
            try {
                auto usblTs = parseTimestamp(usbl.datetime);
                auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(ts - usblTs);
                if (std::abs(diff.count()) < std::abs(minDiff.count())) {
                    minDiff = diff;
                    nearest = &usbl;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error parsing USBL datetime " << usbl.datetime << ": " << e.what() << std::endl;
                continue;
            }
        }
        return nearest;
    }

    // Функция для визуализации результата
    void visualizeResult(const std::string& yamlFile, const std::string& frameFolder)
    {
        YAML::Node root;
        try {
            root = YAML::LoadFile(yamlFile);
        }
        catch (const std::exception& e) {
            std::cerr << "YAML Load error: " << e.what() << std::endl;
            writeLog("YAML Load error: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            return;
        }

        for (const auto& entry : root) {
            try {
                auto timestamp_ms = entry["timestamp"].as<long long>();
                auto accel = entry["accel"].as<std::vector<double>>();
                auto gyro = entry["gyro"].as<std::vector<double>>();
                auto relativeCoords = entry["relativeCoords"].as<std::vector<double>>();
                auto azimuth = entry["azimuth"].as<float>();
                auto localDepth = entry["localDepth"].as<float>();
                auto remoteDepth = entry["remoteDepth"].as<float>();
                auto imageName = entry["image"].as<std::string>();

                // Загрузка изображения
                std::filesystem::path imagePath = std::filesystem::path(frameFolder) / imageName;
                cv::Mat image = cv::imread(imagePath.string());
                if (image.empty()) {
                    std::cerr << "Failed to load image: " << imagePath << std::endl;
                    writeLog("Failed to load image: " + imagePath.string(), mrcv::LOGTYPE::ERROR);
                    continue;
                }

                // Центр изображения
                int cx = image.cols / 2;
                int cy = image.rows / 2;

                // Угол рысканья (yaw) из IMU (предполагается, что gyro[2] — угол в радианах)
                float yaw = static_cast<float>(gyro[2]);
                // Примечание: Если gyro[2] — угловая скорость (рад/с), нужно интегрировать:
                // yaw = integrate_gyro_z(gyro[2], previous_yaw, time_delta);

                // Относительные координаты USBL
                float x = relativeCoords[0]; // X (вправо/влево)
                float y = relativeCoords[1]; // Y (вперёд/назад)
                float z = relativeCoords[2]; // Z (относительная глубина)

                // Пересчёт координат с учётом yaw
                cv::Point2f relativePoint(
                    x * std::cos(-yaw) - y * std::sin(-yaw), // X' в системе камеры
                    x * std::sin(-yaw) + y * std::cos(-yaw)  // Y' в системе камеры
                );

                // Масштабирование для проекции на изображение
                float scale = 10.0f; // Масштаб (настраиваемый)
                cv::Point2f pointer(cx + relativePoint.x * scale, cy - relativePoint.y * scale);

                // Проверка направления с использованием азимута
                float azimuth_rad = azimuth * (CV_PI / 180.0f); // Азимут в радианах

                // Определяем поле зрения камеры (120° по горизонтали)
                float fov_horizontal = 120.0f * (CV_PI / 180.0f); // Угол обзора в радианах
                float focal_length = (image.cols / 2.0f) / std::tan(fov_horizontal / 2.0f);

                // Проверяем, находится ли устройство в зоне видимости
                bool is_in_fov = false;
                if (relativePoint.y > 0) { // Устройство впереди
                    // Проверяем, попадает ли азимут в угол обзора
                    if (std::abs(azimuth_rad) < fov_horizontal / 2.0f) {
                        // Проверяем границы изображения
                        if (pointer.x >= 0 && pointer.x < image.cols && pointer.y >= 0 && pointer.y < image.rows) {
                            is_in_fov = true;
                        }
                    }
                }

                // Определяем выше/ниже на основе глубин
                bool is_above = remoteDepth < localDepth; // Меньшая глубина => выше

                if (is_in_fov) {
                    // Отрисовка точки (красный круг)
                    cv::circle(image, pointer, 4, cv::Scalar(0, 0, 255), -1);
                }
                else {
                    // Определяем направление стрелки
                    bool is_left = azimuth_rad < 0; // Отрицательный азимут => левее

                    // Положение и угол стрелки
                    cv::Point2f arrow_pos;
                    float arrow_angle = 0.0f;

                    if (relativePoint.y <= 0) { // Устройство позади
                        // Стрелка на верхнем или нижнем краю
                        if (is_above) {
                            arrow_pos = cv::Point2f(is_left ? image.cols * 0.25f : image.cols * 0.75f, 10);
                            arrow_angle = is_left ? 135.0f : 45.0f; // Вверх-влево/вправо
                        }
                        else {
                            arrow_pos = cv::Point2f(is_left ? image.cols * 0.25f : image.cols * 0.75f, image.rows - 10);
                            arrow_angle = is_left ? -135.0f : -45.0f; // Вниз-влево/вправо
                        }
                    }
                    else {
                        // Устройство впереди, но вне поля зрения
                        if (is_left) {
                            arrow_pos = cv::Point2f(10, cy);
                            arrow_angle = 180.0f; // Влево
                        }
                        else {
                            arrow_pos = cv::Point2f(image.cols - 10, cy);
                            arrow_angle = 0.0f; // Вправо
                        }
                        // Корректировка по высоте
                        if (is_above) {
                            arrow_pos.y = std::max(10.0f, arrow_pos.y - image.rows * 0.25f);
                        }
                        else {
                            arrow_pos.y = std::min(float(image.rows - 10), arrow_pos.y + image.rows * 0.25f);
                        }
                    }

                    // Отрисовка красной стрелки
                    float arrow_length = 20.0f;
                    cv::Point2f arrow_end(
                        arrow_pos.x + arrow_length * std::cos(arrow_angle * CV_PI / 180.0f),
                        arrow_pos.y - arrow_length * std::sin(arrow_angle * CV_PI / 180.0f)
                    );
                    cv::arrowedLine(image, arrow_pos, arrow_end, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.3);
                }

                // Подпись с временной меткой и информацией
                std::string label = "t: " + std::to_string(timestamp_ms) +
                    ", Azimuth: " + std::to_string(azimuth) +
                    ", Depth: " + std::to_string(remoteDepth) + "/" + std::to_string(localDepth);
                cv::putText(image, label, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                // Показываем изображение
                cv::imshow("Visualized Frame", image);
                cv::waitKey(0);

            }
            catch (const std::exception& e) {
                std::cerr << "Error while parsing entry: " << e.what() << std::endl;
                writeLog("Error while parsing entry: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
                continue;
            }
        }
        cv::destroyAllWindows();
    }

    int fuseSensorData(const std::string& usblPath,
        const std::string& imuPath,
        const std::string& camFolder,
        const std::string& outYAML, bool visFlag)
    {
        writeLog("The fusion function has been started", mrcv::LOGTYPE::INFO);

        std::vector<USBLData> usblData = loadAcousticCSV(usblPath);
        if (usblData.empty()) {
            std::cerr << "Error: No USBL data loaded from " << usblPath << std::endl;
            writeLog("Error: No USBL data loaded from " + usblPath, mrcv::LOGTYPE::ERROR);
            return 1;
        }

        std::vector<IMUData> imuData = loadIMUData(imuPath);
        if (imuData.empty()) {
            std::cerr << "Error: No IMU data loaded from " << imuPath << std::endl;
            writeLog("Error: No IMU data loaded from " + imuPath, mrcv::LOGTYPE::ERROR);
            return 2;
        }

        std::vector<CameraFrame> frames = loadCameraFrames(camFolder);
        if (frames.empty()) {
            std::cerr << "Error: No valid camera frames loaded from " << camFolder << std::endl;
            writeLog("Error: No valid camera frames loaded from " + camFolder, mrcv::LOGTYPE::ERROR);
            return 3;
        }

        std::vector<FusedData> fused;

        for (const auto& usbl : usblData) {
            try {
                auto ts = parseTimestamp(usbl.datetime);

                const IMUData* imuNearest = findNearestIMU(ts, imuData);
                const CameraFrame* camNearest = findNearestFrame(ts, frames);

                if (imuNearest && camNearest) {
                    FusedData data;
                    data.timestamp = ts;

                    std::copy(std::begin(imuNearest->accel), std::end(imuNearest->accel), data.accel);
                    std::copy(std::begin(imuNearest->gyro), std::end(imuNearest->gyro), data.gyro);

                    data.position[0] = 0;
                    data.position[1] = 0;
                    data.position[2] = 0;

                    data.relativeCoords[0] = usbl.x;
                    data.relativeCoords[1] = usbl.y;
                    data.relativeCoords[2] = usbl.z;

                    data.featurePoints = {};

                    // Сохраняем дополнительные параметры USBL
                    data.azimuth = usbl.azimuth;
                    data.localDepth = usbl.localDepth;
                    data.remoteDepth = usbl.remoteDepth;

                    data.imageFilename = camNearest->filename;

                    fused.push_back(data);
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing USBL entry with datetime " << usbl.datetime << ": " << e.what() << std::endl;
                writeLog("Error processing USBL entry with datetime " + usbl.datetime + ": " + e.what(), mrcv::LOGTYPE::EXCEPTION);
                continue;
            }
        }

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
                out << YAML::Key << "azimuth" << YAML::Value << d.azimuth;
                out << YAML::Key << "localDepth" << YAML::Value << d.localDepth;
                out << YAML::Key << "remoteDepth" << YAML::Value << d.remoteDepth;
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
            writeLog("YAML write error: " + std::string(e.what()), mrcv::LOGTYPE::EXCEPTION);
            return 4;
        }
        writeLog("Fusion function is complete", mrcv::LOGTYPE::INFO);
        return 0;
    }

}