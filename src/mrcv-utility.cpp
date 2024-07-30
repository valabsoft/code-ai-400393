#include <mrcv/mrcv.h>

namespace mrcv
{
    /**
     * @brief Функция генерации уникального имени файла.
     * @param fileName - маска имени файла (префикс имени).
     * @param fileExtension - расширение файл в формате ".abc".
     * @return - Уникальное имя файла в формате filename_%d%m%Y_%H%M%S.abc.
     */
    std::string generateUniqueFileName(std::string fileName, std::string fileExtension)
    {
        struct tm currentTime;
        time_t nowTime = time(0);
        localtime_s(&currentTime, &nowTime);

        std::ostringstream outStringStream;
        std::string fullFileName = fileName + "_%d%m%Y_%H%M%S" + fileExtension;
        outStringStream << std::put_time(&currentTime, fullFileName.c_str());
        return outStringStream.str();
    }

    /**
     * @brief Функция генерации уникального имени лог-файла.
     * @return - Уникальное имя файла в формате dd-mm-yyyy.log.
     */
    std::string generateUniqueLogFileName()
    {
        struct tm currentTime;
        time_t nowTime = time(0);
        localtime_s(&currentTime, &nowTime);

        std::ostringstream outStringStream;
        std::string fullFileName = "%d-%m-%Y.log";
        outStringStream << std::put_time(&currentTime, fullFileName.c_str());
        return outStringStream.str();
    }

    /**
     * @brief Функция для создания текстового файла-лога работы функций библиотеки
     * @param logText - текст сообщения для записи в лог-файл
     * @param logType - тип сообщения в лог-файле
     */
    void writeLog(std::string logText, LOGTYPE logType = LOGTYPE::INFO)
    {
        if (!IS_DEBUG_LOG_ENABLED)
            return;

        try
        {
            std::filesystem::path pathToLogDirectory = std::filesystem::current_path() / "log";
            std::filesystem::directory_entry directoryEntry{ pathToLogDirectory };

            // Проверяем существование папки log в рабочем каталоге
            bool isLogDirectoryExists = directoryEntry.exists();

            if (!isLogDirectoryExists)
            {
                // Если папка log не существует, создаем ее
                isLogDirectoryExists = std::filesystem::create_directory(pathToLogDirectory);
                if (!isLogDirectoryExists)
                {
                    return;
                }
            }

            // Определяем тип записи
            std::string logTypeAbbreviation;
            switch (logType)
            {
            case mrcv::LOGTYPE::DEBUG:
                logTypeAbbreviation = "DEBG";
                break;
            case mrcv::LOGTYPE::ERROR:
                logTypeAbbreviation = "ERRR";
                break;
            case mrcv::LOGTYPE::EXCEPTION:
                logTypeAbbreviation = "EXCP";
                break;
            case mrcv::LOGTYPE::INFO:
                logTypeAbbreviation = "INFO";
                break;
            case mrcv::LOGTYPE::WARNING:
                logTypeAbbreviation = "WARN";
                break;
            default:
                logTypeAbbreviation = "INFO";
                break;
            }

            // Определяем временную метку
            struct tm currentTime;
            time_t nowTime = time(0);
            localtime_s(&currentTime, &nowTime);

            std::ostringstream outStringStream;
            outStringStream << std::put_time(&currentTime, "%H:%M:%S");
            std::string logTime = outStringStream.str();

            // Генерируем уникальное имя файла в формате dd-mm-yyyy.log
            std::string logFileName = generateUniqueLogFileName();
            std::filesystem::path pathToLogFile = pathToLogDirectory / logFileName;

            std::ofstream logFile; // Идентификатор лог-файла

            if (std::filesystem::exists(pathToLogFile))
            {
                // Если файл лога существует, открываем файл для дозаписи и добавляем строку в конец            
                logFile.open(pathToLogFile.c_str(), std::ios_base::app);
            }
            else
            {
                // Если файл лога не существует, создаем его и добавляем строчку
                logFile.open(pathToLogFile.c_str(), std::ios_base::out);
            }

            if (logFile.is_open())
            {
                logFile << logTime << " | " << logTypeAbbreviation << " | " << logText << std::endl;
                logFile.close();
            }
        }
        catch (...)
        {
            return;
        }
    }

    /**
     * @brief Функция загрузки изображения.
     * 
     * Функция используется для загрузки изображения с носителя и отображения загруженного изображения в модальном окне.
     * 
     * @param image - объект cv::Mat для хранения загруженного изображения.
     * @param pathToImage - полный путь к файлу с изображением.
     * @param showImage - флаг, отвечающий за отображение модального окна (false по умолчанию).
     * @return - код результата работы функции. 0 - Success; 1 - Невозможно открыть изображение; -1 - Unhandled Exception.
     */
    int readImage(cv::Mat& image, std::string pathToImage, bool showImage)
    {
        try
        {
            image = cv::imread(pathToImage, cv::IMREAD_COLOR);

            if (image.empty())
            {
                std::cout << "Could not read the image: " << pathToImage << std::endl;
                return 1;
            }

            if (showImage)
            {
                cv::imshow(pathToImage, image);
                cv::waitKey(0);
            }
        }
        catch (...)
        {
            return -1; // Unhandled Exception
        }                

        return 0; // SUCCESS
    }

    /**
     * @brief Функция записи видеопотока на диск.
     * 
     * Функция может использоваться дле реализации работы видеорегистратора.
     * 
     * @param cameraID - ID камеры.
     * @param recorderInterval - Интервал записи в секундах.
     * @param fileName - Маска фала.
     * @param codec - Кодек, используемый для создания видеофайла.
     * @return - код результата работы функции. 0 - Success; 1 - ID камеры задан неверно; 2 - Интервал захвата меньше минимального; 3 - Не удалось захватить камеру; 4 - Не удалось создать объектс cv::VideoWriter; -1 - Unhandled Exception.
     */
    int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec)
    {
        writeLog("Вызов ф-ции recordVideo(" + std::to_string(cameraID) + ", " + std::to_string(recorderInterval) +
            ", \"" + fileName + "\", " + std::to_string((int)codec) + ")");
        writeLog("Старт записи видео");
        
        try
        {
            if (cameraID < 0)
            {
                writeLog("ID камеры задан неверно", LOGTYPE::ERROR);
                return 1;
            }   

            if (recorderInterval < UTILITY_DEFAULT_RECORDER_INTERVAL)
            {
                writeLog("Интервал записи не может быть меньше UTILITY_DEFAULT_RECORDER_INTERVAL сек.", LOGTYPE::ERROR);
                return 2;
            }                

            if (fileName.empty())
                fileName = UTILITY_DEFAULT_RECORDER_FILENAME;

            // Создаем объект для записи видео
            cv::VideoWriter videoWriter;
            std::string fileExtension;

            // Определяем параметры videoWriter
            // и расширение выходного файла в зависимости от кодека
            int fourccCode;
            switch (codec)
            {
            case CODEC::MJPG:
                fourccCode = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                fileExtension = ".mp4";
                writeLog("Тип кодека: MJPG");
                break;
            case CODEC::mp4v:
                fourccCode = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                fileExtension = ".mp4";
                writeLog("Тип кодека: mp4v");
                break;
            case CODEC::h265:
                fourccCode = cv::VideoWriter::fourcc('h', '2', '6', '5');
                fileExtension = ".mp4";
                writeLog("Тип кодека: h265");
                break;
            case CODEC::XVID:
                fourccCode = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
                fileExtension = ".avi";
                writeLog("Тип кодека: XVID");
                break;
            default:
                fourccCode = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
                fileExtension = ".avi";
                writeLog("Тип кодека: XVID");
                break;
            }

            // Генерируем имя файла с привязкой к текущему времени
            std::string fullFileName = generateUniqueFileName(fileName, fileExtension);
            writeLog("Имя выходного файла: " + fullFileName);

            // Создаем объект для захвата камеры
            cv::VideoCapture videoCapture(cameraID);

            if (!videoCapture.isOpened())
            {
                writeLog("Не удалось захватить камеру", LOGTYPE::ERROR);
                return 3;
            }   

            // Разрешение камеры
            cv::Size cameraResolution(
                (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH),
                (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

            // FPS камеры
            int cameraFPS = (int)videoCapture.get(cv::CAP_PROP_FPS);
            if (cameraFPS == 0)
                cameraFPS = UTILITY_DEFAULT_CAMERA_FPS;

            writeLog("Создание объекта cv::VideoWriter():");
            writeLog("\tfullFileName: " + fullFileName);
            writeLog("\tfourccCode: " + std::to_string(fourccCode));
            writeLog("\tcameraFPS: " + std::to_string(cameraFPS));
            writeLog("\tcameraResolution (width): " + std::to_string(cameraResolution.width));
            writeLog("\tcameraResolution (height): " + std::to_string(cameraResolution.height));

            // Объект для записи видеопотока
            videoWriter = cv::VideoWriter(fullFileName, fourccCode, cameraFPS, cameraResolution);

            if (!videoWriter.isOpened())
            {
                writeLog("Не удалось создать объект cv::VideoWriter", LOGTYPE::ERROR);
                
                // Освобождение объекта записи видеопотока
                videoWriter.release();

                // Освобождение объекта захвата камеры
                if (videoCapture.isOpened())
                    videoCapture.release();
                
                return 4;
            }

            clock_t timerStart = clock();
            cv::Mat videoFrame;

            // Цикл записи видеопотока в файл
            while ((clock() - timerStart) < (recorderInterval * CLOCKS_PER_SEC))
            {
                videoCapture >> videoFrame;
                videoWriter.write(videoFrame);
            }

            // Освобождение объекта записи видеопотока
            writeLog("Освобождение объекта cv::VideoWriter()");
            videoWriter.release();

            // Освобождение объекта захвата камеры
            writeLog("Освобождение объекта cv::VideoCapture()");
            if (videoCapture.isOpened())
                videoCapture.release();

            // Возврат кода нормального завершения работы
            writeLog("Стоп записи видео.");
            return 0;
        }
        catch (...)
        {
            writeLog("Неизвестное исключение...", LOGTYPE::EXCEPTION);
            return -1;
        }
        
    }
    
    /**
     * @brief Функция вывода информации о текущей сборке OpenCV.
     * @return Строка с диагностической информацией.
     */
    std::string getOpenCVBuildInformation()
    {
        return cv::getBuildInformation().c_str();
    }
}