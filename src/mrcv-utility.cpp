#include <mrcv/mrcv.h>

namespace mrcv
{
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
     * @brief Функция записи видеопотока на диск.
     * 
     * Функция может использоваться дле реализации работы видеорегистратора.
     * 
     * @param cameraID - ID камеры.
     * @param recorderInterval - Интервал записи в секундах.
     * @param fileName - Маска фала.
     * @param codec - Кодек, используемый для создания видеофайла.
     * @return - код результата работы функции. 0 - Success; 1 - ID камеры задан неверно; 2 - Интервал захвата меньше минимального; 3 - Не удалось захватить камеру; -1 - Unhandled Exception.
     */
    int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec)
    {
        try
        {
            if (cameraID < 0)
                return 1; // Код ошибки - ID камеры задан неверно

            if (recorderInterval < UTILITY_DEFAULT_RECORDER_INTERVAL)
                return 2; // Код ошибки - Интервал записи не может быть меньше UTILITY_DEFAULT_RECORDER_INTERVAL сек.

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
                break;
            case CODEC::mp4v:
                fourccCode = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                fileExtension = ".mp4";
                break;
            case CODEC::h265:
                fourccCode = cv::VideoWriter::fourcc('h', '2', '6', '5');
                fileExtension = ".mp4";
                break;
            case CODEC::XVID:
                fourccCode = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
                fileExtension = ".avi";
                break;
            default:
                fourccCode = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
                fileExtension = ".avi";
                break;
            }

            // Генерируем имя файла с привязкой к текущему времени
            std::string fullFileName = generateUniqueFileName(fileName, fileExtension);

            // Создаем объект для захвата камеры
            cv::VideoCapture videoCapture(cameraID);

            if (!videoCapture.isOpened())
                return 3; // Код ошибки - Не удалось захватить камеру

            // Разрешение камеры
            cv::Size cameraResolution((int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH),
                (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

            // FPS камеры
            int cameraFPS = (int)videoCapture.get(cv::CAP_PROP_FPS);
            if (cameraFPS == 0)
                cameraFPS = UTILITY_DEFAULT_CAMERA_FPS;

            // Объект для записи видеопотока
            videoWriter = cv::VideoWriter(fullFileName, fourccCode, cameraFPS, cameraResolution);

            clock_t timerStart = clock();
            cv::Mat videoFrame;

            // Цикл записи видеопотока в файл
            while ((clock() - timerStart) < (recorderInterval * CLOCKS_PER_SEC))
            {
                videoCapture >> videoFrame;
                videoWriter.write(videoFrame);
            }

            // Освобождение объекта записи видеопотока
            videoWriter.release();

            // Освобождение объекта захвата камеры
            if (videoCapture.isOpened())
                videoCapture.release();

            // Возврат кода нормального завершения работы
            return 0;
        }
        catch (...)
        {
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