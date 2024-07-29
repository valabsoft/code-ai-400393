#include <mrcv/mrcv.h>

namespace mrcv
{
    int readImage(std::string pathToImage)
    {
        cv::Mat img = cv::imread(pathToImage, cv::IMREAD_COLOR);

        if (img.empty())
        {
            std::cout << "Could not read the image: " << pathToImage << std::endl;
            return 1;
        }

        cv::imshow(pathToImage, img);
        cv::waitKey(0);

        return 0;
    }

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

    int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec)
    {
        if (cameraID < 0)
            return -1; // Код ошибки - ID камеры задан неверно

        if (recorderInterval < UTILITY_DEFAULT_RECORDER_INTERVAL)
            return -2; // Код ошибки - Интервал записи не может быть меньше UTILITY_DEFAULT_RECORDER_INTERVAL сек.

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
            return -3; // Код ошибки - Не удалось захватить камеру

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

    std::string getOpenCVBuildInformation()
    {
        return cv::getBuildInformation().c_str();
    }
}