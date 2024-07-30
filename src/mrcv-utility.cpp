#include <mrcv/mrcv.h>

namespace mrcv
{
    /**
     * @brief ������� ��������� ����������� ����� �����.
     * @param fileName - ����� ����� ����� (������� �����).
     * @param fileExtension - ���������� ���� � ������� ".abc".
     * @return - ���������� ��� ����� � ������� filename_%d%m%Y_%H%M%S.abc.
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
     * @brief ������� ��������� ����������� ����� ���-�����.
     * @return - ���������� ��� ����� � ������� dd-mm-yyyy.log.
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
     * @brief ������� ��� �������� ���������� �����-���� ������ ������� ����������
     * @param logText - ����� ��������� ��� ������ � ���-����
     * @param logType - ��� ��������� � ���-�����
     */
    void writeLog(std::string logText, LOGTYPE logType = LOGTYPE::INFO)
    {
        if (!IS_DEBUG_LOG_ENABLED)
            return;

        try
        {
            std::filesystem::path pathToLogDirectory = std::filesystem::current_path() / "log";
            std::filesystem::directory_entry directoryEntry{ pathToLogDirectory };

            // ��������� ������������� ����� log � ������� ��������
            bool isLogDirectoryExists = directoryEntry.exists();

            if (!isLogDirectoryExists)
            {
                // ���� ����� log �� ����������, ������� ��
                isLogDirectoryExists = std::filesystem::create_directory(pathToLogDirectory);
                if (!isLogDirectoryExists)
                {
                    return;
                }
            }

            // ���������� ��� ������
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

            // ���������� ��������� �����
            struct tm currentTime;
            time_t nowTime = time(0);
            localtime_s(&currentTime, &nowTime);

            std::ostringstream outStringStream;
            outStringStream << std::put_time(&currentTime, "%H:%M:%S");
            std::string logTime = outStringStream.str();

            // ���������� ���������� ��� ����� � ������� dd-mm-yyyy.log
            std::string logFileName = generateUniqueLogFileName();
            std::filesystem::path pathToLogFile = pathToLogDirectory / logFileName;

            std::ofstream logFile; // ������������� ���-�����

            if (std::filesystem::exists(pathToLogFile))
            {
                // ���� ���� ���� ����������, ��������� ���� ��� �������� � ��������� ������ � �����            
                logFile.open(pathToLogFile.c_str(), std::ios_base::app);
            }
            else
            {
                // ���� ���� ���� �� ����������, ������� ��� � ��������� �������
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
     * @brief ������� �������� �����������.
     * 
     * ������� ������������ ��� �������� ����������� � �������� � ����������� ������������ ����������� � ��������� ����.
     * 
     * @param image - ������ cv::Mat ��� �������� ������������ �����������.
     * @param pathToImage - ������ ���� � ����� � ������������.
     * @param showImage - ����, ���������� �� ����������� ���������� ���� (false �� ���������).
     * @return - ��� ���������� ������ �������. 0 - Success; 1 - ���������� ������� �����������; -1 - Unhandled Exception.
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
     * @brief ������� ������ ����������� �� ����.
     * 
     * ������� ����� �������������� ��� ���������� ������ �����������������.
     * 
     * @param cameraID - ID ������.
     * @param recorderInterval - �������� ������ � ��������.
     * @param fileName - ����� ����.
     * @param codec - �����, ������������ ��� �������� ����������.
     * @return - ��� ���������� ������ �������. 0 - Success; 1 - ID ������ ����� �������; 2 - �������� ������� ������ ������������; 3 - �� ������� ��������� ������; 4 - �� ������� ������� ������� cv::VideoWriter; -1 - Unhandled Exception.
     */
    int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec)
    {
        writeLog("����� �-��� recordVideo(" + std::to_string(cameraID) + ", " + std::to_string(recorderInterval) +
            ", \"" + fileName + "\", " + std::to_string((int)codec) + ")");
        writeLog("����� ������ �����");
        
        try
        {
            if (cameraID < 0)
            {
                writeLog("ID ������ ����� �������", LOGTYPE::ERROR);
                return 1;
            }   

            if (recorderInterval < UTILITY_DEFAULT_RECORDER_INTERVAL)
            {
                writeLog("�������� ������ �� ����� ���� ������ UTILITY_DEFAULT_RECORDER_INTERVAL ���.", LOGTYPE::ERROR);
                return 2;
            }                

            if (fileName.empty())
                fileName = UTILITY_DEFAULT_RECORDER_FILENAME;

            // ������� ������ ��� ������ �����
            cv::VideoWriter videoWriter;
            std::string fileExtension;

            // ���������� ��������� videoWriter
            // � ���������� ��������� ����� � ����������� �� ������
            int fourccCode;
            switch (codec)
            {
            case CODEC::MJPG:
                fourccCode = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                fileExtension = ".mp4";
                writeLog("��� ������: MJPG");
                break;
            case CODEC::mp4v:
                fourccCode = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                fileExtension = ".mp4";
                writeLog("��� ������: mp4v");
                break;
            case CODEC::h265:
                fourccCode = cv::VideoWriter::fourcc('h', '2', '6', '5');
                fileExtension = ".mp4";
                writeLog("��� ������: h265");
                break;
            case CODEC::XVID:
                fourccCode = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
                fileExtension = ".avi";
                writeLog("��� ������: XVID");
                break;
            default:
                fourccCode = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
                fileExtension = ".avi";
                writeLog("��� ������: XVID");
                break;
            }

            // ���������� ��� ����� � ��������� � �������� �������
            std::string fullFileName = generateUniqueFileName(fileName, fileExtension);
            writeLog("��� ��������� �����: " + fullFileName);

            // ������� ������ ��� ������� ������
            cv::VideoCapture videoCapture(cameraID);

            if (!videoCapture.isOpened())
            {
                writeLog("�� ������� ��������� ������", LOGTYPE::ERROR);
                return 3;
            }   

            // ���������� ������
            cv::Size cameraResolution(
                (int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH),
                (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

            // FPS ������
            int cameraFPS = (int)videoCapture.get(cv::CAP_PROP_FPS);
            if (cameraFPS == 0)
                cameraFPS = UTILITY_DEFAULT_CAMERA_FPS;

            writeLog("�������� ������� cv::VideoWriter():");
            writeLog("\tfullFileName: " + fullFileName);
            writeLog("\tfourccCode: " + std::to_string(fourccCode));
            writeLog("\tcameraFPS: " + std::to_string(cameraFPS));
            writeLog("\tcameraResolution (width): " + std::to_string(cameraResolution.width));
            writeLog("\tcameraResolution (height): " + std::to_string(cameraResolution.height));

            // ������ ��� ������ �����������
            videoWriter = cv::VideoWriter(fullFileName, fourccCode, cameraFPS, cameraResolution);

            if (!videoWriter.isOpened())
            {
                writeLog("�� ������� ������� ������ cv::VideoWriter", LOGTYPE::ERROR);
                
                // ������������ ������� ������ �����������
                videoWriter.release();

                // ������������ ������� ������� ������
                if (videoCapture.isOpened())
                    videoCapture.release();
                
                return 4;
            }

            clock_t timerStart = clock();
            cv::Mat videoFrame;

            // ���� ������ ����������� � ����
            while ((clock() - timerStart) < (recorderInterval * CLOCKS_PER_SEC))
            {
                videoCapture >> videoFrame;
                videoWriter.write(videoFrame);
            }

            // ������������ ������� ������ �����������
            writeLog("������������ ������� cv::VideoWriter()");
            videoWriter.release();

            // ������������ ������� ������� ������
            writeLog("������������ ������� cv::VideoCapture()");
            if (videoCapture.isOpened())
                videoCapture.release();

            // ������� ���� ����������� ���������� ������
            writeLog("���� ������ �����.");
            return 0;
        }
        catch (...)
        {
            writeLog("����������� ����������...", LOGTYPE::EXCEPTION);
            return -1;
        }
        
    }
    
    /**
     * @brief ������� ������ ���������� � ������� ������ OpenCV.
     * @return ������ � ��������������� �����������.
     */
    std::string getOpenCVBuildInformation()
    {
        return cv::getBuildInformation().c_str();
    }
}