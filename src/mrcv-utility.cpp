#include <mrcv/mrcv.h>

namespace mrcv
{
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
     * @brief ������� ������ ����������� �� ����.
     * 
     * ������� ����� �������������� ��� ���������� ������ �����������������.
     * 
     * @param cameraID - ID ������.
     * @param recorderInterval - �������� ������ � ��������.
     * @param fileName - ����� ����.
     * @param codec - �����, ������������ ��� �������� ����������.
     * @return - ��� ���������� ������ �������. 0 - Success; 1 - ID ������ ����� �������; 2 - �������� ������� ������ ������������; 3 - �� ������� ��������� ������; -1 - Unhandled Exception.
     */
    int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec)
    {
        try
        {
            if (cameraID < 0)
                return 1; // ��� ������ - ID ������ ����� �������

            if (recorderInterval < UTILITY_DEFAULT_RECORDER_INTERVAL)
                return 2; // ��� ������ - �������� ������ �� ����� ���� ������ UTILITY_DEFAULT_RECORDER_INTERVAL ���.

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

            // ���������� ��� ����� � ��������� � �������� �������
            std::string fullFileName = generateUniqueFileName(fileName, fileExtension);

            // ������� ������ ��� ������� ������
            cv::VideoCapture videoCapture(cameraID);

            if (!videoCapture.isOpened())
                return 3; // ��� ������ - �� ������� ��������� ������

            // ���������� ������
            cv::Size cameraResolution((int)videoCapture.get(cv::CAP_PROP_FRAME_WIDTH),
                (int)videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

            // FPS ������
            int cameraFPS = (int)videoCapture.get(cv::CAP_PROP_FPS);
            if (cameraFPS == 0)
                cameraFPS = UTILITY_DEFAULT_CAMERA_FPS;

            // ������ ��� ������ �����������
            videoWriter = cv::VideoWriter(fullFileName, fourccCode, cameraFPS, cameraResolution);

            clock_t timerStart = clock();
            cv::Mat videoFrame;

            // ���� ������ ����������� � ����
            while ((clock() - timerStart) < (recorderInterval * CLOCKS_PER_SEC))
            {
                videoCapture >> videoFrame;
                videoWriter.write(videoFrame);
            }

            // ������������ ������� ������ �����������
            videoWriter.release();

            // ������������ ������� ������� ������
            if (videoCapture.isOpened())
                videoCapture.release();

            // ������� ���� ����������� ���������� ������
            return 0;
        }
        catch (...)
        {
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