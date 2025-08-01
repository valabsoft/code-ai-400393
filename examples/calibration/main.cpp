#include <mrcv/mrcv.h>
#include <thread>
#include <chrono>

using namespace std::chrono_literals;

int main()
{
    // Структура для хранения конфигурации процедуры калибровки
    mrcv::CalibrationConfig config;

    // Пути к конфигурационному файлу
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "files";
    std::filesystem::path configPath = path / "config.dat";

    // Чтение конфигуарции процедуры калибровки
    mrcv::readCalibrartionConfigFile(configPath.u8string(), config);
    
    // Логирование данных
    mrcv::writeLog("Path to file: " + configPath.u8string());
    mrcv::writeLog("Images count: " + std::to_string(config.image_count));
    mrcv::writeLog("Path to images: " + config.folder_name);
    mrcv::writeLog("Chessboard columns count: " + std::to_string(config.keypoints_c));
    mrcv::writeLog("Chessboard rows count: " + std::to_string(config.keypoints_r));
    mrcv::writeLog("Square size: " + std::to_string(config.square_size));

    mrcv::CalibrationParametersMono monoParL;
    mrcv::CalibrationParametersMono monoParR;
    mrcv::CalibrationParametersStereo stereoPar;

    cv::Mat frameLeft;
    cv::Mat frameRight;

    cv::VideoCapture capLeft;
    cv::VideoCapture capRight;

    std::filesystem::path leftFramePath = currentPath / config.folder_name / "L";
    std::filesystem::path rightFramePath = currentPath / config.folder_name / "R";

    if (!std::filesystem::is_directory(currentPath / config.folder_name))
    {
        mrcv::writeLog("Create folder: " + (currentPath / config.folder_name).u8string());
        std::filesystem::create_directory(currentPath / config.folder_name);
    }
    else
    {
        if (std::filesystem::is_directory(leftFramePath))
        {
            std::filesystem::remove_all(leftFramePath);                        
        }
        mrcv::writeLog("Create folder: " + leftFramePath.u8string());
        std::filesystem::create_directory(leftFramePath);

        if (std::filesystem::is_directory(rightFramePath))
        {
            std::filesystem::remove_all(rightFramePath);            
        }
        mrcv::writeLog("Create folder: " + rightFramePath.u8string());
        std::filesystem::create_directory(rightFramePath);
    }

    std::string fullPathToLeftFrameImage;
    std::string fullPathToRightFrameImage;

    int leftCameraID = 1;
    int rightCameraID = 2;

    std::cout << "Camera test started..." << std::endl;

    if (!capLeft.open(leftCameraID, cv::CAP_DSHOW))
    {
        mrcv::writeLog("Can't open the camera ID = " + std::to_string(leftCameraID), mrcv::LOGTYPE::ERROR);
        return EXIT_FAILURE;
    }
    if (!capRight.open(rightCameraID, cv::CAP_DSHOW))
    {
        mrcv::writeLog("Can't open the camera ID = " + std::to_string(rightCameraID), mrcv::LOGTYPE::ERROR);
        return EXIT_FAILURE;
    }

    std::cout << "Camera test finished." << std::endl;

    std::cout << "Press any key to strat the image grabbing." << std::endl;
    std::cin.get();
    std::cout << "Image grabbing procedure started..." << std::endl;    

    for (int i = 0; i < config.image_count; i++)
    {
        std::cout << "Iteration: " << std::to_string(i + 1) << std::endl;
        // Пауза 2 секунды
        std::this_thread::sleep_for(2s);
        capLeft >> frameLeft;
        capRight >> frameRight;

        // Для баслеровских камер
        // cv::cvtColor(frameLeft, frameLeft, cv::COLOR_BayerRG2RGB);
        // cv::cvtColor(frameRight, frameRight, cv::COLOR_BayerRG2RGB);

        fullPathToLeftFrameImage = (leftFramePath / (std::to_string(i) + ".png")).u8string();
        fullPathToRightFrameImage = (rightFramePath / (std::to_string(i) + ".png")).u8string();

        cv::imwrite(fullPathToLeftFrameImage, frameLeft);
        cv::imwrite(fullPathToRightFrameImage, frameRight);
    }
    std::cout << "Image grabbing procedure finished." << std::endl;
    
    std::cout << "Press any key to strat the calibration." << std::endl;
    std::cin.get();
    std::cout << "Calibration procedure started..." << std::endl;

    std::vector<cv::String> imagesLeft;
    std::vector<cv::String> imagesRight;
    
    //std::filesystem::path calibrationFile("files\\calibration.xml");    
    //auto calibrationPath = currentPath / calibrationFile;
    
    std::filesystem::path calibrationPath = path / "calibration.xml";
    
    mrcv::cameraCalibrationStereo(imagesLeft, imagesRight, leftFramePath.u8string() + "/", rightFramePath.u8string() + "/", stereoPar, config.keypoints_c, config.keypoints_r, config.square_size);
    mrcv::writeCalibrationParametersStereo(calibrationPath.u8string(), stereoPar);

    std::cout << "Calibration procedure finished." << std::endl;
    std::cout << "The calibration files saved to: " << calibrationPath.u8string() << std::endl;
    std::cout << "Press any key to exit.";
    std::cin.get();
    
    return EXIT_SUCCESS;
}
