#include "mrcv/mrcv.h"

namespace mrcv
{
    /**
     * @brief Функция поворота изображения на заданный угол.
     * Поворачивает изображение на определённый угол с использованием центральной точки.
     *
     * @param imageInput - входное (исходное) изображение cv::Mat.
     * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
     * @param angle - угол поворота в градусах.
     * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
     */

    int mrcv::rotateImage(cv::Mat& imageInput, cv::Mat& imageOutput, double angle)
    {
        try
        {
            // Определяем центр изображения
            cv::Point2f center(imageInput.cols / 2.0, imageInput.rows / 2.0);

            // Создаём матрицу для поворота
            cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

            // Поворачиваем изображение
            cv::warpAffine(imageInput, imageOutput, rotationMatrix, imageInput.size());
        }
        catch (...)
        {
            return -1; // Unhandled Exception
        }

        return 0; // SUCCESS
    }

    /**
     * @brief Функция отражения изображения.
     * Отражает изображение по горизонтали, вертикали или обеим осям.
     *
     * @param imageInput - входное (исходное) изображение cv::Mat.
     * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
     * @param flipCode - Код отражения: 0 - вертикальное отражение; 1 - горизонтальное отражение; -1 - обе стороны.
     * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
     */

    int mrcv::flipImage(cv::Mat& imageInput, cv::Mat& imageOutput, int flipCode)
    {
        try
        {

            // Применяем функцию отражения
            cv::flip(imageInput, imageOutput, flipCode);
        }
        catch (...)
        {
            return -1; // Unhandled Exception
        }

        return 0; // SUCCESS
    }

    /**
     * @brief Функция аугментации изображений.
     * Выполняет аугментацию для набора входных изображений на основе заданных методов и сохраняет результат.
     *
     * @param inputImagesAugmetation - вектор входных изображений (cv::Mat) для аугментации.
     * @param outputImagesAugmetation - вектор для сохранения выходных (преобразованных) изображений.
     * @param augmetationMethod - вектор методов аугментации (mrcv::AUGMENTATION_METHOD) для применения.
     * @return Код результата выполнения функции. 0 - успех; -1 - исключение (OpenCV или файловой системы).
     *
     * Функция проверяет наличие директории для сохранения изображений и создает её при необходимости. Для каждого изображения
     * выполняется указанная операция (например, поворот или отражение) с последующей проверкой и сохранением результата в директорию.
     */


    int mrcv::augmetation(std::vector<cv::Mat>& inputImagesAugmetation, std::vector<cv::Mat>& outputImagesAugmetation,
        std::vector<mrcv::AUGMENTATION_METHOD> augmetationMethod)
    {
        std::set<std::string> methodsUsed; // Уникальные методы аугментации
        int savedFilesCount = 0;

        try
        {
            std::string outputFolder = "files\\augmented_images/";
            if (!std::filesystem::exists(outputFolder)) {
                std::filesystem::create_directories(outputFolder);
            }

            for (int q = 0; q < (int)augmetationMethod.size(); q++)
            {
                for (size_t i = 0; i < inputImagesAugmetation.size(); i++)
                {
                    cv::Mat image = inputImagesAugmetation[i];

                    if (image.empty()) {
                        continue;
                    }

                    cv::Mat resultImage;
                    std::string methodName;
                    int status = 0;

                    switch (augmetationMethod.at(q))
                    {
                    case mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL:
                        status = mrcv::flipImage(image, resultImage, 1);
                        methodName = "flipHorizontal";
                        break;
                    case mrcv::AUGMENTATION_METHOD::FLIP_VERTICAL:
                        status = mrcv::flipImage(image, resultImage, 0);
                        methodName = "flipVertical";
                        break;
                    case mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL_AND_VERTICAL:
                        status = mrcv::flipImage(image, resultImage, -1);
                        methodName = "flipHorizontalandVertical";
                        break;
                    case mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_90:
                        status = mrcv::rotateImage(image, resultImage, 90);
                        methodName = "rotate90";
                        break;
                    case mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_45:
                        status = mrcv::rotateImage(image, resultImage, 45);
                        methodName = "rotate45";
                        break;
                    case mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_315:
                        status = mrcv::rotateImage(image, resultImage, 315);
                        methodName = "rotate315";
                        break;
                    case mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_270:
                        status = mrcv::rotateImage(image, resultImage, 270);
                        methodName = "rotate270";
                        break;
                    default:
                        resultImage = image.clone();
                        methodName = "none";
                        break;
                    }

                    if (resultImage.empty()) {
                        continue;
                    }

                    outputImagesAugmetation.push_back(resultImage);

                    std::stringstream ss;
                    ss << outputFolder << "augmented_" << i << "_" << methodName << ".bmp";

                    bool isSaved = cv::imwrite(ss.str(), resultImage);
                    if (isSaved) {
                        savedFilesCount++;
                        methodsUsed.insert(methodName); // Добавляем метод в set
                    }
                }
            }

            // Формируем строку с названиями методов
            std::ostringstream methodsStream;
            for (const auto& method : methodsUsed) {
                methodsStream << method << ", ";
            }
            std::string methodsString = methodsStream.str();
            if (!methodsString.empty()) {
                methodsString.pop_back(); // Удаляем последнюю запятую
                methodsString.pop_back(); // Удаляем пробел
            }

            // Записываем в лог успешное завершение с названиями методов
            writeLog("Augmentation completed successfully. Methods used: " + methodsString + ". Files saved: " + std::to_string(savedFilesCount), mrcv::LOGTYPE::INFO);
        }
        catch (const cv::Exception& ex)
        {
            writeLog("Augmentation failed: " + std::string(ex.what()), mrcv::LOGTYPE::ERROR);
            return -1;
        }
        catch (const std::filesystem::filesystem_error& ex)
        {
            writeLog("Filesystem error: " + std::string(ex.what()), mrcv::LOGTYPE::ERROR);
            return -1;
        }
        catch (...)
        {
            writeLog("Unhandled exception occurred during augmentation.", mrcv::LOGTYPE::EXCEPTION);
            return -1;
        }

        return 0;
    }

#ifdef MRCV_CUDA_ENABLED 
    int mrcv::rotateImageCuda(cv::Mat& imageInput, cv::Mat& imageOutput, double angle)
    {
        try
        {
            // Определяем центр изображения
            cv::Point2f center(imageInput.cols / 2.0, imageInput.rows / 2.0);

            // Создаём матрицу для поворота
            cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

            // Поворачиваем изображение
            cv::cuda::warpAffine(imageInput, imageOutput, rotationMatrix, imageInput.size());
        }
        catch (...)
        {
            return -1; // Unhandled Exception
        }

        return 0; // SUCCESS
    }

    int mrcv::flipImageCuda(cv::Mat& imageInput, cv::Mat& imageOutput, int flipCode)
    {
        try
        {

            // Применяем функцию отражения
            cv::cuda::flip(imageInput, imageOutput, flipCode);
        }
        catch (...)
        {
            return -1; // Unhandled Exception
        }

        return 0; // SUCCESS
    }

    int mrcv::augmetationCuda(std::vector<cv::Mat>& inputImagesAugmetation, std::vector<cv::Mat>& outputImagesAugmetation, std::vector<mrcv::AUGMENTATION_METHOD> augmetationMethod)
    {
        std::set<std::string> methodsUsed; // Уникальные методы аугментации
        int savedFilesCount = 0;

        try
        {
            std::string outputFolder = "files\\augmented_images/";
            if (!std::filesystem::exists(outputFolder)) {
                std::filesystem::create_directories(outputFolder);
            }

            for (int q = 0; q < (int)augmetationMethod.size(); q++)
            {
                for (size_t i = 0; i < inputImagesAugmetation.size(); i++)
                {
                    cv::Mat image = inputImagesAugmetation[i];

                    if (image.empty()) {
                        continue;
                    }

                    cv::Mat resultImage;
                    std::string methodName;
                    int status = 0;

                    switch (augmetationMethod.at(q))
                    {
                    case mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL:
                        status = mrcv::flipImageCuda(image, resultImage, 1);
                        methodName = "flipHorizontal";
                        break;
                    case mrcv::AUGMENTATION_METHOD::FLIP_VERTICAL:
                        status = mrcv::flipImageCuda(image, resultImage, 0);
                        methodName = "flipVertical";
                        break;
                    case mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL_AND_VERTICAL:
                        status = mrcv::flipImageCuda(image, resultImage, -1);
                        methodName = "flipHorizontalandVertical";
                        break;
                    case mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_90:
                        status = mrcv::rotateImageCuda(image, resultImage, 90);
                        methodName = "rotate90";
                        break;
                    case mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_45:
                        status = mrcv::rotateImageCuda(image, resultImage, 45);
                        methodName = "rotate45";
                        break;
                    case mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_315:
                        status = mrcv::rotateImageCuda(image, resultImage, 315);
                        methodName = "rotate315";
                        break;
                    case mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_270:
                        status = mrcv::rotateImageCuda(image, resultImage, 270);
                        methodName = "rotate270";
                        break;
                    default:
                        resultImage = image.clone();
                        methodName = "none";
                        break;
                    }

                    if (resultImage.empty()) {
                        continue;
                    }

                    outputImagesAugmetation.push_back(resultImage);

                    std::stringstream ss;
                    ss << outputFolder << "augmented_" << i << "_" << methodName << ".bmp";

                    bool isSaved = cv::imwrite(ss.str(), resultImage);
                    if (isSaved) {
                        savedFilesCount++;
                        methodsUsed.insert(methodName); // Добавляем метод в set
                    }
                }
            }

            // Формируем строку с названиями методов
            std::ostringstream methodsStream;
            for (const auto& method : methodsUsed) {
                methodsStream << method << ", ";
            }
            std::string methodsString = methodsStream.str();
            if (!methodsString.empty()) {
                methodsString.pop_back(); // Удаляем последнюю запятую
                methodsString.pop_back(); // Удаляем пробел
            }

            // Записываем в лог успешное завершение с названиями методов
            writeLog("Augmentation completed successfully. Methods used: " + methodsString + ". Files saved: " + std::to_string(savedFilesCount), mrcv::LOGTYPE::INFO);
        }
        catch (const cv::Exception& ex)
        {
            writeLog("Augmentation failed: " + std::string(ex.what()), mrcv::LOGTYPE::ERROR);
            return -1;
        }
        catch (const std::filesystem::filesystem_error& ex)
        {
            writeLog("Filesystem error: " + std::string(ex.what()), mrcv::LOGTYPE::ERROR);
            return -1;
        }
        catch (...)
        {
            writeLog("Unhandled exception occurred during augmentation.", mrcv::LOGTYPE::EXCEPTION);
            return -1;
        }

        return 0;
    }
#endif
}

