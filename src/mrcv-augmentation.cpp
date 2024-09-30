#include "mrcv/mrcv.h"

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
     * @brief Функция предварительной обработки изображений (автоматическая коррекция контраста и яркости, резкости)
     * Функция автоматической предобработки изображения, кооррекции яркости и контраста, резкости.
     *
     *
     * @param image - изображение cv::Mat, над которым происходит преобразование.
     * @param metodImagePerProcessingBrightnessContrast - вектор параметров, которые опрределяют, какие преобразования и в какой последовательноси  проводить.
     *  none  - без изменений
     * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
     */

int mrcv::augmetation(std::vector<cv::Mat>& inputImagesAugmetation, std::vector<cv::Mat>& outputImagesAugmetation,
    std::vector<mrcv::AUGMENTATION_METHOD> augmetationMethod)
{
    try
    {
        // Проверяем наличие директории и создаем её, если не существует
        std::string outputFolder = "files\\augmented_images/";
        if (!std::filesystem::exists(outputFolder)) {
            std::filesystem::create_directories(outputFolder); // Создать директорию для сохранения изображений
            std::cout << "Directory created: " << outputFolder << std::endl;
        }

        for (int q = 0; q < (int)augmetationMethod.size(); q++)
        {
            for (size_t i = 0; i < inputImagesAugmetation.size(); i++)  // Для всех входных изображений
            {
                cv::Mat image = inputImagesAugmetation[i];  // Извлекаем текущее изображение

                // Проверка, загружено ли изображение
                if (image.empty()) {
                    std::cerr << "Error: Input image at index " << i << " is empty or failed to load." << std::endl;
                    continue; // Пропуск этой итерации, если изображение пустое
                }

                cv::Mat resultImage;
                std::string methodName;
                int status = 0; // Инициализация переменной status

                switch (augmetationMethod.at(q))
                {
                case mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL:
                    status = mrcv::flipImage(image, resultImage, 1);  // Горизонтальное отражение
                    methodName = "flipHorizontal";
                    break;
                case mrcv::AUGMENTATION_METHOD::FLIP_VERTICAL:
                    status = mrcv::flipImage(image, resultImage, 0);  // Вертикальное отражение
                    methodName = "flipVertical";
                    break;
                case mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL_AND_VERTICAL:
                    status = mrcv::flipImage(image, resultImage, -1);  // Горизонтальное и вертикальное отражение
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
                    resultImage = image.clone();  // Если метод не применён, просто копируем изображение
                    methodName = "none";
                    break;
                }

                // Проверка на пустое изображение после обработки
                if (resultImage.empty()) {
                    std::cerr << "Error: Resulting image is empty after applying " << methodName << " on image " << i << std::endl;
                    continue; // Пропуск, если результат пустой
                }

                // Сохранение результата
                outputImagesAugmetation.push_back(resultImage);

                // Создание уникального имени файла для сохранения
                std::stringstream ss;
                ss << outputFolder << "augmented_" << i << "_" << methodName << ".bmp"; // Формат BMP

                // Сохранение изображения на диск
                bool isSaved = cv::imwrite(ss.str(), resultImage);
                if (!isSaved) {
                    std::cerr << "Error: Failed to save image " << ss.str() << std::endl;
                }
                else {
                    std::cout << "Image saved as: " << ss.str() << std::endl;
                }
            }
        }
    }
    catch (const cv::Exception& ex)
    {
        std::cerr << "OpenCV Exception: " << ex.what() << std::endl;
        return -1; // OpenCV exception
    }
    catch (const std::filesystem::filesystem_error& ex)
    {
        std::cerr << "Filesystem error: " << ex.what() << std::endl;
        return -1; // Filesystem exception
    }
    catch (...)
    {
        std::cerr << "Unhandled exception occurred." << std::endl;
        return -1; // Unhandled Exception
    }

    return 0; // SUCCESS
}

