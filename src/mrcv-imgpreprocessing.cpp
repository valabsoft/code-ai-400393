#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>
cv::Mat mrcv::getErrorImage(std::string textError)
{
    cv::Mat errorImage = cv::Mat::zeros(600, 960, CV_8UC3);
    cv::putText(errorImage,
        textError,
        //                    {int(coef_size * 50), HR_Img_.rows - int(coef_size * 150)},
        { 25, 150 },
        cv::FONT_HERSHEY_SIMPLEX,              // int 	fontFace
        1.15,              //double 	fontScale
        cv::Scalar(47, 20, 162),
        3,               // thickness,
        cv::LINE_8,    // lineType = //cv::LINE_4 //cv::LINE_8 //cv::LINE_AA
        false);
    return errorImage;
}

int mrcv::increaseImageContrastEqualizeHist(cv::Mat& imageInput, cv::Mat& imageOutput)
{
    try
    {
        if (imageInput.empty())
        {
            imageOutput = mrcv::getErrorImage("increaseImageContrastEqualizeHist:: Image is Empty");
            return 1; // 1 - Пустое изображение
        }
        ///////////////////////////////////////////////////////////////////////
        // equalizeHist - Гистограммная эквализация изображения
        ///////////////////////////////////////////////////////////////////////
        if (imageInput.channels() == 3) // если цветное изображение
        {
            // Разделяем изображение на 3 канала (B, G и R)
            std::vector<cv::Mat> bgr_planes;
            split(imageInput, bgr_planes);
            // Применяем выравнивание к гистограммам всех каналов
            equalizeHist(bgr_planes[0], bgr_planes[0]);
            equalizeHist(bgr_planes[1], bgr_planes[1]);
            equalizeHist(bgr_planes[2], bgr_planes[2]);
            // Объединяем выровненные каналы в выровненное цветное изображение
            merge(bgr_planes, imageOutput);
        }
        else if (imageInput.channels() == 1) // если монохромное (серое) изображение
        {
            equalizeHist(imageInput, imageOutput);
        }
        else  // неизветсный формат изображения
        {
            imageOutput = mrcv::getErrorImage("increaseImageContrastEqualizeHist:: Unknown Image Format");
            return 2; // 2 - Неизвестный формат изображения
        }
    }
    catch (...)
    {
        imageOutput = mrcv::getErrorImage("increaseImageContrastEqualizeHist:: Unhandled Exception");
        return -1; // Unhandled Exception
    }
    return EXIT_SUCCESS;
}

int mrcv::increaseImageContrastCLAHE(cv::Mat& imageInput, cv::Mat& imageOutput, double clipLimit, cv::Size gridSize)
{
    try
    {
        if (imageInput.empty())
        {
            imageOutput = mrcv::getErrorImage("increaseImageContrastCLAHE:: Image is Empty");
            return 1; // 1 - Пустое изображение
        }
        ///////////////////////////////////////////////////////////////////////
        // Метод предобработки изображений: Contrast Limited Adaptive Histogram Equalization
        ///////////////////////////////////////////////////////////////////////
        if (imageInput.channels() == 3) // если цветное изображение
        {
            cv::Ptr <cv::CLAHE> clahe = cv::createCLAHE(clipLimit, gridSize);
            //  Разделяем изображение на 3 канала (B, G и R)
            std::vector<cv::Mat> bgr_planes;
            split(imageInput, bgr_planes);
            // Применяем выравнивание к гистограммам всех каналов
            clahe->apply(bgr_planes[0], bgr_planes[0]);
            clahe->apply(bgr_planes[1], bgr_planes[1]);
            clahe->apply(bgr_planes[2], bgr_planes[2]);
            // Объединяем выровненные каналы в выровненное цветное изображение
            merge(bgr_planes, imageOutput);
        }
        else if (imageInput.channels() == 1) // если монохромное (серое) изображение
        {
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, gridSize);
            clahe->apply(imageInput, imageOutput);
        }
        else  // неизветсный формат изображения
        {
            imageOutput = mrcv::getErrorImage("increaseImageContrastCLAHE:: Unknown Image Format");
            return 2; // 2 - Неизвестный формат изображения
        }
    }
    catch (...)
    {
        imageOutput = mrcv::getErrorImage("increaseImageContrastCLAHE:: Unhandled Exception");
        return -1; // Unhandled Exception
    }
    return EXIT_SUCCESS;
}


int mrcv::increaseImageContrastСolorLabCLAHE(cv::Mat& imageInput, cv::Mat& imageOutput, double clipLimit, cv::Size gridSize)
{
    try
    {
        if (imageInput.empty())
        {
            imageOutput = mrcv::getErrorImage("increaseImageContrastCLAHE:: Image is Empty");
            return 1; // 1 - Пустое изображение
        }
        ///////////////////////////////////////////////////////////////////////
        // CLAHE (через Color Lab) цветного изображения
        ///////////////////////////////////////////////////////////////////////
        if (imageInput.channels() == 3) // если цветное изображение
        {
            cv::Ptr <cv::CLAHE> clahe = cv::createCLAHE(2, cv::Size(8, 8));
            cv::Mat img_lab;
            cv::Mat imageDouble;
            // Конвертация изображения в Lab
            cvtColor(imageInput, img_lab, cv::COLOR_BGR2Lab);
            //  Разделяем изображение на 3 канала (B, G и R)
            std::vector<cv::Mat> lab_planes;
            split(img_lab, lab_planes);
            lab_planes[0] = lab_planes[0];
            // Применяем выравнивание к гистограммам всех каналов
            clahe->apply(lab_planes[0], lab_planes[0]);
            lab_planes[0] = lab_planes[0];
            // Объединяем выровненные каналы в выровненное цветное изображение
            merge(lab_planes, img_lab);
            cvtColor(img_lab, imageOutput, cv::COLOR_Lab2BGR);
        }
        else if (imageInput.channels() == 1) // если монохромное (серое) изображение
        {
            imageOutput = mrcv::getErrorImage("preprocessingImage:: Unknown Image Format");
            return 2; // 2 - Неизвестный формат изображения
        }
        else  // неизветсный формат изображения
        {
            imageOutput = mrcv::getErrorImage("preprocessingImage:: Unknown Image Format");
            return 2; // 2 - Неизвестный формат изображения
        }
    }
    catch (...)
    {
        imageOutput = mrcv::getErrorImage("increaseImageContrastCLAHE:: Unhandled Exception");
        return -1; // Unhandled Exception
    }
    return EXIT_SUCCESS;
}

int mrcv::changeImageBrightness(cv::Mat& imageInput, cv::Mat& imageOutput, double gamma)
{
    try
    {
        if (imageInput.empty())
        {
            imageOutput = mrcv::getErrorImage("changeImageBrightness:: Image is Empty");
            return 1; // 1 - Пустое изображение
        }
        ///////////////////////////////////////////////////////////////////////
        // Гамма-коррекция
        ///////////////////////////////////////////////////////////////////////
        cv::Mat imageDouble;
        // Конвертация изображения в double
        imageInput.convertTo(imageDouble, CV_64F, 1. / 255, 0);
        // Коррекция
        pow(imageDouble, gamma, imageDouble);
        // Конвертация изображения в исходный форма
        imageDouble = imageDouble * 255;
        imageDouble.convertTo(imageOutput, CV_8U);
    }
    catch (...)
    {
        imageOutput = mrcv::getErrorImage("changeImageBrightness:: Unhandled Exception");
        return -1; // Unhandled Exception
    }
    return EXIT_SUCCESS;
}

int mrcv::preprocessingImage(cv::Mat& image, std::vector<mrcv::IMG_PREPROCESSING_METHOD> metodImagePerProcessing, const std::string& pathToFileCameraParametrs)
{
    try
    {
        if (image.empty())
        {
            image = mrcv::getErrorImage("preprocessingImage:: Image is Empty");
            return 1; // 1 - Пустое изображение
        }

        for (int q = 0; q < (int)metodImagePerProcessing.size(); q++)
        {
            switch (metodImagePerProcessing.at(q))
            {
            case mrcv::IMG_PREPROCESSING_METHOD::NONE:
            {
                // ... без преобразования
                break;
            }
            case mrcv::IMG_PREPROCESSING_METHOD::EQUALIZEHIST:
            {
                // Повышение контраста с помощью метода Эквализация Гистограмм изображения (equalizeHist)
                int status = mrcv::increaseImageContrastEqualizeHist(image, image);
                break;
            } // case
            case mrcv::IMG_PREPROCESSING_METHOD::CLAHE:
            {
                // Повышение контраста с помощью метода Адаптивной гистограммой эквализации (Contrast Limited Adaptive Histogram Equalization)
                int status = mrcv::increaseImageContrastCLAHE(image, image, 2, { 8,8 });
                break;
            } // case
            case mrcv::IMG_PREPROCESSING_METHOD::BGRTOGRAY:
            {
                if (image.channels() == 3) // если цветное изображение)
                {
                    // Преобразование цветного изображения в серое
                    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
                }
                else if (image.channels() == 1) // если монохромное (серое) изображение
                {
                    // ... без преобразования
                }
                else  // неизветсный формат изображения
                {
                    image = mrcv::getErrorImage("preprocessingImage:: Unknown Image Format");
                    return 2; // 2 - Неизвестный формат изображения
                }
                break;
            } // case
            case mrcv::IMG_PREPROCESSING_METHOD::COLORLABCLAHE:
            {
                int status = mrcv::increaseImageContrastСolorLabCLAHE(image, image, 2, { 8,8 });
                break;
            } // case
            case mrcv::IMG_PREPROCESSING_METHOD::SHARPENING01:
            {
                ///////////////////////////////////////////////////////////////
                // Фильтр повышения резкости
                ///////////////////////////////////////////////////////////////
                // Применение оператора Лапласа
                cv::Mat edges;
                cv::Laplacian(image, edges, -1);
                image = image - 2 * edges;
                break;
            }// case
            case mrcv::IMG_PREPROCESSING_METHOD::SHARPENING02:
            {
                ///////////////////////////////////////////////////////////////
                // Фильтр повышения резкости
                ///////////////////////////////////////////////////////////////
                // Применение оператора Гаусса
                cv::Mat low;
                cv::GaussianBlur(image, low, cv::Size(9, 9), 0, 0);
                image = image + 4 * (image - low);
                break;
            }// case
            case mrcv::IMG_PREPROCESSING_METHOD::BRIGHTNESSLEVELUP:
            {
                // Повышение яркости на один уровень
                int status = mrcv::changeImageBrightness(image, image, 0.8);
                break;
            }// case
            case mrcv::IMG_PREPROCESSING_METHOD::BRIGHTNESSLEVELDOWN:
            {
                // Понижение яркости на один уровень
                int status = mrcv::changeImageBrightness(image, image, 1.25);
                break;
            }// case
            case mrcv::IMG_PREPROCESSING_METHOD::NOISEFILTERINGMEDIANFILTER:
            {
                medianBlur(image, image, 3);
                break;
            }// case
            case mrcv::IMG_PREPROCESSING_METHOD::NOISEFILTERINGAVARAGEFILTER:
            {
                GaussianBlur(image, image, cv::Size(3, 3), 0, 0);
                break;
            }// case
            case mrcv::IMG_PREPROCESSING_METHOD::CORRECTIONGEOMETRICDEFORMATION:
            {
                cv::Mat map11;
                cv::Mat map12;
                int status = mrcv::readCameraParametrsFromFile(pathToFileCameraParametrs.c_str(), map11, map12);
                cv::remap(image, image, map11, map12, cv::INTER_LINEAR);
                break;
            }// case
            } // switch
        } // for;
    }
    catch (...)
    {
        image = mrcv::getErrorImage("preprocessingImage:: Unhandled Exception");
        return -1; // Unhandled Exception
    }
    return EXIT_SUCCESS;
}

int mrcv::readCameraParametrsFromFile(const char* pathToFileCameraParametrs, cv::Mat& map11, cv::Mat& map12)
{
    try
    {
        cv::Mat M1;
        cv::Mat D1;
        cv::Mat R1;
        cv::Mat P1;
        cv::Mat M1n = P1.clone();;
        cv::Size  imageSize;
        cv::FileStorage fs(pathToFileCameraParametrs, cv::FileStorage::READ);
        if (fs.isOpened())
        {
            fs["M1"] >> M1;
            fs["D1"] >> D1;
            fs["imageSize"] >> imageSize;
            fs["R1"] >> R1;
            fs["P1"] >> P1;
            fs.release();
        }
        else
        {
            fs.releaseAndGetString();
        }
        // Расчёт первая карт точек
        cv::initUndistortRectifyMap(M1, D1, R1, M1n, imageSize, CV_16SC2, map11, map12);
    }
    catch (...)
    {
        return -1; // Unhandled Exception
    }
    return EXIT_SUCCESS;
}
