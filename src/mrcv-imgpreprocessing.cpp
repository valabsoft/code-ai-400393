#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>

cv::Mat mrcv::getErrorImage(std::string textError)
{
    cv::Mat errorImage = cv::Mat::zeros(600, 960, CV_8UC3);
    cv::putText(errorImage,
        textError,
        { 25, 150 },
        cv::FONT_HERSHEY_SIMPLEX,  // int 	fontFace
        1.0,                        //double 	fontScale
        cv::Scalar(47, 20, 162),
        3,                          // thickness,
        cv::LINE_8,                 // lineType = //cv::LINE_4 //cv::LINE_8 //cv::LINE_AA
        false);
    return errorImage;
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
        // ////////////////////
        // Гамма-коррекция
        // ////////////////////
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
    return 0; // SUCCESS
}

int mrcv::sharpeningImage01(cv::Mat& imageInput, cv::Mat& imageOutput, double gainFactorHighFrequencyComponent)
{
    try
    {
        if (imageInput.empty())
        {
            imageOutput = mrcv::getErrorImage("sharpeningImage01:: Image is Empty");
            return 1; // 1 - Пустое изображение
        }
        // ===========
        // Фильтр повышения резкости 01
        // ===========
        cv::Mat edges;
        cv::Laplacian(imageInput, edges, -1); // Применение оператора Лапласа
        imageOutput = imageInput - gainFactorHighFrequencyComponent * edges;
    }
    catch (...)
    {
        imageOutput = mrcv::getErrorImage("sharpeningImage01:: Unhandled Exception");
        return -1; // Unhandled Exception
    }
    return 0; // SUCCESS
}

int mrcv::sharpeningImage02(cv::Mat& imageInput, cv::Mat& imageOutput, cv::Size filterSize, double sigmaFilter, double gainFactorHighFrequencyComponent)
{
    try
    {
        if (imageInput.empty())
        {
            imageOutput = mrcv::getErrorImage("sharpeningImage02:: Image is Empty");
            return 1; // 1 - Пустое изображение
        }
        // ////////////////////
        // Фильтр повышения резкости
        // ////////////////////
        cv::Mat low;
        cv::GaussianBlur(imageInput, low, filterSize, sigmaFilter, sigmaFilter); // Применение оператора Гаусса
        imageOutput = imageInput + gainFactorHighFrequencyComponent * (imageInput - low);
    }
    catch (...)
    {
        imageOutput = mrcv::getErrorImage("sharpeningImage02:: Unhandled Exception");
        return -1; // Unhandled Exception
    }
    return 0; // SUCCESS
}

int mrcv::readCameraParametrsFromFile(const char* pathToFileCameraParametrs, mrcv::cameraParameters& cameraParameters)
{
    try
    {
        cv::FileStorage fs(pathToFileCameraParametrs, cv::FileStorage::READ);
        if (fs.isOpened())
        {
            fs["M1"] >> cameraParameters.M1;
            fs["D1"] >> cameraParameters.D1;
            fs["R"] >> cameraParameters.R;
            fs["T"] >> cameraParameters.T;
            fs["R1"] >> cameraParameters.R1;
            fs["P1"] >> cameraParameters.P1;
            fs["imageSize"] >> cameraParameters.imageSize;
            fs["rms"] >> cameraParameters.rms;
            fs["avgErr"] >> cameraParameters.avgErr;
            fs.release();
        }
        else
        {
            fs.releaseAndGetString();
        }
        // Расчёт первая карт точек
        cv::Mat M1n;   // новая матрица камеры 3x3
        cv::initUndistortRectifyMap(cameraParameters.M1, cameraParameters.D1, cameraParameters.R1, M1n, cameraParameters.imageSize,
            CV_16SC2, cameraParameters.map11, cameraParameters.map12);
    }
    catch (...)
    {
        return -1; // Unhandled Exception
    }
    return 0; // SUCCESS
}

int mrcv::preprocessingImage(cv::Mat& image, std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> metodImagePerProcessing, const std::string& pathToFileCameraParametrs)
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
            case mrcv::METOD_IMAGE_PERPROCESSIN::NONE:
            {
                // ... без преобразования
                break;
            }
            case mrcv::METOD_IMAGE_PERPROCESSIN::CONVERTING_BGR_TO_GRAY:
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
            case mrcv::METOD_IMAGE_PERPROCESSIN::BRIGHTNESS_LEVEL_UP:
            {
                // Повышение яркости на один уровень
                int status = mrcv::changeImageBrightness(image, image, 0.8);
                mrcv::writeLog("\t BRIGHTNESS_LEVEL_UP, state = " + std::to_string(status));
                break;
            }// case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BRIGHTNESS_LEVEL_DOWN:
            {
                // Понижение яркости на один уровень
                int status = mrcv::changeImageBrightness(image, image, 1.25);
                mrcv::writeLog("\t BRIGHTNESS_LEVEL_DOWN, state = " + std::to_string(status));
                break;
            }// case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_01_YCBCR_EQUALIZEHIST:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST, mrcv::COLOR_MODEL::CM_YCBCR);
                mrcv::writeLog("\t BALANCE_CONTRAST_01_YCBCR_EQUALIZEHIST, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_02_YCBCR_CLAHE:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE, mrcv::COLOR_MODEL::CM_YCBCR, 2, { 8,8 });
                mrcv::writeLog("\t BALANCE_CONTRAST_02_YCBCR_CLAHE, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING, mrcv::COLOR_MODEL::CM_YCBCR, {}, {}, 3);
                mrcv::writeLog("\t BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION, mrcv::COLOR_MODEL::CM_YCBCR, {}, {}, {}, -1, 2);
                mrcv::writeLog("\t BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION, state = " + std::to_string(status));
                break;
            }// case


            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_05_HSV_EQUALIZEHIST:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST, mrcv::COLOR_MODEL::CM_HSV);
                mrcv::writeLog("\t BALANCE_CONTRAST_05_HSV_EQUALIZEHIST, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_06_HSV_CLAHE:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE, mrcv::COLOR_MODEL::CM_HSV, 2, { 8,8 });
                mrcv::writeLog("\t BALANCE_CONTRAST_06_HSV_CLAHE, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING, mrcv::COLOR_MODEL::CM_HSV, {}, {}, 3);
                mrcv::writeLog("\t BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION, mrcv::COLOR_MODEL::CM_HSV, {}, {}, {}, -1, 2);
                mrcv::writeLog("\t BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION, state = " + std::to_string(status));
                break;
            }// case

            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_09_LAB_EQUALIZEHIST:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST, mrcv::COLOR_MODEL::CM_LAB);
                mrcv::writeLog("\t BALANCE_CONTRAST_09_LAB_EQUALIZEHIST, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_10_LAB_CLAHE:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE, mrcv::COLOR_MODEL::CM_LAB, 2, { 8,8 });
                mrcv::writeLog("\t BALANCE_CONTRAST_10_LAB_CLAHE, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING, mrcv::COLOR_MODEL::CM_LAB, {}, {}, 3);
                mrcv::writeLog("\t BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION, mrcv::COLOR_MODEL::CM_LAB, {}, {}, {}, -1, 2);
                mrcv::writeLog("\t BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION, state = " + std::to_string(status));
                break;
            }// case

            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_13_RGB_EQUALIZEHIST:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST, mrcv::COLOR_MODEL::CM_RGB);
                mrcv::writeLog("\t BALANCE_CONTRAST_09_LAB_EQUALIZEHIST, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_14_RGB_CLAHE:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE, mrcv::COLOR_MODEL::CM_RGB, 2, { 8,8 });
                mrcv::writeLog("\t BALANCE_CONTRAST_10_LAB_CLAHE, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING, mrcv::COLOR_MODEL::CM_RGB, {}, {}, 3);
                mrcv::writeLog("\t BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING, state = " + std::to_string(status));
                break;
            } // case
            case mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION:
            {
                int status = mrcv::increaseImageContrast(image, image,
                    mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION, mrcv::COLOR_MODEL::CM_RGB, {}, {}, {}, -1, 2);
                mrcv::writeLog("\t BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION, state = " + std::to_string(status));
                break;
            }// case
            case mrcv::METOD_IMAGE_PERPROCESSIN::SHARPENING_01:
            {
                int status = mrcv::sharpeningImage01(image, image, 2);
                mrcv::writeLog("\t SHARPENING_01, state = " + std::to_string(status));
                break;
            }// case
            case mrcv::METOD_IMAGE_PERPROCESSIN::SHARPENING_02:
            {
                int status = mrcv::sharpeningImage02(image, image, cv::Size(9, 9), 0, 4);
                mrcv::writeLog("\t SHARPENING_02, state = " + std::to_string(status));
                break;
            }// case
            case mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_01_MEDIAN_FILTER:
            {
                medianBlur(image, image, 3);
                mrcv::writeLog("\t NOISE_FILTERING_01_MEDIAN_FILTER, state = 0");
                break;
            }// case
            case mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_02_AVARAGE_FILTER:
            {
                GaussianBlur(image, image, cv::Size(3, 3), 0, 0);
                mrcv::writeLog("\t NOISE_FILTERING_02_AVARAGE_FILTER, state = 0 ");
                break;
            }// case
            case mrcv::METOD_IMAGE_PERPROCESSIN::CORRECTION_GEOMETRIC_DEFORMATION:
            {
                mrcv::cameraParameters cameraParameters;
                int status = mrcv::readCameraParametrsFromFile(pathToFileCameraParametrs.c_str(), cameraParameters);
                cv::remap(image, image, cameraParameters.map11, cameraParameters.map12, cv::INTER_LINEAR);
                mrcv::writeLog("\t CORRECTION_GEOMETRIC_DEFORMATION, state = " + std::to_string(status));
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
    return 0; // SUCCESS
}

int mrcv::increaseImageContrast(cv::Mat& imageInput, cv::Mat& imageOutput,
    mrcv::METOD_INCREASE_IMAGE_CONTRAST metodIncreaseContrast, mrcv::COLOR_MODEL colorSpace,
    double clipLimitCLAHE, cv::Size gridSizeCLAHE, float percentContrastBalance,
    double mContrastExtantion, double eContrastExtantion)
{
    try
    {
        if (imageInput.empty())
        {
            imageOutput = mrcv::getErrorImage("increaseImageContrast:: Image is Empty");
            return 1; // 1 - Пустое изображение
        }

        if (imageInput.channels() == 3) // если цветное изображение
        {
            cv::Mat imageOtherModel;
            std::vector<cv::Mat> planes;

            switch (colorSpace)
            {
            case mrcv::COLOR_MODEL::CM_HSV:
            {
                int qc = 2; // Корректируемая координата цветового пространства
                cvtColor(imageInput, imageOtherModel, cv::COLOR_BGR2HSV);  // Конвертация изображения в другое цветовое пространство
                split(imageOtherModel, planes);                          //  Разделяем изображение на 3 канала

                switch (metodIncreaseContrast)
                {
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST:
                {
                    // Повышение контраста с помощью метода гистограммой эквализации (Histogram Equalization)
                    equalizeHist(planes[qc], planes[qc]);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE:
                {
                    cv::Ptr <cv::CLAHE> clahe = cv::createCLAHE(clipLimitCLAHE, gridSizeCLAHE);
                    // Повышение контраста с помощью метода Адаптивной гистограммой эквализации (Contrast Limited Adaptive Histogram Equalization)
                    clahe->apply(planes[qc], planes[qc]);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING:
                {
                    int state = mrcv::contrastBalancing(planes[qc], percentContrastBalance);
                    if (state != 0) mrcv::writeLog(" BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION:
                {
                    int state = mrcv::contrastExtantion(planes[qc], mContrastExtantion, eContrastExtantion);
                    if (state != 0) mrcv::writeLog(" BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
                    break;
                }  // case

                } // switch (metodIncreaseContrast)

                merge(planes, imageOtherModel);    // Объединение каналлов
                cvtColor(imageOtherModel, imageOutput, cv::COLOR_HSV2BGR); // Конвертация изображения в RGB
                break;
            } // case: CM_HSV
            case mrcv::COLOR_MODEL::CM_LAB:
            {
                int qc = 0; // Корректируемая координата цветового пространства
                cvtColor(imageInput, imageOtherModel, cv::COLOR_BGR2Lab);  // Конвертация изображения в другое цветовое пространство
                split(imageOtherModel, planes);                          //  Разделяем изображение на 3 канала

                switch (metodIncreaseContrast)
                {
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST:
                {
                    equalizeHist(planes[qc], planes[qc]);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE:
                {
                    cv::Ptr <cv::CLAHE> clahe = cv::createCLAHE(clipLimitCLAHE, gridSizeCLAHE);
                    clahe->apply(planes[qc], planes[qc]);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING:
                {
                    int state = mrcv::contrastBalancing(planes[qc], percentContrastBalance);
                    if (state != 0) mrcv::writeLog(" BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION:
                {
                    int state = mrcv::contrastExtantion(planes[qc], mContrastExtantion, eContrastExtantion);
                    if (state != 0) mrcv::writeLog(" BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
                    break;
                }  // case

                } // switch (metodIncreaseContrast)

                merge(planes, imageOtherModel);    // Объединение каналлов
                cvtColor(imageOtherModel, imageOutput, cv::COLOR_Lab2BGR); // Конвертация изображения обратно в RGB
                break;
            } // case: CM_LAB
            case mrcv::COLOR_MODEL::CM_YCBCR:
            {
                int qc = 0; // Корректируемая координата цветового пространства
                cvtColor(imageInput, imageOtherModel, cv::COLOR_BGR2YCrCb);  // Конвертация изображения в другое цветовое пространство
                split(imageOtherModel, planes);                          //  Разделяем изображение на 3 канала

                switch (metodIncreaseContrast)
                {
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST:
                {
                    equalizeHist(planes[qc], planes[qc]);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE:
                {
                    cv::Ptr <cv::CLAHE> clahe = cv::createCLAHE(clipLimitCLAHE, gridSizeCLAHE);
                    clahe->apply(planes[qc], planes[qc]);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING:
                {
                    int state = mrcv::contrastBalancing(planes[qc], percentContrastBalance);
                    if (state != 0) mrcv::writeLog(" BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION:
                {
                    int state = mrcv::contrastExtantion(planes[qc], mContrastExtantion, eContrastExtantion);
                    if (state != 0) mrcv::writeLog(" BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
                    break;
                }  // case
                } // switch (metodIncreaseContrast)

                merge(planes, imageOtherModel);    // Объединение каналлов
                cvtColor(imageOtherModel, imageOutput, cv::COLOR_YCrCb2BGR); // Конвертация изображения обратно в RGB
                break;
            } // case: CM_YCBCR
            case mrcv::COLOR_MODEL::CM_RGB:
            {
                cvtColor(imageInput, imageOtherModel, cv::COLOR_BGR2YCrCb);  // Конвертация изображения в другое цветовое пространство
                split(imageOtherModel, planes);                          //  Разделяем изображение на 3 канала

                switch (metodIncreaseContrast)
                {
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST:
                {
                    equalizeHist(planes[0], planes[0]);
                    equalizeHist(planes[1], planes[1]);
                    equalizeHist(planes[2], planes[2]);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE:
                {
                    cv::Ptr <cv::CLAHE> clahe = cv::createCLAHE(clipLimitCLAHE, gridSizeCLAHE);
                    clahe->apply(planes[0], planes[0]);
                    clahe->apply(planes[1], planes[1]);
                    clahe->apply(planes[2], planes[2]);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING:
                {
                    int state0 = mrcv::contrastBalancing(planes[0], percentContrastBalance);
                    int state1 = mrcv::contrastBalancing(planes[1], percentContrastBalance);
                    int state2 = mrcv::contrastBalancing(planes[2], percentContrastBalance);
                    if (state0 != 0) mrcv::writeLog(" BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING, state[0]  = " + std::to_string(state0), mrcv::LOGTYPE::ERROR);
                    if (state1 != 0) mrcv::writeLog(" BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING, state[1] = " + std::to_string(state1), mrcv::LOGTYPE::ERROR);
                    if (state2 != 0) mrcv::writeLog(" BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING, state[2] = " + std::to_string(state2), mrcv::LOGTYPE::ERROR);
                    break;
                }  // case
                case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION:
                {
                    int state0 = mrcv::contrastExtantion(planes[0], mContrastExtantion, eContrastExtantion);
                    int state1 = mrcv::contrastExtantion(planes[1], mContrastExtantion, eContrastExtantion);
                    int state2 = mrcv::contrastExtantion(planes[2], mContrastExtantion, eContrastExtantion);
                    if (state0 != 0) mrcv::writeLog(" BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION, state[0] = " + std::to_string(state0), mrcv::LOGTYPE::ERROR);
                    if (state1 != 0) mrcv::writeLog(" BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION, state[1] = " + std::to_string(state1), mrcv::LOGTYPE::ERROR);
                    if (state2 != 0) mrcv::writeLog(" BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION, state[2] = " + std::to_string(state2), mrcv::LOGTYPE::ERROR);
                    break;
                }  // case
                } // switch (metodIncreaseContrast)
                merge(planes, imageOtherModel);    // Объединение каналлов
                cvtColor(imageOtherModel, imageOutput, cv::COLOR_YCrCb2BGR); // Конвертация изображения обратно в RGB
                break;
            } // case: CM_RGB
            break;
            } // switch
        } // if
        else if (imageInput.channels() == 1) // если монохромное (серое) изображение
        {
            switch (metodIncreaseContrast)
            {
            case mrcv::METOD_INCREASE_IMAGE_CONTRAST::EQUALIZE_HIST:
            {
                equalizeHist(imageInput, imageOutput);
                break;
            }  // case
            case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CLAHE:
            {
                cv::Ptr <cv::CLAHE> clahe = cv::createCLAHE(clipLimitCLAHE, gridSizeCLAHE);
                clahe->apply(imageInput, imageOutput);
                break;
            }  // case
            case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_BALANCING:
            {
                imageOutput = imageInput.clone();
                int state = mrcv::contrastBalancing(imageOutput, percentContrastBalance);
                break;
            }  // case
            case mrcv::METOD_INCREASE_IMAGE_CONTRAST::CONTRAST_EXTENSION:
            {
                imageOutput = imageInput.clone();
                int state = mrcv::contrastExtantion(imageOutput, mContrastExtantion, eContrastExtantion);
                break;
            }  // case
            } // switch (metodIncreaseContrast)
        }
        else  // неизветсный формат изображения
        {
            imageOutput = mrcv::getErrorImage("increaseImageContrast:: Unknown Image Format");
            return 2; // 2 - Неизвестный формат изображения
        }
    }
    catch (...)
    {
        imageOutput = mrcv::getErrorImage("increaseImageContrast:: Unhandled Exception");
        return -1; // Unhandled Exception
    }
    return 0; // SUCCESS
}

int mrcv::contrastBalancing(cv::Mat& planeArray, float percent)
{
    try
    {
        if (planeArray.empty())
        {
            return 1; // 1 - Пустой массив
        }
        // ////////////////////
        // Сontrast Balance
        // ////////////////////
        mrcv::writeLog(" percent = " + std::to_string(percent), mrcv::LOGTYPE::DEBUG);
        if (percent > 0 && percent < 100)
        {
            float ratio = percent / 200.0f;
            cv::Mat flat;
            // Нахождение значения нижнего и верхнего процентилей на основе входного процентиля
            planeArray.reshape(1, 1).copyTo(flat);
            cv::sort(flat, flat, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
            int lowValue = flat.at<uchar>(cvFloor(((float)flat.cols) * ratio));
            int highValue = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - ratio)));
            // Насыщение ниже нижнего процентиля и выше верхнего процентиля
            planeArray.setTo(lowValue, planeArray < lowValue);
            planeArray.setTo(highValue, planeArray > highValue);
            // Масштабирования диапазона значений массива
            normalize(planeArray, planeArray, 0, 255, cv::NORM_MINMAX);
        }
        else  // если выход за диапазон percent
        {
            return 3; // 3 - выход за диапазон percent
        }
    }
    catch (...)
    {
        return -1; // Unhandled Exception
    }
    return 0; // SUCCESS
}

int mrcv::contrastExtantion(cv::Mat& planeArray, double m, double e)
{
    try
    {
        if (planeArray.empty())
        {
            return 1; // 1 - Пустой массив
        }
        cv::Mat  planeDouble;
        planeArray.convertTo(planeDouble, CV_32FC3, 1.f / 255);

        if (m < 0) m = cv::mean(planeDouble)[0]; // Если параметр < 0 тогда m = средняя якость
        planeDouble = m / (planeDouble + 1e-16);
        cv::pow(planeDouble, e, planeDouble);
        planeDouble = 1 / (1 + planeDouble);
        planeDouble.convertTo(planeArray, CV_8UC3, 255);
    }
    catch (...)
    {
        return -1; // Unhandled Exception
    }
    return 0; // SUCCESS
}