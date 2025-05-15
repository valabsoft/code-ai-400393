#include <mrcv/mrcv.h>

#include <iostream>

int main(int, char*[])
{
    // ////////////////////
    // 0. Инициализация параметров
    // ////////////////////
    int state; // для ошибок функций
    int  numberFrames; // количество кадров видео файле
    bool exitCode; // код выхода из цикла

    const  std::string filePathModelYoloNeuralNet = "./files/NeuralNet/yolov5n-seg.onnx";  // путь к файлу модель нейронной сети
    const  std::string filePathClasses = "./files/NeuralNet/yolov5.names";      // путь к файлу списоком обнаруживамых класов
    const  std::string pathToFileCameraParametrs = "./files/(66a)_(960p)_NewCamStereoModule_Air.xml";  // путь к файлу с параметрами стереокамеры
    cv::String videoPathCamera01 = "./files/SV_01_left_20.mp4"; // путь к видео файлу камера 01
    cv::String videoPathCamera02 = "./files/SV_02_right_20.mp4"; // путь к видео файлу камера 02
    cv::String pathSaveVideoFile;

    std::vector<cv::Mat> inputVideoFramesCamera01;  // входное цветное RGB изображение камеры 01
    std::vector<cv::Mat> inputVideoFramesCamera02;  // входное цветное RGB изображение камеры 02
    std::vector<cv::Mat> outputVideoFrames;         // последовательность изображений с резултатом
    std::vector<cv::Mat> disparityMapVideoFrames;   // последовательность карты диспаратности
    std::vector<cv::Mat> outDisparityMapVideoFrames;   // последовательность изображений карты диспаратности

    std::vector<cv::Mat> inputVideoFramesCamera01Remap;        // выровненное (ректифицированное) изображения камеры 01
    std::vector<cv::Mat> inputVideoFramesCamera02Remap;        // выровненное (ректифицированное) изображения камеры 02

    std::vector<mrcv::pointsData> points3DVideoFrames;              // данные об облаке 3D точек

    mrcv::settingsMetodDisparity settingsMetodDisparity;  // вводные настройки метода
    settingsMetodDisparity.smbNumDisparities = 240;     // Максимальное несоответствие минус минимальное несоответствие. Этот параметр должен быть кратен 16.
    settingsMetodDisparity.smbBlockSize = 5;            // Соответствующий размер блока. Это должно быть нечетное число >=1
    settingsMetodDisparity.smbPreFilterCap = 17;        // Значение усечения для предварительно отфильтрованных пикселей
    settingsMetodDisparity.smbMinDisparity = 0;         // Минимально возможное значение несоответствия
    settingsMetodDisparity.smbTextureThreshold = 0;
    settingsMetodDisparity.smbUniquenessRatio = 27;    // Предел в процентах, при котором наилучшее (минимальное) вычисленное значение функции стоимости должно “победить” второе наилучшее значение чтобы считать найденное совпадение правильным. Обычно достаточно значения в диапазоне 5-15
    settingsMetodDisparity.smbSpeckleWindowSize = 68;  // Максимальный размер областей сглаживания диспропорций для учета их шумовых пятен и исключения smbSpeckleRange
    settingsMetodDisparity.smbSpeckleRange = 21;       // Максимальное изменение диспропорций в пределах каждого подключенного компонента
    settingsMetodDisparity.smbDisp12MaxDiff = 21;

    settingsMetodDisparity.metodDisparity = mrcv::METOD_DISPARITY::MODE_HH; // метод поиска карты дииспаратности
    int limitOutPoints = 8000;              // лимит на количество точек на выходе алгоритма поиска облака 3D точек
    // параметры области для отсеивания выбросов {x_min, y_min, z_min, x_max, y_max, z_max}
    std::vector<double> limitsOutlierArea = { -4.0e3, -4.0e3, 250, 4.0e3, 4.0e3, 3.0e3 };

    double coefFilterSigma = 3.1; // коэвициент кратности с.к.о. для фильтра отсеивания выбрасов (по умолчанию 2.5*sigma)

    std::vector<std::vector<cv::Mat>> replyMasksVideoFrames;        // вектор бинарных масок сегментов обнаруженных объектов

    std::vector<cv::Mat> outputImage3dSceeneVideoFrames;  // 3D сцена

    mrcv::parameters3dSceene parameters3dSceene; // параметры 3D сцены
    parameters3dSceene.angX = -25;
    parameters3dSceene.angY = -45;
    parameters3dSceene.angZ = -35;
    parameters3dSceene.tX = -200;
    parameters3dSceene.tY = 200;
    parameters3dSceene.tZ = -600;
    parameters3dSceene.dZ = -1000;
    // запись в лог файл
    mrcv::writeLog();
    mrcv::writeLog("=== НОВЫЙ ЗАПУСК ===");

    // //////////////////
    // Время (начало)
    // //////////////////
    int64 time = cv::getTickCount();
    int delay;
    // //////////////////

    // ////////////////////
    // 1. Загрузка изображений из видео файла
    // ////////////////////
    // 1.1 Камера 01
    state = 0;
    exitCode = true;
    state = mrcv::readVideoFile(videoPathCamera01, exitCode, inputVideoFramesCamera01);
    if (state == 0)
    {
        mrcv::writeLog("1.1 Чтение видео из файла Камера 01 (успешно)");
    }
    else
    {
        mrcv::writeLog("Чтение видео из файла Камера 01, status = " + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }
    mrcv::writeLog("    количество кадров файла Камера 01 =  " + std::to_string(inputVideoFramesCamera01.size()));
    mrcv::writeLog("    видео файл Камера 01 =  " + videoPathCamera01);

    // 1.2 Камера 02
    state = 0;
    exitCode = true;
    state = mrcv::readVideoFile(videoPathCamera02, exitCode, inputVideoFramesCamera02);
    if (state == 0)
    {
        mrcv::writeLog("1.2 Чтение видео из файла Камера 02 (успешно)");
    }
    else
    {
        mrcv::writeLog("Чтение видео из файла Камера 02, status = " + std::to_string(state));
    }
    mrcv::writeLog("    количество кадров файла Камера 02 =  " + std::to_string(inputVideoFramesCamera02.size()));
    mrcv::writeLog("    видео файл Камера 02 =  " + videoPathCamera02);

    numberFrames = inputVideoFramesCamera01.size(); // определение количества кадров

    // 1.3 Выод на экран
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        cv::Mat outputStereoPair;
        state |= mrcv::makingStereoPair(inputVideoFramesCamera01.at(frame), inputVideoFramesCamera02.at(frame), outputStereoPair);
        state |= mrcv::showImage(outputStereoPair, "readVideoFile", 0.75);
        cv::waitKey(40);
    }

    if (state == 0)
    {
        mrcv::writeLog("1.3 Вывод стерео видео на экран (успешно)");
    }
    else
    {
        mrcv::writeLog("1.3 Вывод стерео видео на экран, status = " + std::to_string(state));
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма №01 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 2. Загрузка параметров камеры
    // ////////////////////
    mrcv::cameraStereoParameters cameraParameters;
    state = mrcv::readCameraStereoParametrsFromFile(pathToFileCameraParametrs.c_str(), cameraParameters);
    // ////////////////////
    if (state == 0)
    {
        mrcv::writeLog("2. Загрузка параметров стереокамеры из файла (успешно)");
    }
    else
    {
        mrcv::writeLog("2. Загрузка параметров стереокамеры из файла, readCameraStereoParametrsFromFile, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма №02 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // Обработка в цикле каждого изображения видео последовательности
    // ////////////////////
    // ////////////////////
    // 3 Подготовка изображений (коррекция искажений и выравнивание)
    // ////////////////////
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        cv::Mat inputImageCamera01Remap;        // выровненное (ректифицированное) изображения камеры 01
        cv::Mat inputImageCamera02Remap;        // выровненное (ректифицированное) изображения камеры 02

        state |= mrcv::convetingUndistortRectify(inputVideoFramesCamera01.at(frame), inputImageCamera01Remap,
            cameraParameters.map11, cameraParameters.map12);
        state |= mrcv::convetingUndistortRectify(inputVideoFramesCamera02.at(frame), inputImageCamera02Remap,
            cameraParameters.map21, cameraParameters.map22);

        inputVideoFramesCamera01Remap.push_back(inputImageCamera01Remap);
        inputVideoFramesCamera02Remap.push_back(inputImageCamera02Remap);
    }

    if (state == 0)
    {
        mrcv::writeLog("3. Выравнивание изображения (камера 01 & камера 02) (успешно)");
    }
    else
    {
        mrcv::writeLog("3. Выравнивание изображения (камера 01 & камера 02), convetingUndistortRectify, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма №03 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 4. Предобработка изображения
    // ////////////////////
    // выбор методов предобработки
    std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> metodImagePerProcessinBrightnessContrast =
    {
            mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_01_MEDIAN_FILTER,
            mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_02_AVARAGE_FILTER,
            mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_10_LAB_CLAHE,
            mrcv::METOD_IMAGE_PERPROCESSIN::NONE,
            mrcv::METOD_IMAGE_PERPROCESSIN::BRIGHTNESS_LEVEL_UP,
            mrcv::METOD_IMAGE_PERPROCESSIN::BRIGHTNESS_LEVEL_UP,
            mrcv::METOD_IMAGE_PERPROCESSIN::NONE,
    };
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {

        state |= mrcv::preprocessingImage(inputVideoFramesCamera01Remap.at(frame), metodImagePerProcessinBrightnessContrast,
            pathToFileCameraParametrs);
        state |= mrcv::preprocessingImage(inputVideoFramesCamera02Remap.at(frame), metodImagePerProcessinBrightnessContrast,
            pathToFileCameraParametrs);
    }

    if (state == 0)
    {
        mrcv::writeLog("4. Предобработка изображения (камера 01 & камера 02) завершена (успешно)");
    }
    else
    {
        mrcv::writeLog("4. Предобработка изображения (камера 01 & камера 02), preprocessingImage, state = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма №04 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 5. Поиск 3D точек сцены
    // ////////////////////
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        mrcv::pointsData points3D;              // данные об облаке 3D точек
        cv::Mat disparityMap;                   // карта диспаратности
        state |= mrcv::find3dPointsADS(inputVideoFramesCamera01Remap.at(frame), inputVideoFramesCamera02Remap.at(frame), points3D,
            settingsMetodDisparity, disparityMap, cameraParameters, limitOutPoints, limitsOutlierArea);

        cv::Mat outDisparityMap;
        double minVal, maxVal;
        minMaxLoc(disparityMap, &minVal, &maxVal);
        disparityMap.convertTo(outDisparityMap, CV_8UC1, 255 / (maxVal - minVal));
        applyColorMap(outDisparityMap, outDisparityMap, cv::COLORMAP_TURBO);

        outDisparityMapVideoFrames.push_back(outDisparityMap);
        disparityMapVideoFrames.push_back(disparityMap);
        points3DVideoFrames.push_back(points3D);
    }

    if (state == 0)
    {
        mrcv::writeLog("5. Облако 3D точек сцены найдено (успешно)");
    }
    else
    {
        mrcv::writeLog("5. Облако 3D точек сцены, find3dPointsADS, status = " + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма №05 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 5.2 Запись результирующего видео
    // ////////////////////
    pathSaveVideoFile = "./files/DisparityMap.mp4";
    exitCode = 1;
    state = 0;
    state = mrcv::writeVideoFile(pathSaveVideoFile, exitCode, outDisparityMapVideoFrames);
    if (state == 0)
    {
        mrcv::writeLog("5.2 Запись видео карты диспаратности (успешно)");
    }
    else
    {
        mrcv::writeLog("5.2 Запись видео карты диспаратности, writeVideoFile, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }
    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма № 05.2 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 6. Сегментация изображения (по результатам обнаружения и распознания объектов)
    // ////////////////////
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        std::vector<cv::Mat> replyMasks;        // вектор бинарных масок сегментов обнаруженных объектов
        cv::Mat outputImage;                    // изображение с резултатом

        state |= mrcv::detectingSegmentsNeuralNet(inputVideoFramesCamera01Remap.at(frame), outputImage, replyMasks,
            filePathModelYoloNeuralNet, filePathClasses);

        replyMasksVideoFrames.push_back(replyMasks);
        outputVideoFrames.push_back(outputImage);
    }

    if (state == 0)
    {
        mrcv::writeLog("6. Сегментация изображения (успешно)");
    }
    else
    {
        mrcv::writeLog("6. Сегментация изображения, detectingSegmentsNeuralNet, status = " + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма №06 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 7. Определения координат 3D точек в сегментах идентифицированных объектов
    // ////////////////////
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        state = mrcv::matchSegmentsWith3dPoints(points3DVideoFrames.at(frame), replyMasksVideoFrames.at(frame));
    }

    if (state == 0)
    {
        mrcv::writeLog("7. Сопоставление координат и сегментов (успешно)");
    }
    else
    {
        mrcv::writeLog("7. Сопоставление координат и сегментов, matchSegmentsWith3dPoints, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма №07 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 8. Нанесения координат 3D центра сегмента на изображени в виде текста
    // ////////////////////
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        state |= mrcv::addToImageCenter3dSegments(outputVideoFrames.at(frame), outputVideoFrames.at(frame), points3DVideoFrames.at(frame));
    }

    if (state == 0)
    {
        mrcv::writeLog("8. Нанесение координат 3D центра сегмента на результирующие изображение (успешно)");
    }
    else
    {
        mrcv::writeLog("8. Нанесение координат 3D центра сегмента на результирующие изображение, drawCenter3dSegments, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма №08 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 8.2 Запись результирующего видео
    // ////////////////////
    pathSaveVideoFile = "./files/Result.mp4";
    exitCode = 1;
    state = 0;
    state = mrcv::writeVideoFile(pathSaveVideoFile, exitCode, outputVideoFrames);
    if (state == 0)
    {
        mrcv::writeLog("8.2 Запись результирующего видео (успешно)");
    }
    else
    {
        mrcv::writeLog("8.2 Запись результирующего видео, writeVideoFile, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }
    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма № 08.2 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 9. Получение 3D сцены
    // ////////////////////
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        cv::Mat outputImage3dSceene;  // 3D сцена
        parameters3dSceene.angX++;
        parameters3dSceene.angY++;
        parameters3dSceene.angZ++;
        state = mrcv::getImage3dSceene(points3DVideoFrames.at(frame), parameters3dSceene, cameraParameters, outputImage3dSceene);

        outputImage3dSceeneVideoFrames.push_back(outputImage3dSceene);
    }

    if (state == 0)
    {
        mrcv::writeLog("9. Проекция 3D сцены на 2D изображение для вывода на экран (успешно)");
    }
    else
    {
        mrcv::writeLog("9. Проекция 3D сцены на 2D изображение для вывода на экран, getImage3dSceene, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма № 09 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 9.2 Запись результирующего видео
    // ////////////////////
    pathSaveVideoFile = "./files/3dSceene.mp4";
    exitCode = 1;
    state = 0;
    state = mrcv::writeVideoFile(pathSaveVideoFile, exitCode, outputImage3dSceeneVideoFrames);
    if (state == 0)
    {
        mrcv::writeLog("9.2 Запись видео облака 3D точек сцены (успешно)");
    }
    else
    {
        mrcv::writeLog("9.2 Запись видео облака 3D точек сцены, writeVideoFile, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }
    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма № 09.2 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 10. Функции алгоритма оценки параметров идентифицированных объектов заданной формы (Оценка параметров и формы идентифицированного объекта)
    // ////////////////////
    std::vector<std::vector<mrcv::primitiveData>> primitivesVideoFrames;
    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        std::cout << "frame = " << frame << std::endl;

        std::vector<mrcv::primitiveData> primitives;
        primitives.resize(points3DVideoFrames.at(frame).numSegments);

        for (int numberSelectedSegment = 0; numberSelectedSegment < (int)points3DVideoFrames.at(frame).numSegments; ++numberSelectedSegment)
        {
            state |= mrcv::detectObjectPrimitives(points3DVideoFrames.at(frame), primitives[numberSelectedSegment], numberSelectedSegment, coefFilterSigma);

            std::cout << "Segment = " << numberSelectedSegment << ", "
                << "numPointsInSegment = " << primitives[numberSelectedSegment].numPointsInSegment << ", "
                << "primitiveTypeName = " << primitives[numberSelectedSegment].primitiveTypeName << ", "
                << std::endl;
        }
        primitivesVideoFrames.push_back(primitives);
    }

    if (state == 0)
    {
        mrcv::writeLog("10. Оценка параметров и формы идентифицированных объектов (успешно)");
    }
    else
    {
        mrcv::writeLog("10. Оценка параметров и формы идентифицированных объектов, status = " + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма № 10 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 11. Функции алгоритма прорисовки примитивов идентифицированных объектов заданной формы
    // ////////////////////
    std::vector<cv::Mat> outputImageGeneralProjectionVideoFrames;
    //    std::vector<cv::Mat> outputImageProjectionXYVideoFrames;
    //    std::vector<cv::Mat> outputImageProjectionYZVideoFrames;
    //    std::vector<cv::Mat> outputImageProjectionXZVideoFrames;

    state = 0;
    for (int frame = 0; frame < numberFrames; ++frame)
    {
        mrcv::outputPrimitivesImages outputPrimitivesImages;
        state |= mrcv::drawPrimitives(outputVideoFrames.at(frame), outputPrimitivesImages, primitivesVideoFrames.at(frame));

        outputImageGeneralProjectionVideoFrames.push_back(outputPrimitivesImages.outputImageGeneralProjection);
    }

    if (state == 0)
    {
        mrcv::writeLog("11. Прорисовка примитивов идентифицированных объектов заданной формы  (успешно)");
    }
    else
    {
        mrcv::writeLog("11. Прорисовка примитивов идентифицированных объектов заданной формы , status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }

    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма № 11 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    // ////////////////////
    // 11.2 Запись результирующего видео примитивов
    // ////////////////////
    pathSaveVideoFile = "./files/ResultPrimitives.mp4";
    exitCode = 1;
    state = 0;
    state = mrcv::writeVideoFile(pathSaveVideoFile, exitCode, outputImageGeneralProjectionVideoFrames);
    if (state == 0)
    {
        mrcv::writeLog("11.2 Запись результирующего видео примитивов (успешно)");
    }
    else
    {
        mrcv::writeLog("11.2 Запись результирующего видео примитивов, writeVideoFile, status = "
            + std::to_string(state), mrcv::LOGTYPE::DEBUG);
    }
    // ////////////////////
    // Время (Конец)
    // ////////////////////
    time = cv::getTickCount() - time;
    delay = int(time * 1000 / cv::getTickFrequency());
    mrcv::writeLog(" Время работы алгоритма № 11.2 = " + std::to_string(delay) + " (ms)");
    // ////////////////////
    time = cv::getTickCount();
    // ////////////////////

    cv::waitKey(0);

    return 0;
}