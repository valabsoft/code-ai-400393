#include <mrcv/mrcv.h>

#include <iostream>

int main()
{
    // ////////////////////
    // 0. Инициализация параметров
    // ////////////////////
    int state;                              // для ошибок функций
    cv::Mat inputImageCamera01;             // входное цветное RGB изображение камеры 01
    cv::Mat inputImageCamera02;             // входное цветное RGB изображение камеры 02
    cv::Mat outputImage = inputImageCamera01.clone(); // изображение с резултатом
    cv::Mat inputImageCamera01Remap;        // выровненное (ректифицированное) изображения камеры 01
    cv::Mat inputImageCamera02Remap;        // выровненное (ректифицированное) изображения камеры 02
    mrcv::pointsData points3D;              // данные об облаке 3D точек

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

    cv::Mat disparityMap;                   // карта диспаратности
    settingsMetodDisparity.metodDisparity = mrcv::METOD_DISPARITY::MODE_SGBM; // метод поиска карты дииспаратности
    int limitOutPoints = 18000;              // лимит на количество точек на выходе алгоритма поиска облака 3D точек
    // параметры области для отсеивания выбросов {x_min, y_min, z_min, x_max, y_max, z_max}
    std::vector<double> limitsOutlierArea = { -4.0e3, -4.0e3, 250, 4.0e3, 4.0e3, 3.0e3 };
    std::vector<cv::Mat> replyMasks;        // вектор бинарных масок сегментов обнаруженных объектов
    const  std::string filePathModelYoloNeuralNet = "./files/NeuralNet/yolov5n-seg.onnx";  // путь к файлу моддель нейронной сери
    const  std::string filePathClasses = "./files/NeuralNet/yolov5.names";      // путь к файлу списоком обнаруживамых класов
    const  std::string pathToFileCameraParametrs = "./files/(66a)_(960p)_NewCamStereoModule_Air.xml";  // путь к файлу с параметрами стереокамеры
    cv::String filePathOutputImage01 = "./files/L1000.bmp";                   // путь к файлу изображения камера 01
    cv::String filePathOutputImage02 = "./files/R1000.bmp";                   // путь к файлу изображения камера 02
    cv::Mat outputImage3dSceene;  // 3D сцена
    mrcv::parameters3dSceene parameters3dSceene; // параметры 3D сцены
    parameters3dSceene.angX = 25;
    parameters3dSceene.angY = 45;
    parameters3dSceene.angZ = 35;
    parameters3dSceene.tX = -200;
    parameters3dSceene.tY = 200;
    parameters3dSceene.tZ = -600;
    parameters3dSceene.dZ = -1000;
    double coefFilterSigma = 3.3; // коэвициент кратности с.к.о. для фильтра отсеивания выбрасов (по умолчанию 2.5*sigma)

    mrcv::writeLog(); // запись в лог файл
    mrcv::writeLog("=== НОВЫЙ ЗАПУСК ===");

    // ////////////////////
    // 1. Загрузка изображения
    // ////////////////////
    inputImageCamera01 = cv::imread(filePathOutputImage01, cv::IMREAD_COLOR);
    inputImageCamera02 = cv::imread(filePathOutputImage02, cv::IMREAD_COLOR);
    if (!inputImageCamera01.empty() && !inputImageCamera02.empty())
    {
        mrcv::writeLog("1. Загрузка изображений из файла (успешно)");
    }
    mrcv::writeLog("    загружено изображение: " + filePathOutputImage01 + " :: " + std::to_string(inputImageCamera01.size().width) + "x"
        + std::to_string(inputImageCamera01.size().height) + "x" + std::to_string(inputImageCamera01.channels()));
    mrcv::writeLog("    загружено изображение: " + filePathOutputImage02 + " :: " + std::to_string(inputImageCamera02.size().width) + "x"
        + std::to_string(inputImageCamera02.size().height) + "x" + std::to_string(inputImageCamera02.channels()));

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
        mrcv::writeLog("readCameraStereoParametrsFromFile, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    // ////////////////////
    // 3. Предобработка изображения
    // ////////////////////
    // выбор методов предобработки
    std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> metodImagePerProcessinBrightnessContrast =
    {
            mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_01_MEDIAN_FILTER,
            mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_02_AVARAGE_FILTER,
            mrcv::METOD_IMAGE_PERPROCESSIN::NONE,
            mrcv::METOD_IMAGE_PERPROCESSIN::SHARPENING_01,
            mrcv::METOD_IMAGE_PERPROCESSIN::BRIGHTNESS_LEVEL_UP,
            mrcv::METOD_IMAGE_PERPROCESSIN::BRIGHTNESS_LEVEL_UP,
            mrcv::METOD_IMAGE_PERPROCESSIN::NONE,
    };

    state = mrcv::preprocessingImage(inputImageCamera01, metodImagePerProcessinBrightnessContrast, pathToFileCameraParametrs);
    if (state == 0)
    {
        mrcv::writeLog("3. Предобработка изображения (камера 01) завершена (успешно)");
    }
    else
    {
        mrcv::writeLog("3. Предобработка изображения (камера 01), preprocessingImage, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    state = mrcv::preprocessingImage(inputImageCamera02, metodImagePerProcessinBrightnessContrast, pathToFileCameraParametrs);
    if (state == 0)
    {
        mrcv::writeLog("3. Предобработка изображения (камера 02) завершена (успешно)");
    }
    else
    {
        mrcv::writeLog("3. Предобработка изображения (камера 02), preprocessingImage, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }


    // ////////////////////
    // 4. Функции для определения координат 3D точек в сегментах идентифицированных объектов и  восстановления 3D сцены по двумерным изображениям
    // ////////////////////
    state = mrcv::find3dPointsInObjectsSegments(inputImageCamera01, inputImageCamera02, cameraParameters,
        inputImageCamera01Remap, inputImageCamera02Remap, settingsMetodDisparity, disparityMap,
        points3D, replyMasks, outputImage, outputImage3dSceene, parameters3dSceene,
        filePathModelYoloNeuralNet, filePathClasses, limitOutPoints, limitsOutlierArea);
    if (state == 0)
    {
        mrcv::writeLog("4. Определения координат 3D точек обнаруженных объектов (успешно)");
    }
    else
    {
        mrcv::writeLog("4.  Определения координат 3D точек обнаруженных объектов, find3dPointsInObjectsSegments, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    // ////////////////////
    // 5. Функции алгоритма оценки параметров идентифицированных объектов заданной формы (Оценка параметров и формы идентифицированного объекта)
    // ////////////////////
    std::vector<mrcv::primitiveData> primitives;
    primitives.resize(points3D.numSegments);


    for (int numberSelectedSegment = 0; numberSelectedSegment < (int)points3D.numSegments; ++numberSelectedSegment)
    {
        state = mrcv::detectObjectPrimitives(points3D, primitives[numberSelectedSegment], numberSelectedSegment, coefFilterSigma);

        // Вывод данных о примитивах в лог-файл
        if (state == 0)
        {
            mrcv::writeLog("5. Оценка параметров и формы идентифицированного объекта №"
                + std::to_string(numberSelectedSegment) + " (успешно)");
        }
        else
        {
            mrcv::writeLog("5. Оценка параметров и формы идентифицированного объекта №"
                + std::to_string(numberSelectedSegment) + ", status =" + std::to_string(state), mrcv::LOGTYPE::ERROR);
        }

        mrcv::writeLog("    primitive type №:" + std::to_string(primitives[numberSelectedSegment].primitiveType)
            + ", " + primitives[numberSelectedSegment].primitiveTypeName + ",");
        mrcv::writeLog("    количесво точек в сегменте = " + std::to_string(primitives[numberSelectedSegment].numPointsInSegment) + ",");
        mrcv::writeLog("    параметры примитива: ");

        for (int qi = 0; qi < (int)primitives[numberSelectedSegment].primitiveParameter.size(); ++qi)
        {
            mrcv::writeLog("    " + std::to_string(primitives[numberSelectedSegment].primitiveParameter[qi]));
        }
    }

    // ////////////////////
    // 6. Функции алгоритма прорисовки примитивов идентифицированных объектов заданной формы
    // ////////////////////
    mrcv::outputPrimitivesImages outputPrimitivesImages;

    state = mrcv::drawPrimitives(outputImage, outputPrimitivesImages, primitives);

    if (state == 0)
    {
        mrcv::writeLog("6. Прорисовка примитивов идентифицированных объектов заданной формы  (успешно)");
    }
    else
    {
        mrcv::writeLog("6.Прорисовка примитивов идентифицированных объектов заданной формы , status =" + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    // ////////////////////
    // 6.1 Вывод исходного изображения
    // ////////////////////
    cv::Mat outputStereoPair;
    state = mrcv::makingStereoPair(inputImageCamera01, inputImageCamera02, outputStereoPair);
    if (state != 0) mrcv::writeLog("makingStereoPair (outputStereoPair) status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    state = mrcv::showImage(outputStereoPair, "SourceStereoImage");
    if (state == 0)
    {
        mrcv::writeLog("6.1 Вывод исходного изображения (успешно)");
    }
    else
    {
        mrcv::writeLog("6.1 Вывод исходного изображения, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    // ////////////////////
    // 6.2 Вывод карты диспаратности
    // ////////////////////
    state = mrcv::showDispsarityMap(disparityMap, "disparityMap", 0.75);
    if (state == 0)
    {
        mrcv::writeLog("6.2 Вывод карты диспаратности (успешно)");
    }
    else
    {
        mrcv::writeLog("6.2 Вывод карты диспаратности, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    // ////////////////////
    // 6.3 Вывод проекции 3D сцены на экран
    // ////////////////////
    state = mrcv::showImage(outputImage3dSceene, "outputImage3dSceene", 0.75);
    if (state == 0)
    {
        mrcv::writeLog("6.3 Вывод проекции 3D сцены на экран (успешно)");
    }
    else
    {
        mrcv::writeLog("6.3 Вывод проекции 3D сцены на экран, status =  " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    // ////////////////////
    // 6.4 Вывод результата в виде изображения с выделенными сегментами и 3D координатами центров этих сегментов
    // ////////////////////
    state = mrcv::showImage(outputImage, "outputImage", 0.75);

    if (state == 0)
    {
        mrcv::writeLog("6.4 Вывод изображения с 3D координатами центров сегментов (успешно)");
    }
    else
    {
        mrcv::writeLog("6.4 Вывод изображения с 3D координатами центров сегментов, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    // ////////////////////
    // 6.5 Вывод изображения примитивов
    // ////////////////////
    state = mrcv::showImage(outputPrimitivesImages.outputImageGeneralProjection, "outputImagedrawPrimitives", 1);
    if (state == 0)
    {
        mrcv::writeLog("6.5 Вывод изображения примитивов (успешно)");
    }
    else
    {
        mrcv::writeLog("6.5  Вывод изображения примитивов, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
    }

    // ////////////////////
    // Запись изображений с примитивами в файл
    // ////////////////////
    cv::imwrite("./files/outputImageGeneralProjection.jpg", outputPrimitivesImages.outputImageGeneralProjection);
    cv::imwrite("./files/projectionXY.jpg", outputPrimitivesImages.outputImageProjectionXY);
    cv::imwrite("./files/projectionYZ.jpg", outputPrimitivesImages.outputImageProjectionYZ);
    cv::imwrite("./files/projectionXZ.jpg", outputPrimitivesImages.outputImageProjectionXZ);

    // ////////////////////
    // 6.6 Вывод проекции 3D сцены на экран
    // ////////////////////

    for (int prim = 0; prim < int(primitives.size()); prim++)
    {

        mrcv::pointsData points3DObject;              // данные об облаке 3D точек
        cv::Mat outputImage3dObject;

        points3DObject.numPoints0 = primitives.at(prim).numPointsInSegment;
        points3DObject.numPoints = primitives.at(prim).numPointsInSegment;

        points3DObject.vu0.resize(primitives.at(prim).numPointsInSegment);
        points3DObject.xyz0.resize(primitives.at(prim).numPointsInSegment);
        points3DObject.rgb0.resize(primitives.at(prim).numPointsInSegment);

        for (int point = 0; point < primitives.at(prim).numPointsInSegment; point++)
        {
            points3DObject.vu0.at(point).push_back(primitives.at(prim).segmentPoints2Dvu.at(point).x);
            points3DObject.vu0.at(point).push_back(primitives.at(prim).segmentPoints2Dvu.at(point).y);

            points3DObject.xyz0.at(point).push_back(primitives.at(prim).segmentPoints3Dxyz.at(point).x);
            points3DObject.xyz0.at(point).push_back(primitives.at(prim).segmentPoints3Dxyz.at(point).y);
            points3DObject.xyz0.at(point).push_back(primitives.at(prim).segmentPoints3Dxyz.at(point).z);

            points3DObject.rgb0.at(point).push_back(primitives.at(prim).segmentPointsRGB.at(point)[0]);
            points3DObject.rgb0.at(point).push_back(primitives.at(prim).segmentPointsRGB.at(point)[1]);
            points3DObject.rgb0.at(point).push_back(primitives.at(prim).segmentPointsRGB.at(point)[2]);
        }

        state = mrcv::getImage3dSceene(points3DObject, parameters3dSceene, cameraParameters, outputImage3dObject);

        if (state == 0)
        {
            mrcv::writeLog("6.5 Проекция 3D сцены на 2D изображение для вывода на экран (успешно)");
        }
        else
        {
            mrcv::writeLog("6.6 getImage3dSceene, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
        }

        state = mrcv::showImage(outputImage3dObject, "outputImage3dObject # " + std::to_string(prim), 0.75);
        if (state == 0)
        {
            mrcv::writeLog("6.5 Вывод проекции 3D сцены на экран (успешно)");
        }
        else
        {
            mrcv::writeLog("6.5 Вывод проекции 3D сцены на экран, status =  " + std::to_string(state), mrcv::LOGTYPE::ERROR);
        }
    }

    cv::waitKey(0);
    return 0;
}