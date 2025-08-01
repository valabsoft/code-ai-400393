#include <mrcv/mrcv.h>

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
	cv::Mat disparityMap;                   // карта диспаратности
	settingsMetodDisparity.metodDisparity = mrcv::METOD_DISPARITY::MODE_SGBM; // метод поиска карты дииспаратности
	int limitOutPoints = 8000;              // лимит на количество точек на выходе алгоритма поиска облака 3D точек
	// параметры области для отсеивания выбросов {x_min, y_min, z_min, x_max, y_max, z_max}
	std::vector<double> limitsOutlierArea = { -4.0e3, -4.0e3, 450, 4.0e3, 4.0e3, 3.0e3 };
	std::vector<cv::Mat> replyMasks;        // вектор бинарных масок сегментов обнаруженных объектов
	
	//auto currentPath = std::filesystem::current_path();

	//std::filesystem::path path = currentPath / "files";
	//const std::filesystem::path filePathModelYoloNeuralNet = path / "NeuralNet" / "yolov5n-seg.onnx";
	//const std::filesystem::path filePathClasses = path / "NeuralNet" / "yolov5.names";
	
	const  std::string filePathModelYoloNeuralNet = "./files/NeuralNet/yolov5n-seg.onnx";  // путь к файлу моддель нейронной сери
	const  std::string filePathClasses = "./files/NeuralNet/yolov5.names";      // путь к файлу списоком обнаруживамых класов
	
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
	state = mrcv::readCameraStereoParametrsFromFile("./files/(66a)_(960p)_NewCamStereoModule_Air.xml", cameraParameters);
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
	// 3.A Функции для определения координат 3D точек в сегментах идентифицированных объектов и  восстановления 3D сцены по двумерным изображениям
	// ////////////////////
	state = mrcv::find3dPointsInObjectsSegments(inputImageCamera01, inputImageCamera02, cameraParameters,
		inputImageCamera01Remap, inputImageCamera02Remap, settingsMetodDisparity, disparityMap,
		points3D, replyMasks, outputImage, outputImage3dSceene, parameters3dSceene,
		filePathModelYoloNeuralNet, filePathClasses, limitOutPoints, limitsOutlierArea);

	// ////////////////////
	// 4. Вывод данных
	// ////////////////////
	// 4.1 Загрузка и вывод изображения экспериментального стенда
	// ////////////////////
	cv::Mat fotoExperimantalStand = cv::imread("./files/experimantalStand.jpg", cv::IMREAD_COLOR);
	state = mrcv::showImage(fotoExperimantalStand, "fotoExperimantalStand");
	// ////////////////////
	// 4.2 Вывод исходного изображения
	// ////////////////////
	cv::Mat outputStereoPair;
	state = mrcv::makingStereoPair(inputImageCamera01, inputImageCamera02, outputStereoPair);
	if (state != 0) mrcv::writeLog("makingStereoPair (outputStereoPair) status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	state = mrcv::showImage(outputStereoPair, "SourceStereoImage");
	if (state == 0)
	{
		mrcv::writeLog("4.2 Вывод исходного изображения (успешно)");
	}
	else
	{
		mrcv::writeLog("4.2 Вывод исходного изображения, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	}
	// ////////////////////
	// 4.3 Вывод стереопары после ректификации
	// ////////////////////
	cv::Mat outputStereoPairRemap;
	state = mrcv::makingStereoPair(inputImageCamera01Remap, inputImageCamera02Remap, outputStereoPairRemap);
	state = mrcv::showImage(outputStereoPairRemap, "outputStereoPairRemap");
	if (state == 0)
	{
		mrcv::writeLog("4.3 Вывод стереопары после ректификации (успешно)");
	}
	else
	{
		mrcv::writeLog("4.3 Вывод стереопары после ректификации, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	}
	// ////////////////////
	// 4.4 Вывод карты диспаратности
	// ////////////////////
	state = mrcv::showDispsarityMap(disparityMap, "disparityMap");
	if (state == 0)
	{
		mrcv::writeLog("4.4 Вывод карты диспаратности (успешно)");
	}
	else
	{
		mrcv::writeLog("4.4 Вывод карты диспаратности, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	}
	// ////////////////////
	// 4.5 Вывод бинарных изображений сегментов
	// ////////////////////
	for (int qs = 0; qs < points3D.numSegments; ++qs)
	{
		state = mrcv::showImage(replyMasks.at(qs), "replyMasks " + std::to_string(qs), 0.5);
	}
	if (state == 0)
	{
		mrcv::writeLog("4.5 Вывод бинарных изображений сегментов (успешно)");
	}
	else
	{
		mrcv::writeLog("4.5 Вывод бинарных изображений сегментов, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	}
	// ////////////////////
	//  4.6 Сохранение данных о 3D точках после сегментации в текстовый файл
	// ////////////////////
	cv::String pathToFiledPointsInObjectsSegments = "./files/3DPointsInObjectsSegments.txt";
	state = mrcv::saveInFile3dPointsInObjectsSegments(points3D, pathToFiledPointsInObjectsSegments);
	if (state == 0)
	{
		mrcv::writeLog("4.6 Сохранение данных о 3D точках в текстовый файл (успешно)");
		mrcv::writeLog("    путь к файлу " + pathToFiledPointsInObjectsSegments);
	}
	else
	{
		mrcv::writeLog("4.6 Сохранение данных о 3D точках в текстовый файл, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	}
	// ////////////////////
	// 4.7 Вывод результата в виде изображения с выделенными сегментами и 3D координатами центров этих сегментов
	// ////////////////////
	state = mrcv::showImage(outputImage, "outputImage", 1.0);

	if (state == 0)
	{
		mrcv::writeLog("4.7 Вывод изображения с 3D координатами центров сегментов (успешно)");
	}
	else
	{
		mrcv::writeLog("4.7 Вывод изображения с 3D координатами центров сегментов, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	}
	// ////////////////////
	// 4.8 Вывод проекции 3D сцены на экран
	// ////////////////////
	state = mrcv::showImage(outputImage3dSceene, "outputImage3dSceene", 1.0);
	if (state == 0)
	{
		mrcv::writeLog("4.8 Вывод проекции 3D сцены на экран (успешно)");
	}
	else
	{
		mrcv::writeLog("4.8 Вывод проекции 3D сцены на экран, status =  " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	}

	cv::waitKey(0);
	return 0;
}
