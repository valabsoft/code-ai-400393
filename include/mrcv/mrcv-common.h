#pragma once

#include <stdio.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include "mrcv-segmentation.h"

#if _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#define Pi 3,1415926535

namespace mrcv
{
	// Флаг отладочного лога, если false - лог не создается
	static const bool IS_DEBUG_LOG_ENABLED = true;
	
	// Маска файла для функции записи видео
	static const std::string UTILITY_DEFAULT_RECORDER_FILENAME = "video";
	
	// Интервал записи видео файла по умолчанию
	static const int UTILITY_DEFAULT_RECORDER_INTERVAL = 5;
	
	// FPS камеры по умолчанию
	static const int UTILITY_DEFAULT_CAMERA_FPS = 25;

	// Константы для задачи OBJCOURSE
	static const float OBJCOURSE_FONT_SCALE = 0.7f;
	static const int OBJCOURSE_THICKNESS = 1;
	static cv::Scalar OBJCOURSE_BLACK = cv::Scalar(0, 0, 0);
	static cv::Scalar OBJCOURSE_YELLOW = cv::Scalar(0, 255, 255);
	static cv::Scalar OBJCOURSE_RED = cv::Scalar(0, 0, 255);
	static cv::Scalar OBJCOURSE_GREEN = cv::Scalar(0, 255, 0);
	static const bool OBJCOURSE_DRAW_LABEL = true;
	
	// Виды кодеков
	enum class CODEC
	{
		XVID,
		MJPG,
		mp4v,
		h265
	};
	
	// Виды записей в лог-файле
	enum class LOGTYPE
	{
		DEBUG,		// Отладака			DEBG
		ERROR,		// Ошибка			ERRR
		EXCEPTION,	// Исключение		EXCP
		INFO,		// Информация		INFO
		WARNING		// Предупреждение	WARN
	};
	
	// Методы коррекции конраста изобрадения (цветовые пространства) для функции increaseImageContrast
	enum class METOD_INCREASE_IMAGE_CONTRAST
	{
		EQUALIZE_HIST,       // метод Гистограммная  эквализация  (Histogram Equalization)
		CLAHE,               // метод Адаптивная Гистограммная  эквализация (Contrast Limited Adaptive Histogram Equalization)
		CONTRAST_BALANCING,  // метод Баланса контрастности, основанный на фильтрации крайних значений
		CONTRAST_EXTENSION,  // метод Расширения контрастности, основанный на логарифмическом преобразовании
	};

	// Цветовые модели (цветовые пространства) для функции increaseImageContrast
	enum class COLOR_MODEL
	{
		CM_RGB,    //
		CM_HSV,    //
		CM_LAB,    //
		CM_YCBCR,  //
	};

	//  Методы предобработки изображения для функции preprocessingImage
	enum class METOD_IMAGE_PERPROCESSIN
	{
		NONE,                               // без изменений
		CONVERTING_BGR_TO_GRAY,             // преобразование типа изображения к из цветноко BGR к монохромному (серому)
		// Коррекция яркости
		BRIGHTNESS_LEVEL_UP,                // увеличение уровня яркости на один уровень
		BRIGHTNESS_LEVEL_DOWN,              // уменьшение уровня яркости на один уровен
		// Методы коррекции контрастности
		// YCBCR
		BALANCE_CONTRAST_01_YCBCR_EQUALIZEHIST,        // повышение контрастности, метод 01
		BALANCE_CONTRAST_02_YCBCR_CLAHE,               // повышение контрастности, метод 02
		BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING,  // повышение контрастности, метод 03
		BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION,  // повышение контрастности, метод 04
		// HSV
		BALANCE_CONTRAST_05_HSV_EQUALIZEHIST,          // повышение контрастности, метод 05
		BALANCE_CONTRAST_06_HSV_CLAHE,                 // повышение контрастности, метод 06
		BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING,    // повышение контрастности, метод 07
		BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION,    // повышение контрастности, метод 08
		// LAB
		BALANCE_CONTRAST_09_LAB_EQUALIZEHIST,          // повышение контрастности, метод 09
		BALANCE_CONTRAST_10_LAB_CLAHE,                 // повышение контрастности, метод 10
		BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING,    // повышение контрастности, метод 11
		BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION,    // повышение контрастности, метод 12
		// RGB
		BALANCE_CONTRAST_13_RGB_EQUALIZEHIST,          // повышение контрастности, метод 13
		BALANCE_CONTRAST_14_RGB_CLAHE,                 // повышение контрастности, метод 14
		BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING,    // повышение контрастности, метод 15
		BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION,    // повышение контрастности, метод 16
		// Резкость
		SHARPENING_01,                      // повышение резкости, метод 01
		SHARPENING_02,                      // повышение резкости, метод 02
		// Фильтрация шума
		NOISE_FILTERING_01_MEDIAN_FILTER,   // фильтрация изображения от импульсных шумов
		NOISE_FILTERING_02_AVARAGE_FILTER,  // фильтрация изображения от шумов
		CORRECTION_GEOMETRIC_DEFORMATION    // коррекция геометрических искажений
	};

	//  Параметры камеры
	struct cameraParameters
	{
		// Внутренние параметры камеры
		cv::Mat M1;         // матрица камеры 3x3
		cv::Mat D1;         // вектор коэффициентов искажения , коэффициенты радиальной и тангенциальной дисторсии
		// Внешиние параметры камеры
		cv::Mat R;          // матрица поворота 3x3 камеры относительно абсолютной системы координат
		cv::Mat T;          // вектор смещения  камеры  относительно абсолютной системы координат
		//  Матрицы проекции - проецирует 3D точки, заданные в исправленной системе координат камеры, на исправленное 2D изображение камеры
		cv::Mat R1;         // Матрица поворота 3x3 для выполнения процедуры выравнивания (ректификации) для первой
		cv::Mat P1;         // Матрицы проекции 3x4 в новых (выравненных) системах координат для камеры
		// Карта переназначения используются для быстрого преобразования изображения
		cv::Mat map11;      // карта 01 для переназначения камеры
		cv::Mat map12;      // карта 02 для переназначения камеры
		// дополнительные параметры
		cv::Size imageSize; // размер изображения
		double rms;         // ошибка перепроецирорования
		double avgErr;      // средняя ошибка
		char pathToFileCameraParametrs; // путь к файлу c параметрами камеры (c которого загружены данные).
	};
	
	// Список методов построение карты расхождений
	enum class METOD_DISPARITY
	{
		MODE_NONE,      // без построения каты диспаратности (загружается уже готовая карта диспаратности)
		MODE_BM,        // самый быстрый и простой
		MODE_SGBM,      // обычный режим
		MODE_SGBM_3WAY, // алгоритм выполняется быстрее обычного режима
		MODE_HH,        // выполнение полномасштабного двухпроходного алгоритма
		MODE_HH4        // долгий
	};

	// Настройки метода поиска расхождений (диспаратности) для поиска 3D точек
	struct settingsMetodDisparity
	{
		mrcv::METOD_DISPARITY metodDisparity;
		int smbNumDisparities = 192;     // Максимальное несоответствие минус минимальное несоответствие. Этот параметр должен быть кратен 16.
		int smbBlockSize = 9;            // Соответствующий размер блока. Это должно быть нечетное число >=1
		int smbPreFilterCap = 17;        // Значение усечения для предварительно отфильтрованных пикселей
		int smbMinDisparity = 0;         // Минимально возможное значение несоответствия
		int smbTextureThreshold = 0;
		int smbUniquenessRatio = 27;    // Предел в процентах, при котором наилучшее (минимальное) вычисленное значение функции стоимости должно “победить” второе наилучшее значение чтобы считать найденное совпадение правильным. Обычно достаточно значения в диапазоне 5-15
		int smbSpeckleWindowSize = 68;  // Максимальный размер областей сглаживания диспропорций для учета их шумовых пятен и исключения smbSpeckleRange
		int smbSpeckleRange = 21;       // Максимальное изменение диспропорций в пределах каждого подключенного компонента
		int smbDisp12MaxDiff = 21;
	};
	// Параметры стереокамеры
	struct cameraStereoParameters
	{
		// Внутренние параметры камер 01 и 02 стереокамеры
		cv::Mat M1;     // матрица камеры 3x3  (камеры 01 стереокамеры)
		cv::Mat D1;     // вектор коэффициентов искажения (камеры 01 стереокамеры), коэффициенты радиальной и тангенциальной дисторсии
		cv::Mat M2;     // матрица камеры 3x3  (камеры 02 стереокамеры)
		cv::Mat D2;     // вектор коэффициентов искажения (камеры 02 стереокамеры), коэффициенты радиальной и тангенциальной дисторсии
		// Внешиние параметры камер стереокамеры
		cv::Mat E;      // существенная матрица стереокамеры
		cv::Mat F;      // фундаментальная матрица стереокамеры
		cv::Mat R;      // матрица поворота 3x3 камеры 02 относительно камеры 01
		cv::Mat T;      // вектор смещения  камеры 02 относительно камеры 01
		//  Матрицы проекции - проецирует 3D точки, заданные в исправленной системе координат камеры, на исправленное 2D изображение камеры
		cv::Mat R1;     // Матрица поворота 3x3 для выполнения процедуры выравнивания (ректификации) для первой 01
		cv::Mat R2;     // Матрица поворота 3x3 для выполнения процедуры выравнивания (ректификации) для первой 02
		cv::Mat P1;     // Матрицы проекции 3x4 в новых (выравненных) системах координат для камеры 01
		cv::Mat P2;     // Матрицы проекции 3x4 в новых (выравненных) системах координат для камеры 02
		cv::Mat Q;      // Матрица 4x4 преобразования перспективы
		// Карта переназначения используются для быстрого преобразования изображения
		cv::Mat map11;  // карта 01 для переназначения камеры 01
		cv::Mat map12;  // карта 02 для переназначения камеры 01
		cv::Mat map21;  // карта 01 для переназначения камеры 02
		cv::Mat map22;  // карта 02 для переназначения камеры 02
		// дополнительные параметры
		cv::Size imageSize;     // размер изображения
		double rms;             // ошибка перепроецирования
		double avgErr;          // средняя ошибка
		char pathToFileStereoCameraParametrs; // путь к файлу c параметрами камеры (c которого загружены данные).
	};

	// Данные для хранения информации о облаке 3D точек
	struct pointsData     // Структура хранения данных о 3D точках и проекциях
	{
		int numPoints0 = -1;                          // Количество точек (до уменьшения)
		int numPoints = -1;                          // Количество точек (после уменьшения, во всех сегментах)
		// Для точек до сегментации сразу полсе обнаружения 3D точек (внешний вектор - номер точки, внутренний - её параметры)
		std::vector<std::vector<int>> vu0;            // 2D координаты точки на изображении  (пиксель)
		std::vector<std::vector<double>> xyz0;        // 3D координаты точки на пространстве (мм)
		std::vector<std::vector<int>> rgb0;           // цвет 3D точки (все) (вектора r, g, b)
		// Для точек после после уменьшения и распределения по сегментам (внешний вектор номер точки, внутренний её параметры)
		std::vector<std::vector<int>> vu;            // 2D координаты точки на изображении (после уменьшения) [vu -> yx] (пиксель)
		std::vector<std::vector<double>> xyz;        // 3D координаты точки на пространстве (после уменьшения) (мм)
		std::vector<std::vector<int>> rgb;           // цвет точки (после уменьшения)
		std::vector<int> segment;                    // Для каждой точки номер сегмента
		// Для сегмента
		int numSegments = -1;                        // Количество сегментов
		std::vector<int> numPointsInSegments;             // Количесво точек в сегменте
		std::vector<std::vector<int>> pointsInSegments;   // Для каждого сегмента номер точки
		std::vector<cv::Point2d> center2dSegments;        // Геометрический 2D центр сегмента
		std::vector<cv::Point3d> center3dSegments;        // Геометрический 3D центр сегмента
	};

	// Параметры проектирования 3D сцены на 2D изображени для вывода на экрани или в видео файл
	struct parameters3dSceene  // Структура параметров проектирования 3D сцены на 2D изображении для вывода на экран
	{
		double angX = 0;    // угол поворота вокруг оси x центра вращения (градусы)
		double angY = 0;    // угол поворота вокруг оси y центра вращения (градусы)
		double angZ = 0;    // угол поворота вокруг оси z центра вращения (градусы)
		double tX = 0;      // смещение вдоль оси x центра вращения (мм)
		double tY = 0;      // смещение вдоль оси y центра вращения (мм)
		double tZ = 0;      // смещение вдоль оси z центра вращения  (мм)
		double scale = 1;   // Масштаб
		double dZ = -3000;  // смещение центра вращения от камер вдоль оси z (мм)
	};
	
	// Структура для хранения параметров калибровки одиночной камеры
	struct CalibrationParametersMono
	{
		cv::Mat cameraMatrix;     // Матрица камеры
		cv::Mat distCoeffs;       // Вектор коэффициентов дисторсии
		cv::Mat rvecs;            // Кортеж векторов поворота для перехода из базиса объекта в базис камеры
		cv::Mat tvecs;            // Кортеж векторов смещения для перехода из базиса объекта в базис камеры
		cv::Mat stdDevIntrinsics; // Вектор оценок внутренних параметров камеры
		cv::Mat stdDevExtrinsics; // Вектор оценок внешних параметров камеры
		cv::Mat perViewErrors;    // Вектор среднеквадратической ошибки перепроецирования для каждого вида
		double RMS;               // Значение среднеквадратической ошибки перепроецирования
	};

	// Структура для хранения параметров калибровки стерео камеры
	struct CalibrationParametersStereo {
		cv::Mat cameraMatrixL;	// Матрица левой камеры
		cv::Mat cameraMatrixR;	// Матрица правой камеры
		cv::Mat distCoeffsL;	// Вектор коэффициентов дисторсии левой камеры
		cv::Mat distCoeffsR;	// Вектор коэффициентов дисторсии правой камеры
		cv::Mat R;				// Матрица поворотов
		cv::Mat T;				// Вектор смещений
		cv::Mat E;				// Матрица существенных параметров
		cv::Mat F;				// Фундаментальная матрица
		cv::Mat rvecs;			// Кортеж векторов поворота для перехода из базиса объекта в базис камеры
		cv::Mat tvecs;			// Кортеж векторов смещения для перехода из базиса объекта в базис камеры
		cv::Mat perViewErrors;	// Вектор среднеквадратической ошибки перепроецирования для каждого вида
		double RMS;				// Значение среднеквадратической ошибки перепроецирования
	};

	// Структура конфигурационного файла для калибровки
	struct CalibrationConfig
	{
		std::string folder_name = "../calibration_images/";	// Путь к конфигурационному файлу
		int keypoints_c = 9;								// Число ключевых точек вдоль одного столбца калибровочной доски
		int keypoints_r = 6;								// Число ключевых точек вдоль одной строки калибровочной доски
		float square_size = 20.1;							// Размер квадрата калибровочной доски в мм
		int image_count = 50;								// Общее число пар изображений в фотосете
	};


	// Структура trianTricks предназначена для повышения производительности обучения
	struct trainTricks {
		unsigned int freeze_epochs = 0;					// Замораживает магистраль нейронной сети во время первых freeze_epochs, по умолчанию 0;
		std::vector<unsigned int> decay_epochs = { 0 };	// При каждом decay_epochs скорость обучения будет снижаться на 90 процентов, по умолчанию 0;
		float dice_ce_ratio = (float)0.5;				// Вес выпадения кубиков в общем проигрыше, по умолчанию 0,5;
		float horizontal_flip_prob = (float)0.0;		// Вероятность увеличения поворота по горизонтали, по умолчанию 0;
		float vertical_flip_prob = (float)0.0;			// Вероятность увеличения поворота по вертикали, по умолчанию 0;
		float scale_rotate_prob = (float)0.0;			// Вероятность выполнения поворота и увеличения масштаба, по умолчанию 0;
		float scale_limit = (float)0.1;
		float rotate_limit = (float)45.0;
		int interpolation = cv::INTER_LINEAR;
		int border_mode = cv::BORDER_CONSTANT;
	};

	enum class AUGMENTATION_METHOD
	{
		NONE,
		FLIP_HORIZONTAL,
		FLIP_VERTICAL,
		ROTATE_IMAGE_90,
		ROTATE_IMAGE_45,
		ROTATE_IMAGE_270,
		ROTATE_IMAGE_315,
		FLIP_HORIZONTAL_AND_VERTICAL,
		TEST
	};

	enum class DISPARITY_TYPE
	{
		ALL,
		BASIC_DISPARITY,
		BASIC_HEATMAP,
		FILTERED_DISPARITY,
		FILTERED_HEATMAP,
	};
}
