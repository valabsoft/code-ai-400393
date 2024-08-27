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

#include "mrcv-FPN.h"

#if _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace mrcv
{
	// Флаг отладочного лога, если false - лог не создается
	const bool IS_DEBUG_LOG_ENABLED = true;
	
	// Маска файла для функции записи видео
	const std::string UTILITY_DEFAULT_RECORDER_FILENAME = "video";
	
	// Интервал записи видео файла по умолчанию
	const int UTILITY_DEFAULT_RECORDER_INTERVAL = 5;
	
	// FPS камеры по умолчанию
	const int UTILITY_DEFAULT_CAMERA_FPS = 25;
	
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
}
