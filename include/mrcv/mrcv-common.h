#pragma once

#include <stdio.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <regex>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

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
}
