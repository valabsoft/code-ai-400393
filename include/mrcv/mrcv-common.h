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
	// ���� ����������� ����, ���� false - ��� �� ���������
	const bool IS_DEBUG_LOG_ENABLED = true;
	
	// ����� ����� ��� ������� ������ �����
	const std::string UTILITY_DEFAULT_RECORDER_FILENAME = "video";
	
	// �������� ������ ����� ����� �� ���������
	const int UTILITY_DEFAULT_RECORDER_INTERVAL = 5;
	
	// FPS ������ �� ���������
	const int UTILITY_DEFAULT_CAMERA_FPS = 25;
	
	// ���� �������
	enum class CODEC
	{
		XVID,
		MJPG,
		mp4v,
		h265
	};
	
	// ���� ������� � ���-�����
	enum class LOGTYPE
	{
		DEBUG,		// ��������			DEBG
		ERROR,		// ������			ERRR
		EXCEPTION,	// ����������		EXCP
		INFO,		// ����������		INFO
		WARNING		// ��������������	WARN
	};
	
	// ��������� ��� �������� ���������� ���������� ��������� ������
	struct CalibrationParametersMono
	{
		cv::Mat cameraMatrix;     // ������� ������
		cv::Mat distCoeffs;       // ������ ������������� ���������
		cv::Mat rvecs;            // ������ �������� �������� ��� �������� �� ������ ������� � ����� ������
		cv::Mat tvecs;            // ������ �������� �������� ��� �������� �� ������ ������� � ����� ������
		cv::Mat stdDevIntrinsics; // ������ ������ ���������� ���������� ������
		cv::Mat stdDevExtrinsics; // ������ ������ ������� ���������� ������
		cv::Mat perViewErrors;    // ������ �������������������� ������ ����������������� ��� ������� ����
		double RMS;               // �������� �������������������� ������ �����������������
	};
	
	// ��������� ��� �������� ���������� ���������� ������ ������
	struct CalibrationParametersStereo {
		cv::Mat cameraMatrixL;	// ������� ����� ������
		cv::Mat cameraMatrixR;	// ������� ������ ������
		cv::Mat distCoeffsL;	// ������ ������������� ��������� ����� ������
		cv::Mat distCoeffsR;	// ������ ������������� ��������� ������ ������
		cv::Mat R;				// ������� ���������
		cv::Mat T;				// ������ ��������
		cv::Mat E;				// ������� ������������ ����������
		cv::Mat F;				// ��������������� �������
		cv::Mat rvecs;			// ������ �������� �������� ��� �������� �� ������ ������� � ����� ������
		cv::Mat tvecs;			// ������ �������� �������� ��� �������� �� ������ ������� � ����� ������
		cv::Mat perViewErrors;	// ������ �������������������� ������ ����������������� ��� ������� ����
		double RMS;				// �������� �������������������� ������ �����������������
	};
}
