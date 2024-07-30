#pragma once

#include <mrcv/export.h>

#include <iomanip>
#include <chrono>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <mrcv/mrcv-common.h>

namespace mrcv
{
	MRCV_EXPORT int add(int a, int b);
	MRCV_EXPORT int readImage(cv::Mat& image, std::string pathToImage, bool showImage = false);
	MRCV_EXPORT std::string getOpenCVBuildInformation();
	MRCV_EXPORT int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec);
}