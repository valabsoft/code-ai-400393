#pragma once

#include <mrcv/export.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <fstream>

#include <stdio.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <regex>
#include <vector>
#include <iterator>
#include <cstdio>

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
	MRCV_EXPORT int getImagesFromYandex(std::string query, int minWidth, int minHeight, std::string nameTemplate, std::string outputFolder, bool separateDataset, unsigned int trainsetPercentage, unsigned int countFoto, bool money, std::string key, std::string secretKey);
}
