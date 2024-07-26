#pragma once

#include <mrcv/export.h>

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

namespace mrcv
{
	MRCV_EXPORT int add(int a, int b);
	MRCV_EXPORT int imread(std::string pathtoimage);
	MRCV_EXPORT  std::string openCVInfo();
}