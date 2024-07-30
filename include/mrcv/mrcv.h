#pragma once

#include <mrcv/export.h>

#include <iostream>

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


namespace mrcv
{
	MRCV_EXPORT int add(int a, int b);
	MRCV_EXPORT int imread(std::string pathtoimage);
	
    MRCV_EXPORT int getImagesFromYandex(std::string zapros,  int minwidth, int minheight, std::string nametemplate,  std::string outputfolder, bool separatedataset, int trainsetpercentage, unsigned int countfoto,bool money, std::string key="", std::string secretKey="");


}
