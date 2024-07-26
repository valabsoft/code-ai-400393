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

//namespace fs = std::filesystem;

namespace mrcv
{
	MRCV_EXPORT int add(int a, int b);
	MRCV_EXPORT int imread(std::string pathtoimage);
	
	MRCV_EXPORT std::string readFile(const std::string& fileName);
	MRCV_EXPORT int parserUrl(const char* nameFind);
	//MRCV_EXPORT std::vector< int > substringFind(std::string text,std::string word);
	MRCV_EXPORT std::vector< std::string > urlFind(std::string text);
	MRCV_EXPORT int saveFile(std::string nameFile, std::vector< std::string > arrUrl);
	MRCV_EXPORT void downloadFoto(std::vector< std::string > arrUrl, std::string patch);
	MRCV_EXPORT void delSmal(std::string filename,int rows,int cols);
	MRCV_EXPORT void copyFile(std::string filepath,std::string target, int percent);

}
