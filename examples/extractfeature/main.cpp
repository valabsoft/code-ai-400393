#include"mrcv/mrcv.h"

int main() 
{
	std::filesystem::path imageFiles("files\\Images\\");
	std::filesystem::path fuseFile("files\\fuseData.yaml");
	std::filesystem::path extractedFeaturesFile("files\\extractedData.yaml");

	auto currentPath = std::filesystem::current_path();

	auto fuseTupleSavePath = currentPath / fuseFile;
	auto datasetPath = currentPath / imageFiles;
	auto extracedFeaturesPath = currentPath / extractedFeaturesFile;

	int res = mrcv::extractFeatureVector(fuseTupleSavePath.u8string(), datasetPath.u8string(), extracedFeaturesPath.u8string());

	return 0;
}