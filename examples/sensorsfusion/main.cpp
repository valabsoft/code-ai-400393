#include"mrcv/mrcv.h"

int main() 
{
	std::filesystem::path imageFiles("files\\Images\\");
	std::filesystem::path imuFile("files\\imuLog.csv");
	std::filesystem::path usblFile("files\\usblLog.csv");
	std::filesystem::path fuseFile("files\\fuseData.yaml");

	auto currentPath = std::filesystem::current_path();

	auto datasetPath = currentPath / imageFiles;
	auto imuDataPath = currentPath / imuFile;
	auto usblDataPath = currentPath / usblFile;
	auto fuseTupleSavePath = currentPath / fuseFile;

	int res = mrcv::fuseSensorData(usblDataPath.u8string(), imuDataPath.u8string(), datasetPath.u8string(), fuseTupleSavePath.u8string(), true);

	return 0;
}