#include"mrcv/mrcv.h"

int main() 
{
	//std::filesystem::path imageFiles("files\\Images\\");
	//std::filesystem::path imuFile("files\\imuLog.csv");
	//std::filesystem::path usblFile("files\\usblLog.csv");
	//std::filesystem::path fuseFile("files\\fuseData.yaml");

	auto currentPath = std::filesystem::current_path();

	//auto datasetPath = currentPath / imageFiles;
	//auto imuDataPath = currentPath / imuFile;
	//auto usblDataPath = currentPath / usblFile;
	//auto fuseTupleSavePath = currentPath / fuseFile;

	std::filesystem::path path = currentPath / "files";
	
	std::filesystem::path datasetPath = path / "Images";
	std::filesystem::path imuDataPath = path / "imuLog.csv" ;
	std::filesystem::path usblDataPath = path / "usblLog.csv";
	std::filesystem::path fuseTupleSavePath = path / "fuseData.yaml";

	int res = mrcv::fuseSensorData(usblDataPath.u8string(), imuDataPath.u8string(), datasetPath.u8string(), fuseTupleSavePath.u8string(), true);

	return 0;
}
