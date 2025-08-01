#include"mrcv/mrcv.h"

int main() 
{

	auto currentPath = std::filesystem::current_path();

	std::filesystem::path path = currentPath / "files";
	std::filesystem::path fuseTupleSavePath = path / "fuseData.yaml";
	std::filesystem::path datasetPath = path / "Images" / "";
	std::filesystem::path extracedFeaturesPath = path / "extractedData.yaml";


	int res = mrcv::extractFeatureVector(fuseTupleSavePath.u8string(), datasetPath.u8string(), extracedFeaturesPath.u8string());

	return 0;
}
