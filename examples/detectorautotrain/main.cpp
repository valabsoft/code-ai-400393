#include"mrcv/mrcv.h"

int main()
{
	std::filesystem::path imageFiles("files\\onwater\\");
	std::filesystem::path vocClassesFile("files\\onwater\\voc_classes.txt");
	std::filesystem::path modelSaveFile("files\\onwater_autodetector.pt");
	std::filesystem::path pretrainedModelFile("files\\yolo4_tiny.pt");

	auto currentPath = std::filesystem::current_path();

	auto datasetPath = currentPath / imageFiles;
	auto vocClassesPath = currentPath / vocClassesFile;
	auto modelSavePath = currentPath / modelSaveFile;
	auto pretrainedModelPath = currentPath / pretrainedModelFile;

	mrcv::Detector detector;
	detector.Initialize(0, 416, 416, vocClassesPath.string());
	detector.AutoTrain(datasetPath.u8string(), ".jpg", { 10, 15, 30 }, { 4, 8 }, { 0.001, 1.0E-4F }, pretrainedModelPath.u8string(), modelSavePath.u8string());

	return 0;
}