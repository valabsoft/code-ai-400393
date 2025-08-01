#include"mrcv/mrcv.h"




int main()
{
	auto currentPath = std::filesystem::current_path();
	
	std::filesystem::path path = currentPath / "files";
	std::filesystem::path datasetPath = path / "onwater";
	std::filesystem::path vocClassesPath = path / "onwater" / "voc_classes.txt";
	std::filesystem::path modelSavePath = path / "onwater_autodetector.pt";
	std::filesystem::path pretrainedModelPath = path / "yolo4_tiny.pt";

	mrcv::Detector detector;
	detector.Initialize(0, 416, 416, vocClassesPath.string());
	
	auto start = std::chrono::high_resolution_clock::now();
	detector.AutoTrain(datasetPath.u8string(), ".jpg", { 10, 15, 30 }, { 4, 8 }, { 0.001, 1.0E-4F }, pretrainedModelPath.u8string(), modelSavePath.u8string());
	
	auto end = std::chrono::high_resolution_clock::now();
	
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	
	std::cout << "Time taken by autotrain: " << duration.count() << "ms" << std::endl;

	return 0;
}
