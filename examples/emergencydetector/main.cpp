#include"mrcv/mrcv.h"

int main()
{
	std::filesystem::path imageFile("files\\pipes\\test\\images\\burst_augment_56_blur45_jpg.rf.714ba15fef8cf1f9ebcc19bbdb07dd2a.jpg");
	std::filesystem::path vocClassesFile("files\\pipes\\voc_classes.txt");
	std::filesystem::path weightFile("files\\emergency_detector.pt");

	auto currentPath = std::filesystem::current_path();

	auto imagePath = currentPath / imageFile;
	auto vocClassesPath = currentPath / vocClassesFile;
	auto weightPath = currentPath / weightFile;

	cv::Mat image = cv::imread(imagePath.u8string());

	mrcv::Detector detector;
	// Инициализация структуры модели
	detector.Initialize(0, 416, 416, vocClassesPath.string());
	// Загрузка весов обученной модели
	detector.LoadWeight(weightPath.u8string());
	// Детекция объектов на изображении
	detector.Predict(image, true, 0.1);

	return 0;
}