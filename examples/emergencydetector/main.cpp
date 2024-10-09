#include"mrcv/mrcv.h"

int main()
{
	// TODO: VA 09-10-2024: Исправить работу с файлами!!!
	// Все файлы, являющиеся ресурсами должны быть помещены в папку files (см. пример disparitymap)
	
	// // Загрузка тестовых изображений
	// std::filesystem::path fileImageLeft("files\\example_left.jpg");
	// std::filesystem::path fileImageRight("files\\example_right.jpg");
	// auto currentPath = std::filesystem::current_path();
	// // Получение полного пути
	// auto pathImageLeft = currentPath / fileImageLeft;
	// auto pathImageRight = currentPath / fileImageRight;

	// Имена файлов тестовых изображений изменить
	// После исправления лишние комментарии убрать

	std::string rootPath = "D:/Games/mrcv/examples/emergencydetector/";
	cv::Mat image = cv::imread(rootPath + "pipes/test/images/burst_augment_56_blur45_jpg.rf.714ba15fef8cf1f9ebcc19bbdb07dd2a.jpg");
	//cv::Mat image = cv::imread(rootPath + "newpipes/test/images/featured-image-burst-pipe-jpeg_jpg.rf.bccc93feba9ee9f6e2a4822174fbeae9.jpg");

	mrcv::Detector detector;
	// Инициализация структуры модели
	detector.Initialize(0, 416, 416, rootPath + "pipes/voc_classes.txt");
	//detector.Train(rootPath + "pipes/", ".jpg", 15, 4, 1.0E-4F, rootPath + "emergency_detector.pt", rootPath + "yolo4_tiny.pt");
	
	// Загрузка весов обученной модели
	detector.LoadWeight(rootPath + "pipes/emergency_detector.pt");
	// Детекция объектов на изображении
	detector.Predict(image, true, 0.1);

	return 0;
}