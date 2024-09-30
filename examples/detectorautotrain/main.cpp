#include"mrcv/mrcv.h"

int main()
{
	std::string rootPath = "D:/Games/mrcv/examples/detectorautotrain/";
	cv::Mat image = cv::imread(rootPath + "underwater/val/images/0aefb953c94f5389_jpg.rf.168cd98b455de9e9d9fdeec9849d2459.jpg");

	mrcv::Detector detector;
	detector.Initialize(0, 416, 416, rootPath + "underwater/voc_classes.txt");
	detector.AutoTrain(rootPath + "underwater", ".jpg", { 10, 15, 30}, { 4, 8 }, {0.001, 1.0E-4F}, rootPath + "underwater_autodetector.pt", rootPath + "yolo4_tiny.pt");

	return 0;
}