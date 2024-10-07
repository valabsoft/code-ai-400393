#include"mrcv/mrcv.h"

int main()
{
	std::string rootPath = "D:/Games/mrcv/examples/emergencydetector/";
	cv::Mat image = cv::imread(rootPath + "pipes/val/images/burst_augment_1_rotated45_jpg.rf.530bc03709cdda3bb8d222209a69f4f5.jpg");

	mrcv::Detector detector;
	detector.Initialize(0, 416, 416, rootPath + "pipes/voc_classes.txt");
	detector.LoadWeight(rootPath + "pipes/emergency_autodetector.pt");
	detector.Predict(image, true, 0.1);

	return 0;
}