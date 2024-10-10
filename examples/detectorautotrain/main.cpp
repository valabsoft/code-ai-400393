#include"mrcv/mrcv.h"

int main()
{
	std::string rootPath = "D:/Games/mrcv/examples/detectorautotrain/";

	mrcv::Detector detector;
	detector.Initialize(0, 416, 416, rootPath + "onwater/voc_classes.txt");
	detector.AutoTrain(rootPath + "onwater/", ".jpg", { 10, 15, 30}, { 4, 8 }, {0.001, 1.0E-4F}, rootPath + "yolo4_tiny.pt", rootPath + "onwater_autodetector.pt");
	return 0;
}