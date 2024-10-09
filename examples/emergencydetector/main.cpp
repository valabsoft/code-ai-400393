#include"mrcv/mrcv.h"

int main()
{
	std::string rootPath = "D:/Games/mrcv/examples/emergencydetector/";
	cv::Mat image = cv::imread(rootPath + "pipes/test/images/burst_augment_56_blur45_jpg.rf.714ba15fef8cf1f9ebcc19bbdb07dd2a.jpg");
	//cv::Mat image = cv::imread(rootPath + "newpipes/test/images/featured-image-burst-pipe-jpeg_jpg.rf.bccc93feba9ee9f6e2a4822174fbeae9.jpg");

	mrcv::Detector detector;
	// ������������� ��������� ������
	detector.Initialize(0, 416, 416, rootPath + "pipes/voc_classes.txt");
	//detector.Train(rootPath + "pipes/", ".jpg", 15, 4, 1.0E-4F, rootPath + "emergency_detector.pt", rootPath + "yolo4_tiny.pt");
	
	// �������� ����� ��������� ������
	detector.LoadWeight(rootPath + "pipes/emergency_detector.pt");
	// �������� �������� �� �����������
	detector.Predict(image, true, 0.1);

	return 0;
}