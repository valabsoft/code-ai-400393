#include <mrcv/mrcv.h>

int main()
{
	std::filesystem::path imageFile("files\\");

	auto currentPath = std::filesystem::current_path();

	auto imagePath = currentPath / imageFile;

	cv::Mat img1 = cv::imread(imagePath.u8string() + "1.png");
	cv::Mat img2 = cv::imread(imagePath.u8string() + "2.png");

	std::cout << "Similarity: " << mrcv::compareImages(img1, img2, 1) << std::endl;
	return 0;
}
