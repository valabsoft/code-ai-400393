#include <mrcv/mrcv.h>

int main()
{
	auto currentPath = std::filesystem::current_path();

	std::filesystem::path path = currentPath / "files" / "";
	std::filesystem::path imagePath = path;

	cv::Mat img1 = cv::imread(imagePath.u8string() + "img1.png");
	cv::Mat img2 = cv::imread(imagePath.u8string() + "img2.png");

	std::cout << "Similarity: " << mrcv::compareImages(img1, img2, 1) << std::endl;
	return 0;
}
