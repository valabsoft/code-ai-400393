#include <mrcv/mrcv.h>

int main() {
	std::filesystem::path imagesFile("files//");
	auto currentPath = std::filesystem::current_path();
	auto imagesPath = currentPath / imagesFile;

	auto imagePath = imagesPath.u8string() + "../images/test/43.jpg";
	cv::Mat image = cv::imread(imagePath);

	mrcv::Segmentor segmentor;

	segmentor.Initialize(-1, 512, 320, { "background","ship" }, "resnet34", imagesPath.u8string() + "../weights/resnet34.pt");
	segmentor.LoadWeight(imagesPath.u8string() + "../weights/segmentor.pt");
	segmentor.Predict(image, "ship");

	return 0;
}
