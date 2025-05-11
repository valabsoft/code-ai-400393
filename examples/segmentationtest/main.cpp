#include <mrcv/mrcv.h>

int main() {

	std::filesystem::path weightsFile("file\\weights\\resnet34.pt");
    std::filesystem::path segmentorFile("file\\weights\\segmentor.pt");
    std::filesystem::path imageFile("file\\images\\43.jpg");

    auto currentPath = std::filesystem::current_path();

    auto weightsPath = currentPath / weightsFile;
    auto segmentorPath = currentPath / segmentorFile;
    auto imagePath = currentPath / imageFile;

	cv::Mat image = cv::imread(imagePath.u8string());

	mrcv::Segmentor segmentor;

	segmentor.Initialize(512, 320, { "background","ship" }, "resnet34", weightsPath.u8string());
	segmentor.LoadWeight(segmentorPath.u8string());
	segmentor.Predict(image, "ship");

	return 0;
}
