#include <mrcv/mrcv.h>

int main() {

    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "files";
    
    std::filesystem::path weightsPath = path / "weights" / "resnet34.pt";
    std::filesystem::path segmentorPath = path / "weights" / "segmentor.pt";
    std::filesystem::path imagePath = path / "images" / "source.jpg";

	cv::Mat image = cv::imread(imagePath.u8string());

	mrcv::Segmentor segmentor;

	segmentor.Initialize(512, 320, { "background","ship" }, "resnet34", weightsPath.u8string());
	segmentor.LoadWeight(segmentorPath.u8string());
	segmentor.Predict(image, "ship", true);

	return 0;
}
