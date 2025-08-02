#include <mrcv/mrcv.h>

int main()
{
    auto currentPath = std::filesystem::current_path();

    std::filesystem::path path = currentPath / "files";
    
    std::filesystem::path weightsPath = path / "weights" / "resnet34.pt";
    std::filesystem::path dataPath = path / "images";
    std::filesystem::path savePath = path / "weights" / "segmentor.pt";

    mrcv::Segmentor segmentor;
    segmentor.Initialize(512, 320, { "background","ship" }, "resnet34", weightsPath.u8string());
    segmentor.Train(0.0003, 20, 4, dataPath.u8string(), ".jpg", savePath.u8string());

    return 0;
}
