#include <mrcv/mrcv.h>

int main()
{
    std::filesystem::path weightsFile("file\\weights\\resnet34.pt");
    std::filesystem::path dataFile("file\\images");
    std::filesystem::path saveFile("file\\weights\\segmentor.pt");

    auto currentPath = std::filesystem::current_path();

    auto weightsPath = currentPath / weightsFile;
    auto dataPath = currentPath / dataFile;
    auto savePath = currentPath / saveFile;


    mrcv::Segmentor segmentor;
    segmentor.Initialize(512, 320, { "background","ship" }, "resnet34", weightsPath.u8string());
    segmentor.Train(0.0003, 20, 4, dataPath.u8string(), ".jpg", savePath.u8string());

    return 0;
}
