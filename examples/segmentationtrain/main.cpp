#include <mrcv/mrcv.h>

int main()
{
    std::filesystem::path imagesFile("files//");
    auto currentPath = std::filesystem::current_path();
    auto imagesPath = currentPath / imagesFile;

    mrcv::Segmentor segmentor;
    segmentor.Initialize(-1, 512, 320, { "background","ship" }, "resnet34", imagesPath.u8string() + "../weights/resnet34.pt");
    segmentor.Train(0.0003, 20, 4, imagesPath.u8string() + "../images", ".jpg", imagesPath.u8string() + "../weights/segmentor.pt");

    return 0;
}
