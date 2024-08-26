#include <mrcv/mrcv.h>

int main()
{
    mrcv::Segmentor segmentor;
    segmentor.Initialize(-1, 512, 320, {"background","ship"}, "resnet34", "../weights/resnet34.pt");
    segmentor.Train(0.0003, 20, 4, "../images", ".jpg", "../weights/segmentor.pt");

    return 0;
}
