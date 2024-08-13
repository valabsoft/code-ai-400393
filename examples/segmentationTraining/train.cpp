#include <iostream>
#include <mrcv/mrcv.h>

int main()
{
    //int gpu_num = torch::getNumGPUs();
   // std::cout << gpu_num;
    mrcv::Segmentor segmentor;
    segmentor.Initialize(-1, 512, 320, {"background","ship"}, "resnet34", "../weights/resnet34.pt");
    segmentor.Train(0.0003, 20, 4, "../images", ".jpg", "../weights/segmentor.pt");

    return 0;
}
