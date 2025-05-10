#include <iostream>
#include <mrcv/mrcv-yolov5.h>

int main(void)
{
    try
    {
        mrcv::YOLOv5GenerateConfig(mrcv::YOLOv5Model::YOLOv5s, "yolov5s-coco.yaml", 80);
        std::cout << "Configuration file generated successfully!" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    try
    {
        mrcv::YOLOv5GenerateHyperparameters(mrcv::YOLOv5Model::YOLOv5s, 640, 640, "yolov5s-hyp.yaml", 80);
        std::cout << "Configuration file generated successfully!" << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}
