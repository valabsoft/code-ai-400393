#include <string>
#include <mrcv/mrcv-yolov5.h>

int main(void)
{
    std::string inputDir = "images"; // Папка с изображениями
    std::string outputDir = "labels"; // Папка для сохранения разметки

    mrcv::YOLOv5LabelerProcessing(inputDir, outputDir);

    return 0;
}
