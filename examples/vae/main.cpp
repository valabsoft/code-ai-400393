#include <mrcv/mrcv.h>
#include <mrcv/mrcv-vae.h>

int main()
{
    // Пути к файлам модели
   std::filesystem::path imageFile("files\\images");
   std::filesystem::path modelFile("files\\ship.onnx");
   std::filesystem::path classFile("files\\ship.names");
   std::filesystem::path resultFile("files\\result");
    

    auto currentPath = std::filesystem::current_path();

    auto imagePath = currentPath / imageFile;
    auto modelPath = currentPath / modelFile;
    auto classPath = currentPath / classFile;
    auto resultPath = currentPath / resultFile;

    std::string imagePathSrt{ imagePath.u8string() };
    std::string modelPathStr{ modelPath.u8string() };
    std::string classPathStr{ classPath.u8string() };
    std::string resultPathStr{ resultPath.u8string() };

    int height = 640;
    int width = 640;

    cv::Mat genImage = mrcv::neuralNetworkAugmentationAsMat(imagePathSrt, height, width, 200, 2, 1000, 32, 3E-4);

    mrcv::semiAutomaticLabeler(genImage, height, width, resultPathStr, modelPathStr, classPathStr);
    
    return EXIT_SUCCESS;
}