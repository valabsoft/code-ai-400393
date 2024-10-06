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

   int height = 640;
   int width = 640;

   cv::Mat genImage = mrcv::neuralNetworkAugmentationAsMat(imagePath.u8string(), height, width, 800, 8, 800, 32, 3E-4);

   cv::Mat colorGenImage;
   cv::cvtColor(genImage, colorGenImage, cv::COLOR_GRAY2BGR);

   cv::imshow("", colorGenImage);
   cv::waitKey(0);

   cv::imwrite(resultPath.u8string() + "/generated.jpg", colorGenImage);

   mrcv::semiAutomaticLabeler(resultPath.u8string() + "/generated.jpg", height, width, resultPath.u8string(), modelPath.u8string(), classPath.u8string());
    
   return EXIT_SUCCESS;
}