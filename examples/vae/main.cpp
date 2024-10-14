#include <mrcv/mrcv.h>
#include <mrcv/mrcv-vae.h>

int main()
{
    // Пути к файлам модели
   std::filesystem::path imagesFile("files\\images");
   std::filesystem::path modelFile("files\\ship.onnx");
   std::filesystem::path classFile("files\\ship.names");
   std::filesystem::path resultFile("files\\result");
    
   auto currentPath = std::filesystem::current_path();

   auto imagesPath = currentPath / imagesFile;
   auto modelPath = currentPath / modelFile;
   auto classPath = currentPath / classFile;
   auto resultPath = currentPath / resultFile;
   
   // Размер изображений в датасете
   int height = 640;
   int width = 640;
   
   // Генерация изображения с помощью vae 
   cv::Mat genImage = mrcv::neuralNetworkAugmentationAsMat(imagesPath.u8string(), height, width, 200, 2, 1200, 16, 3E-4);

   // Перевод изображения в цветной формат
   cv::Mat colorGenImage;
   cv::cvtColor(genImage, colorGenImage, cv::COLOR_GRAY2BGR);

   // Сохранение сгенерированного изображения
   cv::imwrite(resultPath.u8string() + "/generated.jpg", colorGenImage);

   // Полуавтоматическая разметка 
   mrcv::semiAutomaticLabeler(colorGenImage, 640, 640, resultPath.u8string(), modelPath.u8string(), classPath.u8string());
    
   return EXIT_SUCCESS;
}