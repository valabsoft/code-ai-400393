#include <mrcv/mrcv.h>
#include <mrcv/mrcv-vae.h>

int main()
{
    // Пути к файлам модели    
   auto currentPath = std::filesystem::current_path();
   std::filesystem::path path = currentPath / "files";
   
   std::filesystem::path imagesPath = path / "images";
   std::filesystem::path modelPath = path / "ship.onnx";
   std::filesystem::path classPath = path / "ship.names";
   std::filesystem::path resultPath = path / "result";
   
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
