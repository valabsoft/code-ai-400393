#include <mrcv/mrcv.h>

int main()
{
    std::filesystem::path imagesFile("files//");
    auto currentPath = std::filesystem::current_path();
    auto imagesPath = currentPath / imagesFile;

    std::string queryString = "sunken+ships"; // строка запроса, для двойного запроса использовать "+" без пробелов  Ex: sunken+ships
    int minWidth = 300;
    int minHeight = 300;
    std::string templateName = "ships"; // шаблон файла     
    std::string outputFolder = imagesPath.u8string() + "images/"; // путь к изображениям
    bool separateDataset = true;
    unsigned int trainsetPercentage = 70;
    unsigned int countFoto = 30;
    bool money = false;

    std::string key = "";
    std::string secretKey = "";

    int result = mrcv::getImagesFromYandex(queryString, minWidth, minHeight, templateName, outputFolder, separateDataset, trainsetPercentage, countFoto, money, key, secretKey);

    std::cout << "Результат работы функции " << result << '\n';

    return 0;
} 
