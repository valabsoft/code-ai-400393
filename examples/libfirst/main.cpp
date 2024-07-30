#include <mrcv/mrcv.h>

int main()
{
    std::string zapros="hares"; // строка запроса, для двойного запроса использовать + без пробелов  зайцы+ушастые
    int minWidth=300;
    int minHeight=300;
    std::string nameTemplate="hares"; // шаблон файла     
    std::string outputFolder="/home/oleg/kodII/mrcv/images/"; // путь к изображениям
    bool separateDataset = true;
    int trainsetPercentage = 70;
    unsigned int countFoto =30;
    bool money = false;
    
    std::string key="";
    std::string secretKey ="";    


    int rez=mrcv::getImagesFromYandex(zapros, minWidth, minHeight, nameTemplate,  outputFolder, separateDataset, trainsetPercentage,countFoto,money,key,secretKey);
   
	std::cout << "Результат работы функции " << rez << '\n';
    
    return 0;
} 
