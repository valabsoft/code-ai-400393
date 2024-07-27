#include <mrcv/mrcv.h>


int main()
{
    std::string patchBody="/home/oleg/kodII/mrcv/body.html"; // заменить на свой путь
    std::string outputfolder="/home/oleg/kodII/mrcv/images/"; // заменить на свой путь
    std::string nametemplate="tank"; // 
    int count =5;
    int trainsetpercentage = 70;
    int minwidth=300;
    int minheight=300;
    bool separatedataset = true;
    std::string text="машинки";
    
int rez=mrcv::getImagesFromYandex(patchBody, count, minwidth, minheight, nametemplate, outputfolder, separatedataset,trainsetpercentage);
//int rez=mrcv::getImagesFromYandex(text, count, minwidth, minheight, nametemplate, outputfolder, separatedataset,trainsetpercentage);
    return 0;
} 
