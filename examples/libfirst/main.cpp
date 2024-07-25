#include <mrcv/mrcv.h>


int main()
{
    std::string patchBody="/home/oleg/kodII/mrcv/body.html"; // заменить на свой путь
    std::string patchFoto="/home/oleg/kodII/mrcv/images/"; // заменить на свой путь
    std::string text= mrcv::readFile(patchBody); // читаем скачанный файл
    std::vector< std::string > arrUrl;
    arrUrl =mrcv::urlFind(text); //поиск в файле url

    int rez=mrcv::saveFile("url.txt",arrUrl); // запись в файл список url

    mrcv::downloadFoto(arrUrl,patchFoto); // качаем все фото из списка

    mrcv::delSmal(patchFoto,300,300); // удаляем мелкие фото (ширина, высота)
    mrcv::copyFile(patchFoto,patchFoto,70); // раскладываем файлы по папкам в процентном отношении 70%

    return 0;
} 
