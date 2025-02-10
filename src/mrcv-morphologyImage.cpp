#include <mrcv/mrcv.h>

namespace mrcv
{
/**
* @brief функция морфологического открытия.
* @param image - исходное фото, out - путь для нового файла, element - размер преобразования.
* @return - результат работы функции.
*/
int openingMorphological(cv::Mat image,std::string out,cv::Mat element)
{

    cv::Mat output;
    // Opening
    cv::morphologyEx(image, output,
                 cv::MORPH_OPEN, element,
                 cv::Point(-1, -1), 2);
    imwrite(out,output);
    return 0;
}
/**
* @brief функция морфологического закрытия.
* @param image - исходное фото, out - путь для нового файла, element - размер преобразования.
* @return - результат работы функции.
*/
int closingMorphological(cv::Mat image,std::string out,cv::Mat element)
{
    cv::Mat output;
    // Closing
    cv::morphologyEx(image, output,
                 cv::MORPH_CLOSE, element,
                 cv::Point(-1, -1), 2);
    imwrite(out,output);
     return 0;
}
/**
* @brief функция морфологического градиента.
* @param image - исходное фото, out - путь для нового файла, element - размер преобразования.
* @return - результат работы функции.
*/
int gradientMorphological(cv::Mat image,std::string out,cv::Mat element)
{
    cv::Mat output;
    // Gradient
    cv::morphologyEx(image, output,
                 cv::MORPH_GRADIENT, element,
                 cv::Point(-1, -1), 1);
    imwrite(out,output);
    return 0;

}
/**
* @brief функция морфологической эрозии.
* @param image - исходное фото, out - путь для нового файла, element - размер преобразования.
* @return - результат работы функции.
*/
int erodeMorphological(cv::Mat image,std::string out,cv::Mat element)
{
    cv::Mat erod;
    // For Erosion
    cv::erode(image, erod, element,
          cv::Point(-1, -1), 1);
    imwrite(out,erod);
    return 0;
}
/**
* @brief функция морфологического расширения.
* @param image - исходное фото, out - путь для нового файла, element - размер преобразования.
* @return - результат работы функции.
*/
int dilationMorphological(cv::Mat image,std::string out,cv::Mat element)
{
    cv::Mat dill;
    // For Dilation
    cv::dilate(image, dill, element,
           cv::Point(-1, -1), 1);
    imwrite(out,dill);
    return 0;
}
/**
* @brief главная функция.
* @param image - исходное фото, out - путь для нового файла, metod - метод преобразования , morph_size - размер преобразования.
* @return - результат работы функции.
*/
int morphologyImage(cv::Mat image,std::string out,mrcv::METOD_MORF metod,int morph_size)
{
    cv::Mat element = getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2 * morph_size + 1,
             2 * morph_size + 1),
        cv::Point(morph_size,
              morph_size));
    switch (metod)
    {
    case mrcv::METOD_MORF::OPEN:
        openingMorphological(image, out, element);
        break;
    case mrcv::METOD_MORF::CLOSE:
        closingMorphological(image, out, element);
        break;
    case mrcv::METOD_MORF::DILAT:
        dilationMorphological(image, out, element);
        break;
    case mrcv::METOD_MORF::ERODE:
        erodeMorphological(image, out, element);
        break;
    case mrcv::METOD_MORF::GRADIENT:
        gradientMorphological(image, out, element);
        break;
    }
return 0;

}
	
	
	
}
