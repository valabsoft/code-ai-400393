#include <mrcv/mrcv.h>

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "usage: mrcv-imread <cv::Mat image>, <std::string pathToImage>, <bool showImage>" << std::endl;
        return -1;
    }
    cv::Mat image;
    mrcv::readImage(image, argv[1], true);
    return 0;
}