#include <mrcv/mrcv.h>

#include <iostream>

int main(int, char*[])
{
    std::cout << mrcv::getOpenCVBuildInformation() << std::endl;
    return 0;
}