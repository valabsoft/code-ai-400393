#include <mrcv/mrcv.h>

#include <iostream>

int main(int, char*[])
{
    std::cout << mrcv::openCVInfo() << std::endl;
    return 0;
}