#include <mrcv/mrcv.h>

#include <iostream>

int main(int, char*[])
{
    mrcv::MRCVPoint point;
    point.setX(10);
    point.setY(20);
    std::cout << "P" << point.gerCoordinates() << std::endl;
    return 0;
}