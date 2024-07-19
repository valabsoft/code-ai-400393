#include <mrcv/mrcv.h>

#include <iostream>

int main(int, char*[])
{
    auto sum = mrcv::add(3, 2);
    std::cout << sum << std::endl;
    return 0;
}