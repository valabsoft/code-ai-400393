#include <mrcv/mrcv.h>

#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "usage: mrcv-imread <path_to_image>" << std::endl;
        return -1;
    }

    mrcv::imread(argv[1]);
    return 0;
}