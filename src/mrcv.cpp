#include <mrcv/mrcv.h>

namespace mrcv
{
    int add(int a, int b)
    {
        return a + b;
    }

    int imread(std::string pathtoimage)
    {
        cv::Mat img = cv::imread(pathtoimage, cv::IMREAD_COLOR);

        if (img.empty())
        {
            std::cout << "Could not read the image: " << pathtoimage << std::endl;
            return 1;
        }

        cv::imshow(pathtoimage, img);
        cv::waitKey(0);

        return 0;
    }


}
