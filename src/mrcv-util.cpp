#include <mrcv/mrcv.h>

namespace mrcv
{
    std::string openCVInfo()
    {
        return cv::getBuildInformation().c_str();
    }
}