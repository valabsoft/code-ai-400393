#include <mrcv/mrcv.h>

int main()
{
 cv::Mat img1 = cv::imread("/home/oleg/install/code-ai-400393-developer/examples/compareImages/files/1.png");
 cv::Mat img2 = cv::imread("/home/oleg/install/code-ai-400393-developer/examples/compareImages/files/2.png");
 
 std::cout << "Similarity: " << mrcv::compareImages(img1,img2,1) << std::endl;
 return 0;


}
