#include <mrcv/mrcv.h>

int main()
{
    int morph_size = 1;

    cv::Mat image = cv::imread("/home/oleg/install/code-ai-400393-developer/examples/morphologyImage/files/opening.png", cv::IMREAD_GRAYSCALE);
    //image = imread("D:\\QtProect\\closing.png", IMREAD_GRAYSCALE);
   // image = imread("D:\\QtProect\\gradient.png", IMREAD_GRAYSCALE);
   // Mat image = imread("files/j.png", IMREAD_GRAYSCALE);

    std::string out = "/home/oleg/install/code-ai-400393-developer/examples/morphologyImage/files/out.png";
    // Check if the image is created successfully or not
    if (!image.data) {
        std::cout << "Could not open or"
             << " find the image\n";
        return 0;
    }
    int result = mrcv::morphologyImage(image,out,mrcv::METOD_MORF::OPEN,morph_size);

}
