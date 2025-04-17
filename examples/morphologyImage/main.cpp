#include <mrcv/mrcv.h>

int main()
{
    std::filesystem::path imagesFile("files//");
    auto currentPath = std::filesystem::current_path();
    auto imagesPath = currentPath / imagesFile;

    int morph_size = 1;

    cv::Mat image = cv::imread(imagesPath.u8string() + "/opening.png", cv::IMREAD_GRAYSCALE);
    std::string out = imagesPath.u8string() + "out.png";
    // Check if the image is created successfully or not
    if (!image.data) {
        std::cout << "Could not open or"
            << " find the image\n";
        return 0;
    }
    int result = mrcv::morphologyImage(image, out, mrcv::METOD_MORF::OPEN, morph_size);

}
