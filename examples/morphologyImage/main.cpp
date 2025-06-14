#include <mrcv/mrcv.h>

int main()
{
    std::filesystem::path imageFile("files\\");

    auto currentPath = std::filesystem::current_path();

    auto imagePath = currentPath / imageFile;

    int morph_size = 1;

    cv::Mat image = cv::imread(imagePath.u8string() + "opening.png", cv::IMREAD_GRAYSCALE);
    std::string out = imagePath.u8string() + "out.png";
    // Check if the image is created successfully or not
    if (!image.data) {
        std::cout << "Could not open or"
            << " find the image\n";
        return 0;
    }
    int result = mrcv::morphologyImage(image, out, mrcv::METOD_MORF::OPEN, morph_size);

}
