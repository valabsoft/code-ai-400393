#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(augmentation_test, augmentation)
{
    std::vector<cv::Mat> inputImagesAugmetation(10);
    
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "augmentation";

    inputImagesAugmetation[0] = cv::imread((path / "img0.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[1] = cv::imread((path / "img1.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[2] = cv::imread((path / "img2.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[3] = cv::imread((path / "img3.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[4] = cv::imread((path / "img4.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[5] = cv::imread((path / "img5.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[6] = cv::imread((path / "img6.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[7] = cv::imread((path / "img7.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[8] = cv::imread((path / "img8.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[9] = cv::imread((path / "img9.jpg").u8string(), cv::IMREAD_COLOR);

    std::vector<cv::Mat> inputImagesCopy = inputImagesAugmetation;
    std::vector<cv::Mat> outputImagesAugmetation;

    std::vector<mrcv::AUGMENTATION_METHOD> augmetationMethod =
    {
        mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_90,
        mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL,
        mrcv::AUGMENTATION_METHOD::FLIP_VERTICAL,
        mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_45,
        mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_315,
        mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_270,
        mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL_AND_VERTICAL,
    };

    int exitcode = mrcv::augmetation(inputImagesAugmetation, outputImagesAugmetation, augmetationMethod);
    
    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}

TEST(augmentation_test, batchAugmentation)
{

    std::vector<cv::Mat> inputImagesAugmetation(10);
    auto currentPath = std::filesystem::current_path();
    std::filesystem::path path = currentPath / "data" / "augmentation";

    inputImagesAugmetation[0] = cv::imread((path / "img0.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[1] = cv::imread((path / "img1.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[2] = cv::imread((path / "img2.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[3] = cv::imread((path / "img3.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[4] = cv::imread((path / "img4.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[5] = cv::imread((path / "img5.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[6] = cv::imread((path / "img6.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[7] = cv::imread((path / "img7.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[8] = cv::imread((path / "img8.jpg").u8string(), cv::IMREAD_COLOR);
    inputImagesAugmetation[9] = cv::imread((path / "img9.jpg").u8string(), cv::IMREAD_COLOR);

    std::vector<cv::Mat> inputImagesCopy = inputImagesAugmetation;
    std::vector<cv::Mat> outputImagesAugmetation;

    mrcv::BatchAugmentationConfig config;
    config.keep_original = true;
    config.total_output_count = 100;
    config.random_seed = 42;

    config.method_weights = {
        {mrcv::AUGMENTATION_METHOD::FLIP_HORIZONTAL, 0.2},
        {mrcv::AUGMENTATION_METHOD::ROTATE_IMAGE_90, 0.2},
        {mrcv::AUGMENTATION_METHOD::BRIGHTNESS_CONTRAST_ADJUST, 0.3},
        {mrcv::AUGMENTATION_METHOD::PERSPECTIVE_WARP, 0.2},
        {mrcv::AUGMENTATION_METHOD::COLOR_JITTER, 0.1},
    };

    std::vector<cv::Mat> batchOutput;
    int exitcode = mrcv::batchAugmentation(inputImagesCopy, batchOutput, config, (path / "batchOutput").u8string());
    
    EXPECT_EQ(exitcode, EXIT_SUCCESS);
}