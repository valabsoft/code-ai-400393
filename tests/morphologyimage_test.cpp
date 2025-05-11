#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(morphologyimage_test, morphologyimage)
{
	auto currentPath = std::filesystem::current_path();
	std::filesystem::path path = currentPath / "data" / "morphologyimage";
	
    cv::Mat imgInput = cv::imread((path / "img01.png").u8string(), cv::IMREAD_GRAYSCALE);
    std::string imgOutput = (path / "img02.png").u8string();
    
    int exitcode = mrcv::morphologyImage(imgInput, imgOutput, mrcv::METOD_MORF::OPEN, 1);

	EXPECT_EQ(exitcode, EXIT_SUCCESS);
}