#include <mrcv/mrcv.h>
#include <gtest/gtest.h>

TEST(imgpreprocessing_test, imgpreprocessing)
{
	auto currentPath = std::filesystem::current_path();
	std::filesystem::path path = currentPath / "data" / "imgpreprocessing";

	// Загрузка исходных данных
	cv::Mat imageIn;
	cv::Mat imageOut;

	imageIn = cv::imread((path / "test-img.jfif").u8string(), cv::IMREAD_COLOR);
	imageOut = imageIn.clone();

	// Набор методов предобработки
	std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> metodImagePerProcessinBrightnessContrast =
	{
		mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_01_MEDIAN_FILTER,
		mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_10_LAB_CLAHE,
		mrcv::METOD_IMAGE_PERPROCESSIN::SHARPENING_02,
		mrcv::METOD_IMAGE_PERPROCESSIN::BRIGHTNESS_LEVEL_DOWN,
		mrcv::METOD_IMAGE_PERPROCESSIN::NONE,
		mrcv::METOD_IMAGE_PERPROCESSIN::CORRECTION_GEOMETRIC_DEFORMATION,
	};
	// Запуск предобработки
	int exitcode = mrcv::preprocessingImage(imageOut, metodImagePerProcessinBrightnessContrast, (path / "camera-parameters.xml").u8string());

	EXPECT_EQ(exitcode, EXIT_SUCCESS);
}