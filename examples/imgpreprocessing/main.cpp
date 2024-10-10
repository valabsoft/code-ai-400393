#include <mrcv/mrcv.h>

int main()
{
	mrcv::writeLog(" ");
	mrcv::writeLog(" === НОВЫЙ ЗАПУСК === ");
	///////////////////////////////////////////////////////////////////////////
	// Загрузка изображения
	///////////////////////////////////////////////////////////////////////////
	cv::Mat imageIn;
	cv::Mat imageOut;
	cv::String imageInputFilePath = "./files/img02.jfif";
	imageIn = cv::imread(imageInputFilePath, cv::IMREAD_COLOR);
	imageOut = imageIn.clone();
	mrcv::writeLog("    загружено изображение: " + imageInputFilePath + " :: "
		+ std::to_string(imageIn.size().width) + "x"
		+ std::to_string(imageIn.size().height) + "x"
		+ std::to_string(imageIn.channels()));

	///////////////////////////////////////////////////////////////////////////
	// Функция предварительной обработки изображений (автоматическая коррекция контраста и яркости, резкости)
	///////////////////////////////////////////////////////////////////////////
	//    std::vector<mrcv::metodImagePerProcessin> metodImagePerProcessinBrightnessContrast =
	//        { mrcv::metodImagePerProcessin::NOISE_FILTERING_01_MEDIAN_FILTER,
	//         mrcv::metodImagePerProcessin::BALANCE_CONTRAST_02_CLAHE,
	//         mrcv::metodImagePerProcessin::SHARPENING_02,
	//         mrcv::metodImagePerProcessin::BRIGHTNESS_LEVEL_DOWN,
	//         mrcv::metodImagePerProcessin::BALANCE_CONTRAST_04_COLOR_BALANCING,
	//         mrcv::metodImagePerProcessin::NONE,
	//         mrcv::metodImagePerProcessin::CORRECTION_GEOMETRIC_DEFORMATION,
	//         };

	//        std::vector<mrcv::metodImagePerProcessin> metodImagePerProcessinBrightnessContrast =
	//        { mrcv::metodImagePerProcessin::NOISE_FILTERING_01_MEDIAN_FILTER,
	//         mrcv::metodImagePerProcessin::BALANCE_CONTRAST_01_YCBCR_EQUALIZEHIST,
	//         mrcv::metodImagePerProcessin::BRIGHTNESS_LEVEL_UP,
	//         mrcv::metodImagePerProcessin::SHARPENING_02,
	//         mrcv::metodImagePerProcessin::BRIGHTNESS_LEVEL_DOWN,
	//         mrcv::metodImagePerProcessin::BRIGHTNESS_LEVEL_DOWN,
	//         mrcv::metodImagePerProcessin::NONE,
	//         mrcv::metodImagePerProcessin::CORRECTION_GEOMETRIC_DEFORMATION,
	//         };


	///////////////////////////////////////////////////////////////////////////
	// Предобработка изображения
	///////////////////////////////////////////////////////////////////////////
	// выбор методов предобработки
	std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> metodImagePerProcessinBrightnessContrast =
	{ mrcv::METOD_IMAGE_PERPROCESSIN::NOISE_FILTERING_01_MEDIAN_FILTER,
	 mrcv::METOD_IMAGE_PERPROCESSIN::BALANCE_CONTRAST_10_LAB_CLAHE,
	 mrcv::METOD_IMAGE_PERPROCESSIN::SHARPENING_02,
	 mrcv::METOD_IMAGE_PERPROCESSIN::BRIGHTNESS_LEVEL_DOWN,
	 mrcv::METOD_IMAGE_PERPROCESSIN::NONE,
	 mrcv::METOD_IMAGE_PERPROCESSIN::CORRECTION_GEOMETRIC_DEFORMATION,
	};
	int state = mrcv::preprocessingImage(imageOut, metodImagePerProcessinBrightnessContrast, "./files/camera-parameters.xml");
	if (state == 0)
	{
		mrcv::writeLog(" Предобработка изображения завершена (успешно)");
	}
	else
	{
		mrcv::writeLog(" preprocessingImage, state = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
	}

	///////////////////////////////////////////////////////////////////////////
	// Сохранение в файл
	///////////////////////////////////////////////////////////////////////////
	cv::String imageOutputFilePath = "./files/outImages/test.png";
	cv::imwrite(imageOutputFilePath, imageOut);
	mrcv::writeLog("\t результат преодобработки сохранён: " + imageInputFilePath);
	///////////////////////////////////////////////////////////////////////////
	// Вывод результата на экран
	///////////////////////////////////////////////////////////////////////////
	// Уменьшение чтобы поместилось на экран
	double CoefShowWindow = 0.5;
	cv::resize(imageIn, imageIn, cv::Size(double(imageIn.cols * CoefShowWindow), double(imageIn.rows * CoefShowWindow)), 0, 0, cv::INTER_LINEAR);
	cv::resize(imageOut, imageOut, cv::Size(double(imageOut.cols * CoefShowWindow), double(imageOut.rows * CoefShowWindow)), 0, 0, cv::INTER_LINEAR);
	cv::namedWindow("imageIn", cv::WINDOW_AUTOSIZE);
	imshow("imageIn", imageIn);
	cv::namedWindow("imageOut", cv::WINDOW_AUTOSIZE);
	imshow("imageOut", imageOut);
	cv::waitKey(0);

	return 0;
}
