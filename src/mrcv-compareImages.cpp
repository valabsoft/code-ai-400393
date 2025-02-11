#include <mrcv/mrcv.h>

namespace mrcv
{
	/**
	* @brief функция сравнения изображения.
	* @param img1 - исходное фото 1, img1 - исходное фото 2, methodCompare - метод сравнения.
	* @return - различия фотографий в процентном соотношении.
	*/
	double compareImages(cv::Mat img1,cv::Mat img2,bool methodCompare)
	{
	    if (methodCompare)
	    {
		    cv::Mat hsv1, hsv2;
		    cvtColor(img1, hsv1, cv::COLOR_BGR2HSV);
		    cvtColor(img2, hsv2, cv::COLOR_BGR2HSV);
		    int hBins = 50, sBins = 60;
		    int histSize[] = {hBins, sBins};
		    float hRanges[] = {0, 180};
		    float sRanges[] = {0, 256};
		    const float* ranges[] = {hRanges, sRanges};
		    int channels[] = {0, 1};
		    cv::Mat hist1, hist2;
		    calcHist(&hsv1, 1, channels, cv::Mat(), hist1, 2, histSize, ranges, true, false);
		    normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

		    calcHist(&hsv2, 1, channels, cv::Mat(), hist2, 2, histSize, ranges, true, false);
		    normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
		    return compareHist(hist1, hist2, cv::HISTCMP_CORREL);
	    }
	    else
	    {
		double errorL2 = norm(img1, img2, cv::NORM_L2);
		return 1 - errorL2 / (img1.rows  * img1.cols);
	    }
	    return 0,0;
	}
}
	
	
	
