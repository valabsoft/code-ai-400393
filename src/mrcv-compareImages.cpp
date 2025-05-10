#include <mrcv/mrcv.h>

namespace mrcv
{
	/**
	* @brief функция сравнения изображения.
	* @param img1 - исходное изображение 1
    * @param img2 - исходное исходное изображение 2
    * @param imethodCompare - метод сравнения.
	* @return - различия фотографий в процентном соотношении.
	*/
	double compareImages(cv::Mat img1,cv::Mat img2, bool methodCompare)
	{
		if (methodCompare)
		{
			cv::Mat hsv1, hsv2;
			cvtColor(img1, hsv1, cv::COLOR_BGR2HSV);
			cvtColor(img2, hsv2, cv::COLOR_BGR2HSV);
			int hBins = 50, sBins = 60;
			int histSize[] = { hBins, sBins };
			float hRanges[] = { 0, 180 };
			float sRanges[] = { 0, 256 };
			const float* ranges[] = { hRanges, sRanges };
			int channels[] = { 0, 1 };
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
			return 1 - errorL2 / (img1.rows * img1.cols);
		}
		return 0;
	}
#ifdef MRCV_CUDA_ENABLED 
    double compareImagesCuda(cv::Mat img1, cv::Mat img2, bool methodCompare) {
        try {
            // Проверка корректности изображений
            if (img1.empty() || img2.empty()) {
                writeLog("Одно из изображений пустое!", mrcv::LOGTYPE::ERROR);
                return 0.0;
            }
            if (img1.size() != img2.size()) {
                writeLog("Изображения имеют разные размеры!", mrcv::LOGTYPE::ERROR);
                return 0.0;
            }
            if (img1.type() != img2.type()) {
                writeLog("Изображения имеют разные типы!", mrcv::LOGTYPE::ERROR);
                return 0.0;
            }

            // Загрузка изображений на GPU
            writeLog("Загрузка img1 на GPU...", mrcv::LOGTYPE::INFO);
            cv::cuda::GpuMat d_img1, d_img2;
            d_img1.upload(img1);
            writeLog("Загрузка img2 на GPU...", mrcv::LOGTYPE::INFO);
            d_img2.upload(img2);

            if (methodCompare) {
                // Преобразование в цветовое пространство HSV на GPU
                writeLog("Преобразование в HSV...", mrcv::LOGTYPE::INFO);
                cv::cuda::GpuMat d_hsv1, d_hsv2;
                cv::cuda::cvtColor(d_img1, d_hsv1, cv::COLOR_BGR2HSV);
                cv::cuda::cvtColor(d_img2, d_hsv2, cv::COLOR_BGR2HSV);

                // Параметры гистограммы
                int hBins = 50;
                float hRanges[] = { 0, 180 };
                const float* ranges[] = { hRanges };

                // Создание буферов для гистограмм CUDA
                writeLog("Создание буферов для гистограмм...", mrcv::LOGTYPE::INFO);
                cv::cuda::GpuMat d_hist1, d_hist2;
                std::vector<cv::cuda::GpuMat> d_hsv1_channels, d_hsv2_channels;

                // Разделение каналов HSV
                writeLog("Разделение каналов HSV...", mrcv::LOGTYPE::INFO);
                cv::cuda::split(d_hsv1, d_hsv1_channels);
                cv::cuda::split(d_hsv2, d_hsv2_channels);

                // Вычисление гистограмм для канала H
                writeLog("Вычисление гистограмм...", mrcv::LOGTYPE::INFO);
                cv::cuda::calcHist(d_hsv1_channels[0], d_hist1);
                cv::cuda::calcHist(d_hsv2_channels[0], d_hist2);

                // Нормализация гистограмм
                writeLog("Нормализация гистограмм...", mrcv::LOGTYPE::INFO);
                cv::cuda::GpuMat d_hist1_norm, d_hist2_norm;
                cv::cuda::normalize(d_hist1, d_hist1_norm, 0, 1, cv::NORM_MINMAX, CV_32F);
                cv::cuda::normalize(d_hist2, d_hist2_norm, 0, 1, cv::NORM_MINMAX, CV_32F);

                // Выгрузка гистограмм на CPU
                writeLog("Выгрузка гистограмм на CPU...", mrcv::LOGTYPE::INFO);
                cv::Mat h_hist1, h_hist2;
                d_hist1_norm.download(h_hist1);
                d_hist2_norm.download(h_hist2);

                // Проверка типов гистограмм
                writeLog("Проверка типов гистограмм...", mrcv::LOGTYPE::INFO);
                if (h_hist1.type() != CV_32F || h_hist2.type() != CV_32F) {
                    writeLog("Преобразование гистограмм в CV_32F...", mrcv::LOGTYPE::INFO);
                    cv::Mat h_hist1_float, h_hist2_float;
                    h_hist1.convertTo(h_hist1_float, CV_32F);
                    h_hist2.convertTo(h_hist2_float, CV_32F);
                    h_hist1 = h_hist1_float;
                    h_hist2 = h_hist2_float;
                }

                // Сравнение гистограмм
                writeLog("Сравнение гистограмм...", mrcv::LOGTYPE::INFO);
                double result = cv::compareHist(h_hist1, h_hist2, cv::HISTCMP_CORREL);
                writeLog("Результат корреляции: " + std::to_string(result), mrcv::LOGTYPE::INFO);
                return result;
            }
            else {
                // Вычисление L2-нормы разности на GPU
                writeLog("Вычисление разности изображений...", mrcv::LOGTYPE::INFO);
                cv::cuda::GpuMat d_diff;
                cv::cuda::subtract(d_img1, d_img2, d_diff);

                writeLog("Вычисление L2-нормы...", mrcv::LOGTYPE::INFO);
                double errorL2 = cv::cuda::norm(d_diff, cv::NORM_L2);

                // Нормализация по размеру изображения
                double result = 1.0 - errorL2 / (img1.rows * img1.cols);
                writeLog("Результат L2-нормы: " + std::to_string(result), mrcv::LOGTYPE::INFO);
                return result;
            }
        }
        catch (const cv::Exception& e) {
            writeLog("OpenCV ошибка: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            return 0.0;
        }
        catch (const std::exception& e) {
            writeLog("Стандартная ошибка: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            return 0.0;
        }
        catch (...) {
            writeLog("Неизвестная ошибка", mrcv::LOGTYPE::ERROR);
            return 0.0;
        }

        return 0.0;
    }
#endif
}
	
	
	
