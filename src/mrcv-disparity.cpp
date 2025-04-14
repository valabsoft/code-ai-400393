#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>

namespace mrcv
{
	int disparityMap(cv::Mat &map, const cv::Mat& imageLeft, const cv::Mat& imageRight, int minDisparity, int numDisparities, int blockSize, double lambda, double sigma, DISPARITY_TYPE disparityType, int colorMap, bool saveToFile, bool showImages)
	{
        ///////////////////////////////////////////////////////////////////////
        // Преобразование в GrayScale
		cv::Mat imageLeftGS;
		cv::Mat imageRightGS;
        cv::Mat imageDisparityLeft;
        cv::Mat imageDisparityRight;

		cv::cvtColor(imageLeft, imageLeftGS, cv::COLOR_BGR2GRAY);
		cv::cvtColor(imageRight, imageRightGS, cv::COLOR_BGR2GRAY);

        ///////////////////////////////////////////////////////////////////////
        // Построение карты диспаратности (базовый метод)    
        cv::Ptr<cv::StereoBM> leftMatcher = cv::StereoBM::create(numDisparities, blockSize);
        cv::Ptr<cv::StereoMatcher> rightMatcher = cv::ximgproc::createRightMatcher(leftMatcher);

        leftMatcher->compute(imageLeftGS, imageRightGS, imageDisparityLeft);
        rightMatcher->compute(imageRightGS, imageLeftGS, imageDisparityRight);

        cv::Mat disparityLeft;
        cv::Mat disparityRight;

        imageDisparityLeft.convertTo(disparityLeft, CV_32F, 1.0);
        imageDisparityRight.convertTo(disparityRight, CV_32F, 1.0);

        disparityLeft = (disparityLeft / 16.0f - (float)minDisparity) / ((float)numDisparities);
        disparityRight = (disparityRight / 16.0f - (float)minDisparity) / ((float)numDisparities);

        if (showImages)
        {
            cv::namedWindow("Image Left", cv::WINDOW_AUTOSIZE);
            cv::imshow("Image Left", imageLeft);

            cv::namedWindow("Image Right", cv::WINDOW_AUTOSIZE);
            cv::imshow("Image Right", imageRight);
            
            cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
            cv::imshow("Disparity", disparityLeft);
        }

        ///////////////////////////////////////////////////////////////////////
        // Расцвечивание карты диспаратности
        cv::Mat disparityTMP;
        disparityLeft.convertTo(disparityTMP, CV_8UC3, 255.0);

        // Запоминаем результат преобразования
        cv::Mat disparity = disparityTMP.clone();

        cv::Mat heatmap;

        cv::applyColorMap(disparityTMP, heatmap, colorMap);
        if (showImages)
        {
            cv::namedWindow("Heatmap", cv::WINDOW_AUTOSIZE);
            cv::imshow("Heatmap", heatmap);
        }

        ///////////////////////////////////////////////////////////////////////
        // Построение карты диспаратности (фильтрация)
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> prtWLSFilter = cv::ximgproc::createDisparityWLSFilter(leftMatcher);

        prtWLSFilter->setLambda(lambda);
        prtWLSFilter->setSigmaColor(sigma);

        cv::Mat filteredDisparityT;

        // Создание фильтра сглаживания
        prtWLSFilter->filter(imageDisparityLeft, imageLeft, filteredDisparityT, imageDisparityRight);

        cv::Mat filteredDisparity;
        cv::ximgproc::getDisparityVis(filteredDisparityT, filteredDisparity, 1.0);
        if (showImages)
        {
            cv::namedWindow("Disparity Filtered", cv::WINDOW_AUTOSIZE);
            cv::imshow("Disparity Filtered", filteredDisparity);
        }

        cv::Mat filteredHeatmap;
        cv::applyColorMap(filteredDisparity, filteredHeatmap, colorMap);
        if (showImages)
        {
            cv::namedWindow("Heatmap Filtered", cv::WINDOW_AUTOSIZE);
            cv::imshow("Heatmap Filtered", filteredHeatmap);
        }        

        if (saveToFile)
        {

            auto currentPath = std::filesystem::current_path();
            auto disparityPath = currentPath / "disparity";

            if (std::filesystem::is_directory(disparityPath))
            {
                std::filesystem::remove_all(disparityPath);
            }
            std::filesystem::create_directory(disparityPath);

            std::filesystem::path fileDisparity("disparity\\disparity-basic.jpg");
            std::filesystem::path fileHeatmap("disparity\\heatmap-basic.jpg");
            std::filesystem::path fileDisparityFiltered("disparity\\disparity-filtered.jpg");
            std::filesystem::path fileHeatmapFiltered("disparity\\heatmap-filtered.jpg");            

            auto pathDisparity = currentPath / fileDisparity;
            auto pathHeatmap = currentPath / fileHeatmap;
            auto pathDisparityFiltered = currentPath / fileDisparityFiltered;
            auto pathHeatmapFiltered = currentPath / fileHeatmapFiltered;

            cv::imwrite(pathDisparity.u8string(), disparity);
            cv::imwrite(pathHeatmap.u8string(), heatmap);
            cv::imwrite(pathDisparityFiltered.u8string(), filteredDisparity);
            cv::imwrite(pathHeatmapFiltered.u8string(), filteredHeatmap);
        }

        switch (disparityType)
        {
        case DISPARITY_TYPE::BASIC_DISPARITY:
            disparity.copyTo(map);
            break;
        case DISPARITY_TYPE::BASIC_HEATMAP:
            heatmap.copyTo(map);
            break;
        case DISPARITY_TYPE::FILTERED_DISPARITY:
            filteredDisparity.copyTo(map);
            break;
        case DISPARITY_TYPE::FILTERED_HEATMAP:
            filteredHeatmap.copyTo(map);
            break;
        case DISPARITY_TYPE::ALL:
            filteredHeatmap.copyTo(map);
            break;
        default:
            filteredHeatmap.copyTo(map);
            break;
        }        

		return EXIT_SUCCESS;
	}

    int disparityMapCuda(cv::cuda::GpuMat& map, const cv::Mat& imageLeft, const cv::Mat& imageRight, int minDisparity, int numDisparities, int blockSize, double lambda, double sigma, DISPARITY_TYPE disparityType, int colorMap, bool saveToFile, bool showImages)
    {
        // Проверка доступности CUDA
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
            std::cerr << "CUDA-устройство не найдено!" << std::endl;
            return EXIT_FAILURE;
        }

        ///////////////////////////////////////////////////////////////////////
        // Преобразование в GrayScale на GPU
        cv::cuda::GpuMat d_imageLeft, d_imageRight, d_imageLeftGS, d_imageRightGS;
        d_imageLeft.upload(imageLeft);
        d_imageRight.upload(imageRight);

        cv::cuda::cvtColor(d_imageLeft, d_imageLeftGS, cv::COLOR_BGR2GRAY);
        cv::cuda::cvtColor(d_imageRight, d_imageRightGS, cv::COLOR_BGR2GRAY);

        // Перенос результатов на CPU
        cv::Mat imageLeftGS, imageRightGS;
        d_imageLeftGS.download(imageLeftGS);
        d_imageRightGS.download(imageRightGS);

        ///////////////////////////////////////////////////////////////////////
        // Построение карты диспаратности (на CPU)
        cv::Ptr<cv::StereoBM> leftMatcher = cv::StereoBM::create(numDisparities, blockSize);
        leftMatcher->setMinDisparity(minDisparity);
        cv::Ptr<cv::StereoMatcher> rightMatcher = cv::ximgproc::createRightMatcher(leftMatcher);

        cv::Mat h_disparityLeft, h_disparityRight;
        leftMatcher->compute(imageLeftGS, imageRightGS, h_disparityLeft);
        rightMatcher->compute(imageRightGS, imageLeftGS, h_disparityRight);

        // Нормализация на CPU
        cv::Mat h_disparityLeft32F, h_disparityRight32F;
        h_disparityLeft.convertTo(h_disparityLeft32F, CV_32F);
        h_disparityRight.convertTo(h_disparityRight32F, CV_32F);

        h_disparityLeft32F = (h_disparityLeft32F / 16.0f - (float)minDisparity) / ((float)numDisparities);
        h_disparityRight32F = (h_disparityRight32F / 16.0f - (float)minDisparity) / ((float)numDisparities);

        if (showImages) {
            cv::namedWindow("Image Left", cv::WINDOW_AUTOSIZE);
            cv::imshow("Image Left", imageLeft);

            cv::namedWindow("Image Right", cv::WINDOW_AUTOSIZE);
            cv::imshow("Image Right", imageRight);

            cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
            cv::imshow("Disparity", h_disparityLeft32F);
        }

        ///////////////////////////////////////////////////////////////////////
        // Расцвечивание карты диспаратности (на CPU)
        cv::Mat h_disparityTMP, h_heatmap;
        h_disparityLeft32F.convertTo(h_disparityTMP, CV_8UC1, 255.0);
        cv::applyColorMap(h_disparityTMP, h_heatmap, colorMap);

        if (showImages) {
            cv::namedWindow("Heatmap", cv::WINDOW_AUTOSIZE);
            cv::imshow("Heatmap", h_heatmap);
        }

        ///////////////////////////////////////////////////////////////////////
        // Фильтрация диспаратности (на CPU)
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wlsFilter = cv::ximgproc::createDisparityWLSFilter(leftMatcher);
        wlsFilter->setLambda(lambda);
        wlsFilter->setSigmaColor(sigma);

        cv::Mat h_filteredDisparityT;
        wlsFilter->filter(h_disparityLeft, imageLeft, h_filteredDisparityT, h_disparityRight);

        cv::Mat h_filteredDisparity;
        cv::ximgproc::getDisparityVis(h_filteredDisparityT, h_filteredDisparity, 1.0);

        cv::Mat h_filteredHeatmap;
        cv::applyColorMap(h_filteredDisparity, h_filteredHeatmap, colorMap);

        if (showImages) {
            cv::namedWindow("Disparity Filtered", cv::WINDOW_AUTOSIZE);
            cv::imshow("Disparity Filtered", h_filteredDisparity);

            cv::namedWindow("Heatmap Filtered", cv::WINDOW_AUTOSIZE);
            cv::imshow("Heatmap Filtered", h_filteredHeatmap);
        }

        ///////////////////////////////////////////////////////////////////////
        // Сохранение результатов
        if (saveToFile) {
            auto currentPath = std::filesystem::current_path();
            auto disparityPath = currentPath / "disparity";

            if (std::filesystem::is_directory(disparityPath)) {
                std::filesystem::remove_all(disparityPath);
            }
            std::filesystem::create_directory(disparityPath);

            std::filesystem::path fileDisparity("disparity\\disparity-basic.jpg");
            std::filesystem::path fileHeatmap("disparity\\heatmap-basic.jpg");
            std::filesystem::path fileDisparityFiltered("disparity\\disparity-filtered.jpg");
            std::filesystem::path fileHeatmapFiltered("disparity\\heatmap-filtered.jpg");

            auto pathDisparity = currentPath / fileDisparity;
            auto pathHeatmap = currentPath / fileHeatmap;
            auto pathDisparityFiltered = currentPath / fileDisparityFiltered;
            auto pathHeatmapFiltered = currentPath / fileHeatmapFiltered;

            cv::imwrite(pathDisparity.u8string(), h_disparityTMP);
            cv::imwrite(pathHeatmap.u8string(), h_heatmap);
            cv::imwrite(pathDisparityFiltered.u8string(), h_filteredDisparity);
            cv::imwrite(pathHeatmapFiltered.u8string(), h_filteredHeatmap);
        }

        ///////////////////////////////////////////////////////////////////////
        // Загрузка результата на GPU для возврата
        cv::cuda::GpuMat d_disparityTMP, d_heatmap, d_filteredDisparity, d_filteredHeatmap;
        d_disparityTMP.upload(h_disparityTMP);
        d_heatmap.upload(h_heatmap);
        d_filteredDisparity.upload(h_filteredDisparity);
        d_filteredHeatmap.upload(h_filteredHeatmap);

        switch (disparityType) {
        case DISPARITY_TYPE::BASIC_DISPARITY:
            d_disparityTMP.copyTo(map);
            break;
        case DISPARITY_TYPE::BASIC_HEATMAP:
            d_heatmap.copyTo(map);
            break;
        case DISPARITY_TYPE::FILTERED_DISPARITY:
            d_filteredDisparity.copyTo(map);
            break;
        case DISPARITY_TYPE::FILTERED_HEATMAP:
            d_filteredHeatmap.copyTo(map);
            break;
        case DISPARITY_TYPE::ALL:
            d_filteredHeatmap.copyTo(map);
            break;
        default:
            d_filteredHeatmap.copyTo(map);
            break;
        }

        if (showImages) {
            cv::waitKey(0);
        }

        return EXIT_SUCCESS;
    }
}