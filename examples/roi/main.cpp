#include <mrcv/mrcv.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <ctime>

static std::pair<float, float> generateCoordinates(int time, int genType = 0, int R = 500, float timeFracture = 1, std::pair<float, float> imgSize = { 1400,1080 })
{
	float dt = (float)time / timeFracture;
	switch (genType)
	{
	case 0:
		return { time + 200, std::sin(dt) * R + R + (imgSize.second / 2 - R) };
	case 1:
		return { std::sin(dt) * R + imgSize.first / 2, std::cos(dt) * R + imgSize.second / 2 };
	}
}
static cv::Point toPoint(std::pair<float, float> point) {
	return { (int)point.first, (int)point.second };
}

int main()
{
	//Инициализация входых перменных
	std::pair<float, float> imgSize = { 1440,1080 };  //размер изображения
	int predictorTrainPointsNum = 50;  //количество точек для обучения модели предсказания положения объекта интереса
	int totalPointsNum = 1000;  //общее количество точек перемещение объекта (кадров)    
	float objectSize = 250.0f; //размер объекта интереса
	int maxError = 30;  //максимальное допустимое отклонение предсказанной координаты от реального значения
	bool drawObj = 0;
	int failSafeNum = 50;
	//инициализация данных для генератора перемещения
	int genType = 1;  //Вид генерации траектории 0 - синус, 1 - круг 
	int R = 300;  //Радиус (размер) генерируемой траектории
	float timeFracture = 50;  //Управление частотой генерируемых траекторий: чем больше значение, тем медленнее объект движется по траектории
	//Инициализация предиктора положения объекта интереса
	int hiddenSize = 20;  //скрытый размер (25 - оптимальное значение)
	int layersNum = 1;  //количество реккурентных слоев модели (1 - оптимальное значение)
	mrcv::Predictor predictor(hiddenSize, layersNum, predictorTrainPointsNum, imgSize);
	//Инициализация оптимизатора размера региона интереса
	size_t epochs = 500;  //Количество эпох для обучения оптимизатора
	size_t sampleSize = 1000;  //Количество генерируемых синтетических данных для обучения оптимизатора
	mrcv::Optimizer optimizer(sampleSize, epochs);
	//Генерация данных для обучения предиктора(первые <predictorTrainPointsNum> кадров видеопотока)
	std::pair<float, float> realCoordinate;
	std::vector<std::pair<float, float>> coordinates;
	for (int i = 1; i <= predictorTrainPointsNum; ++i)
	{
		realCoordinate = generateCoordinates(i, genType, R, timeFracture, imgSize);
		coordinates.emplace_back(realCoordinate);
	}
	//Обучение модели предиктора положения    
	predictor.trainLSTMNet(coordinates);
	//основной цикл
	std::time_t start = std::time(nullptr);
	std::pair<float, float> tmpRealCoordinate = realCoordinate;
	std::pair<float, float> predictedCoordinate;
	std::pair<float, float> tmpPredictedCoordinate = realCoordinate;
	cv::Mat imgR(imgSize.second, imgSize.first, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat roi(objectSize, objectSize, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat imageFull(imgR.rows, imgR.cols + roi.cols, CV_8UC3, cv::Scalar(0, 124, 0));
	roi.copyTo(imageFull(cv::Rect(0, 0, roi.cols, roi.rows)));
	imgR.copyTo(imageFull(cv::Rect(roi.cols, 0, imgR.cols, imgR.rows)));
	float successfullCounter = 0;
	float error = 0;
	float errorSum = 0;
	float avgError = 0;
	int successedPredictions = 0;
	bool workState = false;
	bool goodState = false;
	bool ROISizeAquired = false;
	float roiSize = 0;
	float movingAvgError = 0;
	std::vector<float> errors;
	for (int i = predictorTrainPointsNum + 1; i <= totalPointsNum; i++)
	{
		tmpPredictedCoordinate = predictedCoordinate;
		predictedCoordinate = predictor.predictNextCoordinate();
		cv::line(imgR, toPoint(tmpPredictedCoordinate), toPoint(predictedCoordinate), cv::Scalar(255, 0, 0), 1, 8, 0);
		if (drawObj)
		{
			cv::Mat imgTemp(imgSize.second, imgSize.first, CV_8UC3, cv::Scalar(255, 255, 255));
			imgR = imgTemp;
			cv::circle(imgR, { (int)realCoordinate.first,(int)realCoordinate.second }, objectSize / 2, cv::Scalar(0, 124, 0), 1);
		}
		if (workState && !ROISizeAquired)
		{
			roiSize = optimizer.optimizeRoiSize(realCoordinate,
				predictedCoordinate,
				objectSize,
				movingAvgError / 2);

			std::cout << "Optimized ROI size: " << roiSize << std::endl;
			ROISizeAquired = true;
		}
		if (workState)
		{
			roi = mrcv::extractROI(imgR, toPoint(predictedCoordinate), { (int)roiSize, (int)roiSize });
		}
		cv::Mat imageTemp(imgR.rows, imgR.cols + roi.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		imageFull = imageTemp;
		roi.copyTo(imageFull(cv::Rect(0, 0, roi.cols, roi.rows)));
		imgR.copyTo(imageFull(cv::Rect(roi.cols, 0, imgR.cols, imgR.rows)));
		if (workState)
		{
			cv::putText(imageFull, "ROI acting", { 10, 600 }, cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 124, 0), 2);
		}
		else
		{
			cv::putText(imageFull, "ROI training", { 10, 600 }, cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 124, 124), 2);
		}
		cv::imshow("image", imageFull);
		cv::waitKey(1);
		tmpRealCoordinate = realCoordinate;
		realCoordinate = generateCoordinates(i, genType, R, timeFracture, imgSize);
		predictor.continueTraining(realCoordinate);
		cv::line(imgR, toPoint(tmpRealCoordinate), toPoint(realCoordinate), cv::Scalar(0, 0, 255), 1, 8, 0);
		//вычисление ошибки
		error = std::sqrt(std::pow(realCoordinate.first - predictedCoordinate.first, 2) + std::pow((realCoordinate.second - predictedCoordinate.second), 2));
		errorSum += error;
		avgError = errorSum / (i - predictorTrainPointsNum);
		errors.push_back(error);
		if (errors.size() > failSafeNum / 2)
		{
			errors.erase(errors.begin());
		}
		for (int s = 0; s < errors.size(); s++)
		{
			movingAvgError += errors[s];
		}
		movingAvgError = movingAvgError / errors.size();
		goodState = movingAvgError < maxError;
		if (goodState)
		{
			successedPredictions++;
		}
		else
		{
			successedPredictions = 0;
		}
		workState = successedPredictions > failSafeNum;
		workState = goodState;
		if (1)
		{
			std::cout << "Point number: " << i << "  Average error:  " << avgError << "  Moving average error:  " << movingAvgError << std::endl;
		}
	}
	float successRate = (float)successedPredictions / (totalPointsNum - predictorTrainPointsNum);
	std::time_t stop = std::time(nullptr);
	std::time_t timeElapsed = stop - start;
	std::cout << "Time elapsed: " << timeElapsed << std::endl;
	std::cout << "FPS: " << (double)(totalPointsNum - predictorTrainPointsNum) / (double)timeElapsed << std::endl;
	std::cout << "Average error: " << avgError << std::endl;
	std::cout << "Success rate: " << successRate << std::endl;
	cv::imshow("image", imageFull);
	cv::waitKey();
	return 0;
}