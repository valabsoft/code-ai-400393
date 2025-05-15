#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>

namespace mrcv {
	int mrcv::makingStereoPair(cv::Mat& inputImageCamera01, cv::Mat& inputImageCamera02, cv::Mat& outputStereoPair)
	{
		try
		{
			// ////////////////////
			// Проверка данных
			// ////////////////////
			if (inputImageCamera01.empty() || inputImageCamera02.empty())
			{
				outputStereoPair = mrcv::getErrorImage("makingStereoPair:: Image is Empty");
				return 1; // 1 - Пустое изображение
			}
			// ////////////////////
			// Расчёт размеров исходного изображения
			int maxV = std::max(inputImageCamera01.rows, inputImageCamera02.rows);
			int maxU = inputImageCamera01.cols + inputImageCamera02.cols;
			// Создание стереопары
			cv::Mat3b imagePair(maxV, maxU, cv::Vec3b(0, 0, 0));
			// Копирование
			inputImageCamera01.copyTo(imagePair(cv::Rect(0, 0, inputImageCamera01.cols, inputImageCamera01.rows)));
			inputImageCamera02.copyTo(imagePair(cv::Rect(inputImageCamera01.cols, 0, inputImageCamera02.cols, inputImageCamera02.rows)));
			outputStereoPair = imagePair.clone();
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::showImage(cv::Mat& inputImage, const cv::String windowName, double CoefShowWindow)
	{
		try
		{
			// ////////////////////
			// Проверка полноты данных
			// ////////////////////
			if (inputImage.empty())
			{
				inputImage = mrcv::getErrorImage("showImage:: Image is Empty");
				mrcv::writeLog("showImage:: Image is Empty, status =  1", mrcv::LOGTYPE::ERROR);
				return 1; // 1 - Пустое изображение
			}
			// ////////////////////
			// Вывод результата на экран
			// ////////////////////
			cv::Mat outputImage;
			// Уменьшение чтобы поместилось на экран
			cv::resize(inputImage, outputImage, cv::Size(double(inputImage.cols * CoefShowWindow),
				double(inputImage.rows * CoefShowWindow)), 0, 0, cv::INTER_LINEAR);
			namedWindow(windowName, cv::WINDOW_AUTOSIZE);
			imshow(windowName, outputImage);
			cv::waitKey(10);
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::readCameraStereoParametrsFromFile(const char* pathToFileCameraParametrs, mrcv::cameraStereoParameters& cameraParameters)
	{
		try
		{
			cv::FileStorage fs(pathToFileCameraParametrs, cv::FileStorage::READ);
			if (fs.isOpened())
			{
				fs["M1"] >> cameraParameters.M1;
				fs["D1"] >> cameraParameters.D1;
				fs["M2"] >> cameraParameters.M2;
				fs["D2"] >> cameraParameters.D2;
				fs["E"] >> cameraParameters.E;
				fs["F"] >> cameraParameters.F;
				fs["R"] >> cameraParameters.R;
				fs["T"] >> cameraParameters.T;
				fs["R1"] >> cameraParameters.R1;
				fs["R2"] >> cameraParameters.R2;
				fs["P1"] >> cameraParameters.P1;
				fs["P2"] >> cameraParameters.P2;
				fs["Q"] >> cameraParameters.Q;
				fs["imageSize"] >> cameraParameters.imageSize;
				fs["rms"] >> cameraParameters.rms;
				fs["avgErr"] >> cameraParameters.avgErr;
				fs.release();
			}
			else
			{
				fs.releaseAndGetString();
			}
			cv::stereoRectify(
				cameraParameters.M1,
				cameraParameters.D1,
				cameraParameters.M2,
				cameraParameters.D2,
				cameraParameters.imageSize,
				cameraParameters.R,
				cameraParameters.T,
				cameraParameters.R1,
				cameraParameters.R2,
				cameraParameters.P1,
				cameraParameters.P2,
				cameraParameters.Q,
				cv::CALIB_ZERO_DISPARITY,
				//0,
				-1,
				cameraParameters.imageSize);

			// Расчёт карт точек
			cv::Mat M1n = cameraParameters.P1.clone(); // новая матрица камеры 3x3
			cv::Mat M2n = cameraParameters.P2.clone(); // новая матрица камеры 3x3
			cv::initUndistortRectifyMap(cameraParameters.M1, cameraParameters.D1, cameraParameters.R1, M1n, cameraParameters.imageSize,
				CV_16SC2, cameraParameters.map11, cameraParameters.map12);
			cv::initUndistortRectifyMap(cameraParameters.M2, cameraParameters.D2, cameraParameters.R2, M2n, cameraParameters.imageSize,
				CV_16SC2, cameraParameters.map21, cameraParameters.map22);
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::convetingUndistortRectify(cv::Mat& imageInput, cv::Mat& imageOutput, cv::Mat& map11, cv::Mat& map12)
	{
		try
		{
			if (imageInput.empty())
			{
				imageOutput = mrcv::getErrorImage("convetingUndistortRectify:: Image is Empty");
				return 1; // 1 - Пустое изображение
			}
			cv::remap(imageInput, imageOutput, map11, map12, cv::INTER_LINEAR);
		}
		catch (...)
		{
			return -1;  // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::find3dPointsADS(cv::Mat& inputImageCamera01, cv::Mat& inputImageCamera02, mrcv::pointsData& points3D,
		mrcv::settingsMetodDisparity& settingsMetodDisparity, cv::Mat& disparityMap,
		mrcv::cameraStereoParameters& cameraParameters, int limit3dPoints, std::vector<double> limitsOutlierArea)
	{
		try
		{
			// ////////////////////
			// Проверка полноты данных
			// ////////////////////
			if (inputImageCamera01.empty() || inputImageCamera02.empty())
			{
				return 1; // 1 - Пустое изображение
			}
			if (settingsMetodDisparity.metodDisparity == mrcv::METOD_DISPARITY::MODE_NONE && disparityMap.empty())
			{
				return 2; // 2 - Пустая исходная карта диспаратности, елсли MODE_NONE
			}
			if (cameraParameters.imageSize.height == 0 || cameraParameters.imageSize.width == 0)
			{
				return 3; // 3 - Ошибка в параметрах калибровки
			}
			// ////////////////////

			cv::Mat imgage01Gray;   // Изображение, камера 01
			cv::Mat imgage02Gray;   // Изображение, камера 02
			cv::Mat disparity;     // Карта диспаратности
			cv::Mat xyz_AllPoints; // Карта глубины
			// Преобразование цветного изображения в серое
			cv::cvtColor(inputImageCamera01, imgage01Gray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(inputImageCamera02, imgage02Gray, cv::COLOR_BGR2GRAY);

			// ///////////////////////////////////
			// Получение карты расхождения (диспаратности)
			// ///////////////////////////////////
			if (settingsMetodDisparity.metodDisparity == mrcv::METOD_DISPARITY::MODE_BM) // BM
			{
				cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(settingsMetodDisparity.smbNumDisparities, settingsMetodDisparity.smbBlockSize);
				bm->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
				bm->setPreFilterSize(7);
				bm->setPreFilterCap(settingsMetodDisparity.smbPreFilterCap);        // 31 -> 15 +++
				bm->setMinDisparity(settingsMetodDisparity.smbMinDisparity);
				bm->setTextureThreshold(settingsMetodDisparity.smbTextureThreshold);    // 10   +++
				bm->setUniquenessRatio(settingsMetodDisparity.smbUniquenessRatio);     // 15   +++
				bm->setSpeckleWindowSize(settingsMetodDisparity.smbSpeckleWindowSize);  // 100 +++
				bm->setSpeckleRange(settingsMetodDisparity.smbSpeckleRange);        // 32
				bm->setDisp12MaxDiff(settingsMetodDisparity.smbDisp12MaxDiff);        // 1
				// Расчёт карты диспаратности
				bm->compute(imgage01Gray, imgage02Gray, disparityMap);
			}
			else if (settingsMetodDisparity.metodDisparity == mrcv::METOD_DISPARITY::MODE_SGBM ||
				settingsMetodDisparity.metodDisparity == mrcv::METOD_DISPARITY::MODE_SGBM_3WAY ||
				settingsMetodDisparity.metodDisparity == mrcv::METOD_DISPARITY::MODE_HH ||
				settingsMetodDisparity.metodDisparity == mrcv::METOD_DISPARITY::MODE_HH4)
			{
				// SGBM
				int sgbmWinSize = settingsMetodDisparity.smbBlockSize;
				cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(64,    //int minDisparity
					settingsMetodDisparity.smbNumDisparities,     //int numDisparities
					sgbmWinSize,      //int BlockSize
					0,    //int P1 = 0
					0,   //int P2 = 0
					0,     //int disp12MaxDiff = 0
					0,     //int preFilterCap = 0
					40,      //int uniquenessRatio = 0
					200,    //int speckleWindowSize = 0
					2,     //int speckleRange = 0
					false);  //bool fullDP = false
				switch (settingsMetodDisparity.metodDisparity)
				{
				case mrcv::METOD_DISPARITY::MODE_SGBM:
					sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
					break;
				case mrcv::METOD_DISPARITY::MODE_HH:
					sgbm->setMode(cv::StereoSGBM::MODE_HH);
					break;
				case mrcv::METOD_DISPARITY::MODE_SGBM_3WAY:
					sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
					break;
				case mrcv::METOD_DISPARITY::MODE_HH4:
					sgbm->setMode(cv::StereoSGBM::MODE_HH4);
					break;
				}// switch
				//  Настройка SGBM
				sgbm->setPreFilterCap(settingsMetodDisparity.smbPreFilterCap);
				int cn = imgage01Gray.channels();
				sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
				sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
				sgbm->setMinDisparity(settingsMetodDisparity.smbMinDisparity);
				sgbm->setNumDisparities(settingsMetodDisparity.smbNumDisparities);
				sgbm->setUniquenessRatio(settingsMetodDisparity.smbUniquenessRatio);
				sgbm->setSpeckleWindowSize(settingsMetodDisparity.smbSpeckleWindowSize);
				sgbm->setSpeckleRange(settingsMetodDisparity.smbSpeckleRange);
				sgbm->setDisp12MaxDiff(settingsMetodDisparity.smbDisp12MaxDiff);
				// Расчёт карты диспаратности
				sgbm->compute(imgage01Gray, imgage02Gray, disparityMap);
			} // if

			// //////////////////////
			// Создание карты глубины
			// //////////////////////
			disparityMap.convertTo(disparityMap, CV_32F, 0.0625f); // 1/16 = 0.0625f
			cv::reprojectImageTo3D(disparityMap, xyz_AllPoints, cameraParameters.Q, true, CV_32F);

			// //////////////////////
			// Отсеивание выбросов
			// //////////////////////
			std::vector<std::vector<int>> vu0;            // 2D координаты точки на изображении
			std::vector<std::vector<double>> xyz;        // 3D координаты точки на пространсве
			std::vector<std::vector<int>> rgb;           // цвет 3D точки

			for (int v = 0; v < xyz_AllPoints.rows; v++)
			{
				for (int u = 0; u < xyz_AllPoints.cols; u++)
				{
					cv::Vec3f xyz3D = xyz_AllPoints.at<cv::Vec3f>(v, u);
					if (xyz3D[0] < limitsOutlierArea[0]) continue;
					if (xyz3D[1] < limitsOutlierArea[1]) continue;
					if (xyz3D[2] < limitsOutlierArea[2]) continue;

					if (xyz3D[0] > limitsOutlierArea[3]) continue;
					if (xyz3D[1] > limitsOutlierArea[4]) continue;
					if (xyz3D[2] > limitsOutlierArea[5]) continue;

					// Данные о 3D координате точки и её 2D проекции
					vu0.push_back({ v, u });
					xyz.push_back(std::vector<double>({ xyz3D[0], xyz3D[1], xyz3D[2] }));
					// Данные о цвете точки
					cv::Vec3b rgb2 = inputImageCamera01.at<cv::Vec3b>(v, u);
					rgb.push_back(std::vector<int>({ rgb2[0], rgb2[1], rgb2[2] }));
				}
			}

			// //////////////////////
			//  Занесение резултатов до уменьшения количества точек
			// //////////////////////
			points3D.vu0 = vu0;
			points3D.xyz0 = xyz;
			points3D.rgb0 = rgb;
			points3D.numPoints0 = vu0.size();

			// //////////////////////
			//  Уменьшение кооличесва точек
			// //////////////////////
			int step = 0;  // счетчик для прореживания точек
			int churn = points3D.numPoints0 / limit3dPoints;
			if (churn < 1) churn = 1;
			for (int qi = 0; qi < points3D.numPoints0; qi++)
			{
				step++;
				if (step < churn) continue;
				step = 0;
				points3D.vu.push_back(points3D.vu0.at(qi));
				points3D.xyz.push_back(points3D.xyz0.at(qi));
				points3D.rgb.push_back(points3D.rgb0.at(qi));
				points3D.segment.push_back(-1);
			}
			points3D.numPoints = points3D.vu.size();
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::detectingSegmentsNeuralNet(cv::Mat& imageInput, cv::Mat& imageOutput, std::vector<cv::Mat>& replyMasks,
		const std::string filePathToModelYoloNeuralNet, const std::string filePathToClasses)
	{
		try
		{
			// ////////////////////
			// Проверка входных данных
			// ////////////////////
			if (imageInput.empty())
			{
				imageOutput = mrcv::getErrorImage("detectingSegmentsNeuralNet:: Image is Empty");
				return 1; // 1 - Пустое изображение
			}
			// ////////////////////

			bool reply = false;
			std::string replyAll = "-";
			mrcv::neuralNetSegmentator* segmentator = new mrcv::neuralNetSegmentator(filePathToModelYoloNeuralNet, filePathToClasses);

			segmentator->process(imageInput);
			replyMasks = segmentator->getMasks();
			imageOutput = segmentator->getImage();
		}
		catch (...)
		{

			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::matchSegmentsWith3dPoints(mrcv::pointsData& points3D, std::vector<cv::Mat>& replyMasks)
	{
		try
		{
			// ////////////////////
			// Проверка входных данных
			// ////////////////////
			if (points3D.numPoints0 < 1 || points3D.numPoints < 1) return 1;
			if (replyMasks.empty()) return 2;
			// ////////////////////

			// Инициализация векторов
			points3D.numSegments = replyMasks.size();
			points3D.pointsInSegments.resize(points3D.numSegments);
			points3D.numPointsInSegments.resize(points3D.numSegments, -1);
			points3D.center2dSegments.resize(points3D.numSegments);
			points3D.center3dSegments.resize(points3D.numSegments);

			// ////////////////////
			// Главный цикл перебора обнаруженных сегментов
			// ////////////////////
			for (int qs = 0; qs < points3D.numSegments; ++qs)
			{
				// ////////////////////
				// Морфологичкие обработка над сегментом
				// ////////////////////
				cv::Mat se1 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
				morphologyEx(replyMasks.at(qs), replyMasks.at(qs), cv::MORPH_CLOSE, se1);

				// ////////////////////
				// Сопоставление сегментов и 3D точек
				// ////////////////////
				for (int qp = 0; qp < points3D.numPoints; ++qp)
				{
					int v = points3D.vu.at(qp).at(0);
					int u = points3D.vu.at(qp).at(1);

					if (replyMasks.at(qs).at<uchar>(v, u) != 0)
					{
						points3D.segment.at(qp) = qs;
						points3D.pointsInSegments.at(qs).push_back(qp);
					}
				} // for (qp)
				points3D.numPointsInSegments.at(qs) = points3D.pointsInSegments.at(qs).size();
			} // for (qs)

			// ////////////////////
			// Расчёт (среднего значения) геометрического центра сегмента
			// ////////////////////
			double meanV = 0;
			double meanU = 0;
			double meanX = 0;
			double meanY = 0;
			double meanZ = 0;
			for (int qs = 0; qs < (int)replyMasks.size(); ++qs)
			{
				for (int qp = 0; qp < (int)points3D.numPoints; ++qp)
				{
					if (points3D.segment.at(qp) == qs)
					{
						double v = points3D.vu[qp][0];
						double u = points3D.vu[qp][1];
						double x = points3D.xyz[qp][0];
						double y = points3D.xyz[qp][1];
						double z = points3D.xyz[qp][2];
						meanV = meanV + v;
						meanU = meanU + u;
						meanX = meanX + x;
						meanY = meanY + y;
						meanZ = meanZ + z;
					}
				}
				meanV = meanV / points3D.numPointsInSegments.at(qs);
				meanU = meanU / points3D.numPointsInSegments.at(qs);
				meanX = meanX / points3D.numPointsInSegments.at(qs);
				meanY = meanY / points3D.numPointsInSegments.at(qs);
				meanZ = meanZ / points3D.numPointsInSegments.at(qs);
				points3D.center2dSegments.at(qs).x = (int)meanU;
				points3D.center2dSegments.at(qs).y = (int)meanV;
				points3D.center3dSegments.at(qs).x = meanX;
				points3D.center3dSegments.at(qs).y = meanY;
				points3D.center3dSegments.at(qs).z = meanZ;
			} // for (qs)
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::addToImageCenter3dSegments(cv::Mat& inputImage, cv::Mat& outputImage, mrcv::pointsData& points3D)
	{
		try
		{
			// ////////////////////
			// Проверка входных данных
			// ////////////////////
			if (inputImage.empty()) return 1; // 1 - Пустое изображение
			if (points3D.numSegments < 1) return 3;
			// ////////////////////

			outputImage = inputImage.clone();

			const float FONT_SCALE = 0.5;
			const int   THICKNESS = 1;
			const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
			const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
			const cv::Scalar RED = cv::Scalar(0, 0, 255);
			const cv::Scalar GREEN = cv::Scalar(0, 255, 0);

			for (int qs = 0; qs < (int)points3D.numSegments; ++qs)
			{
				std::stringstream label;
				label
					//                << "x = " << points3D.center3dSegments[qs].x
					//                << ", y = " << points3D.center3dSegments[qs].y
					<< " z = " << points3D.center3dSegments[qs].z;

				//            cv::String label;
				//            label =  "x = " + std::to_string(points3D.center3dSegments[qs].x) +
				//                     ", y = " + std::to_string(points3D.center3dSegments[qs].y) +
				//                     ", z = " + std::to_string(points3D.center3dSegments[qs].z);

				int left = points3D.center2dSegments[qs].x;
				int top = points3D.center2dSegments[qs].y;

				// Нанесение текста
				int baseline;
				cv::Size label_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS, &baseline);
				top = std::max(top, label_size.height);
				cv::Point tlc = cv::Point(left, top);
				cv::Point brc = cv::Point(left + label_size.width, top + label_size.height + baseline);
				// Нанесение подложки под текс
				cv::rectangle(outputImage, tlc, brc, YELLOW, cv::FILLED);
				// Нанесение текста
				cv::putText(outputImage, label.str(), cv::Point(left, top + label_size.height),
					cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, RED, THICKNESS);
			} // for (qs)
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::showDispsarityMap(cv::Mat& disparityMap, const cv::String windowName, double CoefShowWindow)
	{
		try
		{
			// ////////////////////
			// Проверка полноты данных
			// ////////////////////
			if (disparityMap.empty())
			{
				disparityMap = mrcv::getErrorImage("showDispsarityMap:: Image is Empty");
				return 1; // 1 - Пустое изображение
			}
			// ////////////////////
			//    // ////////////////////
			//    // Вывод результата на экран
			//    // ////////////////////
			cv::Mat outDisparity;
			double minVal, maxVal;

			minMaxLoc(disparityMap, &minVal, &maxVal);
			disparityMap.convertTo(outDisparity, CV_8UC1, 255 / (maxVal - minVal));

			applyColorMap(outDisparity, outDisparity, cv::COLORMAP_TURBO);

			cv::resize(outDisparity, outDisparity, cv::Size(int(outDisparity.cols * CoefShowWindow),
				int(outDisparity.rows * CoefShowWindow)), 0, 0, cv::INTER_LINEAR);
			namedWindow(windowName, cv::WINDOW_AUTOSIZE);
			cv::imshow(windowName, outDisparity);
			cv::waitKey(10);

		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::getImage3dSceene(mrcv::pointsData& points3D, mrcv::parameters3dSceene& parameters3dSceene,
		mrcv::cameraStereoParameters& cameraParameters, cv::Mat& outputImage3dSceene)
	{
		try
		{
			// ////////////////////
			// Проверка полноты данных
			// ////////////////////
			if (points3D.numPoints0 < 1 || points3D.numPoints < 1)
			{
				return 1; // 1 - Пустое данные о точках
			}

			if (cameraParameters.imageSize.height == 0 || cameraParameters.imageSize.width == 0)
			{
				return 3; // 3 - Ошибка в параметрах калибровки
			}
			// ////////////////////

			cv::Size2i imgSize = cameraParameters.imageSize;
			outputImage3dSceene = cv::Mat::zeros(imgSize, CV_8UC3);

			double angX = parameters3dSceene.angX;
			double angY = parameters3dSceene.angY;
			double angZ = parameters3dSceene.angZ;
			double tX = parameters3dSceene.tX;
			double tY = parameters3dSceene.tY;
			double tZ = parameters3dSceene.tZ;
			double scale = parameters3dSceene.scale;
			double dZ = parameters3dSceene.dZ;

			cv::Mat Rx = cv::Mat(3, 3, CV_64F);
			cv::Mat Ry = cv::Mat(3, 3, CV_64F);
			cv::Mat Rz = cv::Mat(3, 3, CV_64F);
			cv::Mat R = cv::Mat(3, 3, CV_64F);
			cv::Mat T = cv::Mat(3, 1, CV_64F);
			cv::Mat xyz = cv::Mat(3, 1, CV_64F);
			cv::Mat xyz1 = cv::Mat(4, 1, CV_64F); // xyz в однородных координатах
			cv::Mat uv1 = cv::Mat(3, 1, CV_64F);  // uv в однородных координатах
			cv::Mat uv = cv::Mat(2, 1, CV_64F);
			cv::Mat P = cv::Mat(3, 4, CV_64F);

			std::vector <std::vector<double>> Rvec;
			Rvec.resize(3);

			// ang перевод в радианы
			double coefPI = 3.1416 / 180;
			angX = coefPI * angX;
			angY = coefPI * angY;
			angZ = coefPI * angZ;

			// T
			T.at<double>(0, 0) = tX;
			T.at<double>(1, 0) = tY;
			T.at<double>(2, 0) = tZ;

			// Rx
			Rvec[0] = { 1,          0,           0 };
			Rvec[1] = { 0,  cos(angX),  -sin(angX) };
			Rvec[2] = { 0,  sin(angX),   cos(angX) };

			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					Rx.at<double>(i, j) = Rvec[i][j];

			//Ry
			Rvec[0] = { cos(angY),  0,  sin(angY) };
			Rvec[1] = { 0,  1,          0 };
			Rvec[2] = { -sin(angY),  0,  cos(angY) };

			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					Ry.at<double>(i, j) = Rvec[i][j];

			//Rz
			Rvec[0] = { cos(angZ),  -sin(angZ),   0 };
			Rvec[1] = { sin(angZ),   cos(angZ),   0 };
			Rvec[2] = { 0,           0,   1 };

			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					Rz.at<double>(i, j) = Rvec[i][j];

			// ======
			R = Rx * Ry * Rz;
			// ======

			// P
			Rvec.resize(3);
			Rvec[0] = { 1, 0, 0, 0 };
			Rvec[1] = { 0, 1, 0, 0 };
			Rvec[2] = { 0, 0, 1, -dZ };

			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 4; j++)
					P.at<double>(i, j) = Rvec[i][j];

			P = cameraParameters.M1 * P;

			for (int qp = 0; qp < int(points3D.numPoints0); qp++)
			{
				xyz.at<double>(0, 0) = scale * points3D.xyz0.at(qp).at(0);
				xyz.at<double>(1, 0) = scale * points3D.xyz0.at(qp).at(1);
				xyz.at<double>(2, 0) = scale * points3D.xyz0.at(qp).at(2);

				xyz = R * (xyz + T);

				xyz1.at<double>(0, 0) = xyz.at<double>(0, 0);
				xyz1.at<double>(1, 0) = xyz.at<double>(1, 0);
				xyz1.at<double>(2, 0) = xyz.at<double>(2, 0);
				xyz1.at<double>(3, 0) = 1;

				uv1 = P * xyz1;
				uv.at<double>(0, 0) = uv1.at<double>(0, 0) / uv1.at<double>(2, 0);
				uv.at<double>(1, 0) = uv1.at<double>(1, 0) / uv1.at<double>(2, 0);

				int r, c;
				r = round(uv.at<double>(1, 0));
				c = round(uv.at<double>(0, 0));

				if ((r > 0) && (c > 0) && (r < imgSize.height) && (c < imgSize.width))
				{
					int u;
					int v;
					v = points3D.vu0.at(qp).at(0);
					u = points3D.vu0.at(qp).at(1);

					outputImage3dSceene.at<cv::Vec3b>(r, c) = cv::Vec3b(points3D.rgb0.at(qp).at(0), points3D.rgb0.at(qp).at(1), points3D.rgb0.at(qp).at(2));
				}
			}
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::saveInFile3dPointsInObjectsSegments(pointsData& points3D, const cv::String pathToFile)
	{
		try
		{
			// ////////////////////
			// Проверка полноты данных
			// ////////////////////
			if (points3D.numPoints0 < 1 || points3D.numPoints < 1)
			{
				return 1; // 1 - Нет точек
			}
			// ////////////////////
			//  сохранение данных о 3D точках после сегментации в текстовый файл
			// ////////////////////
			std::ofstream out3;          // поток для записи
			out3.open(pathToFile, std::ios_base::out); // окрываем файл для записи
			if (out3.is_open())
				for (int qp = 0; qp < points3D.numPoints; qp++)
				{
					// пропуск точки если она не пренадлежин ни одному сегменту
					if (points3D.segment.at(qp) == -1) continue;
					out3 << points3D.vu.at(qp)[0] << " \t "
						<< points3D.vu.at(qp)[1] << " \t "
						<< points3D.xyz.at(qp)[0] << " \t "
						<< points3D.xyz.at(qp)[1] << " \t "
						<< points3D.xyz.at(qp)[2] << " \t "
						<< points3D.rgb.at(qp)[0] << " \t "
						<< points3D.rgb.at(qp)[1] << " \t "
						<< points3D.rgb.at(qp)[2] << " \t "
						<< points3D.segment.at(qp) << std::endl;
				}
			out3.close();
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	int mrcv::find3dPointsInObjectsSegments(cv::Mat& inputImageCamera01, cv::Mat& inputImageCamera02,
		mrcv::cameraStereoParameters& cameraParameters,
		cv::Mat& inputImageCamera01Remap, cv::Mat& inputImageCamera02Remap,
		mrcv::settingsMetodDisparity& settingsMetodDisparity, cv::Mat& disparityMap,
		mrcv::pointsData& points3D, std::vector<cv::Mat>& replyMasks, cv::Mat& outputImage,
		cv::Mat& outputImage3dSceene, mrcv::parameters3dSceene& parameters3dSceene,
		const std::string filePathToModelYoloNeuralNet, const std::string filePathToClasses,
		int limitOutPoints, std::vector<double> limitsOutlierArea)
	{
		try
		{
			int state; // для ошибок функций
			// ////////////////////
			// A1. Подготовка изображений (коррекция искажений и выравнивание)
			// ////////////////////
			state = mrcv::convetingUndistortRectify(inputImageCamera01, inputImageCamera01Remap, cameraParameters.map11, cameraParameters.map12);
			// ////////////////////
			if (state == 0)
			{
				mrcv::writeLog("A1. Выравнивание изображения камера 01 (успешно)");
			}
			else
			{
				mrcv::writeLog("convetingUndistortRectify 01, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
			}

			// ////////////////////
			state = mrcv::convetingUndistortRectify(inputImageCamera02, inputImageCamera02Remap, cameraParameters.map21, cameraParameters.map22);
			// ////////////////////
			if (state == 0)
			{
				mrcv::writeLog("A1. Выравнивание изображения камера 02 (успешно)");
			}
			else
			{
				mrcv::writeLog("convetingUndistortRectify 02, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
			}

			// ////////////////////
			// A2. Поиск точек
			// ////////////////////
			state = mrcv::find3dPointsADS(inputImageCamera01Remap, inputImageCamera02Remap, points3D, settingsMetodDisparity, disparityMap,
				cameraParameters, limitOutPoints, limitsOutlierArea);
			// ////////////////////
			if (state == 0)
			{
				mrcv::writeLog("A2. Облако 3D точек сцены найдено (успешно)");
				mrcv::writeLog("    points3D.numPoints0 = " + std::to_string(points3D.numPoints0));
				mrcv::writeLog("    points3D.numPoints = " + std::to_string(points3D.numPoints));
				mrcv::writeLog("    points3D.numSegments = " + std::to_string(points3D.numSegments));
			}
			else
			{
				mrcv::writeLog("find3dPointsADS, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
			}

			if (replyMasks.empty()) // Если данные о сегменте не введены то используется алгоритм сегментации
			{
				// ////////////////////
				// A3. Сегментация изображения (по результатам обнаружения и распознания объектов)
				// ////////////////////
				state = mrcv::detectingSegmentsNeuralNet(inputImageCamera01Remap, outputImage, replyMasks, filePathToModelYoloNeuralNet, filePathToClasses);
				// ////////////////////
				if (state == 0)
				{
					mrcv::writeLog("A3. Сегментация изображения (успешно)");
					mrcv::writeLog("    путь к модели нейронной сети " + filePathToModelYoloNeuralNet);
					mrcv::writeLog("    replyMasks.size() =  " + std::to_string(replyMasks.size()));
				}
				else
				{
					mrcv::writeLog("detectingSegmentsNeuralNet, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
				}
			}
			// ////////////////////
			// A4. Определения координат 3D точек в сегментах идентифицированных объектов
			// ////////////////////
			state = mrcv::matchSegmentsWith3dPoints(points3D, replyMasks);
			// ////////////////////
			if (state == 0)
			{
				mrcv::writeLog("A4. Сопоставление координат и сегментов (успешно)");
				mrcv::writeLog("    points3D.numSegments = " + std::to_string(points3D.numSegments));
				for (int qs = 0; qs < points3D.numSegments; ++qs)
				{
					mrcv::writeLog("    точек в сегменте " + std::to_string(qs) + " = " + std::to_string(points3D.numPointsInSegments.at(qs)) +
						"; 3D центр: " + std::to_string(points3D.center3dSegments.at(qs).x) + ", " +
						std::to_string(points3D.center3dSegments.at(qs).y) + ", " + std::to_string(points3D.center3dSegments.at(qs).z));
				}
			}
			else
			{
				mrcv::writeLog("matchSegmentsWith3dPoints, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
			}

			// ////////////////////
			// A5. Нанесения координат 3D центра сегмента на изображени в виде текста
			// ////////////////////
			state = mrcv::addToImageCenter3dSegments(outputImage, outputImage, points3D);
			// ////////////////////
			if (state == 0)
			{
				mrcv::writeLog("A5. Нанесение координат 3D центра сегмента на результирующие изображение (успешно)");
			}
			else
			{
				mrcv::writeLog("drawCenter3dSegments, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
			}
			// ////////////////////
			// A6. Получение 3D сцены
			// ////////////////////
			state = mrcv::getImage3dSceene(points3D, parameters3dSceene, cameraParameters, outputImage3dSceene);
			// ////////////////////
			if (state == 0)
			{
				mrcv::writeLog("A6. Проекция 3D сцены на 2D изображение для вывода на экран (успешно)");
			}
			else
			{
				mrcv::writeLog("getImage3dSceene, status = " + std::to_string(state), mrcv::LOGTYPE::ERROR);
			}

		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	mrcv::neuralNetSegmentator::neuralNetSegmentator(const std::string model, const std::string classes)
	{
		if (!initializationNetwork(model, classes))
		{
			mrcv::writeLog("Neural network been inited!");
			mrcv::writeLog("  Input width: " + std::to_string(input_width) + "; Input height: " + std::to_string(input_height));
		}
		else
		{
			mrcv::writeLog("Failed to init neural network!", mrcv::LOGTYPE::ERROR);
		}
	}

	int mrcv::neuralNetSegmentator::readСlasses(const std::string file_path)
	{
		std::ifstream classes_file(file_path);
		std::string line;
		srand(time(0));

		if (!classes_file) {
			mrcv::writeLog("network: Failed to open classes names!", mrcv::LOGTYPE::ERROR);
			return ENOENT;
		}
		while (std::getline(classes_file, line)) {
			classes.push_back(line);
			masksColorsSet.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
		}
		classes_file.close();
		return 0;
	}

	int mrcv::neuralNetSegmentator::initializationNetwork(const std::string model_path, const std::string classes_path) {
		int err = readСlasses(classes_path);
		if (err == 0) {
			try {
				network = cv::dnn::readNetFromONNX(model_path);
				if (network.empty()) {
					mrcv::writeLog("Не удалось загрузить модель ONNX!", mrcv::LOGTYPE::ERROR);
					return ENETDOWN;
				}

				// Проверка доступности CUDA
#ifndef MRCV_CUDA_ENABLED
				if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
					network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
					network.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
					mrcv::writeLog("Бэкенд CUDA успешно инициализирован.");
				}
				else {
					mrcv::writeLog("CUDA недоступен, используется CPU.", mrcv::LOGTYPE::WARNING);
					network.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
					network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
				}
#else
				network.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
				network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
#endif 
			}
			catch (const cv::Exception& e) {
				mrcv::writeLog("Ошибка инициализации сети: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
				return -1;
			}
		}
		return err;
	}

	// Для подписей
	void mrcv::neuralNetSegmentator::letterBox(const cv::Mat& img, cv::Mat& out, cv::Vec4d& params,
		cv::Size& newShape, bool autoShape = false, bool scaleFill = false,
		bool scaleUp = true, int stride = 32)
	{
		newShape = cv::Size(input_width, input_height);
		cv::Size shape = img.size();
		float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
		if (!scaleUp)
		{
			r = std::min(r, 1.0f);
		}

		float ratio[2] = { r, r };
		int new_un_pad[2] = { (int)std::round((float)shape.width * r),
							 (int)std::round((float)shape.height * r) };
		auto dw = (float)(newShape.width - new_un_pad[0]);
		auto dh = (float)(newShape.height - new_un_pad[1]);

		if (autoShape)
		{
			dw = (float)((int)dw % stride);
			dh = (float)((int)dh % stride);
		}
		else if (scaleFill)
		{
			dw = 0.0f;
			dh = 0.0f;
			new_un_pad[0] = newShape.width;
			new_un_pad[1] = newShape.height;
			ratio[0] = (float)newShape.width / (float)shape.width;
			ratio[1] = (float)newShape.height / (float)shape.height;
		}

		if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
		{
			cv::resize(img, out, cv::Size(new_un_pad[0], new_un_pad[1]));
		}
		else {
			out = img.clone();
		}
		dw /= 2.0f;
		dh /= 2.0f;
		int top = int(std::round(dh - 0.1f));
		int bottom = int(std::round(dh + 0.1f));
		int left = int(std::round(dw - 0.1f));
		int right = int(std::round(dw + 0.1f));
		params[0] = ratio[0];
		params[1] = ratio[1];
		params[2] = left;
		params[3] = top;
		cv::copyMakeBorder(out, out, top, bottom, left, right, cv::BORDER_CONSTANT, BLACK);
	}

	std::vector<cv::Mat> mrcv::neuralNetSegmentator::preProcess(cv::Mat& img, cv::Vec4d& params)
	{
		cv::Mat input;
		cv::Mat blob;
		cv::Size newSize = cv::Size(input_width, input_height);
		letterBox(img, input, params, newSize);
		cv::dnn::blobFromImage(input, blob, 1.0 / 255.0, cv::Size(input_width, input_height),
			cv::Scalar(), true, false);
		network.setInput(blob);
		std::vector<std::string> output_layer_names{ "output0", "output1" };
		std::vector<cv::Mat> outputs;
		network.forward(outputs, output_layer_names);
		return outputs;
	}

	void mrcv::neuralNetSegmentator::getMask(const cv::Mat& mask_proposals, const cv::Mat& mask_protos,
		outputSegment& output, const maskParams& maskParams)
	{
		int seg_channels = maskParams.segChannels;
		int net_width = maskParams.netWidth;
		int seg_width = maskParams.segWidth;
		int net_height = maskParams.netHeight;
		int seg_height = maskParams.segHeight;
		float mask_threshold = maskParams.maskThreshold;
		cv::Vec4f params = maskParams.params;
		cv::Size src_img_shape = maskParams.srcImgShape;
		cv::Rect temp_rect = output.box;

		//crop from mask_protos
		int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
		int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
		int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
		int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

		rang_w = MAX(rang_w, 1);
		rang_h = MAX(rang_h, 1);
		if (rang_x + rang_w > seg_width)
		{
			if (seg_width - rang_x > 0)
			{
				rang_w = seg_width - rang_x;
			}
			else
			{
				rang_x -= 1;
			}
		}
		if (rang_y + rang_h > seg_height)
		{
			if (seg_height - rang_y > 0)
			{
				rang_h = seg_height - rang_y;
			}
			else {
				rang_y -= 1;
			}
		}

		std::vector<cv::Range> roi_rangs;
		roi_rangs.push_back(cv::Range(0, 1));
		roi_rangs.push_back(cv::Range::all());
		roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
		roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));
		//crop
		cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
		cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
		cv::Mat matmul_res = (mask_proposals * protos).t();
		cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
		cv::Mat dest, mask;
		//sigmoid
		cv::exp(-masks_feature, dest);
		dest = 1.0 / (1.0 + dest);

		int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
		int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
		int width = ceil(net_width / seg_width * rang_w / params[0]);
		int height = ceil(net_height / seg_height * rang_h / params[1]);

		cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
		mask = mask(temp_rect - cv::Point(left, top)) > mask_threshold;
		output.boxMask = mask;
	}

	// Прорисовка рамки выделяющей объеект.
	void mrcv::neuralNetSegmentator::drawLabel(cv::Mat& img, std::string label, int left, int top)
	{
		// Отображение надписи в верхней части ограничивающего прямоугольника
		int baseline;
		cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
			FONT_SCALE, THICKNESS, &baseline);
		top = std::max(top, label_size.height);
		// Верхний левый угол
		cv::Point tlc = cv::Point(left, top);
		// Нижний правый угол
		cv::Point brc = cv::Point(left + label_size.width,
			top + label_size.height + baseline);
		// Прорисовка прямоугольника
		cv::rectangle(img, tlc, brc, BLACK, cv::FILLED);
		// Нанесение текста в прямоугольник
		cv::putText(img, label, cv::Point(left, top + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS);
	}

	void mrcv::neuralNetSegmentator::drawResult(cv::Mat& img, std::vector<outputSegment> result, std::vector<std::string> class_name)
	{
		cv::Mat mask = img.clone();
		for (int i = 0; i < (int)result.size(); i++)
		{
			cv::rectangle(img, result[i].box, GREEN, 3 * THICKNESS);
			mask(result[i].box).setTo(masksColorsSet[result[i].id], result[i].boxMask);
			std::string label = cv::format("%.2f", result[i].confidence);
			label = class_name[result[i].id] + ": " + label;
			int left = result[i].box.x;
			int top = result[i].box.y;
			drawLabel(img, label, left, top);
		}
		cv::addWeighted(img, 0.5, mask, 0.5, 0, img);
	}

	cv::Mat mrcv::neuralNetSegmentator::postProcess(cv::Mat& img, std::vector<cv::Mat>& outputs,
		const std::vector<std::string>& class_name, cv::Vec4d& params)
	{
		classesIDSet.clear();
		confidencesSet.clear();
		boxesSet.clear();
		classesSet.clear();
		masksSet.clear();
		std::vector<int> class_ids;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		std::vector<std::vector<float>> picked_proposals;

		float* data = (float*)outputs[0].data;

		const int dimensions = class_name.size() + 5 + 32;
		const int rows = 25200;
		for (int i = 0; i < rows; ++i)
		{
			float confidence = data[4];
			if (confidence >= CONFIDENCE_THRESHOLD)
			{
				float* classes_scores = data + 5;
				cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
				cv::Point class_id;
				double max_class_score;
				cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
				if (max_class_score > SCORE_THRESHOLD) {
					float x = (data[0] - params[2]) / params[0];
					float y = (data[1] - params[3]) / params[1];
					float w = data[2] / params[0];
					float h = data[3] / params[1];
					int left = std::max(int(x - 0.5 * w), 0);
					int top = std::max(int(y - 0.5 * h), 0);
					int width = int(w);
					int height = int(h);
					boxes.push_back(cv::Rect(left, top, width, height));
					confidences.push_back(confidence);
					class_ids.push_back(class_id.x);

					std::vector<float> temp_proto(data + class_name.size() + 5, data + dimensions);
					picked_proposals.push_back(temp_proto);
				}
			}
			data += dimensions;
		}

		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
		std::vector<outputSegment> output;
		std::vector<std::vector<float>> temp_mask_proposals;
		cv::Rect holeImgRect(0, 0, img.cols, img.rows);
		for (int i = 0; i < (int)indices.size(); ++i)
		{
			int idx = indices[i];
			cv::Rect box = boxes[idx];

			boxesSet.push_back(box);
			confidencesSet.push_back(confidences[idx]);
			classesIDSet.push_back(class_ids[idx]);
			classesSet.push_back(class_name[class_ids[idx]]);

			outputSegment result;
			result.id = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx] & holeImgRect;
			temp_mask_proposals.push_back(picked_proposals[idx]);
			output.push_back(result);
		}

		maskParams mask_params;
		mask_params.params = params;
		mask_params.srcImgShape = img.size();
		for (int i = 0; i < (int)temp_mask_proposals.size(); ++i)
		{
			getMask(cv::Mat(temp_mask_proposals[i]).t(), outputs[1], output[i], mask_params);
			// =================
			// добавлено для передачи полноразмерной маски
			cv::Size shape = img.size();
			cv::Mat canvas = cv::Mat::zeros(shape, CV_8UC1);
			output.at(i).boxMask.copyTo(canvas(cv::Rect(output[i].box)));
			masksSet.push_back(canvas);
			// =================
		}
		drawResult(img, output, class_name);
		return img;
	}

	// Обрабокта
	cv::Mat mrcv::neuralNetSegmentator::process(cv::Mat& img)
	{
		cv::Mat input = img.clone();
		std::vector<cv::Mat> detections;
		cv::Vec4d params;
		detections = preProcess(input, params);
		cv::Mat res = postProcess(input, detections, neuralNetSegmentator::classes, params);
		// Информация об эффективности.
		std::vector<double> layersTimes;
		double freq = cv::getTickFrequency();
		neuralNetSegmentator::timeInference = network.getPerfProfile(layersTimes) / freq;

		processedImage = res;
		return res;
	}

	cv::Mat mrcv::neuralNetSegmentator::getImage()
	{
		return processedImage;
	}

	std::vector<cv::Mat> mrcv::neuralNetSegmentator::getMasks()
	{
		return masksSet;
	}

	std::vector<int> mrcv::neuralNetSegmentator::getClassIDs()
	{
		return classesIDSet;
	}

	std::vector<float> mrcv::neuralNetSegmentator::getConfidences()
	{
		return confidencesSet;
	}

	std::vector<cv::Rect> mrcv::neuralNetSegmentator::getBoxes()
	{
		return boxesSet;
	}

	std::vector<std::string> mrcv::neuralNetSegmentator::getClasses()
	{
		return classesSet;
	}

	float mrcv::neuralNetSegmentator::getInference()
	{
		return timeInference;
	}


	/**
		 * @brief detectObjectPrimitives
		 * @brief Функции определения информации о параметрах и форме идентифицированного объекта (примитивов объектов)
		 * @param points3D                 - входные и исходные данные для хранения информации о облаке 3D точек
		 * @param primitive                - входные и исходные данные для хранения информации о параметрах и форме идентифицированного объекта (примитивов объектов)
		 * @param numberSelectedSegment    - номер сегмента (идентифицированного объекта) в облаке 3D тосчек
		 * @param coefFilterSigma          - коэвициент кратности с.к.о. для фильтра отсеивания выбрасов (по умолчанию 2.5*sigma)
		 * @return - код результата работы функции. 0 - Success; 2 - пустое облако 3D; 3 - Нет точек в сегменте; 4 - Не верный номер сигмента;
		 * 5 - Недостаточно точек в сегменте; -1 - Неизвестная ошибка.
		 */
	int mrcv::detectObjectPrimitives(mrcv::pointsData& points3D, mrcv::primitiveData& primitive, int numberSelectedSegment, double coefFilterSigma)
	{
		try
		{
			int state; // для ошибок функций

			// ////////////////////
			// Проверка полноты данных
			// ////////////////////
			if (points3D.numPoints < 1)
			{
				return 2; // 2- Нет точек
			}

			if (points3D.numPointsInSegments.at(numberSelectedSegment) < 1)
			{
				return 3; // 3 - Нет точек в сегменте
			}

			if (numberSelectedSegment < 0 || numberSelectedSegment > int(points3D.numPointsInSegments.size() - 1))
			{
				return 4; // 4 - Не верный номер сигмента
			}
			// ////////////////////

			std::vector<std::vector<double>> figureModel;
			std::vector<cv::Point2d> vu;                  // (для конвертации) все 2D точки из облака (на изображении)
			std::vector<cv::Point3d> xyz;                 // (для конвертации) все точеки 3D из облака 3D точек
			std::vector<cv::Vec3b> rgb;                   // (для конвертации) цвет точки из облака в RGB
			std::vector<cv::Point2d> segmentPoints2Dvu;  //  все 2D точки сегмента
			std::vector<cv::Point3d> segmentPoints3Dxyz; //  все точеки 3D точек сегмента
			std::vector<cv::Vec3b> segmentPointsRGB;     // цвет точки сегмента в RGB
			vu.resize(points3D.numPoints);
			xyz.resize(points3D.numPoints);
			rgb.resize(points3D.numPoints);

			// Используемые переменные
			std::vector<int> INDEX = points3D.segment;
			int numClusters = points3D.numSegments;

			//
			for (int qi = 0; qi < points3D.numPoints; qi++)
			{
				vu.at(qi).x = double(points3D.vu.at(qi).at(1));
				vu.at(qi).y = double(points3D.vu.at(qi).at(0));

				xyz.at(qi).x = points3D.xyz.at(qi).at(0);
				xyz.at(qi).y = points3D.xyz.at(qi).at(1);
				xyz.at(qi).z = points3D.xyz.at(qi).at(2);

				rgb.at(qi)[0] = (uchar)points3D.rgb.at(qi).at(0);
				rgb.at(qi)[1] = (uchar)points3D.rgb.at(qi).at(1);
				rgb.at(qi)[2] = (uchar)points3D.rgb.at(qi).at(2);
			}

			for (int qp = 0; qp < int(INDEX.size()); qp++)
			{
				if (INDEX[qp] == numberSelectedSegment)
				{
					segmentPoints2Dvu.push_back(vu[qp]);
					segmentPoints3Dxyz.push_back(xyz[qp]);
					segmentPointsRGB.push_back(rgb[qp]);
				}
			}

			// ////////////////////
			// Фильтрация выбросов
			// ////////////////////
			int numSegmentPoints = segmentPoints3Dxyz.size();
			cv::Mat R = cv::Mat::zeros(numSegmentPoints, numSegmentPoints, CV_64FC1); // матрица расстояний

			// Рассчёт матрицы расстояний
			for (int qi = 0; qi < R.cols; qi++)
			{
				for (int qj = 0; qj < R.rows; qj++)
				{
					double dx = segmentPoints3Dxyz.at(qi).x - segmentPoints3Dxyz.at(qj).x;
					double dy = segmentPoints3Dxyz.at(qi).y - segmentPoints3Dxyz.at(qj).y;
					double dz = segmentPoints3Dxyz.at(qi).z - segmentPoints3Dxyz.at(qj).z;

					R.at<double>(qi, qj) = std::sqrt(std::pow(dx, 2) + std::pow(dy, 2) + std::pow(dz, 2));
				}
			}

			std::vector<double> meanPointDistance(numSegmentPoints, 0);
			double mean2PointDistance;
			double stdMeanPointDistance;

			segmentPoints3Dxyz.clear(); // очиска данных о количестветочек в сегменте

			// Рассчет среднего расстояния для каждой точки
			for (int qj = 0; qj < R.cols; qj++)
			{
				double sum = 0;
				for (int qi = 0; qi < R.cols; qi++)
				{
					sum = sum + R.at<double>(qi, qj);
				}
				meanPointDistance.at(qj) = sum / R.rows;
			}
			// Рассчет среднего расстояния
			double sum = 0;
			for (int qi = 0; qi < R.cols; qi++)
			{
				sum = sum + meanPointDistance.at(qi);
			}
			mean2PointDistance = sum / R.cols;

			// Рассчет с.к.о по среднему расстоянию для каждой точки
			double sumD = 0;
			for (int qi = 0; qi < R.cols; qi++)
			{
				sumD = sumD + std::pow((meanPointDistance.at(qi) - mean2PointDistance), 2);
			}
			stdMeanPointDistance = std::sqrt(sumD / R.cols);

			// Фильтрация
			for (int qsp = 0; qsp < int(numSegmentPoints); qsp++)
			{
				if (meanPointDistance.at(qsp) < (coefFilterSigma * stdMeanPointDistance + mean2PointDistance))
				{
					primitive.segmentPoints2Dvu.push_back(segmentPoints2Dvu[qsp]);
					primitive.segmentPoints3Dxyz.push_back(segmentPoints3Dxyz[qsp]);
					primitive.segmentPointsRGB.push_back(segmentPointsRGB[qsp]);
				}
			}
			primitive.numPointsInSegment = primitive.segmentPoints3Dxyz.size();

			// ////////////////////
			// Определение модели объекта
			// ////////////////////
			primitive.primitiveType = -1;
			primitive.primitiveTypeName = "none";
			primitive.primitiveParameter = { 0, 0, 0, 0, 0 };
			primitive.primitivePoints = {};

			if (points3D.numPointsInSegments.at(numberSelectedSegment) < 5 || points3D.xyz0.size() < 5 || points3D.xyz.size() < 5)
			{
				return 5; // 5 - Недостаточно точек в сегменте
			}


			// Fit plane
			std::vector<cv::Point3d> planePoints;
			std::vector<double> PlaneModel = mrcv::FitPlane(primitive.segmentPoints3Dxyz, planePoints);
			// Fit sphere
			std::vector<double> SphereModel = mrcv::FitSphere(primitive.segmentPoints3Dxyz);
			//Fit cylinder
			std::vector<cv::Point3d> cylinderPoints;
			std::vector<double> CylinderModel = mrcv::FitCylinder(primitive.segmentPoints3Dxyz, PlaneModel, cylinderPoints);

			// ////////////////////
			// Определение вида примитива
			// ////////////////////
			int figureInClaster = 0;
			double minErrorOfModel = 0.0;
			if (PlaneModel[PlaneModel.size() - 1] < SphereModel[SphereModel.size() - 1])
			{
				minErrorOfModel = PlaneModel[PlaneModel.size() - 1];
				figureInClaster = 1;
			}
			else
			{
				minErrorOfModel = SphereModel[SphereModel.size() - 1];
				figureInClaster = 2;
			}

			if (CylinderModel[CylinderModel.size() - 1] < minErrorOfModel)
			{
				minErrorOfModel = CylinderModel[CylinderModel.size() - 1];
				figureInClaster = 3;
			}

			double l1, l2, lMin;
			l1 = sqrt(pow((planePoints[0].x - planePoints[1].x), 2) + pow((planePoints[0].y - planePoints[1].y), 2) + pow((planePoints[0].z - planePoints[1].z), 2));
			l2 = sqrt(pow((planePoints[0].x - planePoints[2].x), 2) + pow((planePoints[0].y - planePoints[2].y), 2) + pow((planePoints[0].z - planePoints[2].z), 2));

			if (l1 < l2)
				lMin = l1;
			else
				lMin = l2;

			switch (figureInClaster)
			{
			case 0:
			{
				primitive.primitiveType = 0;
				primitive.primitiveTypeName = "none";
				primitive.primitiveParameter = { 0, 0, 0, 0, 0 };
				primitive.primitivePoints = {};
				break;
			}
			case 1: // Плоскость
			{
				primitive.primitiveType = 1;
				primitive.primitiveTypeName = "Plane";
				primitive.primitiveParameter = PlaneModel;
				primitive.primitivePoints = planePoints;
				break;
			}
			case 2: // Сфера
			{
				if (SphereModel[3] < lMin)
				{
					primitive.primitiveType = 2;
					primitive.primitiveTypeName = "Sphere";
					primitive.primitiveParameter = SphereModel;
					primitive.primitivePoints = {};
				}
				else
				{
					primitive.primitiveType = 1;
					primitive.primitiveTypeName = "Plane";
					primitive.primitiveParameter = PlaneModel;
					primitive.primitivePoints = planePoints;

				}
				break;
			}
			case 3:  // Цилиндр
			{
				if (CylinderModel[6] < lMin)
				{
					primitive.primitiveType = 3;
					primitive.primitiveTypeName = "Cylinder";
					primitive.primitiveParameter = CylinderModel;
					primitive.primitivePoints = cylinderPoints;

				}
				else
				{
					primitive.primitiveType = 1;
					primitive.primitiveTypeName = "Plane";
					primitive.primitiveParameter = PlaneModel;
					primitive.primitivePoints = planePoints;
				}
			}
			}
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	/**
		 * @brief drawPrimitives
		 * @brief Функции прорисовки на изображение информации примитивов объектов на основе параметров и форме идентифицированных объектов
		 * @param inputImageCamera01Undistort  - выходное (преобразованное) камеры 01 цветное RGB изображение, формата cv::Mat CV_8UC3
		 * @param outputImages                 - структура данных для хранения цветных RGB изображений, формата cv::Mat CV_8UC3
		 *                                       о параметрах и форме идентифицированных объектов
		 * @param primitives                  -  вектор структур данных о данных для хранения информации о параметрах и форме
		 *                                       идентифицированного объекта (примитивов объектов)
		 * @return - код результата работы функции. 0 - Success; -1 - Неизвестная ошибка.
		 */
	int mrcv::drawPrimitives(cv::Mat& inputImageCamera01Undistort, outputPrimitivesImages& outputImages, std::vector<mrcv::primitiveData>& primitives)
	{
		try
		{
			// ////////////////////
			// Проверка на пустоту
			// ////////////////////

			if (inputImageCamera01Undistort.empty())
			{
				return 1; // 1 - пустое изображение
			}

			if (primitives.empty())
			{
				return 2; // 2- нет данных о примитивах
			}

			// ////////////////////
			// Объявление и инициализация параметров
			// ////////////////////
			std::vector<std::vector<double>> figureModel;

			cv::Size imgSize = inputImageCamera01Undistort.size();
			int h = imgSize.height;
			int w = imgSize.width;
			int ho = (h / 2) - 0.5;
			int wo = (w / 2) - 0.5;

			cv::Mat imageDrawPrimitives = cv::Mat::ones(imgSize, CV_8UC3);

			cv::Mat projectionXY = cv::Mat::ones(imgSize, CV_8UC3);
			projectionXY = cv::Scalar(255, 255, 255);
			cv::Mat projectionYZ = cv::Mat::ones(imgSize, CV_8UC3);
			projectionYZ = cv::Scalar(255, 255, 255);
			cv::Mat projectionXZ = cv::Mat::ones(imgSize, CV_8UC3);
			projectionXZ = cv::Scalar(255, 255, 255);

			// ////////////////////
			//
			// ////////////////////
			double xLimMin = 1e+8;
			double xLimMax = -1e+8;
			double yLimMin = 1e+8;
			double yLimMax = -1e+8;
			double zLimMin = 1e+8;
			double zLimMax = -1e+8;
			std::vector<cv::Point3d> xyz;

			for (int prim = 0; prim < int(primitives.size()); prim++)
			{
				xyz = primitives[prim].segmentPoints3Dxyz;

				for (int i = 0; i < int(xyz.size()); i++)
				{
					//if (IDX[i] != 0) {
					if (xyz[i].x < xLimMin)
						xLimMin = xyz[i].x;
					if (xyz[i].x > xLimMax)
						xLimMax = xyz[i].x;
					if (xyz[i].y < yLimMin)
						yLimMin = xyz[i].y;
					if (xyz[i].y > yLimMax)
						yLimMax = xyz[i].y;
					if (xyz[i].z < zLimMin)
						zLimMin = xyz[i].z;
					if (xyz[i].z > zLimMax)
						zLimMax = xyz[i].z;
					//}
				}
			}

			xLimMin = 5e+2;
			xLimMax = -5e+2;
			yLimMin = 5e+2;
			yLimMax = -5e+2;
			zLimMin = 1.2e+2;
			zLimMax = 0;

			// ////////////////////
			// Создание изображения с результатом (прорисовка сетки и др.)
			// ////////////////////

			double xyScale, yzScale, xzScale, xScale, yScale, zScale;
			xScale = std::min(abs((w / 2 - 50) / xLimMax), abs((w / 2 - 50) / xLimMin));
			yScale = std::min(abs((h / 2 - 50) / yLimMax), abs((h / 2 - 50) / yLimMin));
			xyScale = std::min(xScale, yScale);
			zScale = (w - 100) / zLimMax;
			yzScale = std::min(zScale, yScale);
			xScale = std::min(abs((h / 2 - 50) / xLimMax), abs((h / 2 - 50) / xLimMin));
			xzScale = std::min(xScale, zScale);

			cv::Rect rect(49, 49, (w - 100), (h - 100));
			rectangle(projectionXY, rect, cv::Scalar(0, 0, 0));
			rectangle(projectionYZ, rect, cv::Scalar(0, 0, 0));
			rectangle(projectionXZ, rect, cv::Scalar(0, 0, 0));

			cv::Point2d p1, p2;
			p1.x = 49; p1.y = ho;
			p2.x = (w - 51); p2.y = ho;
			line(projectionXY, p1, p2, cv::Scalar(200, 200, 200));
			p1.x = wo; p1.y = 49;
			p2.x = wo; p2.y = (h - 100);
			line(projectionXY, p1, p2, cv::Scalar(200, 200, 200));
			p1.x = 49; p1.y = ho;
			p2.x = (w - 100); p2.y = ho;
			line(projectionYZ, p1, p2, cv::Scalar(200, 200, 200));
			line(projectionXZ, p1, p2, cv::Scalar(200, 200, 200));
			char textZ[10];
			cv::Point2d textPoint;
			textPoint.x = 49;
			textPoint.y = 40;
			putText(projectionXY, "Y", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 0), 2);
			putText(projectionYZ, "Y", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 0), 2);
			putText(projectionXZ, "X", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 0), 2);
			textPoint.x = (w - 42);
			textPoint.y = (h - 51);
			putText(projectionXY, "X", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 0), 2);
			putText(projectionYZ, "Z", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 0), 2);
			putText(projectionXZ, "Z", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 0), 2);
			textPoint.x = 20;
			textPoint.y = (h / 2 - 10);
			putText(projectionXY, "0", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
			putText(projectionYZ, "0", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
			putText(projectionXZ, "0", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
			textPoint.x = 40;
			textPoint.y = (h - 20);
			putText(projectionYZ, "0", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
			putText(projectionXZ, "0", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
			textPoint.x = (h / 2 - 10);
			textPoint.y = (h - 20);
			putText(projectionXY, "0", textPoint, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));

			// ////////////////////
			//
			// ////////////////////
			cv::Mat res4;

			res4 = inputImageCamera01Undistort.clone();

			for (int k = 0; k < int(primitives.size()); k++)
			{
				if (primitives[k].primitiveType == 0 || primitives[k].numPointsInSegment < 5)
				{
					continue; // пропуск, если мало точек
				}

				std::vector<cv::Point3d> planePoints;
				std::vector<cv::Point3d> cylinderPoints;
				std::vector<double> PlaneModel;
				std::vector<double> SphereModel;
				std::vector<double> CylinderModel;

				cv::Point2d pcPoint;

				switch (primitives[k].primitiveType)
				{
					// ////////////////////
					// Fit plane
					// ////////////////////
				case 1:
				{
					PlaneModel = primitives[k].primitiveParameter;
					planePoints = primitives[k].primitivePoints;

					figureModel.push_back(PlaneModel);

					for (int i = 0; i < 6; i++)
					{
						p1.x = wo + planePoints[0].x * xyScale + 0.2 * i * (planePoints[1].x * xyScale - planePoints[0].x * xyScale);
						p1.y = ho + planePoints[0].y * xyScale + 0.2 * i * (planePoints[1].y * xyScale - planePoints[0].y * xyScale);
						p2.x = wo + planePoints[2].x * xyScale + 0.2 * i * (planePoints[3].x * xyScale - planePoints[2].x * xyScale);
						p2.y = ho + planePoints[2].y * xyScale + 0.2 * i * (planePoints[3].y * xyScale - planePoints[2].y * xyScale);
						line(projectionXY, p1, p2, cv::Scalar(50, 50, 50));
						p1.x = wo + planePoints[0].x * xyScale + 0.2 * i * (planePoints[2].x * xyScale - planePoints[0].x * xyScale);
						p1.y = ho + planePoints[0].y * xyScale + 0.2 * i * (planePoints[2].y * xyScale - planePoints[0].y * xyScale);
						p2.x = wo + planePoints[1].x * xyScale + 0.2 * i * (planePoints[3].x * xyScale - planePoints[1].x * xyScale);
						p2.y = ho + planePoints[1].y * xyScale + 0.2 * i * (planePoints[3].y * xyScale - planePoints[1].y * xyScale);
						line(projectionXY, p1, p2, cv::Scalar(50, 50, 50));
					}
					for (int i = 0; i < 6; i++)
					{
						p1.x = 49 + planePoints[0].z * yzScale + 0.2 * i * (planePoints[1].z * yzScale - planePoints[0].z * yzScale);
						p1.y = ho + planePoints[0].y * yzScale + 0.2 * i * (planePoints[1].y * yzScale - planePoints[0].y * yzScale);
						p2.x = 49 + planePoints[2].z * yzScale + 0.2 * i * (planePoints[3].z * yzScale - planePoints[2].z * yzScale);
						p2.y = ho + planePoints[2].y * yzScale + 0.2 * i * (planePoints[3].y * yzScale - planePoints[2].y * yzScale);
						line(projectionYZ, p1, p2, cv::Scalar(50, 50, 50));
						p1.x = 49 + planePoints[0].z * yzScale + 0.2 * i * (planePoints[2].z * yzScale - planePoints[0].z * yzScale);
						p1.y = ho + planePoints[0].y * yzScale + 0.2 * i * (planePoints[2].y * yzScale - planePoints[0].y * yzScale);
						p2.x = 49 + planePoints[1].z * yzScale + 0.2 * i * (planePoints[3].z * yzScale - planePoints[1].z * yzScale);
						p2.y = ho + planePoints[1].y * yzScale + 0.2 * i * (planePoints[3].y * yzScale - planePoints[1].y * yzScale);
						line(projectionYZ, p1, p2, cv::Scalar(50, 50, 50));
					}
					for (int i = 0; i < 6; i++)
					{
						p1.x = 49 + planePoints[0].z * xzScale + 0.2 * i * (planePoints[1].z * xzScale - planePoints[0].z * xzScale);
						p1.y = ho + planePoints[0].x * xzScale + 0.2 * i * (planePoints[1].x * xzScale - planePoints[0].x * xzScale);
						p2.x = 49 + planePoints[2].z * xzScale + 0.2 * i * (planePoints[3].z * xzScale - planePoints[2].z * xzScale);
						p2.y = ho + planePoints[2].x * xzScale + 0.2 * i * (planePoints[3].x * xzScale - planePoints[2].x * xzScale);
						line(projectionXZ, p1, p2, cv::Scalar(50, 50, 50));
						p1.x = 49 + planePoints[0].z * xzScale + 0.2 * i * (planePoints[2].z * xzScale - planePoints[0].z * xzScale);
						p1.y = ho + planePoints[0].x * xzScale + 0.2 * i * (planePoints[2].x * xzScale - planePoints[0].x * xzScale);
						p2.x = 49 + planePoints[1].z * xzScale + 0.2 * i * (planePoints[3].z * xzScale - planePoints[1].z * xzScale);
						p2.y = ho + planePoints[1].x * xzScale + 0.2 * i * (planePoints[3].x * xzScale - planePoints[1].x * xzScale);
						line(projectionXZ, p1, p2, cv::Scalar(50, 50, 50));
					}
					break;
				} // case 1

				// ////////////////////
				// Fit sphere
				// ////////////////////
				case 2:
				{
					SphereModel = primitives[k].primitiveParameter;

					figureModel.push_back(SphereModel);
					pcPoint.x = wo + SphereModel[0] * xyScale;
					pcPoint.y = ho + SphereModel[1] * xyScale;
					circle(projectionXY, pcPoint, SphereModel[3] * xyScale, cv::Scalar(50, 50, 50));
					double xLeft, xRight, yTop, yLeft, yRigth;
					yTop = pcPoint.y - SphereModel[3] * xyScale;
					for (int i = 0; i < 4; i++)
					{
						xLeft = pcPoint.x - sqrt(pow(SphereModel[3] * xyScale, 2) - pow(SphereModel[3] * xyScale - 0.5 * i * SphereModel[3] * xyScale, 2));
						xRight = pcPoint.x + sqrt(pow(SphereModel[3] * xyScale, 2) - pow(SphereModel[3] * xyScale - 0.5 * i * SphereModel[3] * xyScale, 2));
						yLeft = yRigth = yTop + 0.5 * i * SphereModel[3] * xyScale;
						p1.x = xLeft; p1.y = yLeft;
						p2.x = xRight; p2.y = yRigth;
						line(projectionXY, p1, p2, cv::Scalar(50, 50, 50));
						for (int j = 0; j < 3; j++)
						{
							p1.x = xLeft + (xRight - xLeft) * 0.25 * (j + 1);
							p1.y = yLeft;
							p2.x = pcPoint.x - sqrt(pow(SphereModel[3] * xyScale, 2) - pow(SphereModel[3] * xyScale - 0.5 * (i + 1) * SphereModel[3] * xyScale, 2)) + 2 * sqrt(pow(SphereModel[3] * xyScale, 2) - pow(SphereModel[3] * xyScale - 0.5 * (i + 1) * SphereModel[3] * xyScale, 2)) * 0.25 * (j + 1);
							p2.y = yTop + 0.5 * (i + 1) * SphereModel[3] * xyScale;
							line(projectionXY, p1, p2, cv::Scalar(50, 50, 50));
						}
					}

					pcPoint.x = 49 + SphereModel[2] * yzScale;
					pcPoint.y = ho + SphereModel[1] * yzScale;
					circle(projectionYZ, pcPoint, SphereModel[3] * yzScale, cv::Scalar(50, 50, 50));
					yTop = pcPoint.y - SphereModel[3] * yzScale;
					for (int i = 0; i < 4; i++)
					{
						xLeft = pcPoint.x - sqrt(pow(SphereModel[3] * yzScale, 2) - pow(SphereModel[3] * yzScale - 0.5 * i * SphereModel[3] * yzScale, 2));
						xRight = pcPoint.x + sqrt(pow(SphereModel[3] * yzScale, 2) - pow(SphereModel[3] * yzScale - 0.5 * i * SphereModel[3] * yzScale, 2));
						yLeft = yRigth = yTop + 0.5 * i * SphereModel[3] * yzScale;
						p1.x = xLeft; p1.y = yLeft;
						p2.x = xRight; p2.y = yRigth;
						line(projectionYZ, p1, p2, cv::Scalar(50, 50, 50));
						for (int j = 0; j < 3; j++)
						{
							p1.x = xLeft + (xRight - xLeft) * 0.25 * (j + 1);
							p1.y = yLeft;
							p2.x = pcPoint.x - sqrt(pow(SphereModel[3] * yzScale, 2) - pow(SphereModel[3] * yzScale - 0.5 * (i + 1) * SphereModel[3] * yzScale, 2)) + 2 * sqrt(pow(SphereModel[3] * yzScale, 2) - pow(SphereModel[3] * yzScale - 0.5 * (i + 1) * SphereModel[3] * yzScale, 2)) * 0.25 * (j + 1);
							p2.y = yTop + 0.5 * (i + 1) * SphereModel[3] * yzScale;
							line(projectionYZ, p1, p2, cv::Scalar(50, 50, 50));
						}
					}

					pcPoint.x = 49 + SphereModel[2] * xzScale;
					pcPoint.y = ho + SphereModel[0] * xzScale;
					circle(projectionXZ, pcPoint, SphereModel[3] * xzScale, cv::Scalar(50, 50, 50));
					circle(projectionXZ, pcPoint, 0.5 * SphereModel[3] * xzScale, cv::Scalar(50, 50, 50));
					for (int i = 0; i < 4; i++)
					{
						p1.x = pcPoint.x + SphereModel[3] * xzScale * cos(i * CV_PI / 4);
						p1.y = pcPoint.y + SphereModel[3] * xzScale * sin(i * CV_PI / 4);
						p2.x = pcPoint.x + SphereModel[3] * xzScale * cos(i * CV_PI / 4 + CV_PI);
						p2.y = pcPoint.y + SphereModel[3] * xzScale * sin(i * CV_PI / 4 + CV_PI);
						line(projectionXZ, p1, p2, cv::Scalar(50, 50, 50));
					}
					break;
				} // case 2

				// ////////////////////
				// Fit cylinder
				// ////////////////////
				case 3:
				{
					CylinderModel = primitives[k].primitiveParameter;
					cylinderPoints = primitives[k].primitivePoints;

					figureModel.push_back(CylinderModel);
					int nearCenter, farCenter;
					if (CylinderModel[2] < CylinderModel[5])
					{
						nearCenter = 0;
						farCenter = 1;
					}
					else
					{
						nearCenter = 1;
						farCenter = 0;
					}
					for (int i = 0; i < 10; i++)
					{
						p1.x = wo + cylinderPoints[2 * i + nearCenter].x * xyScale;
						p1.y = ho + cylinderPoints[2 * i + nearCenter].y * xyScale;
						p2.x = wo + cylinderPoints[2 * (i + 1) + nearCenter].x * xyScale;
						p2.y = ho + cylinderPoints[2 * (i + 1) + nearCenter].y * xyScale;
						line(projectionXY, p1, p2, cv::Scalar(50, 50, 50));
						if ((cylinderPoints[2 * i + farCenter].z < CylinderModel[2 + 3 * farCenter])
							|| (cylinderPoints[2 * (i + 1) + farCenter].z < CylinderModel[2 + 3 * farCenter]))
						{
							p1.x = wo + cylinderPoints[2 * i + farCenter].x * xyScale;
							p1.y = ho + cylinderPoints[2 * i + farCenter].y * xyScale;
							p2.x = wo + cylinderPoints[2 * (i + 1) + farCenter].x * xyScale;
							p2.y = ho + cylinderPoints[2 * (i + 1) + farCenter].y * xyScale;
							line(projectionXY, p1, p2, cv::Scalar(50, 50, 50));
							p1.x = wo + cylinderPoints[2 * i + farCenter].x * xyScale;
							p1.y = ho + cylinderPoints[2 * i + farCenter].y * xyScale;
							p2.x = wo + cylinderPoints[2 * i + nearCenter].x * xyScale;
							p2.y = ho + cylinderPoints[2 * i + nearCenter].y * xyScale;
							line(projectionXY, p1, p2, cv::Scalar(50, 50, 50));
							p1.x = wo + cylinderPoints[2 * (i + 1) + farCenter].x * xyScale;
							p1.y = ho + cylinderPoints[2 * (i + 1) + farCenter].y * xyScale;
							p2.x = wo + cylinderPoints[2 * (i + 1) + nearCenter].x * xyScale;
							p2.y = ho + cylinderPoints[2 * (i + 1) + nearCenter].y * xyScale;
							line(projectionXY, p1, p2, cv::Scalar(50, 50, 50));
						}
					}
					if (CylinderModel[0] < CylinderModel[3])
					{
						nearCenter = 1;
						farCenter = 0;
					}
					else {
						nearCenter = 0;
						farCenter = 1;
					}
					for (int i = 0; i < 10; i++)
					{
						p1.x = 49 + cylinderPoints[2 * i + nearCenter].z * yzScale;
						p1.y = ho + cylinderPoints[2 * i + nearCenter].y * yzScale;
						p2.x = 49 + cylinderPoints[2 * (i + 1) + nearCenter].z * yzScale;
						p2.y = ho + cylinderPoints[2 * (i + 1) + nearCenter].y * yzScale;
						line(projectionYZ, p1, p2, cv::Scalar(50, 50, 50));
						if ((cylinderPoints[2 * i + farCenter].x > CylinderModel[3 * farCenter])
							|| (cylinderPoints[2 * (i + 1) + farCenter].x > CylinderModel[3 * farCenter]))
						{
							p1.x = 49 + cylinderPoints[2 * i + farCenter].z * yzScale;
							p1.y = ho + cylinderPoints[2 * i + farCenter].y * yzScale;
							p2.x = 49 + cylinderPoints[2 * (i + 1) + farCenter].z * yzScale;
							p2.y = ho + cylinderPoints[2 * (i + 1) + farCenter].y * yzScale;
							line(projectionYZ, p1, p2, cv::Scalar(50, 50, 50));
							p1.x = 49 + cylinderPoints[2 * i + farCenter].z * yzScale;
							p1.y = ho + cylinderPoints[2 * i + farCenter].y * yzScale;
							p2.x = 49 + cylinderPoints[2 * i + nearCenter].z * yzScale;
							p2.y = ho + cylinderPoints[2 * i + nearCenter].y * yzScale;
							line(projectionYZ, p1, p2, cv::Scalar(50, 50, 50));
							p1.x = 49 + cylinderPoints[2 * (i + 1) + farCenter].z * yzScale;
							p1.y = ho + cylinderPoints[2 * (i + 1) + farCenter].y * yzScale;
							p2.x = 49 + cylinderPoints[2 * (i + 1) + nearCenter].z * yzScale;
							p2.y = ho + cylinderPoints[2 * (i + 1) + nearCenter].y * yzScale;
							line(projectionYZ, p1, p2, cv::Scalar(50, 50, 50));
						}
					}
					if (CylinderModel[1] < CylinderModel[4]) {
						nearCenter = 0;
						farCenter = 1;
					}
					else {
						nearCenter = 1;
						farCenter = 0;
					}
					for (int i = 0; i < 10; i++)
					{
						p1.x = 49 + cylinderPoints[2 * i + nearCenter].z * xzScale;
						p1.y = ho + cylinderPoints[2 * i + nearCenter].x * xzScale;
						p2.x = 49 + cylinderPoints[2 * (i + 1) + nearCenter].z * xzScale;
						p2.y = ho + cylinderPoints[2 * (i + 1) + nearCenter].x * xzScale;
						line(projectionXZ, p1, p2, cv::Scalar(50, 50, 50));
						if ((cylinderPoints[2 * i + farCenter].y < CylinderModel[1 + 3 * farCenter])
							|| (cylinderPoints[2 * (i + 1) + farCenter].y < CylinderModel[1 + 3 * farCenter]))
						{
							p1.x = 49 + cylinderPoints[2 * i + farCenter].z * xzScale;
							p1.y = ho + cylinderPoints[2 * i + farCenter].x * xzScale;
							p2.x = 49 + cylinderPoints[2 * (i + 1) + farCenter].z * xzScale;
							p2.y = ho + cylinderPoints[2 * (i + 1) + farCenter].x * xzScale;
							line(projectionXZ, p1, p2, cv::Scalar(50, 50, 50));
							p1.x = 49 + cylinderPoints[2 * i + farCenter].z * xzScale;
							p1.y = ho + cylinderPoints[2 * i + farCenter].x * xzScale;
							p2.x = 49 + cylinderPoints[2 * i + nearCenter].z * xzScale;
							p2.y = ho + cylinderPoints[2 * i + nearCenter].x * xzScale;
							line(projectionXZ, p1, p2, cv::Scalar(50, 50, 50));
							p1.x = 49 + cylinderPoints[2 * (i + 1) + farCenter].z * xzScale;
							p1.y = ho + cylinderPoints[2 * (i + 1) + farCenter].x * xzScale;
							p2.x = 49 + cylinderPoints[2 * (i + 1) + nearCenter].z * xzScale;
							p2.y = ho + cylinderPoints[2 * (i + 1) + nearCenter].x * xzScale;
							line(projectionXZ, p1, p2, cv::Scalar(50, 50, 50));
						}
					}
					break;
				}// case 3
				} // switch
			} // for

			// ////////////////////////////////////
			// Вывод данных
			// ////////////////////////////////////
			outputImages.outputImageProjectionXY = projectionXY;
			outputImages.outputImageProjectionYZ = projectionYZ;
			outputImages.outputImageProjectionXZ = projectionXZ;
			// ////////////////////////////////////

			resize(projectionXY, projectionXY, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
			resize(projectionYZ, projectionYZ, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
			resize(projectionXZ, projectionXZ, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);

			cv::Rect r1(0, 0, w / 2, h / 2);
			cv::Rect r2(w / 2, 0, w / 2, h / 2);
			cv::Rect r3(0, h / 2, w / 2, h / 2);
			cv::Rect r4(w / 2, h / 2, w / 2, h / 2);
			resize(res4, res4, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
			res4.copyTo(imageDrawPrimitives(r1));
			projectionXY.copyTo(imageDrawPrimitives(r2));
			projectionYZ.copyTo(imageDrawPrimitives(r3));
			projectionXZ.copyTo(imageDrawPrimitives(r4));
			p1.x = 0; p1.y = (h / 2) - 0.5;
			p2.x = (w - 1); p2.y = (h / 2) - 0.5;
			line(imageDrawPrimitives, p1, p2, cv::Scalar(0, 0, 0), 2);
			p1.x = (w / 2) - 0.5; p1.y = 0;
			p2.x = (w / 2) - 0.5; p2.y = (h - 1);
			line(imageDrawPrimitives, p1, p2, cv::Scalar(0, 0, 0), 2);
			cv::Rect rectF(0, 0, w, h);
			rectangle(imageDrawPrimitives, rectF, cv::Scalar(0, 0, 0), 2);

			// ////////////////////////////////////
			// Вывод данных
			// ////////////////////////////////////
			// ////////////////////////////////////
			outputImages.outputImageGeneralProjection = imageDrawPrimitives;
			// ////////////////////////////////////
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}
		return 0; // SUCCESS
	}

	/**
		 * @brief FitPlane
		 * @brief Функции определения примитива типа плоскость
		 * @param clastersData      -  координаты 3D точек сегмента (объекта)
		 * @param planePoints       -
		 * @return - std::vector<double> PlaneModel
		 */
	std::vector<double> mrcv::FitPlane(std::vector<cv::Point3d> clastersData, std::vector<cv::Point3d>& planePoints)
	{

		std::vector<double> abd(4, 0);   // z = ax + by + c: [a, b, c, err]
		int MatSize = int(clastersData.size());
		cv::Mat xyzClaster = cv::Mat::zeros(3, MatSize, CV_64F);

		for (int i = 0; i < MatSize; i++)
		{
			xyzClaster.at<double>(0, i) = clastersData[i].x;
			xyzClaster.at<double>(1, i) = clastersData[i].y;
			xyzClaster.at<double>(2, i) = clastersData[i].z;
		}

		cv::Mat A = cv::Mat::zeros(MatSize, 3, CV_64F);
		cv::Mat b = cv::Mat::zeros(MatSize, 1, CV_64F);
		cv::Mat resX = cv::Mat::zeros(3, 1, CV_64F);
		for (int i = 0; i < MatSize; i++)
		{
			A.at<double>(i, 0) = xyzClaster.at<double>(0, i);
			A.at<double>(i, 1) = xyzClaster.at<double>(1, i);
			A.at<double>(i, 2) = 1;
			b.at<double>(i) = xyzClaster.at<double>(2, i);
		}
		resX = mrcv::MNK(A, b);

		abd[0] = resX.at<double>(0);
		abd[1] = resX.at<double>(1);
		abd[2] = resX.at<double>(2);

		double yRad = atan2(abd[0], 1.0);
		cv::Mat Ry = cv::Mat::eye(3, 3, CV_64F);
		Ry.at<double>(0, 0) = cos(yRad);
		Ry.at<double>(0, 2) = sin(yRad);
		Ry.at<double>(2, 0) = -sin(yRad);
		Ry.at<double>(2, 2) = cos(yRad);
		cv::Mat planeRotated0 = Ry * xyzClaster;

		for (int i = 0; i < MatSize; i++)
		{
			A.at<double>(i, 0) = planeRotated0.at<double>(0, i);
			A.at<double>(i, 1) = planeRotated0.at<double>(1, i);
			A.at<double>(i, 2) = 1;
			b.at<double>(i) = planeRotated0.at<double>(2, i);
		}
		resX = mrcv::MNK(A, b);

		double xRad = -atan2(resX.at<double>(1), 1.0);
		cv::Mat Rx = cv::Mat::eye(3, 3, CV_64F);
		Rx.at<double>(1, 1) = cos(xRad);
		Rx.at<double>(1, 2) = -sin(xRad);
		Rx.at<double>(2, 1) = sin(xRad);
		Rx.at<double>(2, 2) = cos(xRad);
		cv::Mat planeRotated1 = Rx * planeRotated0;
		double meanZ = 0.0;
		for (int i = 0; i < MatSize; i++)
			meanZ += planeRotated1.at<double>(2, i);
		meanZ /= MatSize;
		double meanErr = 0.0;
		for (int i = 0; i < MatSize; i++)
			meanErr += abs(planeRotated1.at<double>(2, i) - meanZ);
		meanErr /= MatSize;
		abd[3] = meanErr;

		double zRad = CV_PI / 360;

		cv::Mat Rz = cv::Mat::eye(3, 3, CV_64F);
		Rz.at<double>(0, 0) = cos(zRad);
		Rz.at<double>(0, 1) = -sin(zRad);
		Rz.at<double>(1, 0) = sin(zRad);
		Rz.at<double>(1, 1) = cos(zRad);
		std::vector<std::vector<double>> rotationResults;
		cv::Mat planeRotatedStep = planeRotated1.clone();
		for (int i = 0; i < 180; i++)
		{
			std::vector<double> rotationStep(6, 0);
			planeRotatedStep = Rz * planeRotatedStep;
			double xmin = mrcv::minMatElement(planeRotatedStep.row(0));
			double xmax = mrcv::maxMatElement(planeRotatedStep.row(0));
			double ymin = mrcv::minMatElement(planeRotatedStep.row(1));
			double ymax = mrcv::maxMatElement(planeRotatedStep.row(1));
			double stepSq = (xmax - xmin) * (ymax - ymin);
			rotationStep[0] = stepSq;
			rotationStep[1] = xmin;
			rotationStep[2] = xmax;
			rotationStep[3] = ymin;
			rotationStep[4] = ymax;
			double stepRad = zRad * (i + 1);
			rotationStep[5] = stepRad;
			rotationResults.push_back(rotationStep);
		}

		double errMin = 1e+8;
		int minErrPlaneIdx = 0;
		for (int i = 0; i < 180; i++)
			if (rotationResults[i][0] < errMin)
			{
				errMin = rotationResults[i][0];
				minErrPlaneIdx = i;
			}
		cv::Mat pPoints = cv::Mat::zeros(3, 4, CV_64F);
		for (int i = 0; i < 4; i++)
		{
			pPoints.at<double>(0, i) = rotationResults[minErrPlaneIdx][floor(i / 2) + 1];
			pPoints.at<double>(1, i) = rotationResults[minErrPlaneIdx][i % 2 + 3];
			pPoints.at<double>(2, i) = meanZ;
		}
		cv::Mat Rxb = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat Ryb = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat Rzb = cv::Mat::eye(3, 3, CV_64F);
		Rxb.at<double>(1, 1) = cos(-xRad);
		Rxb.at<double>(1, 2) = -sin(-xRad);
		Rxb.at<double>(2, 1) = sin(-xRad);
		Rxb.at<double>(2, 2) = cos(-xRad);
		Ryb.at<double>(0, 0) = cos(-yRad);
		Ryb.at<double>(0, 2) = sin(-yRad);
		Ryb.at<double>(2, 0) = -sin(-yRad);
		Ryb.at<double>(2, 2) = cos(-yRad);
		Rzb.at<double>(0, 0) = cos(-rotationResults[minErrPlaneIdx][5]);
		Rzb.at<double>(0, 1) = -sin(-rotationResults[minErrPlaneIdx][5]);
		Rzb.at<double>(1, 0) = sin(-rotationResults[minErrPlaneIdx][5]);
		Rzb.at<double>(1, 1) = cos(-rotationResults[minErrPlaneIdx][5]);
		pPoints = Ryb * Rxb * Rzb * pPoints;
		for (int i = 0; i < 4; i++)
		{
			cv::Point3d pp;
			pp.x = pPoints.at<double>(0, i);
			pp.y = pPoints.at<double>(1, i);
			pp.z = pPoints.at<double>(2, i);
			planePoints.push_back(pp);
		}
		return abd;
	}

	/**
		 * @brief FitSphere
		 * @brief Функции определения примитива типа шар
		 * @param clastersData      -  координаты 3D точек сегмента (объекта)
		 * @return -  std::vector<double> SphereModel
		 */
	std::vector<double> mrcv::FitSphere(std::vector<cv::Point3d> clastersData)
	{
		std::vector<double>  SphereModel(5, 0);  //[xc, yc, zc, radius, err]
		int MatSize = int(clastersData.size());
		cv::Mat xClaster = cv::Mat::zeros(MatSize, 1, CV_64F);
		cv::Mat yClaster = cv::Mat::zeros(MatSize, 1, CV_64F);
		cv::Mat zClaster = cv::Mat::zeros(MatSize, 1, CV_64F);
		double xMean = 0.0, yMean = 0.0, zMean = 0.0;
		for (int i = 0; i < MatSize; i++)
		{
			xClaster.at<double>(i) = clastersData[i].x;
			yClaster.at<double>(i) = clastersData[i].y;
			zClaster.at<double>(i) = clastersData[i].z;
			xMean += clastersData[i].x;
			yMean += clastersData[i].y;
			zMean += clastersData[i].z;
		}
		xMean /= double(MatSize);
		yMean /= double(MatSize);
		zMean /= double(MatSize);
		cv::Mat A = cv::Mat::zeros(3, 3, CV_64F);
		cv::Mat b = cv::Mat::zeros(3, 1, CV_64F);
		cv::Mat resX = cv::Mat::zeros(3, 1, CV_64F);
		for (int i = 0; i < MatSize; i++)
		{
			A.at<double>(0, 0) += xClaster.at<double>(i) * (xClaster.at<double>(i) - xMean) / MatSize;
			A.at<double>(0, 1) += xClaster.at<double>(i) * (yClaster.at<double>(i) - yMean) / MatSize;
			A.at<double>(0, 2) += xClaster.at<double>(i) * (zClaster.at<double>(i) - zMean) / MatSize;
			A.at<double>(1, 0) += yClaster.at<double>(i) * (xClaster.at<double>(i) - xMean) / MatSize;
			A.at<double>(1, 1) += yClaster.at<double>(i) * (yClaster.at<double>(i) - yMean) / MatSize;
			A.at<double>(1, 2) += yClaster.at<double>(i) * (zClaster.at<double>(i) - zMean) / MatSize;
			A.at<double>(2, 0) += zClaster.at<double>(i) * (xClaster.at<double>(i) - xMean) / MatSize;
			A.at<double>(2, 1) += zClaster.at<double>(i) * (yClaster.at<double>(i) - yMean) / MatSize;
			A.at<double>(2, 2) += zClaster.at<double>(i) * (zClaster.at<double>(i) - zMean) / MatSize;
			b.at<double>(0) += (pow(xClaster.at<double>(i), 2) + pow(yClaster.at<double>(i), 2) + pow(zClaster.at<double>(i), 2)) * (xClaster.at<double>(i) - xMean) / MatSize;
			b.at<double>(1) += (pow(xClaster.at<double>(i), 2) + pow(yClaster.at<double>(i), 2) + pow(zClaster.at<double>(i), 2)) * (yClaster.at<double>(i) - yMean) / MatSize;
			b.at<double>(2) += (pow(xClaster.at<double>(i), 2) + pow(yClaster.at<double>(i), 2) + pow(zClaster.at<double>(i), 2)) * (zClaster.at<double>(i) - zMean) / MatSize;
		}
		A *= 2;

		resX = mrcv::MNK(A, b);

		double Radius = 0.0;
		for (int i = 0; i < MatSize; i++)
			Radius += (pow((xClaster.at<double>(i) - resX.at<double>(0)), 2) + pow((yClaster.at<double>(i) - resX.at<double>(1)), 2) + pow((zClaster.at<double>(i) - resX.at<double>(2)), 2)) / MatSize;
		Radius = sqrt(Radius);

		SphereModel[0] = resX.at<double>(0);
		SphereModel[1] = resX.at<double>(1);
		SphereModel[2] = resX.at<double>(2);
		SphereModel[3] = Radius;

		double meanErr = 0.0;
		for (int i = 0; i < MatSize; i++)
			meanErr += abs(sqrt(pow((xClaster.at<double>(i) - resX.at<double>(0)), 2) + pow((yClaster.at<double>(i) - resX.at<double>(1)), 2) + pow((zClaster.at<double>(i) - resX.at<double>(2)), 2)) - Radius);
		meanErr /= MatSize;
		SphereModel[4] = meanErr;

		return SphereModel;
	}

	/**
		 * @brief FitCylinder
		 * @brief Функции определения примитива типа цилиндр
		 * @param clastersData      -  координаты 3D точек сегмента (объекта)
		 * @param planePoints       -
		 * @param cylinderPoints    -
		 * @return - std::vector<double> CylinderModel
		 */
	std::vector<double> mrcv::FitCylinder(std::vector< cv::Point3d> clastersData, std::vector<double> planeModel, std::vector<cv::Point3d>& cylinderPoints)
	{
		std::vector<double> CylinderModel(8, 0.0);    //[xc1, yc1, zc1, xc2, yc2, zc2, radius, err]
		int MatSize = int(clastersData.size());
		cv::Mat xyzClaster = cv::Mat::zeros(3, MatSize, CV_64F);
		for (int i = 0; i < MatSize; i++)
		{
			xyzClaster.at<double>(0, i) = clastersData[i].x;
			xyzClaster.at<double>(1, i) = clastersData[i].y;
			xyzClaster.at<double>(2, i) = clastersData[i].z;
		}
		double yRad = atan2(planeModel[0], 1.0);
		cv::Mat Ry = cv::Mat::eye(3, 3, CV_64F);
		Ry.at<double>(0, 0) = cos(yRad);
		Ry.at<double>(0, 2) = sin(yRad);
		Ry.at<double>(2, 0) = -sin(yRad);
		Ry.at<double>(2, 2) = cos(yRad);
		cv::Mat cylinderRotated0 = Ry * xyzClaster;

		std::vector<cv::Point3d> planeData;
		cv::Point3d pointXYZ;
		for (int i = 0; i < MatSize; i++)
		{
			pointXYZ.x = cylinderRotated0.at<double>(0, i);
			pointXYZ.y = cylinderRotated0.at<double>(1, i);
			pointXYZ.z = cylinderRotated0.at<double>(2, i);
			planeData.push_back(pointXYZ);
		}
		std::vector<cv::Point3d> planePoints;
		std::vector<double> planeModel2 = FitPlane(planeData, planePoints);

		double xRad = -atan2(planeModel2[1], 1.0);
		cv::Mat Rx = cv::Mat::eye(3, 3, CV_64F);
		Rx.at<double>(1, 1) = cos(xRad);
		Rx.at<double>(1, 2) = -sin(xRad);
		Rx.at<double>(2, 1) = sin(xRad);
		Rx.at<double>(2, 2) = cos(xRad);
		cv::Mat cylinderRotated1 = Rx * cylinderRotated0;

		double zRad = CV_PI / 360;

		cv::Mat Rz = cv::Mat::eye(3, 3, CV_64F);
		Rz.at<double>(0, 0) = cos(zRad);
		Rz.at<double>(0, 1) = -sin(zRad);
		Rz.at<double>(1, 0) = sin(zRad);
		Rz.at<double>(1, 1) = cos(zRad);
		std::vector<std::vector<double>> rotationResults;
		cv::Mat cylinderRotatedStep = cylinderRotated1.clone();
		for (int i = 0; i < 360; i++)
		{
			std::vector<double> rotationStep(7, 0);
			cylinderRotatedStep = Rz * cylinderRotatedStep;
			cv::Mat circleKoef = cv::Mat::zeros(3, 1, CV_64F);
			cv::Mat Ac = cv::Mat::zeros(MatSize, 3, CV_64F);
			cv::Mat b = cv::Mat::zeros(MatSize, 1, CV_64F);
			for (int i = 0; i < MatSize; i++)
			{
				Ac.at<double>(i, 0) = cylinderRotatedStep.at<double>(0, i);
				Ac.at<double>(i, 1) = cylinderRotatedStep.at<double>(2, i);
				Ac.at<double>(i, 2) = 1;
				b.at<double>(i) = -(pow(cylinderRotatedStep.at<double>(0, i), 2) + pow(cylinderRotatedStep.at<double>(2, i), 2));
			}
			circleKoef = mrcv::MNK(Ac, b);
			double xc1 = -circleKoef.at<double>(0) / 2;
			double zc1 = -circleKoef.at<double>(1) / 2;
			double radius1 = sqrt((pow(xc1, 2) + pow(zc1, 2)) - circleKoef.at<double>(2));
			double ymin = mrcv::minMatElement(cylinderRotatedStep.row(1));
			double ymax = mrcv::maxMatElement(cylinderRotatedStep.row(1));
			double stepRad = zRad * (i + 1);
			double meanErr = 0.0;
			for (int i = 0; i < MatSize; i++)
				meanErr += abs(sqrt(pow((cylinderRotatedStep.at<double>(0, i) - xc1), 2) + pow((cylinderRotatedStep.at<double>(2, i) - zc1), 2)) - radius1);
			meanErr /= MatSize;
			rotationStep[0] = meanErr;
			rotationStep[1] = stepRad;
			rotationStep[2] = xc1;
			rotationStep[3] = zc1;
			rotationStep[4] = ymin;
			rotationStep[5] = ymax;
			rotationStep[6] = radius1;
			rotationResults.push_back(rotationStep);
		}
		double errMin = 1e+8;
		int minErrCylinderIdx = 0;
		for (int i = 0; i < 360; i++)
			if (rotationResults[i][0] < errMin)
			{
				errMin = rotationResults[i][0];
				minErrCylinderIdx = i;
			}
		cv::Mat osn1 = cv::Mat::zeros(3, 1, CV_64F);
		cv::Mat osn2 = cv::Mat::zeros(3, 1, CV_64F);
		osn1.at<double>(0) = rotationResults[minErrCylinderIdx][2];
		osn1.at<double>(1) = rotationResults[minErrCylinderIdx][4];
		osn1.at<double>(2) = rotationResults[minErrCylinderIdx][3];
		osn2.at<double>(0) = rotationResults[minErrCylinderIdx][2];
		osn2.at<double>(1) = rotationResults[minErrCylinderIdx][5];
		osn2.at<double>(2) = rotationResults[minErrCylinderIdx][3];

		cv::Mat Rxb = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat Ryb = cv::Mat::eye(3, 3, CV_64F);
		cv::Mat Rzb = cv::Mat::eye(3, 3, CV_64F);
		Rxb.at<double>(1, 1) = cos(-xRad);
		Rxb.at<double>(1, 2) = -sin(-xRad);
		Rxb.at<double>(2, 1) = sin(-xRad);
		Rxb.at<double>(2, 2) = cos(-xRad);
		Ryb.at<double>(0, 0) = cos(-yRad);
		Ryb.at<double>(0, 2) = sin(-yRad);
		Ryb.at<double>(2, 0) = -sin(-yRad);
		Ryb.at<double>(2, 2) = cos(-yRad);
		Rzb.at<double>(0, 0) = cos(-rotationResults[minErrCylinderIdx][1]);
		Rzb.at<double>(0, 1) = -sin(-rotationResults[minErrCylinderIdx][1]);
		Rzb.at<double>(1, 0) = sin(-rotationResults[minErrCylinderIdx][1]);
		Rzb.at<double>(1, 1) = cos(-rotationResults[minErrCylinderIdx][1]);
		osn1 = Ryb * Rxb * Rzb * osn1;
		osn2 = Ryb * Rxb * Rzb * osn2;
		CylinderModel[0] = osn1.at<double>(0);
		CylinderModel[1] = osn1.at<double>(1);
		CylinderModel[2] = osn1.at<double>(2);
		CylinderModel[3] = osn2.at<double>(0);
		CylinderModel[4] = osn2.at<double>(1);
		CylinderModel[5] = osn2.at<double>(2);
		CylinderModel[6] = rotationResults[minErrCylinderIdx][6];
		CylinderModel[7] = rotationResults[minErrCylinderIdx][0];

		cv::Mat pPoints = cv::Mat::zeros(3, 22, CV_64F);
		for (int i = 0; i <= 10; i++)
		{
			pPoints.at<double>(0, 2 * i) = rotationResults[minErrCylinderIdx][2] + rotationResults[minErrCylinderIdx][6] * cos(i * CV_PI / 5);
			pPoints.at<double>(1, 2 * i) = rotationResults[minErrCylinderIdx][4];
			pPoints.at<double>(2, 2 * i) = rotationResults[minErrCylinderIdx][3] + rotationResults[minErrCylinderIdx][6] * sin(i * CV_PI / 5);
			pPoints.at<double>(0, 2 * i + 1) = rotationResults[minErrCylinderIdx][2] + rotationResults[minErrCylinderIdx][6] * cos(i * CV_PI / 5);
			pPoints.at<double>(1, 2 * i + 1) = rotationResults[minErrCylinderIdx][5];
			pPoints.at<double>(2, 2 * i + 1) = rotationResults[minErrCylinderIdx][3] + rotationResults[minErrCylinderIdx][6] * sin(i * CV_PI / 5);
		}
		pPoints = Ryb * Rxb * Rzb * pPoints;
		for (int i = 0; i < 22; i++)
		{
			cv::Point3d pp;
			pp.x = pPoints.at<double>(0, i);
			pp.y = pPoints.at<double>(1, i);
			pp.z = pPoints.at<double>(2, i);
			cylinderPoints.push_back(pp);
		}

		return CylinderModel;
	}

	/**
		 * @brief MNK
		 * @brief Вспомогательная функции решение системы алгебраических уравнений методом обратной матрицы
		 * @param A - матрица коэффициентов
		 * @param b  - вектор правых частей
		 * @return - cv::Mat x - вектор неизвестрых
		 */
	cv::Mat mrcv::MNK(cv::Mat A, cv::Mat b)
	{
		cv::Mat resX = cv::Mat::zeros(3, 1, CV_64F);
		cv::Mat At, Ainv, Aleft;
		cv::transpose(A, At);
		Ainv = At * A;
		Ainv = Ainv.inv();
		Aleft = Ainv * At;
		resX = Aleft * b;
		return resX;
	}

	/**
		 * @brief minMatElement
		 * @brief Вспомогательная функции поиск минимального значения в матрице
		 * @param data  -  входная матрица-
		 * @return - double min - минимальное значение в матрице
		 */
	double mrcv::minMatElement(cv::Mat data)
	{
		double temp = 1e+8;
		for (int i = 0; i < data.cols; i++)
			if (data.at<double>(i) < temp)
				temp = data.at<double>(i);
		return temp;
	}

	/**
		 * @brief maxMatElement
		 * @brief Вспомогательная функции поиск максимального значения в матрице
		 * @param data  -  входная матрица
		 * @return - double min - минимальное значение в матрице
		 */
	double mrcv::maxMatElement(cv::Mat data)
	{
		double temp = -1e+8;
		for (int i = 0; i < data.cols; i++)
			if (data.at<double>(i) > temp)
				temp = data.at<double>(i);
		return temp;
	}
}