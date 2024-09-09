#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>

namespace mrcv
{
	errno_t ObjCourse::initNN(const std::string pathToModel, const std::string pathToClasses)
	{
		errno_t err = readClasses(pathToClasses);
		if (err == 0)
		{
			_network = cv::dnn::readNetFromONNX(pathToModel);
			if (_network.empty())
			{
				return ENETDOWN;
			}
			else
			{
				_network.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
				_network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
			}
		}

		return err;
	}
    ObjCourse::ObjCourse(const std::string pathToModel, const std::string pathToClasses)
	{
		if (!initNN(pathToModel, pathToClasses))
		{
            writeLog("The neural network has been initiated successfully!");
            writeLog("Input width: " + std::to_string(_inputWidth));
            writeLog("Input height: " + std::to_string(_inputHeight));
		}
		else
		{
            writeLog("The neural network initialization ERROR!");
		}
	}
    ObjCourse::ObjCourse(const std::string pathToModel, const std::string pathToClasses, int width, int height)
	{
		_inputWidth = width;
		_inputHeight = height;

		if (!initNN(pathToModel, pathToClasses))
		{
			writeLog("The neural network has been initiated successfully!");
			writeLog("Input width: " + std::to_string(_inputWidth));
			writeLog("Input height: " + std::to_string(_inputHeight));
		}
		else
		{
			writeLog("The neural network initialization ERROR!");
		}
	}
	errno_t ObjCourse::readClasses(const std::string pathToClasses)
	{
		std::ifstream classesFile(pathToClasses);
		std::string line;

		if (!classesFile)
		{
			writeLog("Failed to open classes names!");
			return ENOENT;
		}
		while (std::getline(classesFile, line))
		{
			_classes.push_back(line);
		}
		classesFile.close();

		return 0;
	}
    void ObjCourse::drawLabel(cv::Mat& img, std::string label, int left, int top)
    {
        // Отображение метки в боундинг боксе
        int baseline;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, OBJCOURSE_FONT_SCALE, OBJCOURSE_THICKNESS, &baseline);
        top = std::max(top, labelSize.height);
        // Левый верхний угол
        cv::Point topLeftCorner = cv::Point(left, top);
        // Правый нижний угол
        cv::Point bottomRightCorner = cv::Point(left + labelSize.width, top + labelSize.height + baseline);        
        // Отрисовка
        cv::rectangle(img, topLeftCorner, bottomRightCorner, OBJCOURSE_BLACK, cv::FILLED);        
        cv::putText(img, label, cv::Point(left, top + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, OBJCOURSE_FONT_SCALE, OBJCOURSE_YELLOW, OBJCOURSE_THICKNESS);
    }
	std::vector<cv::Mat> ObjCourse::preProcess(cv::Mat& img, cv::dnn::Net& net)
	{
		cv::Mat blob;
		cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(_inputWidth, _inputHeight), cv::Scalar(), true, false);
		net.setInput(blob);
		std::vector<cv::Mat> outputs;
		net.forward(outputs, net.getUnconnectedOutLayersNames());
		return outputs;
	}
	cv::Mat ObjCourse::postProcess(cv::Mat& img, std::vector<cv::Mat>& outputs, const std::vector<std::string>& classNames)
	{
        // Начальная инициализация
        cv::Mat ret = img.clone();
        _classesIdSet.clear();
        _confidencesSet.clear();
        _boxesSet.clear();
        _classesSet.clear();

        std::vector<int> classIDs;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Расчет коэффицентов ресайзинга
        float xFactor = img.cols / (float)_inputWidth;
        float yFactor = img.rows / (float)_inputHeight;

        float* data = (float*)outputs[0].data;

        const size_t dimensions = classNames.size() + 5;
        const size_t rows = 25200;        
        for (size_t i = 0; i < rows; ++i)
        {
            float confidence = data[4];
            // Пропускаем маловероятные объекты
            if (confidence >= OBJCOURSE_CONFIDENCE_THRESHOLD)
            {
                float* classesScores = data + 5;                
                cv::Mat scores(1, (int)classNames.size(), CV_32FC1, classesScores);                
                cv::Point classID;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classID);
                // Продолжаем если пройден вероятностный порог
                if (maxClassScore > OBJCOURSE_SCORE_THRESHOLD)
                {
                    // Запоминаем класс ID
                    confidences.push_back(confidence);
                    classIDs.push_back(classID.x);                    
                    float cx = data[0];
                    float cy = data[1];
                    // Размер бокса
                    float w = data[2];
                    float h = data[3];
                    // Координаты боундинг бокса
                    int left = int((cx - 0.5 * w) * xFactor);
                    int top = int((cy - 0.5 * h) * yFactor);
                    int width = int(w * xFactor);
                    int height = int(h * yFactor);
                    // Запоминаем информацию о боксе
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            data += dimensions;
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, OBJCOURSE_SCORE_THRESHOLD, OBJCOURSE_NMS_THRESHOLD, indices);

        int bigestArea = INT_MIN;
        int bigestIndex = -1;
        int boxIndex = -1;

        for (size_t i = 0; i < indices.size(); i++)
        {
            int idx = indices[i];
            cv::Rect box = boxes[idx];

            boxIndex++;
            if (box.area() > bigestArea)
            {
                bigestIndex = boxIndex;
                bigestArea = box.area();
            }
        }

        if (bigestIndex > -1)
        {
            int idx = indices[bigestIndex];
            cv::Rect box = boxes[idx];

            _boxesSet.push_back(box);
            _confidencesSet.push_back(confidences[idx]);
            _classesIdSet.push_back(classIDs[idx]);
            _classesSet.push_back(classNames[classIDs[idx]]);

            int left = box.x;
            int top = box.y;
            int width = box.width;
            int height = box.height;
            
            // Отрисовка боундинг бокса
            cv::rectangle(ret, cv::Point(left, top), cv::Point(left + width, top + height), OBJCOURSE_GREEN, 3 * OBJCOURSE_THICKNESS);
            
            // Получаем метку класса
            std::string label = cv::format("%.2f", confidences[idx]);
            label = classNames[classIDs[idx]] + ": " + label;
            if (OBJCOURSE_DRAW_LABEL)
            {
                // Отрисовка метки класса
                drawLabel(ret, label, left, top);
            }
        }

        return ret;
	}		
	std::string ObjCourse::getInfo(void)
	{
        std::string tmpString = "";
        for (size_t i = 0; i < _classesIdSet.size(); i++)
        {
            tmpString += _classesSet[i];
            tmpString += ": ";
            tmpString += std::to_string(_confidencesSet[i]);
            tmpString += "\n";
        }
        return tmpString;
	}
	cv::Mat ObjCourse::mainProcess(cv::Mat& img)
	{
        std::vector<cv::Mat> detections;
        detections = preProcess(img, _network);
        cv::Mat res = postProcess(img, detections, ObjCourse::_classes);
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency();
        ObjCourse::_inferenceTime = _network.getPerfProfile(layersTimes) / (float)freq;
        return res;
	}
}