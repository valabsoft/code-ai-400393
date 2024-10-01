#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>

namespace mrcv
{
#ifdef _WIN32
    errno_t ObjCourse::initNN(const std::string pathToModel, const std::string pathToClasses)
#else
    error_t ObjCourse::initNN(const std::string pathToModel, const std::string pathToClasses)
#endif
	{
		
#ifdef _WIN32
        errno_t err = readClasses(pathToClasses);
#else
        error_t err = readClasses(pathToClasses);
#endif
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
    ObjCourse::ObjCourse(const std::string pathToModel, const std::string pathToClasses, int width, int height, float scoreThreshold, float nmsThreshold, float confidenceThreshold, float cameraAngle)
    {
        _inputWidth = width;
        _inputHeight = height;

        _scoreThreshold = scoreThreshold;
        _nmsThreshold = nmsThreshold;
        _confidenceThreshold = confidenceThreshold;

        _cameraAngle = cameraAngle;

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
#ifdef _WIN32
    errno_t ObjCourse::readClasses(const std::string pathToClasses)
#else
    error_t ObjCourse::readClasses(const std::string pathToClasses)
#endif
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
        cv::Mat processedImage = img.clone();
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
            if (confidence >= _confidenceThreshold)
            {
                float* classesScores = data + 5;                
                cv::Mat scores(1, (int)classNames.size(), CV_32FC1, classesScores);                
                cv::Point classID;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classID);
                // Продолжаем если пройден вероятностный порог
                if (maxClassScore > _scoreThreshold)
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

        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, _scoreThreshold, _nmsThreshold, indexes);

        mrcv::writeLog("POST PROCESS ===>");
        mrcv::writeLog("boxes.size(): " + std::to_string(boxes.size()));
        mrcv::writeLog("confidences.size(): " + std::to_string(confidences.size()));

        int bigestArea = INT_MIN;
        int bigestIndex = -1;
        int boxIndex = -1;

        for (size_t i = 0; i < indexes.size(); i++)
        {
            int idx = indexes[i];
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
            int idx = indexes[bigestIndex];
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
            cv::rectangle(processedImage, cv::Point(left, top), cv::Point(left + width, top + height), OBJCOURSE_GREEN, 3 * OBJCOURSE_THICKNESS);
            
            // Получаем метку класса
            std::string label = cv::format("%.2f", confidences[idx]);
            label = classNames[classIDs[idx]] + ": " + label;
            if (OBJCOURSE_DRAW_LABEL)
            {
                // Отрисовка метки класса
                drawLabel(processedImage, label, left, top);
            }
        }

        return processedImage;
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
        cv::Mat processResult = postProcess(img, detections, ObjCourse::_classes);
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency();
        ObjCourse::_inferenceTime = _network.getPerfProfile(layersTimes) / (float)freq;
        return processResult;
	}    
    int ObjCourse::findAngle(double resolution, double cameraAngle, int cx)
    {
        _cameraAngle = cameraAngle;
        return (int)((cx * _cameraAngle / resolution) - _cameraAngle / 2);
    }
    std::string ObjCourse::getTimeStamp()
    {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        auto timer = system_clock::to_time_t(now);
        std::tm bt = *std::localtime(&timer);
        std::ostringstream oss;
        oss << std::put_time(&bt, "%d-%m-%Y %H:%M:%S"); // DD-MM-YYYY HH:MM:SS
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return oss.str();
    }
    int ObjCourse::getObjectCount(cv::Mat frame)
    {
        cv::Mat img = mainProcess(frame);

        // Сохранить результат обработки на диск для отладки
        if (IS_DEBUG_LOG_ENABLED)
        {
            std::filesystem::path outputFile("files\\output.bmp");
            auto currentPath = std::filesystem::current_path();
            auto outputPath = currentPath / outputFile;
            cv::imwrite(outputPath.u8string(), img);
        }

        // Результаты работы детектора       
        std::vector<int> ids = getClassIDs();
        std::vector<float> confidences = getConfidences();
        std::vector<cv::Rect> boxes = getBoxes();
        std::vector<std::string> classes = getClasses();

        std::stringstream strIDs;
        for each (auto id in ids)
        {
            strIDs << std::to_string(id) << ";";
        }
        std::stringstream strConfs;
        for each (auto conf in confidences)
        {
            strConfs << std::to_string(conf) << ";";
        }
        writeLog("IDs: " + strIDs.str());
        writeLog("Confidence: " + strConfs.str());
        writeLog("Boxes: " + std::to_string(boxes.size()));
        
        return (int)boxes.size();
    }
    float ObjCourse::getObjectCourse(cv::Mat frame, double frameWidth, double cameraAngle)
    {
        cv::Mat img = mainProcess(frame);
        std::string timestamp = getTimeStamp();

        // Результаты работы детектора       
        std::vector<int> ids = getClassIDs();
        std::vector<float> confidences = getConfidences();
        std::vector<cv::Rect> boxes = getBoxes();
        std::vector<std::string> classes = getClasses();
        
        ///////////////////////////////////////////////////////////////////////
        // Поиск бокса цели с максимальной площадью
        ///////////////////////////////////////////////////////////////////////
        int bigestArea = INT_MIN;
        int bigestIndex = -1;
        int boxIndex = -1;

        for (auto b : boxes)
        {
            boxIndex++;
            if (b.area() > bigestArea)
            {
                bigestIndex = boxIndex;
                bigestArea = b.area();
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Расчет управления
        ///////////////////////////////////////////////////////////////////////        
        
        cv::Point center;
        cv::Point centerN;
        cv::Point centerP;
        cv::Point centerZ;

        std::string direction;
        int angle;
        
        if (boxes.size() > 0)
        {
            // Расчет центра бокса с обнаруженной целью
            center = (boxes[bigestIndex].br() + boxes[bigestIndex].tl()) * 0.5;

            // Угол между прицелом и целью
            angle = findAngle(frameWidth, cameraAngle, center.x);

            // Команда управления лево / право
            direction = center.x > frameWidth / 2 ? "RIGHT" : "LEFT";

            cv::Point boardBoxPt1;
            cv::Point boardBoxPt2;

            float SIGHT_WIDTH = 50; // Размер прицела
            
            // Координаты бокса прицела
            boardBoxPt1.x = (int)(img.cols / 2) - (int)SIGHT_WIDTH;
            boardBoxPt1.y = (int)(img.rows / 2) - (int)SIGHT_WIDTH;
            boardBoxPt2.x = (int)(img.cols / 2) + (int)SIGHT_WIDTH;
            boardBoxPt2.y = (int)(img.rows / 2) + (int)SIGHT_WIDTH;

            // Если цель находится в границах прицела - удерживаем курс
            //if ((boardBoxPt1.x <= center.x) && (center.x <= boardBoxPt2.x) &&
            //    (boardBoxPt1.y <= center.y) && (center.y <= boardBoxPt2.y))
            //{
            //    direction = "HOLD";
            //}

            // Алгоритм удержания цели только по оси абцисс
            if ((boardBoxPt1.x <= center.x) && (center.x <= boardBoxPt2.x))
                direction = "HOLD";

            // Время работы детектора
            std::stringstream ssTime;
            std::string inference;            

            // Строка инфорации
            std::string diagnosticInfo;

            // Время работы детектора
            ssTime.str(std::string()); // Очистка строкового стримера
            ssTime << std::fixed << std::setprecision(2) << getInference();
            inference = ssTime.str();
            
            // Отладочная информация
            diagnosticInfo = "CMD:\t(" + direction + ":" + std::to_string(angle) + ")" + "\tTIME: " + inference + "\t" + timestamp;
            writeLog(diagnosticInfo);

            return angle;
            
        }
        else
        {
            std::string diagnosticInfo = "TIME: " + timestamp + " NOT FOUND...";
            writeLog(diagnosticInfo);

            return 0.0;
        }

        return 0.0;
    }
}