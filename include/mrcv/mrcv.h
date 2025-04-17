#pragma once

#include <mrcv/export.h>
#include <mrcv/mrcv-common.h>
#include <mrcv/mrcv-detector.h>

namespace mrcv
{
	/**
	* @brief функция сравнения изображения.
	* @param img1 - исходное фото 1, img1 - исходное фото 2, methodCompare - метод сравнения.
	* @return - различия фотографий в процентном соотношении.
	*/
	MRCV_EXPORT double compareImages(cv::Mat img1,cv::Mat img2,bool methodCompare);
	
	/**
        * @brief функция морфологического преобразования.
        * @param image - исходное фото, out - путь для нового файла, metod - метод преобразования , morph_size - размер преобразования.
        *  @return - результат работы функции.
        */
        MRCV_EXPORT int morphologyImage(cv::Mat image,std::string out, METOD_MORF metod,int morph_size);
        
	/////////////////////////////////////////////////////////////////////////////
	// Утилиты
	/////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Функция для создания текстового файла-лога работы функций библиотеки
	 * @param logText - текст сообщения для записи в лог-файл
	 * @param logType - тип сообщения в лог-файле
	 */
	MRCV_EXPORT void writeLog(std::string logText, LOGTYPE logType = LOGTYPE::INFO);
	
	/**
	 * @brief Функция для записи строки-разделителя в текстовый лог-файл
	 */
	MRCV_EXPORT void writeLog();
	
	/**
	 * @brief Функция сложения двух целых чисел.
	 * @param a - Первое слагаемое.
	 * @param b - Второе слагаемое.
	 * @return - Резальтат вычсиления выражения a + b
	 */
	MRCV_EXPORT int add(int a, int b);
	
	/**
	 * @brief Функция загрузки изображения.
	 *
	 * Функция используется для загрузки изображения с носителя и отображения загруженного изображения в модальном окне.
	 *
	 * @param image - объект cv::Mat для хранения загруженного изображения.
	 * @param pathToImage - полный путь к файлу с изображением.
	 * @param showImage - флаг, отвечающий за отображение модального окна (false по умолчанию).
	 * @return - код результата работы функции. 0 - Success; 1 - Невозможно открыть изображение; -1 - Unhandled Exception.
	 */
	MRCV_EXPORT int readImage(cv::Mat& image, std::string pathToImage, bool showImage = false);
	
	/**
	 * @brief Функция вывода информации о текущей сборке OpenCV.
	 * @return Строка с диагностической информацией.
	 */
	MRCV_EXPORT std::string getOpenCVBuildInformation();
	
	/**
	 * @brief Функция записи видеопотока на диск.
	 *
	 * Функция может использоваться дле реализации работы видеорегистратора.
	 *
	 * @param cameraID - ID камеры.
	 * @param recorderInterval - Интервал записи в секундах.
	 * @param fileName - Маска фала.
	 * @param codec - Кодек, используемый для создания видеофайла.
	 * @return - код результата работы функции. 0 - Success; 1 - ID камеры задан неверно; 2 - Интервал захвата меньше минимального; 3 - Не удалось захватить камеру; 4 - Не удалось создать объектс cv::VideoWriter; -1 - Unhandled Exception.
	 */
	MRCV_EXPORT int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec);
	
	/**
	* @brief Функция скачаивания файла Яндекса.
	* @param query - Строка запроса для поиска.
	* @param minWidth - Минимальная ширина изображения.
	* @param minHeight - Минимальная высота изображения.
	* @param nameTemplate - Шаблон имени файла.
	* @param outputFolder - Папка для скачивания.
	* @param separateDataset - Флаг разбивки датасета на тренировочную и тестовую выборки.
	* @param trainsetPercentage - Процент для распределения между папками.
	* @param countFoto - Количество необходимых фото для скачивания.
	* @param money - Платный или бесплатный вариант работы.
	* @param key - Яндекс key.
	* @param secretKey - Яндекс secretKey.
	* @return - Результат работы функции.
	*/
	MRCV_EXPORT int getImagesFromYandex(std::string query, int minWidth, int minHeight, std::string nameTemplate, std::string outputFolder, bool separateDataset, unsigned int trainsetPercentage, unsigned int countFoto, bool money, std::string key, std::string secretKey);
	/////////////////////////////////////////////////////////////////////////////
	// Калибровка
	/////////////////////////////////////////////////////////////////////////////
	
	/**
	 * @brief Функция общей калибровки.
	 * @param imagesL - Вектор строк имён изображений левой камеры.
	 * @param imagesR - Вектор строк имён изображений правой камеры.
	 * @param pathToImagesL - Путь к папке с изображениями левой камеры.
	 * @param pathToImagesR - Путь к папке с изображениями правой камеры.
	 * @param calibrationParametersL - Структура для хранения калибровочных параметров левой камеры.
	 * @param calibrationParametersR - Структура для хранения калибровочных параметров правой камеры.
	 * @param calibrationParameters - Структура для хранения калибровочных параметров стерео пары.
	 * @param chessboardColCount - Количество ключевых точек калибровочной доски по столбцам.
	 * @param chessboardRowCount - Количество ключевых точек калибровочной доски по строкам.
	 * @param chessboardSquareSize - Размер поля калиброчно доски в мм.
	 */
	MRCV_EXPORT void cameraCalibration(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathToImagesL, std::string pathToImagesR, CalibrationParametersMono& calibrationParametersL, CalibrationParametersMono& calibrationParametersR, CalibrationParametersStereo& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief Функция калибровки одиночной камеры.
	 * @param images - Вектор строк имён изображений камеры.
	 * @param pathToImages - Путь к папке с изображениями камеры.
	 * @param calibrationParameters - Структура для хранения калибровочных параметров камеры.
	 * @param chessboardColCount - Количество ключевых точек калибровочной доски по столбцам.
	 * @param chessboardRowCount - Количество ключевых точек калибровочной доски по строкам.
	 * @param chessboardSquareSize - Размер поля калиброчно доски в мм.
	 */
	MRCV_EXPORT void cameraCalibrationMono(std::vector<cv::String> images, std::string pathToImages, CalibrationParametersMono& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief Функция калибровки стерео пары.
	 * @param imagesL - Вектор строк имён изображений левой камеры.
	 * @param imagesR - Вектор строк имён изображений правой камеры.
	 * @param pathToImagesL - Путь к папке с изображениями левой камеры.
	 * @param pathToImagesR - Путь к папке с изображениями правой камеры.
	 * @param calibrationParameters - Структура для хранения калибровочных параметров стерео пары.
	 * @param chessboardColCount - Количество ключевых точек калибровочной доски по столбцам.
	 * @param chessboardRowCount - Количество ключевых точек калибровочной доски по строкам.
	 * @param chessboardSquareSize - Размер поля калиброчно доски в мм.
	 */
	MRCV_EXPORT void cameraCalibrationStereo(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathToImagesL, std::string pathToImagesR, CalibrationParametersStereo& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief Функция чтения параметров калибровки одиночной камеры.
	 * @param fileName - Полный путь к файлу калибровочных параметров.
	 * @return
	 */
	MRCV_EXPORT CalibrationParametersMono readCalibrationParametersMono(std::string fileName);

	/**
	 * @brief Функция записи параметров калибровки одиночной камеры.
	 * @param fileName - Полный путь к файлу калибровочных параметров.
	 * @param parameters - Структура для хранения калибровочных параметров.
	 */
	MRCV_EXPORT void writeCalibrationParametersMono(std::string fileName, CalibrationParametersMono parameters);

	/**
	 * @brief Функция записи параметров калибровки стерео пары
	 * @param fileName - Полный путь к файлу калибровочных параметров.
	 * @param parameters - Структура для хранения калибровочных параметров.
	 */
	MRCV_EXPORT void writeCalibrationParametersStereo(std::string fileName, CalibrationParametersStereo parameters);

	/**
	 * @brief Функция чтения параметров калибровки стерео пары
	 * @param fileName - Полный путь к файлу калибровочных параметров.
	 * @return - Структура для хранения калибровочных параметров.
	 */
	MRCV_EXPORT CalibrationParametersStereo readCalibrationParametersStereo(std::string fileName);
	
	/**
	 * @brief Функция чтения конфигурационного файла для калибровки
	 * @param pathToConfigFile - Полный путь к конфигурационному файлу.
	 * @return - Структура для хранения параметров процедуры калибровки.
	 */
	MRCV_EXPORT int readCalibrartionConfigFile(std::string pathToConfigFile, CalibrationConfig& config);
	/////////////////////////////////////////////////////////////////////////////

	MRCV_EXPORT class Segmentor
	{
	public:
		Segmentor() { };
		~Segmentor() { };
		
		/**
		 * @brief Функция инициализации
		 * @param gpu_id - Подключение GPU.
		 * @param width - Ширина пердаваемых изображений.
		 * @param height - Высота пердаваемых изображений.
		 * @param listName - Список классов.
		 * @param encoderName - Имя кодировщика.
		 * @param pretrainedPath - Путь к кодировщику.
		 */
		void Initialize(int gpu_id, int width, int height, std::vector<std::string>&& listName, std::string encoderName, std::string pretrainedPath);

		/**
		 * @brief функция для повышения производительности обучения
		 * @param tricks - Структура дополнений для обучения таких как скорость оюучения, вращение изображения и вес при проигрыше .
		 */
		void SetTrainTricks(trainTricks& tricks);

		/**
		 * @brief Функция обучения модели
		 * @param learning_rate - Темп обучения.
		 * @param epochs - Количество эпох.
		 * @param batch_size - Количество обучающих примеров за одну итерацию.
		 * @param train_val_path - Путь к изображениям для обучения.
		 * @param imageType - Тип изображений.
		 * @param save_path - Путь для сохранения своих весов.
		 */
		void Train(float learning_rate, unsigned int epochs, int batch_size, std::string train_val_path, std::string imageType, std::string save_path);

		/**
		 * @brief функция загрузки своих весов
		 * @param pathWeight - Путь к своим весам.
		 */
		void LoadWeight(std::string pathWeight);

		/**
		 * @brief Функция прогноза
		 * @param image - Тестируемое изображение.
		 * @param which_class - Список классов.
		 */
		void Predict(cv::Mat& image, const std::string& which_class);

	private:
		int width = 512;
		int height = 512;
		std::vector<std::string> listName;
		torch::Device device = torch::Device(torch::kCPU);
		trainTricks tricks;
		FPN fpn{ nullptr };
	};

	MRCV_EXPORT class MRCVPoint
	{
	private:
		int _X;
		int _Y;
	public:
		MRCVPoint();
		void setX(int X);
		void setY(int Y);
		std::string gerCoordinates();
	};

	MRCV_EXPORT class ObjCourse
	{
	public:
		ObjCourse(const std::string pathToModel, const std::string pathToClasses);
		ObjCourse(const std::string pathToModel, const std::string pathToClasses, int width, int height);
		ObjCourse(const std::string pathToModel, const std::string pathToClasses, int width, int height, float scoreThreshold, float nmsThreshold, float confidenceThreshold, float cameraAngle);
		std::vector<float> getConfidences(void) { return _confidencesSet; }
		std::vector<cv::Rect> getBoxes(void) { return _boxesSet; }
		std::vector<int> getClassIDs(void) { return _classesIdSet; }
		std::vector<std::string> getClasses(void) { return _classesSet; }
		float getInference(void) { return _inferenceTime; }
		std::string getInfo(void);
		cv::Mat mainProcess(cv::Mat& img);
		int getObjectCount(cv::Mat frame);
		float getObjectCourse(cv::Mat frame, double frameWidth, double cameraAngle);
	private:
		cv::dnn::Net _network;
		int _inputWidth = 640;
		int _inputHeight = 640;
		float _scoreThreshold = 0.50f;
		float _nmsThreshold = 0.45f;
		float _confidenceThreshold = 0.45f;
		float _cameraAngle = 80;
		std::vector<std::string> _classes;
		std::vector<int> _classesIdSet;
		std::vector<cv::Rect> _boxesSet;
		std::vector<float> _confidencesSet;
		std::vector<std::string> _classesSet;
		float _inferenceTime;
#ifdef _WIN32
		errno_t readClasses(const std::string pathToClasses);
		errno_t initNN(const std::string pathToModel, const std::string pathToClasses);
#else
		error_t readClasses(const std::string pathToClasses);
		error_t initNN(const std::string pathToModel, const std::string pathToClasses);
#endif		
		void drawLabel(cv::Mat& img, std::string label, int left, int top);
		std::vector<cv::Mat> preProcess(cv::Mat& img, cv::dnn::Net& net);
		cv::Mat postProcess(cv::Mat& img, std::vector<cv::Mat>& outputs, const std::vector<std::string>& classNames);
		int findAngle(double resolution, double cameraAngle, int cx);
		std::string getTimeStamp();
	};

	/////////////////////////////////////////////////////////////////////////////
	// Методы препроцессинга изображений
	/////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Функция формирования изображения в случаи ошибки.
	 * Функция формирует изображение с чёрным фоном и наносит текс (текс должен содержать ссылку на место и тип ошибки)
	 * (для информирование оператора в режиме реального времени)
	 * @param textError - текс сообщения, которое будет записано в изображение.
	 * @return - выходное цветное RGB изображение: формата cv::Mat CV_8UC3, с кодом ошибки
	 */
	cv::Mat getErrorImage(std::string textError);
	
	/**
	 * @brief Функция автоматической предобработки изображения, кооррекции яркости.
	 * Функция автоматической коррекции яркости изображения с помощью степенного преобразования (метод Гамма-Коррекции)
	 * @param imageInput -  входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param imageOutput - выходное (преобразованное) цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param gamma -       степень преобразования (1 - без изменений от 1 до 0 - осветление, от 1 до 10 - затемнение изображения)
	 * @return - код результата работы функции: 0 - Success; 1 - Пустое изображение;  -1 - Неизвестная ошибка.
	 */
	int changeImageBrightness(cv::Mat& imageInput, cv::Mat& imageOutput, double gamma);
	
	/**
	 * @brief Функция автоматической кооррекции контраста.
	 * Функция реализует несколько методов коррекции контраста изображения
	 * @param imageInput        -  входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param imageOutput       - выходное (преобразованное) цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param metodIncreaseContrast - выбор метода коррекции конраста изобрадения
	 * EQUALIZE_HIST,       // метод Гистограммная  эквализация  (Histogram Equalization)
	 * CLAHE,               // метод Адаптивная Гистограммная  эквализация (Contrast Limited Adaptive Histogram Equalization)
	 * CONTRAST_BALANCING,  // метод Баланса контрастности, основанный на фильтрации крайних значений
	 * CONTRAST_EXTENSION,  // метод Расширения контрастности, основанный на логарифмическом преобразовании
	 * @param colorSpace        -  выбор цветовой модели (цветовые пространства)
	 * CM_RGB,
	 * CM_HSV,
	 * CM_LAB,
	 * CM_YCBCR,
	 * @param clipLimitCLAHE    - пороговое значение для ограничения контрастности
	 * @param gridSizeCLAHE     - размер сетки. Изображение будет разделено на одинакового размера части в виде матрицы,
	 * @param percentContrastBalance -  параметр медода Баланса контрастности процент отсеивания крайних значений яркости из массива planeArray,
	 * диапазон зачений больше 0 меньше 100
	 * @param mContrastExtantion - параметр медода Расширение контрастности, смещение функции преобразования по оси яркости входного изображения
	 * @param eContrastExtantion - параметр медода Расширение контрастности, отвечает за наклон кривой функции преобразования относительно оси яркости входного изображения
	 * @return - код результата работы функции. 0 - Success;
	 * 1 - Пустое изображение; 2 - Неизвестный формат изображения;  -1 - Неизвестная ошибка.
	 */
	int increaseImageContrast(cv::Mat& imageInput, cv::Mat& imageOutput,
		mrcv::METOD_INCREASE_IMAGE_CONTRAST metodIncreaseContrast, mrcv::COLOR_MODEL colorSpace,
		double clipLimitCLAHE = 2, cv::Size gridSizeCLAHE = cv::Size(9, 9),
		float percentContrastBalance = 5, double mContrastExtantion = -1, double eContrastExtantion = 4);
	
	/**
	 * @brief Повышение резкости изображения. Алгоритм №01 (фильтра Лапласа)
	 * Функция повышения резкости изображения с помощью фильтра Лапласа
	 * @param imageInput -      входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param imageOutput -     выходное (преобразованное) цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param gainFactorHighFrequencyComponent - коэффициент усиления высокочастоной составляющей (чем выше тем выше резкость), рекомендуемое = 2.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; -1 - Неизвестная ошибка.
	 */
	int sharpeningImage01(cv::Mat& imageInput, cv::Mat& imageOutput, double gainFactorHighFrequencyComponent);
	
	/**
	 * @brief Повышение резкости изображения. Алгоритм №02 (фильтра Гаусса)
	 * Функция повышения резкости изображения с помощью фильтра Гаусса
	 * @param imageInput -      входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param imageOutput -     выходное (преобразованное) цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param filterSize  -     размер маски фильтра Гауссу, рекомендуемое = cv::Size(9, 9).
	 * @param sigmaFilter  -    тандартное отклонение филтра Гаусса, елсли 0 - значение по умолчанию
	 * @param gainFactorHighFrequencyComponent - коэффициент усиления высокочастоной составляющей (чем выше тем выше резкость), рекомендуемое = 4.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; -1 - Неизвестная ошибка.
	 */
	int sharpeningImage02(cv::Mat& imageInput, cv::Mat& imageOutput, cv::Size filterSize, double sigmaFilter, double gainFactorHighFrequencyComponent);
	
	/**
	 * @brief Функция чтения параметров камеры из файла.
	 * @param fileNameCameraParameters -    путь к файлу c параметрами камеры.
	 * @param map11 - первая карта точек (x, y) или просто значений x для исправления геометрических искажений
	 * @param map12 - вторая карта значений y, (пустая карта, если map1 равен точкам (x,y)) соответственно.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int readCameraParametrsFromFile(const char* pathToFileCameraParametrs, mrcv::cameraParameters& cameraParameters);
	
	/**
	 * @brief Функция предварительной обработки изображений (автоматическая коррекция контраста и яркости, резкости)
	 * Функция интегрирует в себе остальные функции предобработки изображений
	 * @param image - входное и выходное цветное RGB изображение, формата cv::Mat CV_8UC3, над которым происходит преобразование.
	 * @param metodImagePerProcessingBrightnessContrast - вектор параметров, который определяет, какие преобразования и в какой последовательности проводить.
	 *  NONE                    - без изменений
	 *  CONVERTING_BGR_TO_GRAY,             // преобразование типа изображения к из цветноко BGR к монохромному (серому)
	 *  BRIGHTNESS_LEVEL_UP,                // увеличение уровня яркости на один уровень
	 *  BRIGHTNESS_LEVEL_DOWN,              // уменьшение уровня яркости на один уровень
	 *  BALANCE_CONTRAST_01_YCBCR_EQUALIZEHIST,        // повышение контрастности, метод 01
	 *  BALANCE_CONTRAST_02_YCBCR_CLAHE,               // повышение контрастности, метод 02
	 *  BALANCE_CONTRAST_03_YCBCR_CONTRAST_BALANCING,  // повышение контрастности, метод 03
	 *  BALANCE_CONTRAST_04_YCBCR_CONTRAST_EXTENSION,  // повышение контрастности, метод 04
	 *  BALANCE_CONTRAST_05_HSV_EQUALIZEHIST,          // повышение контрастности, метод 05
	 *  BALANCE_CONTRAST_06_HSV_CLAHE,                 // повышение контрастности, метод 06
	 *  BALANCE_CONTRAST_07_HSV_CONTRAST_BALANCING,    // повышение контрастности, метод 07
	 *  BALANCE_CONTRAST_08_HSV_CONTRAST_EXTENSION,    // повышение контрастности, метод 08
	 *  BALANCE_CONTRAST_09_LAB_EQUALIZEHIST,          // повышение контрастности, метод 09
	 *  BALANCE_CONTRAST_10_LAB_CLAHE,                 // повышение контрастности, метод 10
	 *  BALANCE_CONTRAST_11_LAB_CONTRAST_BALANCING,    // повышение контрастности, метод 11
	 *  BALANCE_CONTRAST_12_LAB_CONTRAST_EXTENSION,    // повышение контрастности, метод 12
	 *  BALANCE_CONTRAST_13_RGB_EQUALIZEHIST,          // повышение контрастности, метод 13
	 *  BALANCE_CONTRAST_14_RGB_CLAHE,                 // повышение контрастности, метод 14
	 *  BALANCE_CONTRAST_15_RGB_CONTRAST_BALANCING,    // повышение контрастности, метод 15
	 *  BALANCE_CONTRAST_16_RGB_CONTRAST_EXTENSION,    // повышение контрастности, метод 16
	 *  SHARPENING_01        - повышение резкости, метод 01
	 *  SHARPENING_02        - повышение резкости, метод 02
	 *  NOISE_FILTERING_01_MEDIAN_FILTER  - фильтрация изображения от импульсных шумов
	 *  NOISE_FILTERING_02_AVARAGE_FILTER - фильтрация изображения от шумов
	 *  CORRECTION_GEOMETRIC_DEFORMATION  - коррекция геометрических искажений
	 * @param fileNameCameraParameters    - путь к файлу c параметрами камеры для исправления геометрических искажений.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int preprocessingImage(cv::Mat& imageIn, std::vector<mrcv::METOD_IMAGE_PERPROCESSIN> metodImagePerProcessingm, const std::string& fileNameCameraParameters);
	
	/**
	 * @brief Функция реализует метод Баланса контрастности
	 * Функция принимает матрицу либо серого изображения, либо одну из координат цветового пространства
	 * @param planeArray -  входное двухмерный массив, формата cv::Mat CV_8UC1
	 * @param percent -     процент отсеивания крайних значений яркости из массива planeArray, диапазон зачений больше 0 меньше 100
	 * @return - код результата работы функции. 0 - Success;
	 * 1 - Пустое массив;  3 - выход за диапазон percent; -1 - Неизвестная ошибка.
	 */
	int contrastBalancing(cv::Mat& planeArray, float percent);
	
	/**
	 * @brief Функция реализует метод Расширение контрастности
	 * Функция принимает матрицу либо серого изображения, либо одну из координат цветового пространства
	 * @param planeArray -  входное двухмерный массив, формата cv::Mat CV_8UC1
	 * @param m - параметр медода, смещение функции преобразования по оси яркости входного изображения
	 * @param e - параметр медода, отвечает за наклон кривой функции преобразования относительно оси яркости входного изображения
	 * @return - код результата работы функции: 0 - Success; 1 - Пустое массив;  -1 - Неизвестная ошибка.
	 */
	int contrastExtantion(cv::Mat& planeArray, double m = -1, double e = 2);
	
	/**
	 * @brief Класс для работы с плотным стерео и кластеризацией.
	 *
	 * Класс `DenseStereo` предоставляет функционал для загрузки данных,
	 * выполнения кластеризации, вывода и визуализации кластеров.
	 */
	MRCV_EXPORT class DenseStereo {
	public:
		/**
		 * @brief Выполняет кластеризацию загруженных данных.
		 *
		 * Функция для выполнения кластеризации данных, хранящихся
		 * в `vuxyzrgb`. Результаты кластеризации сохраняются в `IDX`.
		 */
		void makeClustering();

		/**
		 * @brief Загружает данные из файла.
		 *
		 * Функция считывает данные из указанного файла и сохраняет их
		 * во внутренней структуре `vuxyzrgb`.
		 *
		 * @param filename Имя файла, из которого будут загружены данные.
		 */
		void loadDataFromFile(const std::string& filename);

		/**
		 * @brief Печатает информацию о кластерах.
		 *
		 * Функция выводит на экран информацию о кластерах,
		 * сформированных в результате выполнения кластеризации.
		 */
		void printClusters();

	private:
		/**
		 * @brief Класс для хранения координат точек.
		 *
		 * В этом классе сохраняются трехмерные координаты точек,
		 * используемых в процессе кластеризации.
		 */
		class Vuxyzrgb {
		public:
			std::vector<std::vector<double>> xyz; ///< Трехмерные координаты точек.
		};

		Vuxyzrgb vuxyzrgb; ///< Экземпляр класса для хранения данных.
		std::mutex vuxyzrgb_mutex; ///< Мьютекс для защиты данных `vuxyzrgb`.

		std::vector<int> IDX; ///< Вектор индексов кластеров для каждой точки.
	};

	int flipImage(cv::Mat& imageInput, cv::Mat& imageOutput, int flipCode);

	int rotateImage(cv::Mat& imageInput, cv::Mat& imageOutput, double angle);

	int augmetation(std::vector<cv::Mat>& inputImagesAugmetation, std::vector<cv::Mat>& outputImagesAugmetation,
		std::vector<mrcv::AUGMENTATION_METHOD> augmetationMethod);

	/**
	 *	@brief Класс детектора
	 *
	 *	Класс реализует атрибуты и методы детекции объектов на изображениях подводных объектов,
	 *	а также аварийных ситуаций, связанных с подводными технологическими сооружениями.
	 */
	MRCV_EXPORT class Detector
	{
	private:
		int width = 416;
		int height = 416;
		std::vector<std::string> nameList;
		torch::Device device = torch::Device(torch::kCPU);
		YoloBody_tiny detector{ nullptr };
	public:
		/**
		* @brief Конструктор класса по умолчанию
		*/
		Detector();
		/**
		* @brief Функция инициализации модели детектора
		*
		* @param gpuID			- ID устройтсва, для которого будет инициализирована модель. Значение < 1 - исполнение программы на CPU, 0 и > 0 - исполнение программы на выбранном GPU-устройстве
		* @param width			- Ширина изображения, до которой будет масштабировано исходное изображение
		* @param height			- Высота изображения, до которой будет масштабировано исходное изображение
		* @param nameListPath	- Путь к текстовому файлу, содержащему наименования предполагаемых классов
		*/
		void Initialize(int gpuID, int width, int height, std::string nameListPath);
		/**
		* @brief Функция обучения модели детектора
		*
		* @param trainValPath	- Путь к директории с обучающим и валидационным датасетами
		* @param imageType		- Расширение изображений в директориях. Например, ".jpg", ".png" и т.д.
		* @param numEpoch		- Число эпох, требуемых для обучения модели
		* @param batchSize		- Размер батча, используемого при обучении модели
		* @param learningRate	- Темп обучения модели, который определяет размер шага на каждой итерации, при движении к минимуму функции потерь
		* @param savePath		- Путь, по которому будут сохранены веса обученной модели в формате .pt
		* @param pretrainedPath - Путь к предобученной модели YOLOv4_tiny.pt
		*
		* @return Код результата выполнения функции.
		* Значения:
		*			0	- Success;
		*			1	- Путь к предобученной модели задан не корректно или не существует;
		*			2	- Размер батча превышает число изображений в директории с обучающей выборкой или директория пуста;
		*			3	- Путь к обучающей выборке задан неверно;
		*			-1	- Неизвестная ошибка.
		*/
		int Train(std::string trainValPath, std::string imageType, int numEpochs = 30, int batchSize = 4, float learningRate = 0.0003,
			std::string savePath = "detector.pt", std::string pretrainedPath = "detector.pt");
		/**
		* @brief Функция загрузки весов обученной модели
		*
		* @param - Путь к весам
		*
		* @return - Код результата выполнения функции. 0 - Success; -1 - Неизвестная ошибка.
		*/
		int LoadWeight(std::string weightPath);
		/**
		* @brief Функция загрузки предобученной модели
		*
		* @param - Путь к предобученной модели в формате .pt
		* @return - Код результата выполнения функции. 0 - Success; -1 - Неизвестная ошибка
		*/
		int LoadPretrained(std::string pretrainedPath);
		/**
		* @brief Функция детекции и идентификации объектов по изображению
		*
		* @param image		- Изображение в формате cv::Mat
		* @param show		- Флаг отображения изображения с нанесёнными ограничивающими рамками.
		* @param confTresh	- Пороговое значение интервала "доверительности" для предсказаний модели. Если модель предсказывает, что вероятность присутствия объекта ниже этого значения, такой объект отбрасывается.
		* @param nmsThresh	- Пороговое значение для Non-Maximum Suppression (NMS), алгоритма, который отбрасывает пересекающиеся предсказания объектов. Если несколько рамок сильно пересекаются и показывают высокую вероятность для одного и того же объекта, остаётся только та рамка, которая имеет наибольшую вероятность
		*
		* @return - Код результата выполнения функции. 0 - Success; -1 - Неизвестная ошибка.
		*/
		int Predict(cv::Mat image, bool show = true, float confThresh = 0.3, float nmsThresh = 0.3);
		/**
		* @brief Функция автоматического обучения модели детектора
		*
		* @param trainValPath	- Путь к директории с обучающим и валидационным датасетами
		* @param imageType		- Расширение изображений в директориях. Например, ".jpg", ".png" и т.д.
		* @param epochList		- Кортеж, содержащий перечень значений эпох для обучения модели
		* @param batchSizes		- Кортеж, содержащий перечень размеров батча, используемых при обучении модели
		* @param learningRate	- Кортеж, содержащий перечень значений темпа обучения модели
		* @param savePath		- Путь, по которому будут сохранены веса обученной модели в формате .pt
		* @param pretrainedPath - Путь к предобученной модели YOLOv4_tiny.pt
		*
		* @return Код результата выполнения функции.
		* Значения:
		*			0	- Success;
		*			1	- Путь к предобученной модели задан не корректно или не существует;
		*			2	- Размер батча превышает число изображений в директории с обучающей выборкой или директория пуста;
		*			3	- Путь к обучающей выборке задан неверно;
		*			-1	- Неизвестная ошибка.
		*/
		int AutoTrain(std::string trainValPath, std::string imageType, std::vector<int> epochsList = { 10, 30, 50 }, std::vector<int> batchSizes = { 4, 8, 10 },
			std::vector<float> learningRates = { 0.1, 0.01 }, std::string pretrainedPath = "detector", std::string savePath = "detector.pt");
		/**
		* @brief Функция валидации модели
		*
		* @param valDataPath	- Путь к директории с валидационной выборкой
		* @param imageType		- Расширение изображений в директориях. Например, ".jpg", ".png" и т.д.
		* @param batchSize		- Размер батча, используемого при обучении модели
		*
		* @return Значение функции потерь для текущей выборки
		*/
		float Validate(std::string valDataPath, std::string imageType, int batchSize);
	};
	/**
	 * @brief Функция для построения карты диспаратности
	 * @param map - Буфер для карты диспаратности.
	 * @param imageLeft - Изображение с левой камеры стереопары.
	 * @param imageRight - Изображение с правой камеры стереопары.
	 * @param minDisparity - Минимальный размер блока.
	 * @param numDisparities - Кол-во итераций.
	 * @param blockSize - Размер блока.
	 * @param lambda - Параметр lambda.
	 * @param sigma - Параметр sigma.
	 * @param colorMap - Цветовая схема для расцвечивания карты.
	 * @param disparityType - Тип карты диспаратности.
	 * @param saveToFile - Признак сохранения карты в файл.
	 * @param showImages - Признак отображения изображений в отдельных окнах.
	 * @return - Карта диспаратности в cv::Mat формате.
	 */
	int disparityMap(cv::Mat& map, const cv::Mat& imageLeft, const cv::Mat& imageRight, int minDisparity, int numDisparities, int blockSize, double lambda, double sigma, DISPARITY_TYPE disparityType = DISPARITY_TYPE::ALL, int colorMap = cv::COLORMAP_TURBO, bool saveToFile = false, bool showImages = false);


	/**
	* @brief Класс предиктора
	*
	* Класс реализует методы предсказания положения объекта интереса при поиощи LSTM сети.
	* @param hiddenSize_ - Размер скрытых слоев LSTM сети.
	* @param numLayers_ - Количество слоев LSTM сети.
	* @param pointsNumber_ - Количество точек (пар координат) для обучения сети.
	* @param imgSize_ - Размер изображения, на котором происходит предсказание положения объекта.
	* @param failsafeDeviation_ - Максимальная величина скользящего среднего отклонения, чтобы предсказание считалось успешным.
	* @param failsafeDeviationThreshold_ - Количество успешных предсказаний, чтобы работа сети считалась успешной.
	* @param movingAvgScale_ - Размер выборки для вычисления скользящего среднего отклонения.
	*/
	MRCV_EXPORT class Predictor {
	public:
		Predictor(const int64_t& hiddenSize_,
			const int64_t& numLayers_,
			const unsigned int& pointsNumber_,
			const std::pair<int, int>& imgSize_,
			const float& failsafeDeviation_ = 100,
			const unsigned int& failsafeDeviationThreshold_ = 5,
			const unsigned int& movingAvgScale_ = 25)
			:
			hiddenSize(hiddenSize_),
			numLayers(numLayers_),
			pointsNumber(pointsNumber_),
			imgWidth(imgSize_.first),
			imgHeight(imgSize_.second),
			failsafeDeviation(failsafeDeviation_),
			failsafeDeviationThreshold(failsafeDeviationThreshold_),
			movingAvgScale(movingAvgScale_),
			lstm(torch::nn::LSTM(torch::nn::LSTMOptions(2, hiddenSize_).num_layers(numLayers_))),
			linear(torch::nn::Linear(hiddenSize_, 2)),
			hiddenState(torch::zeros({ numLayers_, 1, hiddenSize_ })),
			cellState(torch::zeros({ numLayers_, 1, hiddenSize_ }))
		{
			lstm->to(torch::kFloat32);
			linear->to(torch::kFloat32);
		};

		/**
		 * @brief Функция обучения LSTM сети
		 *
		 * @param coordinates - Вектор входных координат.
		 * @param imageLeft - Флаг обучения. Используется для дообучения модели. (true - первое обучение, false - дообучение).
		 */
		void trainLSTMNet(const std::vector<std::pair<float, float>> coordinates, bool isTraining = false);

		/**
		 * @brief Функция дообучения LSTM сети
		 *
		 * @param coordinate - Входные координаты.
		 */
		void continueTraining(const std::pair<float, float> coordinate);

		/**
		 * @brief Функция предсказания следующей координаты
		 *
		 * @return - Предсказанные сетью координаты.
		 */
		std::pair<float, float> predictNextCoordinate();

		/**
		 * @brief Функция получения скользящего среднего отклонения предсказания
		 *
		 * @return - Скользящее среднее отклонение предсказания.
		 */
		float getMovingAverageDeviation();

		/**
		 * @brief Функция получения среднего отклонения предсказания
		 *
		 * @return - Среднее отклонение предсказания.
		 */
		float getAverageDeviation();

		/**
		 * @brief Функция получения последнего отклонения предсказания
		 *
		 * @return - Последнее отклонение предсказания.
		 */
		float getLastDeviation();

		/**
		 * @brief Функция получения статуса рабочего состояния
		 *
		 * @return - Состояние модели (true - сеть вышла на рабочий режим, false - сеть в состоянии обучения).
		 */
		bool isWorkState();

	private:
		std::pair<float, float> denormilizeOutput(std::pair<float, float> coords);
		std::pair<float, float> normilizePair(std::pair<float, float> coords);
		std::vector<std::pair<float, float>> normilizeInput(std::vector<std::pair<float, float>> coords);
		void updateDeviations();
		torch::nn::LSTM lstm{ nullptr };
		torch::nn::Linear linear{ nullptr };
		torch::Tensor hiddenState;
		torch::Tensor cellState;
		int64_t inputSize = 2;
		int64_t hiddenSize;
		int64_t numLayers;
		float failsafeDeviation;
		unsigned int failsafeDeviationThreshold;
		unsigned int movingAvgScale;
		int numPredictions = 0;
		float predictionDeviation = 0;
		float averagePredictionDeviation = 0;
		float movingAvgPredictionDeviation = 0;
		bool workState = false;
		std::vector<float> lastPredictionDeviations;
		std::pair<float, float> coordsPred = std::make_pair(0.0f, 0.0f);
		std::pair<float, float> coordsReal = std::make_pair(0.0f, 0.0f);
		unsigned int imgWidth;
		unsigned int imgHeight;
		unsigned int pointsNumber;
		std::vector<torch::Tensor> trainingData;
	};

	/**
	* @brief Класс оптимизатора
	*
	* Класс реализует методы оптимизации размера ROI исходя из размеров объекта, его пермещения и ошибки предсказания положения.
	*
	* @param sampleSize_ - Количество сэмплов, которые будут сгенерированы для обучения сети.
	* @param epochs_ - Количество эпох обучения сети.
	*/
	MRCV_EXPORT class Optimizer {
	public:
		Optimizer(size_t sampleSize_ = 1000,
			size_t epochs_ = 50000)
			:
			sampleSize(sampleSize_),
			epochs(epochs_),
			model(torch::nn::Sequential(
				torch::nn::Linear(3, 3000),
				torch::nn::ReLU(),
				torch::nn::Linear(3000, 1000),
				torch::nn::ReLU(),
				torch::nn::Linear(1000, 1)))
		{
		};

		/**
		 * @brief Функция оптимизации размера ROI
		 *
		 * @param prevCoord - Предыдущие координаты объекта.
		 * @param nextCoord - Следующие (предсказанные) координаты объекта.
		 * @param objectSize - Реальный размер объекта.
		 * @param averagePredictionDeviation - Средняя ошибка предсказания положения объекта.
		 *
		 * @return - Предсказанный сетью размер ROI, возвращает 0, если размер ROI меньше допустимого.
		 */
		float optimizeRoiSize(const std::pair<float, float>& prevCoord,
			const std::pair<float, float>& nextCoord,
			const float& objectSize,
			const float& averagePredictionError
		);
	private:
		void generateSyntheticData();
		void trainModel();
		torch::nn::Sequential model;
		size_t sampleSize;
		size_t epochs;
		float objectSize;
		float averagePredictionDeviation;
		float roiSizeNormFactor;
		std::pair<float, float> prevCoord;
		std::pair<float, float> nextCoord;
		torch::Tensor inputs;
		torch::Tensor targets;
	};

	/**
	* @brief Функция извлечения ROI из изображения
	*
	* @param image - Исходное изображение.
	* @param center - Координаты центра ROI.
	* @param roiSize - Размер ROI.
	*
	* @return - извлеченный ROI.
	*/
	cv::Mat extractROI(const cv::Mat& image, const cv::Point& center, const cv::Size& roiSize);

	// Класс сегментатора
	/**
	 * @brief Класс для решения задачи обнаружения объектов с помощью свёрточные нейронные сети YOLO5
	 * Класс используется для обнаружения объектов и сегментации изображения с получения бинарных масок обнаруженных объектов
	 * @param neuralNetSegmentator() -  конструктор класса, входные параметры:
	 * model - путь к файлй с обученной моделью нейронной сети
	 * classes - путь к файлй со списоком обнаруживамых класов объектов
	 * @param process() - запуск работы: обнаружение и распознание (классификация) объектов
	 * @param getMasks() - получение маски объекта
	 * @param getImage() - получение изображение с метками (результата)
	 * @param getClassIDs() - получение номер класса обнаруженных объекта из списка
	 * @param getConfidences() - получение вероятности с которой определён класса объекта
	 * @param getBoxes() - получение координат рамки выделяющей обнаруженный объект
	 * @param getClasses() - получение названий классов объекта
	 * @param getInference() - получение времени обработки
	 * @return
	 */
	MRCV_EXPORT class neuralNetSegmentator
	{
	private:
		// Структура сегмента
		struct outputSegment
		{
			int id;             // идентификатор класса
			float confidence;   // вероятность
			cv::Rect box;       // рамка сегмента
			cv::Mat boxMask;    // маска сегмента
		};
		// Структура параметров маски
		struct maskParams
		{
			int segChannels = 32;
			int segWidth = 160;
			int segHeight = 160;
			int netWidth = 640;
			int netHeight = 640;
			float maskThreshold = 0.5;
			cv::Size srcImgShape;
			cv::Vec4d params;
		};
		// Параметры обработки
		const float SCORE_THRESHOLD = 0.50;
		const float NMS_THRESHOLD = 0.45;
		const float CONFIDENCE_THRESHOLD = 0.45;
		// Параметры шрифтов
		const float FONT_SCALE = 0.7;
		const int   THICKNESS = 1;
		// Цветовые константы
		const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
		const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
		const cv::Scalar RED = cv::Scalar(0, 0, 255);
		const cv::Scalar GREEN = cv::Scalar(0, 255, 0);
		// Структура нейросети
		cv::dnn::Net network;
		// Ширина и высота входного изображения
		const int input_width = 640;
		const int input_height = 640;
		// Переменные для хранения результатов обработки
		std::vector<std::string> classes;       // вектор распознаваемых классов
		std::vector<int> classesIDSet;          // номер класса обнаруженных объекта из списка
		std::vector<cv::Rect> boxesSet;         // координаты рамки выделяющей обнаруженный объект
		std::vector<float> confidencesSet;      // вероятность с которой определён класса объекта
		std::vector<std::string> classesSet;    // название класса объекта
		std::vector<cv::Scalar> masksColorsSet; // цветные маски объектов
		std::vector<cv::Mat> masksSet;          // бинарные маски объектов
		cv::Mat processedImage;                 // изображение с метками (результат)
		float timeInference;                    // время обработки

		// Получить строковые значения классов
		int readСlasses(const std::string file_path);
		// Инициализация нейросети
		int initializationNetwork(const std::string model_path, const std::string classes_path);
		// Прорисовка
		void letterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, cv::Size& newShape,
			bool autoShape, bool scaleFill, bool scaleUp, int stride);
		// Отрисовка метки
		void drawLabel(cv::Mat& img, std::string label, int left, int top);
		void drawResult(cv::Mat& image, std::vector<outputSegment> result,
			std::vector<std::string> class_name);
		// Предобработка результатов
		std::vector<cv::Mat> preProcess(cv::Mat& img, cv::Vec4d& params);
		void getMask(const cv::Mat& mask_proposals, const cv::Mat& mask_protos,
			outputSegment& output, const maskParams& maskParams);
		// Постобработка результатов
		cv::Mat postProcess(cv::Mat& img, std::vector<cv::Mat>& outputs,
			const std::vector<std::string>& class_name,
			cv::Vec4d& params);
	public:
		neuralNetSegmentator(const std::string model, const std::string classes);
		cv::Mat process(cv::Mat& img);
		cv::Mat getImage(void);
		std::vector<cv::Mat> getMasks(void);
		std::vector<int> getClassIDs(void);
		std::vector<float> getConfidences(void);
		std::vector<cv::Rect> getBoxes(void);
		std::vector<std::string> getClasses(void);
		float getInference(void);
	}; // class  neuralNetSegmentator

	/**
	 * @brief Функция формирования стереопараы
	 * Функция склеивает изображение камеры 01 с изображениеме камеры 02 в одно изображение
	 * @param inputImageCamera01 - входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param inputImageCamera02 - входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param outputImage - исходящее склеенное в стереопару цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; -1 - Неизвестная ошибка.
	 */
	int makingStereoPair(cv::Mat& inputImageCamera01, cv::Mat& inputImageCamera02, cv::Mat& inputStereoImage);
	/**
	 * @brief Функция вывода изображения на экран
	 * Функция масштабирует изображение по указанному коэфициенту и выводи его на экран
	 * @param inputImage - входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param windowName - имя окна вывода
	 * @param CoefShowWindow - коэффициент масштабирования
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; -1 - Неизвестная ошибка.
	 */
	int showImage(cv::Mat& inputImage, const cv::String windowName, double CoefShowWindow = 0.5);
	/**
	 * @brief Функция чтения параметров стереокамеры камеры из xml-файла.
	 * Функция читает параметры камеры из xml-файла и записывает их в структуру данных cameraParameters
	 * @param fileNameCameraParameters - ввод пути к файлу c параметрами камеры.
	 * @param cameraParameters - исхдящие параметры камеры
	 *  Внутренние параметры камер 01 и 02 стереокамеры
	 *     M1 - матрица камеры 3x3  (камеры 01 стереокамеры)
	 *     D1 - вектор коэффициентов искажения (камеры 01 стереокамеры), коэффициенты радиальной и тангенциальной дисторсии
	 *     M2 - матрица камеры 3x3  (камеры 02 стереокамеры)
	 *     D2 - вектор коэффициентов искажения (камеры 02 стереокамеры), коэффициенты радиальной и тангенциальной дисторсии
	 *  Внешиние параметры камер стереокамеры
	 *     E  - существенная матрица стереокамеры
	 *     F  - фундаментальная матрица стереокамеры
	 *     R - матрица поворота 3x3 камеры 02 относительно камеры 01
	 *     T  - вектор смещения  камеры 02 относительно камеры 01
	 *     R1 - Матрица поворота 3x3 для выполнения процедуры выравнивания (ректификации) для первой 01
	 *     R2 - Матрица поворота 3x3 для выполнения процедуры выравнивания (ректификации) для первой 02
	 *  Матрицы проекции - проецирует 3D точки, заданные в исправленной системе координат камеры, на исправленное 2D изображение камеры
	 *     P1 - Матрицы проекции 3x4 в новых (выравненных) системах координат для камеры 01
	 *     P2 - Матрицы проекции 3x4 в новых (выравненных) системах координат для камеры 02
	 *     Q  - Матрица 4x4 преобразования перспективы, отображения несоответствия глубине
	 *  Карта переназначения используются для быстрого преобразования изображения
	 *     map11 - карта 01 для переназначения камеры 01
	 *     map12 - карта 02 для переназначения камеры 01
	 *     map21 - карта 01 для переназначения камеры 02
	 *     map22 - карта 02 для переназначения камеры 02
	 *  Дополнительные параметры
	 *     imageSize - размер изображения
	 *     rms - ошибка перепроецирорования
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int readCameraStereoParametrsFromFile(const char* pathToFileCameraParametrs, mrcv::cameraStereoParameters& cameraParameters);
	/**
	 * @brief Функция для коррекция искажений (дисторсии) и выравнивания (ректификации) изображения .
	 * Функция принимает два изображение и возвращает скоректированное изображение на основе данных в картах точек map11 и map12
	 * @param imageInput - входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param imageOutput - выходное (преобразованное) цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param map11 - входная первая карта точек (x, y) или просто значений x для исправления геометрических искажений
	 * @param map12 - входная вторая карта значений y, (пустая карта, если map1 равен точкам (x,y)) соответственно.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int convetingUndistortRectify(cv::Mat& imageInput, cv::Mat& imageOutput, cv::Mat& map11, cv::Mat& map12);
	/**
	 * @brief Функция для формирования облака 3D точек на основе проекций этих точек на изображения стереокамеры
	 * Функция находит 3D точки по их проекциям на изображение на основе метода построения карты расхождений (диспаратности)
	 * Данные об облаке 3D точек записываются в points3D
	 * @param inputImageCamera01    - входное из камеры 01 цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param inputImageCamera02    - входное из камеры 02 цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param points3D              - входные и исходные данные для хранения информации об облаке 3D точек
	 * @param settingsMetodDisparity - вводные настройки метода поиска расхождений (диспаратности) для поиска 3D точек
	 * @param disparityMap          - входная карта диспаратости, если mrcv::metodDisparity::MODE_NONE,
	 * исходящия карта диспаратности, если другой параметр в mrcv::metodDisparity , кроме mrcv::metodDisparity::MODE_NONE
	 * формат CV_32F
	 * @param cameraParameters      - входящие параметры стереокамеры
	 * @param limit3dPoints         - лимит на количество точек на выходе
	 * @param limitsOutlierArea     - параметры области для отсеивания выбросов {x_min, y_min, z_min, x_max, y_max, z_max}
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение;
	 * 2 - Пустая исходная карта диспаратности, елсли выбран MODE_NONE; 3 - Ошибка в параметрах калибровки; -1 - Неизвестная ошибка.
	 */
	int find3dPointsADS(cv::Mat& inputImageCamera01, cv::Mat& inputImageCamera02, mrcv::pointsData& points3D,
		mrcv::settingsMetodDisparity& settingsMetodDisparity, cv::Mat& disparityMap,
		mrcv::cameraStereoParameters& cameraParameters, int limitOutPoints, std::vector<double> limitsOutlierArea);
	/**
	 * @brief Функция для обнаружения объектов и их сегментов на изображении с помощью нейронной сети
	 * Функия использует готовую нейронную сеть через OpenCV, загружая её из файлов
	 * @param imageInput - входное цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param imageOutput - исходящие цветное RGB изображение с обнаруженными объектами, формата cv::Mat CV_8UC3
	 * @param replyMasks -  изходящий массив с масками обнаруженных объектов на кадый объект в оттельности, формата cv::Mat CV_8UC1
	 * @param filePathToModelYoloNeuralNet - путь к файлй с обученной моделью нейронной сети.
	 * @param filePathToClasses - путь к файлй со списоком обнаруживамых класов объектов
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение;  -1 - Неизвестная ошибка.
	 */
	int detectingSegmentsNeuralNet(cv::Mat& imageInput, cv::Mat& imageOutput, std::vector<cv::Mat>& replyMasks,
		const std::string filePathToModelYoloNeuralNet, const std::string filePathToClasses);
	/**
	 * @brief Функция для определения координат 3D точек в сегментах идентифицированных объектов
	 * Функция сопосталяет сегмент объекта с 3D точками по их проекциям на изображение и записывает результат в points3D
	 * @param points3D - входные и исходные даные для хранения информации о облаке 3D точек
	 * @param replyMasks - изходящий массив с масками обнаруженных объектов на кадый объект в отдельности, формата cv::Mat CV_8UC1
	 * @return - код результата работы функции. 0 - Success; 1 - Нет точек; 2 - Нет сегменации; -1 - Неизвестная ошибка.
	 */
	int matchSegmentsWith3dPoints(mrcv::pointsData& points3D, std::vector<cv::Mat>& replyMasks);
	/**
	 * @brief Функция нанесения координат 3D центра сегмента на изображение в виде текста
	 * @param inputImageCamera01 - входное цветое RGB изображени камеры 01, формата cv::Mat CV_8UC3
	 * @param outputImage - исходящие цветое RGB изображени с нанесённой, формата cv::Mat CV_8UC3
	 * @param points3D - входные и исходные даные для хранения информации о облаке 3D точек
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 3 - Отсутсвуют сегменты  -1 - Неизвестная ошибка.
	 */
	int addToImageCenter3dSegments(cv::Mat& inputImage, cv::Mat& outputImage, mrcv::pointsData& points3D);
	/**
	 * @brief Функция для вывода карты диспаратноси в цвете и в нормированном виде
	 * @param disparityMap - входная карта диспаратости, формата cv::Mat CV_32F
	 * @param windowName - имя окна вывода
	 * @param CoefShowWindow - коэффициент масштабирования
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; -1 - Неизвестная ошибка.
	 */
	int showDispsarityMap(cv::Mat& disparityMap, const cv::String windowName, double CoefShowWindow = 0.5);
	/**
	 * @brief Функция проекции 3D сцены на 2D изображение для вывода на экран
	 * @param points3D           - входные даные для хранения информации о облаке 3D точек
	 * @param parameters3dSceene - параметры 3D сцены
	 *
	 *
	 *
	 *
	 * @param cameraParameters   - входящие параметры стереокамеры
	 * @param outputImage3dSceene - исходящие цветное RGB изображение 3D сцены, формата cv::Mat CV_8UC3
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое данные о точках; 3 - Ошибка в параметрах калибровки;
	 * -1 - Неизвестная ошибка.
	 */
	int getImage3dSceene(mrcv::pointsData& points3D, mrcv::parameters3dSceene& parameters3dSceene,
		mrcv::cameraStereoParameters& cameraParameters, cv::Mat& outputImage3dSceene);
	/**
	 * @brief Функция записи в текстовый файл координат 3D точек в сегментах идентифицированных объектов
	 * @param points3D - входные и исходные даные для хранения информации о облаке 3D точек
	 * @param pathToFile - ввод пути к файлу.
	 * @return - код результата работы функции. 0 - Success; 1 - Нет точек; -1 - Неизвестная ошибка.
	 */
	int saveInFile3dPointsInObjectsSegments(mrcv::pointsData& points3D, const cv::String pathToFile);
	/**
	 * @brief Функции для определения координат 3D точек в сегментах идентифицированных объектов и
	 *  восстановления 3D сцены по двумерным изображениям
	 * @param inputImageCamera01       - входное из камеры 01 цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param inputImageCamera02       - входное из камеры 02 цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param cameraParameters         - входящие параметры стереокамеры
	 * @param inputImageCamera01Remap  - выходное (преобразованное) камеры 01 цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param inputImageCamera02Remap  - выходное (преобразованное) камеры 02 цветное RGB изображение, формата cv::Mat CV_8UC3
	 * @param settingsMetodDisparity   - вводные настройки метода поиска расхождений (диспаратности) для поиска 3D точек
	 * @param disparityMap             - входная карта диспаратости, если mrcv::metodDisparity::MODE_NONE,
	 * исходящия карта диспаратности, если другой параметр в mrcv::metodDisparity , кроме mrcv::metodDisparity::MODE_NONE
	 * формат CV_32F
	 * @param points3D                 - входные и исходные данные для хранения информации о облаке 3D точек
	 * @param replyMasks               - входящий и изходящий массив с масками обнаруженных объектов на кадый объект в отдельности, формата cv::Mat CV_8UC1
	 *  Если данные о сегменте введены то используется алгоритм сегментации не используется.
	 * @param imageOutput              - исходящие цветное RGB изображение с обнаруженными объектами, формата cv::Mat CV_8UC3
	 * @param outputImage3dSceene - исходящие цветное RGB изображение 3D сцены, формата cv::Mat CV_8UC3
	 * @param parameters3dSceene - параметры 3D сцены
	 * @param filePathToModelYoloNeuralNet - путь к файлй с готовой моделью нейронной сети.
	 * @param filePathToClasses - путь к файлй со списоком обнаруживамых класов объектов
	 * @param limit3dPoints         - лимит на количество точек на выходе
	 * @param limitsOutlierArea     - параметры области для отсеивания выбросов {x_min, y_min, z_min, x_max, y_max, z_max}
	 * @return - код результата работы функции. 0 - Success; -1 - Неизвестная ошибка.
	 */
	int find3dPointsInObjectsSegments(cv::Mat& inputImageCamera01, cv::Mat& inputImageCamera02,
		mrcv::cameraStereoParameters& cameraParameters,
		cv::Mat& inputImageCamera01Remap, cv::Mat& inputImageCamera02Remap,
		mrcv::settingsMetodDisparity& settingsMetodDisparity, cv::Mat& disparityMap,
		mrcv::pointsData& points3D, std::vector<cv::Mat>& replyMasks, cv::Mat& outputImage,
		cv::Mat& outputImage3dSceene, mrcv::parameters3dSceene& parameters3dSceene,
		const std::string filePathToModelYoloNeuralNet, const std::string filePathToClasses,
		int limitOutPoints = 3000, std::vector<double> limitsOutlierArea = { -4.0e3, -4.0e3, 450, 4.0e3, 4.0e3, 3.0e3 });
		
#ifdef MRCV_CUDA_ENABLED 
	/**
	 * @brief Функция отражения изображения с помощью CUDA.
	 * Отражает изображение по горизонтали, вертикали или обеим осям.
	 *
	 * @param imageInput - входное (исходное) изображение cv::Mat.
	 * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
	 * @param flipCode - Код отражения: 0 - вертикальное отражение; 1 - горизонтальное отражение; -1 - обе стороны.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 * @note Требуется GPU с поддержкой CUDA и CUDA Toolkit 12.4.
	 */
	MRCV_EXPORT int flipImageCuda(cv::Mat& imageInput, cv::Mat& imageOutput, int flipCode);

	/**
	 * @brief Функция поворота изображения на заданный угол с помощью CUDA.
	 * Поворачивает изображение на определённый угол с использованием центральной точки.
	 *
	 * @param imageInput - входное (исходное) изображение cv::Mat.
	 * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
	 * @param angle - угол поворота в градусах.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 * @note Требуется GPU с поддержкой CUDA и CUDA Toolkit 12.4.
	 */
	MRCV_EXPORT int rotateImageCuda(cv::Mat& imageInput, cv::Mat& imageOutput, double angle);

	/**
	 * @brief Функция аугментации изображений с помощью CUDA.
	 * Выполняет аугментацию для набора входных изображений на основе заданных методов и сохраняет результат.
	 *
	 * @param inputImagesAugmetation - вектор входных изображений (cv::Mat) для аугментации.
	 * @param outputImagesAugmetation - вектор для сохранения выходных (преобразованных) изображений.
	 * @param augmetationMethod - вектор методов аугментации (mrcv::AUGMENTATION_METHOD) для применения.
	 * @return Код результата выполнения функции. 0 - успех; -1 - исключение (OpenCV или файловой системы).
	 *
	 * Функция проверяет наличие директории для сохранения изображений и создает её при необходимости. Для каждого изображения
	 * выполняется указанная операция (например, поворот или отражение) с последующей проверкой и сохранением результата в директорию.
	 * @note Требуется GPU с поддержкой CUDA и CUDA Toolkit 12.4.
	 */
	MRCV_EXPORT int augmetationCuda(std::vector<cv::Mat>& inputImagesAugmetation, std::vector<cv::Mat>& outputImagesAugmetation, std::vector<mrcv::AUGMENTATION_METHOD> augmetationMethod);

	/**
	 * @brief Функция общей калибровки с помощью CUDA.
	 * @param imagesL
	 * @param imagesR
	 * @param pathToImagesL
	 * @param pathToImagesR
	 * @param calibrationParametersL
	 * @param calibrationParametersR
	 * @param calibrationParameters
	 * @param chessboardColCount
	 * @param chessboardRowCount
	 * @param chessboardSquareSize
	 * @note Требуется GPU с поддержкой CUDA и CUDA Toolkit 12.4.
	 */
	MRCV_EXPORT void cameraCalibrationCuda(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathToImagesL, std::string pathToImagesR, CalibrationParametersMono& calibrationParametersL, CalibrationParametersMono& calibrationParametersR, CalibrationParametersStereo& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 *@brief Функция калибровки одиночной камеры с помощью CUDA.
	 * @param images
	 * @param pathToImages
	 * @param calibrationParameters
	 * @param chessboardColCount
	 * @param chessboardRowCount
	 * @param chessboardSquareSize
	 * @note Требуется GPU с поддержкой CUDA и CUDA Toolkit 12.4.
	 */
	MRCV_EXPORT void cameraCalibrationMonoCuda(std::vector<cv::String> images, std::string pathToImages, CalibrationParametersMono& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief Функция калибровки стерео пары с помощью CUDA.
	 * @param imagesL
	 * @param imagesR
	 * @param pathToImagesL
	 * @param pathToImagesR
	 * @param calibrationParameters
	 * @param chessboardColCount
	 * @param chessboardRowCount
	 * @param chessboardSquareSize
	 * @note Требуется GPU с поддержкой CUDA и CUDA Toolkit 12.4.
	 */
	MRCV_EXPORT void cameraCalibrationStereoCuda(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathToImagesL, std::string pathToImagesR, CalibrationParametersStereo& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief Функция сравнения двух изображений с помощью CUDA.
	 * @param img1 - Первое входное изображение (cv::Mat).
	 * @param img2 - Второе входное изображение (cv::Mat).
	 * @param methodCompare - Метод сравнения (true для гистограммы, false для L2-нормы).
	 * @return - Различие между изображениями (корреляция гистограмм или нормализованное L2-расстояние).
	 * @note Требуется GPU с поддержкой CUDA и CUDA Toolkit 12.4.
	 */
	MRCV_EXPORT double compareImagesCuda(cv::Mat img1, cv::Mat img2, bool methodCompare);

	/**
	 * @brief Функция для построения карты диспаратности с помощью CUDA.
	 * @param map - Буфер для карты диспаратности.
	 * @param imageLeft - Изображение с левой камеры стереопары.
	 * @param imageRight - Изображение с правой камеры стереопары.
	 * @param minDisparity - Минимальный размер блока.
	 * @param numDisparities - Кол-во итераций.
	 * @param blockSize - Размер блока.
	 * @param lambda - Параметр lambda.
	 * @param sigma - Параметр sigma.
	 * @param colorMap - Цветовая схема для расцвечивания карты.
	 * @param disparityType - Тип карты диспаратности.
	 * @param saveToFile - Признак сохранения карты в файл.
	 * @param showImages - Признак отображения изображений в отдельных окнах.
	 * @return - Карта диспаратности в cv::Mat формате.
	 * @note Требуется GPU с поддержкой CUDA и CUDA Toolkit 12.4.
	 */
	MRCV_EXPORT int disparityMapCuda(cv::cuda::GpuMat& map, const cv::Mat& imageLeft, const cv::Mat& imageRight, int minDisparity, int numDisparities, int blockSize, double lambda, double sigma, DISPARITY_TYPE disparityType, int colorMap, bool saveToFile, bool showImages);
#endif

}

        
