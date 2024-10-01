#pragma once

#include <mrcv/export.h>
#include <mrcv/mrcv-common.h>
#include <mrcv/mrcv-detector.h>

namespace mrcv
{
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
		void Initialize(int gpu_id, int width, int height, std::vector<std::string>&& name_list, std::string encoder_name, std::string pretrained_path);
		void SetTrainTricks(trainTricks& tricks);
		void Train(float learning_rate, unsigned int epochs, int batch_size, std::string train_val_path, std::string image_type, std::string save_path);
		void LoadWeight(std::string weight_path);
		void Predict(cv::Mat& image, const std::string& which_class);
	private:
		int width = 512;
		int height = 512;
		std::vector<std::string> name_list;
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
	 * 
	 *
	 * @param textError - текс сообщения, которое будет записано в изображение.
	 * @return - изображение с кодом ошибки (для информирование оператора в режиме рельного времени).
	 */
	cv::Mat getErrorImage(std::string textError);
	/**
	 * @brief Функция автоматической предобработки изображения, кооррекции контраста.
	 * Функция автоматической коррекции контраста изображения с помощью метода Эквализации Гистограмм
	 * @param imageInput - входное (исходное) изображение cv::Mat.
	 * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
	 * @param clipLimit - пороговое значение для ограничения контрастности
	 * @param gridSize - Размер сетки. Изображение будет разделено на одинакового размера части в виде матрицы,
	 * параметр определяет количество строк и столбцов
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int increaseImageContrastEqualizeHist(cv::Mat& imageInput, cv::Mat& imageOutput);
	/**
	 * @brief Функция автоматической предобработки изображения, кооррекции контраста.
	 * Функция автоматической коррекции контраста изображения с помощью метода Адаптивной Эквализации Гистограмм
	 * Contrast Limited Adaptive Histogram Equalization
	 * @param imageInput - входное (исходное) изображение cv::Mat.
	 * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
	 * @param clipLimit - пороговое значение для ограничения контрастности
	 * @param gridSize - Размер сетки. Изображение будет разделено на одинакового размера части в виде матрицы,
	 * параметр определяет количество строк и столбцов
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int increaseImageContrastCLAHE(cv::Mat& imageInput, cv::Mat& imageOutput, double clipLimit, cv::Size gridSize);
	/**
	 * @brief Функция автоматической предобработки изображения, кооррекции контраста.
	 * Функция автоматической коррекции контраста изображения с помощью метода Адаптивной Эквализации Гистограмм через цетовое пространство Lab
	 * @param imageInput - входное (исходное) изображение cv::Mat.
	 * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int increaseImageContrastСolorLabCLAHE(cv::Mat& imageInput, cv::Mat& imageOutput, double clipLimit, cv::Size gridSize);
	/**
	 * @brief Функция автоматической предобработки изображения, кооррекции яркости.
	 * Функция автоматической коррекции яркости изображения с помощью степенного преобразовамния (метод Гамма-Коррекции)
	 * @param imageInput - входное (исходное) изображение cv::Mat.
	 * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
	 * @param gamma - степень преобразования (1 - без изменений от 1 до 0 - осветление, от 1 до 10 - затемнение изображения)
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int changeImageBrightness(cv::Mat& imageInput, cv::Mat& imageOutput, double gamma);
	/**
	 * @brief Функция предварительной обработки изображений (автоматическая коррекция контраста и яркости, резкости)
	 * Функция автоматической предобработки изображения, кооррекции яркости и контраста, резкости.
	 * @param image - изображение cv::Mat, над которым происходит преобразование.
	 * @param metodImagePerProcessingBrightnessContrast - вектор параметров, которые опрределяют, какие преобразования и в какой последовательноси  проводить.
	 *  none  - без изменений
	 *  brightnessLevelUp - увеличение уровня яркости
	 *  brightnessLevelDown - уменьшение уровня яркости
	 *  equalizeHist -  повышение контрастности, метод 01
	 *  CLAHE -  повышение контрастности, метод 02
	 *  colorLabCLAHE -  повышение контрастности, метод 03
	 *  BGRtoGray - преобразование типа изображения к из цветноко BGR к монохромному (серому)
	 *  sharpening01 -  повышение резкости, метод 01
	 *  sharpening02-  повышение резкости, метод 02
	 *  noiseFilteringMedianFilter - фильтрация изображения от импульсных шумов
	 *  noiseFilteringAvarageFilter- фильтрация изображения от шумов
	 *  correctionGeometricDeformation - коррекция геометрических искажений
	 * @param fileNameCameraParameters - путь к файлу c параметрами камеры для исправления геометрических искажений.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int preprocessingImage(cv::Mat& imageInput, std::vector<mrcv::IMG_PREPROCESSING_METHOD> metodImagePerProcessingm, const std::string& fileNameCameraParameters);
	/**
	 * @brief Функция чтения параметров камеры из файла.
	 * @param fileNameCameraParameters - путь к файлу c параметрами камеры.
	 * @param map11 - первая карта точек (x, y) или просто значений x для исправления геометрических искажений
	 * @param map12 - вторая карта значений y, (пустая карта, если map1 равен точкам (x,y)) соответственно.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */
	int readCameraParametrsFromFile(const char* pathToFileCameraParametrs, cv::Mat& map11, cv::Mat& map12);
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
			std::vector<float> learningRates = { 0.1, 0.01 }, std::string savePath = "detector.pt", std::string pretrainedPath = "detector");
		/**
		* @brief Функция валидации модели
		*
		* @param valDataPath	- Путь к директории с валидационной выборкой
		* @param imageType		- Расширение изображений в директориях. Например, ".jpg", ".png" и т.д.
		* @param batchSize		- Размер батча, используемого при обучении модели
		*
		* @return Значение функции потерь для текущей выборки
		*/
		float Detector::Validate(std::string valDataPath, std::string imageType, int batchSize);
	};
}