#pragma once

#include <mrcv/export.h>
#include <mrcv/mrcv-common.h>

namespace mrcv
{
	/////////////////////////////////////////////////////////////////////////////
	// Утилиты
	/////////////////////////////////////////////////////////////////////////////
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
}
