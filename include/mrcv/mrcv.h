#pragma once

#include <mrcv/export.h>
#include <mrcv/mrcv-common.h>

namespace mrcv
{
	/////////////////////////////////////////////////////////////////////////////
	// Óòèëèòû
	/////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief Ôóíêöèÿ ñëîæåíèÿ äâóõ öåëûõ ÷èñåë.
	 * @param a - Ïåðâîå ñëàãàåìîå.
	 * @param b - Âòîðîå ñëàãàåìîå.
	 * @return - Ðåçàëüòàò âû÷ñèëåíèÿ âûðàæåíèÿ a + b
	 */
	MRCV_EXPORT int add(int a, int b);
	/**
	 * @brief Ôóíêöèÿ çàãðóçêè èçîáðàæåíèÿ.
	 *
	 * Ôóíêöèÿ èñïîëüçóåòñÿ äëÿ çàãðóçêè èçîáðàæåíèÿ ñ íîñèòåëÿ è îòîáðàæåíèÿ çàãðóæåííîãî èçîáðàæåíèÿ â ìîäàëüíîì îêíå.
	 *
	 * @param image - îáúåêò cv::Mat äëÿ õðàíåíèÿ çàãðóæåííîãî èçîáðàæåíèÿ.
	 * @param pathToImage - ïîëíûé ïóòü ê ôàéëó ñ èçîáðàæåíèåì.
	 * @param showImage - ôëàã, îòâå÷àþùèé çà îòîáðàæåíèå ìîäàëüíîãî îêíà (false ïî óìîë÷àíèþ).
	 * @return - êîä ðåçóëüòàòà ðàáîòû ôóíêöèè. 0 - Success; 1 - Íåâîçìîæíî îòêðûòü èçîáðàæåíèå; -1 - Unhandled Exception.
	 */
	MRCV_EXPORT int readImage(cv::Mat& image, std::string pathToImage, bool showImage = false);
	/**
	 * @brief Ôóíêöèÿ âûâîäà èíôîðìàöèè î òåêóùåé ñáîðêå OpenCV.
	 * @return Ñòðîêà ñ äèàãíîñòè÷åñêîé èíôîðìàöèåé.
	 */
	MRCV_EXPORT std::string getOpenCVBuildInformation();
	/**
	 * @brief Ôóíêöèÿ çàïèñè âèäåîïîòîêà íà äèñê.
	 *
	 * Ôóíêöèÿ ìîæåò èñïîëüçîâàòüñÿ äëå ðåàëèçàöèè ðàáîòû âèäåîðåãèñòðàòîðà.
	 *
	 * @param cameraID - ID êàìåðû.
	 * @param recorderInterval - Èíòåðâàë çàïèñè â ñåêóíäàõ.
	 * @param fileName - Ìàñêà ôàëà.
	 * @param codec - Êîäåê, èñïîëüçóåìûé äëÿ ñîçäàíèÿ âèäåîôàéëà.
	 * @return - êîä ðåçóëüòàòà ðàáîòû ôóíêöèè. 0 - Success; 1 - ID êàìåðû çàäàí íåâåðíî; 2 - Èíòåðâàë çàõâàòà ìåíüøå ìèíèìàëüíîãî; 3 - Íå óäàëîñü çàõâàòèòü êàìåðó; 4 - Íå óäàëîñü ñîçäàòü îáúåêòñ cv::VideoWriter; -1 - Unhandled Exception.
	 */
	MRCV_EXPORT int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec);
	/**
	* @brief Ôóíêöèÿ ñêà÷àèâàíèÿ ôàéëà ßíäåêñà.
	* @param query - Ñòðîêà çàïðîñà äëÿ ïîèñêà.
	* @param minWidth - Ìèíèìàëüíàÿ øèðèíà èçîáðàæåíèÿ.
	* @param minHeight - Ìèíèìàëüíàÿ âûñîòà èçîáðàæåíèÿ.
	* @param nameTemplate - Øàáëîí èìåíè ôàéëà.
	* @param outputFolder - Ïàïêà äëÿ ñêà÷èâàíèÿ.
	* @param separateDataset - Ôëàã ðàçáèâêè äàòàñåòà íà òðåíèðîâî÷íóþ è òåñòîâóþ âûáîðêè.
	* @param trainsetPercentage - Ïðîöåíò äëÿ ðàñïðåäåëåíèÿ ìåæäó ïàïêàìè.
	* @param countFoto - Êîëè÷åñòâî íåîáõîäèìûõ ôîòî äëÿ ñêà÷èâàíèÿ.
	* @param money - Ïëàòíûé èëè áåñïëàòíûé âàðèàíò ðàáîòû.
	* @param key - ßíäåêñ key.
	* @param secretKey - ßíäåêñ secretKey.
	* @return - Ðåçóëüòàò ðàáîòû ôóíêöèè.
	*/
	MRCV_EXPORT int getImagesFromYandex(std::string query, int minWidth, int minHeight, std::string nameTemplate, std::string outputFolder, bool separateDataset, unsigned int trainsetPercentage, unsigned int countFoto, bool money, std::string key, std::string secretKey);
	/////////////////////////////////////////////////////////////////////////////
	// Êàëèáðîâêà
	/////////////////////////////////////////////////////////////////////////////
	/**
     * @brief Ôóíêöèÿ îáùåé êàëèáðîâêè.
     * @param imagesL - Âåêòîð ñòðîê èì¸í èçîáðàæåíèé ëåâîé êàìåðû.
     * @param imagesR - Âåêòîð ñòðîê èì¸í èçîáðàæåíèé ïðàâîé êàìåðû.
     * @param pathToImagesL - Ïóòü ê ïàïêå ñ èçîáðàæåíèÿìè ëåâîé êàìåðû.
     * @param pathToImagesR - Ïóòü ê ïàïêå ñ èçîáðàæåíèÿìè ïðàâîé êàìåðû.
     * @param calibrationParametersL - Ñòðóêòóðà äëÿ õðàíåíèÿ êàëèáðîâî÷íûõ ïàðàìåòðîâ ëåâîé êàìåðû.
     * @param calibrationParametersR - Ñòðóêòóðà äëÿ õðàíåíèÿ êàëèáðîâî÷íûõ ïàðàìåòðîâ ïðàâîé êàìåðû.
     * @param calibrationParameters - Ñòðóêòóðà äëÿ õðàíåíèÿ êàëèáðîâî÷íûõ ïàðàìåòðîâ ñòåðåî ïàðû.
     * @param chessboardColCount - Êîëè÷åñòâî êëþ÷åâûõ òî÷åê êàëèáðîâî÷íîé äîñêè ïî ñòîëáöàì.
     * @param chessboardRowCount - Êîëè÷åñòâî êëþ÷åâûõ òî÷åê êàëèáðîâî÷íîé äîñêè ïî ñòðîêàì.
     * @param chessboardSquareSize - Ðàçìåð ïîëÿ êàëèáðî÷íî äîñêè â ìì.
     */
	MRCV_EXPORT void cameraCalibration(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathToImagesL, std::string pathToImagesR, CalibrationParametersMono& calibrationParametersL, CalibrationParametersMono& calibrationParametersR, CalibrationParametersStereo& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief Ôóíêöèÿ êàëèáðîâêè îäèíî÷íîé êàìåðû.
	 * @param images - Âåêòîð ñòðîê èì¸í èçîáðàæåíèé êàìåðû.
	 * @param pathToImages - Ïóòü ê ïàïêå ñ èçîáðàæåíèÿìè êàìåðû.
	 * @param calibrationParameters - Ñòðóêòóðà äëÿ õðàíåíèÿ êàëèáðîâî÷íûõ ïàðàìåòðîâ êàìåðû.
	 * @param chessboardColCount - Êîëè÷åñòâî êëþ÷åâûõ òî÷åê êàëèáðîâî÷íîé äîñêè ïî ñòîëáöàì.
     * @param chessboardRowCount - Êîëè÷åñòâî êëþ÷åâûõ òî÷åê êàëèáðîâî÷íîé äîñêè ïî ñòðîêàì.
     * @param chessboardSquareSize - Ðàçìåð ïîëÿ êàëèáðî÷íî äîñêè â ìì.
	 */
	MRCV_EXPORT void cameraCalibrationMono(std::vector<cv::String> images, std::string pathToImages, CalibrationParametersMono& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief Ôóíêöèÿ êàëèáðîâêè ñòåðåî ïàðû.
	 * @param imagesL - Âåêòîð ñòðîê èì¸í èçîáðàæåíèé ëåâîé êàìåðû.
     * @param imagesR - Âåêòîð ñòðîê èì¸í èçîáðàæåíèé ïðàâîé êàìåðû.
     * @param pathToImagesL - Ïóòü ê ïàïêå ñ èçîáðàæåíèÿìè ëåâîé êàìåðû.
     * @param pathToImagesR - Ïóòü ê ïàïêå ñ èçîáðàæåíèÿìè ïðàâîé êàìåðû.
	 * @param calibrationParameters - Ñòðóêòóðà äëÿ õðàíåíèÿ êàëèáðîâî÷íûõ ïàðàìåòðîâ ñòåðåî ïàðû.
	 * @param chessboardColCount - Êîëè÷åñòâî êëþ÷åâûõ òî÷åê êàëèáðîâî÷íîé äîñêè ïî ñòîëáöàì.
     * @param chessboardRowCount - Êîëè÷åñòâî êëþ÷åâûõ òî÷åê êàëèáðîâî÷íîé äîñêè ïî ñòðîêàì.
     * @param chessboardSquareSize - Ðàçìåð ïîëÿ êàëèáðî÷íî äîñêè â ìì.
	 */
	MRCV_EXPORT void cameraCalibrationStereo(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathToImagesL, std::string pathToImagesR, CalibrationParametersStereo& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief Ôóíêöèÿ ÷òåíèÿ ïàðàìåòðîâ êàëèáðîâêè îäèíî÷íîé êàìåðû.
	 * @param fileName - Ïîëíûé ïóòü ê ôàéëó êàëèáðîâî÷íûõ ïàðàìåòðîâ.
	 * @return
	 */
	MRCV_EXPORT CalibrationParametersMono readCalibrationParametersMono(std::string fileName);

	/**
	 * @brief Ôóíêöèÿ çàïèñè ïàðàìåòðîâ êàëèáðîâêè îäèíî÷íîé êàìåðû.
	 * @param fileName - Ïîëíûé ïóòü ê ôàéëó êàëèáðîâî÷íûõ ïàðàìåòðîâ.
	 * @param parameters - Ñòðóêòóðà äëÿ õðàíåíèÿ êàëèáðîâî÷íûõ ïàðàìåòðîâ.
	 */
	MRCV_EXPORT void writeCalibrationParametersMono(std::string fileName, CalibrationParametersMono parameters);

	/**
	 * @brief Ôóíêöèÿ çàïèñè ïàðàìåòðîâ êàëèáðîâêè ñòåðåî ïàðû
	 * @param fileName - Ïîëíûé ïóòü ê ôàéëó êàëèáðîâî÷íûõ ïàðàìåòðîâ.
	 * @param parameters - Ñòðóêòóðà äëÿ õðàíåíèÿ êàëèáðîâî÷íûõ ïàðàìåòðîâ.
	 */
	MRCV_EXPORT void writeCalibrationParametersStereo(std::string fileName, CalibrationParametersStereo parameters);

	/**
	 * @brief Ôóíêöèÿ ÷òåíèÿ ïàðàìåòðîâ êàëèáðîâêè ñòåðåî ïàðû
	 * @param fileName - Ïîëíûé ïóòü ê ôàéëó êàëèáðîâî÷íûõ ïàðàìåòðîâ.
	 * @return - Ñòðóêòóðà äëÿ õðàíåíèÿ êàëèáðîâî÷íûõ ïàðàìåòðîâ.
	 */
	MRCV_EXPORT CalibrationParametersStereo readCalibrationParametersStereo(std::string fileName);
	/////////////////////////////////////////////////////////////////////////////

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
		void Clustering();

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
}
