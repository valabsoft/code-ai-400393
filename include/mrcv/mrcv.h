#pragma once

#include <mrcv/export.h>
#include <mrcv/mrcv-common.h>

namespace mrcv
{
	/////////////////////////////////////////////////////////////////////////////
	// �������
	/////////////////////////////////////////////////////////////////////////////
	/**
	 * @brief ������� �������� ���� ����� �����.
	 * @param a - ������ ���������.
	 * @param b - ������ ���������.
	 * @return - ��������� ���������� ��������� a + b
	 */
	MRCV_EXPORT int add(int a, int b);
	/**
	 * @brief ������� �������� �����������.
	 *
	 * ������� ������������ ��� �������� ����������� � �������� � ����������� ������������ ����������� � ��������� ����.
	 *
	 * @param image - ������ cv::Mat ��� �������� ������������ �����������.
	 * @param pathToImage - ������ ���� � ����� � ������������.
	 * @param showImage - ����, ���������� �� ����������� ���������� ���� (false �� ���������).
	 * @return - ��� ���������� ������ �������. 0 - Success; 1 - ���������� ������� �����������; -1 - Unhandled Exception.
	 */
	MRCV_EXPORT int readImage(cv::Mat& image, std::string pathToImage, bool showImage = false);
	/**
	 * @brief ������� ������ ���������� � ������� ������ OpenCV.
	 * @return ������ � ��������������� �����������.
	 */
	MRCV_EXPORT std::string getOpenCVBuildInformation();
	/**
	 * @brief ������� ������ ����������� �� ����.
	 *
	 * ������� ����� �������������� ��� ���������� ������ �����������������.
	 *
	 * @param cameraID - ID ������.
	 * @param recorderInterval - �������� ������ � ��������.
	 * @param fileName - ����� ����.
	 * @param codec - �����, ������������ ��� �������� ����������.
	 * @return - ��� ���������� ������ �������. 0 - Success; 1 - ID ������ ����� �������; 2 - �������� ������� ������ ������������; 3 - �� ������� ��������� ������; 4 - �� ������� ������� ������� cv::VideoWriter; -1 - Unhandled Exception.
	 */
	MRCV_EXPORT int recordVideo(int cameraID, int recorderInterval, std::string fileName, CODEC codec);
	/**
	* @brief ������� ����������� ����� �������.
	* @param query - ������ ������� ��� ������.
	* @param minWidth - ����������� ������ �����������.
	* @param minHeight - ����������� ������ �����������.
	* @param nameTemplate - ������ ����� �����.
	* @param outputFolder - ����� ��� ����������.
	* @param separateDataset - ���� �������� �������� �� ������������� � �������� �������.
	* @param trainsetPercentage - ������� ��� ������������� ����� �������.
	* @param countFoto - ���������� ����������� ���� ��� ����������.
	* @param money - ������� ��� ���������� ������� ������.
	* @param key - ������ key.
	* @param secretKey - ������ secretKey.
	* @return - ��������� ������ �������.
	*/
	MRCV_EXPORT int getImagesFromYandex(std::string query, int minWidth, int minHeight, std::string nameTemplate, std::string outputFolder, bool separateDataset, unsigned int trainsetPercentage, unsigned int countFoto, bool money, std::string key, std::string secretKey);
	/////////////////////////////////////////////////////////////////////////////
	// ����������
	/////////////////////////////////////////////////////////////////////////////
	/**
     * @brief ������� ����� ����������.
     * @param imagesL - ������ ����� ��� ����������� ����� ������.
     * @param imagesR - ������ ����� ��� ����������� ������ ������.
     * @param pathToImagesL - ���� � ����� � ������������� ����� ������.
     * @param pathToImagesR - ���� � ����� � ������������� ������ ������.
     * @param calibrationParametersL - ��������� ��� �������� ������������� ���������� ����� ������.
     * @param calibrationParametersR - ��������� ��� �������� ������������� ���������� ������ ������.
     * @param calibrationParameters - ��������� ��� �������� ������������� ���������� ������ ����.
     * @param chessboardColCount - ���������� �������� ����� ������������� ����� �� ��������.
     * @param chessboardRowCount - ���������� �������� ����� ������������� ����� �� �������.
     * @param chessboardSquareSize - ������ ���� ���������� ����� � ��.
     */
	MRCV_EXPORT void cameraCalibration(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathToImagesL, std::string pathToImagesR, CalibrationParametersMono& calibrationParametersL, CalibrationParametersMono& calibrationParametersR, CalibrationParametersStereo& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief ������� ���������� ��������� ������.
	 * @param images - ������ ����� ��� ����������� ������.
	 * @param pathToImages - ���� � ����� � ������������� ������.
	 * @param calibrationParameters - ��������� ��� �������� ������������� ���������� ������.
	 * @param chessboardColCount - ���������� �������� ����� ������������� ����� �� ��������.
     * @param chessboardRowCount - ���������� �������� ����� ������������� ����� �� �������.
     * @param chessboardSquareSize - ������ ���� ���������� ����� � ��.
	 */
	MRCV_EXPORT void cameraCalibrationMono(std::vector<cv::String> images, std::string pathToImages, CalibrationParametersMono& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief ������� ���������� ������ ����.
	 * @param imagesL - ������ ����� ��� ����������� ����� ������.
     * @param imagesR - ������ ����� ��� ����������� ������ ������.
     * @param pathToImagesL - ���� � ����� � ������������� ����� ������.
     * @param pathToImagesR - ���� � ����� � ������������� ������ ������.
	 * @param calibrationParameters - ��������� ��� �������� ������������� ���������� ������ ����.
	 * @param chessboardColCount - ���������� �������� ����� ������������� ����� �� ��������.
     * @param chessboardRowCount - ���������� �������� ����� ������������� ����� �� �������.
     * @param chessboardSquareSize - ������ ���� ���������� ����� � ��.
	 */
	MRCV_EXPORT void cameraCalibrationStereo(std::vector<cv::String> imagesL, std::vector<cv::String> imagesR, std::string pathToImagesL, std::string pathToImagesR, CalibrationParametersStereo& calibrationParameters, int chessboardColCount, int chessboardRowCount, float chessboardSquareSize);

	/**
	 * @brief ������� ������ ���������� ���������� ��������� ������.
	 * @param fileName - ������ ���� � ����� ������������� ����������.
	 * @return
	 */
	MRCV_EXPORT CalibrationParametersMono readCalibrationParametersMono(std::string fileName);

	/**
	 * @brief ������� ������ ���������� ���������� ��������� ������.
	 * @param fileName - ������ ���� � ����� ������������� ����������.
	 * @param parameters - ��������� ��� �������� ������������� ����������.
	 */
	MRCV_EXPORT void writeCalibrationParametersMono(std::string fileName, CalibrationParametersMono parameters);

	/**
	 * @brief ������� ������ ���������� ���������� ������ ����
	 * @param fileName - ������ ���� � ����� ������������� ����������.
	 * @param parameters - ��������� ��� �������� ������������� ����������.
	 */
	MRCV_EXPORT void writeCalibrationParametersStereo(std::string fileName, CalibrationParametersStereo parameters);

	/**
	 * @brief ������� ������ ���������� ���������� ������ ����
	 * @param fileName - ������ ���� � ����� ������������� ����������.
	 * @return - ��������� ��� �������� ������������� ����������.
	 */
	MRCV_EXPORT CalibrationParametersStereo readCalibrationParametersStereo(std::string fileName);
	/////////////////////////////////////////////////////////////////////////////
}
