#include "mrcv/mrcv.h"

#ifdef _WIN32
#define FILE_SEPARATOR "\\"
#else
#define FILE_SEPARATOR "/"
#endif

namespace mrcv
{
	/**
 * @brief Функция поворота изображения на заданный угол.
 * Поворачивает изображение на определённый угол с использованием центральной точки.
 *
 * @param imageInput - входное (исходное) изображение cv::Mat.
 * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
 * @param angle - угол поворота в градусах.
 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
 */

	int rotateImage(cv::Mat& imageInput, cv::Mat& imageOutput, double angle)
	{
		try
		{
			// Определяем центр изображения
			cv::Point2f center(imageInput.cols / 2.0, imageInput.rows / 2.0);

			// Создаём матрицу для поворота
			cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

			// Поворачиваем изображение
			cv::warpAffine(imageInput, imageOutput, rotationMatrix, imageInput.size());
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}

		return 0; // SUCCESS
	}

	/**
	 * @brief Функция отражения изображения.
	 * Отражает изображение по горизонтали, вертикали или обеим осям.
	 *
	 * @param imageInput - входное (исходное) изображение cv::Mat.
	 * @param imageOutput - выходное (преобразованное) изображение cv::Mat.
	 * @param flipCode - Код отражения: 0 - вертикальное отражение; 1 - горизонтальное отражение; -1 - обе стороны.
	 * @return - код результата работы функции. 0 - Success; 1 - Пустое изображение; 2 - Неизвестный формат изображения; -1 - Неизвестная ошибка.
	 */

	int flipImage(cv::Mat& imageInput, cv::Mat& imageOutput, int flipCode)
	{
		try
		{

			// Применяем функцию отражения
			cv::flip(imageInput, imageOutput, flipCode);
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}

		return 0; // SUCCESS
	}

	int adjustBrightnessContrast(cv::Mat& imageInput, cv::Mat& imageOutput,
		double alpha, double beta)
	{
		try
		{
			imageInput.convertTo(imageOutput, -1, alpha, beta);
		}
		catch (...)
		{
			return -1;
		}
		return 0;
	}

	int addNoise(cv::Mat& imageInput, cv::Mat& imageOutput, double strength)
	{
		try
		{
			cv::Mat noise(imageInput.size(), imageInput.type());
			cv::randn(noise, 0, strength * 255);
			imageOutput = imageInput + noise;
			cv::normalize(imageOutput, imageOutput, 0, 255, cv::NORM_MINMAX);
		}
		catch (...)
		{
			return -1;
		}
		return 0;
	}

	int adjustColorBalance(cv::Mat& imageInput, cv::Mat& imageOutput,
		const std::vector<double>& factors)
	{
		try
		{
			if (factors.size() != 3)
				return -1;
			std::vector<cv::Mat> channels;
			cv::split(imageInput, channels);
			for (int i = 0; i < 3; ++i)
			{
				channels[i].convertTo(channels[i], -1, factors[i], 0);
			}
			cv::merge(channels, imageOutput);
		}
		catch (...)
		{
			return -1;
		}
		return 0;
	}

	int applyGaussianBlur(cv::Mat& imageInput, cv::Mat& imageOutput,
		int kernelSize)
	{
		try
		{
			cv::GaussianBlur(imageInput, imageOutput,
				cv::Size(kernelSize, kernelSize), 0);
		}
		catch (...)
		{
			return -1;
		}
		return 0;
	}

	int randomCrop(cv::Mat& imageInput, cv::Mat& imageOutput,
		double cropRatio)
	{
		try
		{
			int cropWidth = static_cast<int>(imageInput.cols * cropRatio);
			int cropHeight = static_cast<int>(imageInput.rows * cropRatio);
			int x = rand() % (imageInput.cols - cropWidth);
			int y = rand() % (imageInput.rows - cropHeight);
			cv::Rect roi(x, y, cropWidth, cropHeight);
			imageOutput = imageInput(roi).clone();
		}
		catch (...)
		{
			return -1;
		}
		return 0;
	}

	int perspectiveTransform(cv::Mat& imageInput, cv::Mat& imageOutput,
		float strength)
	{
		try
		{
			cv::Point2f src[4], dst[4];
			int width = imageInput.cols;
			int height = imageInput.rows;

			src[0] = cv::Point2f(0, 0);
			src[1] = cv::Point2f(width, 0);
			src[2] = cv::Point2f(width, height);
			src[3] = cv::Point2f(0, height);

			for (int i = 0; i < 4; ++i)
			{
				dst[i] = src[i] + cv::Point2f((rand() % (int)(width * strength)) -
					width * strength / 2,
					(rand() % (int)(height * strength)) -
					height * strength / 2);
			}

			cv::Mat M = cv::getPerspectiveTransform(src, dst);
			cv::warpPerspective(imageInput, imageOutput, M, imageInput.size());
		}
		catch (...)
		{
			return -1;
		}
		return 0;
	}

	/**
	 * @brief Функция аугментации изображений.
	 * Выполняет аугментацию для набора входных изображений на основе заданных методов и сохраняет результат.
	 *
	 * @param inputImagesAugmetation - вектор входных изображений (cv::Mat) для аугментации.
	 * @param outputImagesAugmetation - вектор для сохранения выходных (преобразованных) изображений.
	 * @param augmetationMethod - вектор методов аугментации (AUGMENTATION_METHOD) для применения.
	 * @return Код результата выполнения функции. 0 - успех; -1 - исключение (OpenCV или файловой системы).
	 *
	 * Функция проверяет наличие директории для сохранения изображений и создает её при необходимости. Для каждого изображения
	 * выполняется указанная операция (например, поворот или отражение) с последующей проверкой и сохранением результата в директорию.
	 */


	int augmetation(std::vector<cv::Mat>& inputImagesAugmetation, std::vector<cv::Mat>& outputImagesAugmetation,
		std::vector<AUGMENTATION_METHOD> augmetationMethod)
	{
		std::mt19937 gen(0);
		std::uniform_real_distribution<> dist(0.7, 1.3);
		std::set<std::string> methodsUsed; // Уникальные методы аугментации
		int savedFilesCount = 0;

		try
		{
			std::string outputFolder = "files" FILE_SEPARATOR "augmented_images" FILE_SEPARATOR;
			if (!std::filesystem::exists(outputFolder)) {
				std::filesystem::create_directories(outputFolder);
			}

			for (int q = 0; q < (int)augmetationMethod.size(); q++)
			{
				for (size_t i = 0; i < inputImagesAugmetation.size(); i++)
				{
					cv::Mat image = inputImagesAugmetation[i];

					if (image.empty()) {
						continue;
					}

					cv::Mat resultImage;
					std::string methodName;
					int status = 0;

					switch (augmetationMethod.at(q))
					{
					case AUGMENTATION_METHOD::FLIP_HORIZONTAL:
						status = flipImage(image, resultImage, 1);
						methodName = "flipHorizontal";
						break;
					case AUGMENTATION_METHOD::FLIP_VERTICAL:
						status = flipImage(image, resultImage, 0);
						methodName = "flipVertical";
						break;
					case AUGMENTATION_METHOD::FLIP_HORIZONTAL_AND_VERTICAL:
						status = flipImage(image, resultImage, -1);
						methodName = "flipHorizontalandVertical";
						break;
					case AUGMENTATION_METHOD::ROTATE_IMAGE_90:
						status = rotateImage(image, resultImage, 90);
						methodName = "rotate90";
						break;
					case AUGMENTATION_METHOD::ROTATE_IMAGE_45:
						status = rotateImage(image, resultImage, 45);
						methodName = "rotate45";
						break;
					case AUGMENTATION_METHOD::ROTATE_IMAGE_315:
						status = rotateImage(image, resultImage, 315);
						methodName = "rotate315";
						break;
					case AUGMENTATION_METHOD::ROTATE_IMAGE_270:
						status = rotateImage(image, resultImage, 270);
						methodName = "rotate270";
						break;
					case AUGMENTATION_METHOD::BRIGHTNESS_CONTRAST_ADJUST: {
						double alpha = 0.7 + 0.6 * (rand() % 100) / 100.0;
						double beta = -30 + 60 * (rand() % 100) / 100.0;
						status = adjustBrightnessContrast(image, resultImage, alpha, beta);
						methodName = "brightnessContrast";
						break;
					}
					case AUGMENTATION_METHOD::GAUSSIAN_NOISE:
						status = addNoise(image, resultImage, 0.05);
						methodName = "gaussianNoise";
						break;
					case AUGMENTATION_METHOD::COLOR_JITTER: {
						std::vector<double> factors = {
							0.7 + 0.6 * (rand() % 100) / 100.0,
							0.7 + 0.6 * (rand() % 100) / 100.0,
							0.7 + 0.6 * (rand() % 100) / 100.0
						};
						status = adjustColorBalance(image, resultImage, factors);
						methodName = "colorJitter";
						break;
					}
					case AUGMENTATION_METHOD::GAUSSIAN_BLUR: {
						int kernelSize = 3 + 2 * (rand() % 3);
						status = applyGaussianBlur(image, resultImage, kernelSize);
						methodName = "gaussianBlur";
						break;
					}
					case AUGMENTATION_METHOD::RANDOM_CROP: {
						double ratio = 0.7 + 0.2 * (rand() % 100) / 100.0;
						status = randomCrop(image, resultImage, ratio);
						methodName = "randomCrop";
						break;
					}
					case AUGMENTATION_METHOD::PERSPECTIVE_WARP:
						status = perspectiveTransform(image, resultImage, 0.1f);
						methodName = "perspectiveWarp";
						break;
					default:
						resultImage = image.clone();
						methodName = "none";
						break;
					}

					if (resultImage.empty()) {
						continue;
					}

					outputImagesAugmetation.push_back(resultImage);

					std::stringstream ss;
					ss << outputFolder << "augmented_" << i << "_" << methodName << ".bmp";

					bool isSaved = cv::imwrite(ss.str(), resultImage);
					if (isSaved) {
						savedFilesCount++;
						methodsUsed.insert(methodName); // Добавляем метод в set
					}
				}
			}

			// Формируем строку с названиями методов
			std::ostringstream methodsStream;
			for (const auto& method : methodsUsed) {
				methodsStream << method << ", ";
			}
			std::string methodsString = methodsStream.str();
			if (!methodsString.empty()) {
				methodsString.pop_back(); // Удаляем последнюю запятую
				methodsString.pop_back(); // Удаляем пробел
			}

			// Записываем в лог успешное завершение с названиями методов
			writeLog("Augmentation completed successfully. Methods used: " + methodsString + ". Files saved: " + std::to_string(savedFilesCount), LOGTYPE::INFO);
		}
		catch (const cv::Exception& ex)
		{
			writeLog("Augmentation failed: " + std::string(ex.what()), LOGTYPE::ERROR);
			return -1;
		}
		catch (const std::filesystem::filesystem_error& ex)
		{
			writeLog("Filesystem error: " + std::string(ex.what()), LOGTYPE::ERROR);
			return -1;
		}
		catch (...)
		{
			writeLog("Unhandled exception occurred during augmentation.", LOGTYPE::EXCEPTION);
			return -1;
		}

		return 0;
	}

	int batchAugmentation(const std::vector<cv::Mat>& inputs,
		std::vector<cv::Mat>& outputs,
		const BatchAugmentationConfig& config,
		const std::string& output_dir)
	{
		if (inputs.empty())
			return 1;

		std::mt19937 gen(config.random_seed);
		std::uniform_real_distribution<> prob_dist(0.0, 1.0);

		if (!output_dir.empty())
		{
			try
			{
				if (!std::filesystem::exists(output_dir))
				{
					std::filesystem::create_directories(output_dir);
				}
			}
			catch (...)
			{
				return -1;
			}
		}

		if (config.keep_original)
		{
			for (const auto& img : inputs)
			{
				if (!img.empty())
				{
					outputs.push_back(img.clone());
					if (!output_dir.empty())
					{
						std::string filename = output_dir + "/original_" +
							std::to_string(outputs.size()) +
							".png";
						cv::imwrite(filename, img);
					}
				}
			}
		}

		double total_weight = 0.0;
		for (const auto& [method, weight] : config.method_weights)
		{
			if (weight < 0.0)
				return 2;
			total_weight += weight;
		}

		if (total_weight <= 0.0)
			return 3;

		int remaining =
			config.total_output_count > 0
			? (config.total_output_count -
				(config.keep_original ? static_cast<int>(inputs.size()) : 0))
			: (static_cast<int>(inputs.size()) *
				static_cast<int>(config.method_weights.size()));

		std::map<AUGMENTATION_METHOD, int> method_counts;
		for (const auto& [method, weight] : config.method_weights)
		{
			method_counts[method] =
				static_cast<int>((weight / total_weight) * remaining);
		}

		int successful_augmentations = 0;
		std::map<std::string, int> method_stats;

		for (const auto& [method, count] : method_counts)
		{
			std::vector<AUGMENTATION_METHOD> methods = { method };
			method_stats[augmentationMethodToString(method)] = 0;

			for (int i = 0; i < count; ++i)
			{
				std::uniform_int_distribution<> input_dist(
					0, static_cast<int>(inputs.size()) - 1);
				const cv::Mat& input = inputs[input_dist(gen)];
				if (input.empty())
					continue;

				std::vector<cv::Mat> single_input = { input };
				std::vector<cv::Mat> single_output;

				int status = augmetation(single_input, single_output, methods);
				if (status != 0 || single_output.empty())
					continue;

				outputs.push_back(single_output[0]);
				method_stats[augmentationMethodToString(method)]++;
				successful_augmentations++;

				if (!output_dir.empty())
				{
					std::string filename =
						output_dir + "/" + augmentationMethodToString(method) +
						"_" + std::to_string(outputs.size()) + ".png";
					cv::imwrite(filename, single_output[0]);
				}
			}
		}

		std::cout << "Batch augmentation completed." << std::endl;
		std::cout << "Total:\t" << successful_augmentations << " images." << std::endl;
		for (const auto& [method, count] : method_stats)
		{
			std::cout << "  " << method << ":\t" << count << " images (" << (count * 100.0 / successful_augmentations) << "%)" << std::endl;
		}

		return EXIT_SUCCESS;
	}

	std::string augmentationMethodToString(AUGMENTATION_METHOD method)
	{
		switch (method)
		{
		case AUGMENTATION_METHOD::NONE:
			return "none";
		case AUGMENTATION_METHOD::FLIP_HORIZONTAL:
			return "flip_h";
		case AUGMENTATION_METHOD::FLIP_VERTICAL:
			return "flip_v";
		case AUGMENTATION_METHOD::FLIP_HORIZONTAL_AND_VERTICAL:
			return "flip_both";
		case AUGMENTATION_METHOD::ROTATE_IMAGE_90:
			return "rotate_90";
		case AUGMENTATION_METHOD::ROTATE_IMAGE_45:
			return "rotate_45";
		case AUGMENTATION_METHOD::ROTATE_IMAGE_270:
			return "rotate_270";
		case AUGMENTATION_METHOD::ROTATE_IMAGE_315:
			return "rotate_315";
		case AUGMENTATION_METHOD::BRIGHTNESS_CONTRAST_ADJUST:
			return "brightness";
		case AUGMENTATION_METHOD::GAUSSIAN_NOISE:
			return "noise";
		case AUGMENTATION_METHOD::COLOR_JITTER:
			return "color_jitter";
		case AUGMENTATION_METHOD::GAUSSIAN_BLUR:
			return "blur";
		case AUGMENTATION_METHOD::RANDOM_CROP:
			return "crop";
		case AUGMENTATION_METHOD::PERSPECTIVE_WARP:
			return "perspective";
		case AUGMENTATION_METHOD::TEST:
			return "test";
		default:
			return "unknown";
		}
	}


#ifdef MRCV_CUDA_ENABLED 
	int rotateImageCuda(cv::Mat& imageInput, cv::Mat& imageOutput, double angle)
	{
		try
		{
			// Определяем центр изображения
			cv::Point2f center(imageInput.cols / 2.0, imageInput.rows / 2.0);

			// Создаём матрицу для поворота
			cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

			// Поворачиваем изображение
			cv::cuda::warpAffine(imageInput, imageOutput, rotationMatrix, imageInput.size());
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}

		return 0; // SUCCESS
	}

	int flipImageCuda(cv::Mat& imageInput, cv::Mat& imageOutput, int flipCode)
	{
		try
		{

			// Применяем функцию отражения
			cv::cuda::flip(imageInput, imageOutput, flipCode);
		}
		catch (...)
		{
			return -1; // Unhandled Exception
		}

		return 0; // SUCCESS
	}

	int augmetationCuda(std::vector<cv::Mat>& inputImagesAugmetation, std::vector<cv::Mat>& outputImagesAugmetation, std::vector<AUGMENTATION_METHOD> augmetationMethod)
	{
		std::set<std::string> methodsUsed; // Уникальные методы аугментации
		int savedFilesCount = 0;

		try
		{
			std::string outputFolder = "files\\augmented_images/";
			if (!std::filesystem::exists(outputFolder)) {
				std::filesystem::create_directories(outputFolder);
			}

			for (int q = 0; q < (int)augmetationMethod.size(); q++)
			{
				for (size_t i = 0; i < inputImagesAugmetation.size(); i++)
				{
					cv::Mat image = inputImagesAugmetation[i];

					if (image.empty()) {
						continue;
					}

					cv::Mat resultImage;
					std::string methodName;
					int status = 0;

					switch (augmetationMethod.at(q))
					{
					case AUGMENTATION_METHOD::FLIP_HORIZONTAL:
						status = flipImageCuda(image, resultImage, 1);
						methodName = "flipHorizontal";
						break;
					case AUGMENTATION_METHOD::FLIP_VERTICAL:
						status = flipImageCuda(image, resultImage, 0);
						methodName = "flipVertical";
						break;
					case AUGMENTATION_METHOD::FLIP_HORIZONTAL_AND_VERTICAL:
						status = flipImageCuda(image, resultImage, -1);
						methodName = "flipHorizontalandVertical";
						break;
					case AUGMENTATION_METHOD::ROTATE_IMAGE_90:
						status = rotateImageCuda(image, resultImage, 90);
						methodName = "rotate90";
						break;
					case AUGMENTATION_METHOD::ROTATE_IMAGE_45:
						status = rotateImageCuda(image, resultImage, 45);
						methodName = "rotate45";
						break;
					case AUGMENTATION_METHOD::ROTATE_IMAGE_315:
						status = rotateImageCuda(image, resultImage, 315);
						methodName = "rotate315";
						break;
					case AUGMENTATION_METHOD::ROTATE_IMAGE_270:
						status = rotateImageCuda(image, resultImage, 270);
						methodName = "rotate270";
						break;
					default:
						resultImage = image.clone();
						methodName = "none";
						break;
					}

					if (resultImage.empty()) {
						continue;
					}

					outputImagesAugmetation.push_back(resultImage);

					std::stringstream ss;
					ss << outputFolder << "augmented_" << i << "_" << methodName << ".bmp";

					bool isSaved = cv::imwrite(ss.str(), resultImage);
					if (isSaved) {
						savedFilesCount++;
						methodsUsed.insert(methodName); // Добавляем метод в set
					}
				}
			}

			// Формируем строку с названиями методов
			std::ostringstream methodsStream;
			for (const auto& method : methodsUsed) {
				methodsStream << method << ", ";
			}
			std::string methodsString = methodsStream.str();
			if (!methodsString.empty()) {
				methodsString.pop_back(); // Удаляем последнюю запятую
				methodsString.pop_back(); // Удаляем пробел
			}

			// Записываем в лог успешное завершение с названиями методов
			writeLog("Augmentation completed successfully. Methods used: " + methodsString + ". Files saved: " + std::to_string(savedFilesCount), LOGTYPE::INFO);
		}
		catch (const cv::Exception& ex)
		{
			writeLog("Augmentation failed: " + std::string(ex.what()), LOGTYPE::ERROR);
			return -1;
		}
		catch (const std::filesystem::filesystem_error& ex)
		{
			writeLog("Filesystem error: " + std::string(ex.what()), LOGTYPE::ERROR);
			return -1;
		}
		catch (...)
		{
			writeLog("Unhandled exception occurred during augmentation.", LOGTYPE::EXCEPTION);
			return -1;
		}

		return 0;
	}
#endif
}