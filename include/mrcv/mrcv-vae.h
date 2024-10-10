#pragma once

#include <mrcv/export.h>
#include <mrcv/mrcv-common.h>

namespace mrcv
{
   MRCV_EXPORT struct VariationalAutoEncoder : public torch::nn::Module 
   {
        torch::nn::Linear encoder1{ nullptr }, encoder2{ nullptr };
        torch::nn::Linear mu{ nullptr }, logvar{ nullptr };
        torch::nn::Linear decoder1{ nullptr }, decoder2{ nullptr };
        torch::nn::ReLU relu{ nullptr };

        VariationalAutoEncoder(int64_t inputDim, int64_t hDIM, int64_t zDim) 
        {
            encoder1 = register_module("encoder1", torch::nn::Linear(inputDim, hDIM));
            encoder2 = register_module("encoder2", torch::nn::Linear(hDIM, hDIM));
            mu = register_module("mu", torch::nn::Linear(hDIM, zDim));
            logvar = register_module("logvar", torch::nn::Linear(hDIM, zDim));
            decoder1 = register_module("decoder1", torch::nn::Linear(zDim, hDIM));
            decoder2 = register_module("decoder2", torch::nn::Linear(hDIM, inputDim));

            relu = torch::nn::ReLU();
        }

        std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x) 
        {
            x = torch::relu(encoder1->forward(x));
            x = torch::relu(encoder2->forward(x));
            auto mu = this->mu->forward(x);
            auto logvar = this->logvar->forward(x);
            return { mu, logvar };
        }

        torch::Tensor decode(torch::Tensor z) 
        {
            z = torch::relu(decoder1->forward(z));
            z = torch::sigmoid(decoder2->forward(z));
            return z;
        }

        torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor logvar)
        {
            auto std = torch::exp(0.5 * logvar);
            auto eps = torch::randn_like(std);
            return mu + eps * std;
        }

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) 
        {
            auto [mu, logvar] = encode(x);
            auto z = reparameterize(mu, logvar);
            x = torch::relu(decoder1->forward(z));
            x = decoder2->forward(x);
            return std::make_tuple(x, mu, logvar);
        }
    };

   MRCV_EXPORT class LoadImageDataset : public torch::data::datasets::Dataset<LoadImageDataset> 
   {
        std::vector<std::string> images;
        int numColor;
        int height;
        int width;

    public:
        LoadImageDataset(const std::string& root, int height, int width, int numColor) : numColor(numColor)
        {
            for (const auto& entry : std::filesystem::directory_iterator(root)) 
            {
                if (entry.is_regular_file()) 
                {
                    images.push_back(entry.path().string());
                }
            }
            auto str = std::to_string(images.size());
            writeLog("Loaded Images: " + str, mrcv::LOGTYPE::INFO);
        }

        torch::data::Example<> get(size_t index) override 
        {
           
            auto path = images[index];
            cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

            if (image.empty()) 
            {
                writeLog("Failed to Loade Image: " + path, mrcv::LOGTYPE::ERROR);
                return {};
            }
            if (numColor == 1) 
            {
                cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
            }
            if (numColor == 3) 
            {
                cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            }

            //cv::resize(image, image, cv::Size(width, height));

            torch::Tensor tensor = torch::from_blob(image.data, { image.rows, image.cols, numColor }, torch::kUInt8).permute({ 2, 0, 1 }).to(torch::kFloat).div(255);
            int label = 0;
            return { tensor.clone(), torch::tensor(label) };
        }

        torch::optional<size_t> size() const override 
        {
            return images.size();
        }

        int get_num_images() const
        {
            return images.size();
        }
    };

   // Параметры обработки
   static const float SCORE_THRESHOLD = 0.20;
   static const float NMS_THRESHOLD = 0.45;
   static const float CONFIDENCE_THRESHOLD = 0.15;

   // Параметры шрифтов
   static const float FONT_SCALE = 0.7;
   static const int   THICKNESS = 1;

   // Цветовые константы
   static cv::Scalar BLACK = cv::Scalar(0, 0, 0);
   static cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
   static cv::Scalar RED = cv::Scalar(0, 0, 255);
   static cv::Scalar GREEN = cv::Scalar(0, 255, 0);

   class NNPreLabeler {
   private:
       /** Ширина и высота исходного изображения */
       cv::Size sourceSize;
       /** Структура нейросети */
       cv::dnn::Net network;
       /** Ширина и высота входного изображения */
       int inputWidth;
       int inputHeight;
       /** Вектор распознаваемых классов */
       std::vector<std::string> classes;
       /** Структуры для хранения результатов обработки */
       std::vector<int> classesIdSet;
       std::vector<cv::Rect> boxesSet;
       std::vector<float> confidencesSet;
       std::vector<std::string> classesSet;
       /** Время обработки */
       float inferenceTime;
       /** Получить строковые значения классов */
       int readClasses(const std::string filePath);
       /** Инициализация нейросети */
       int initNetwork(const std::string modelPath,
           const std::string classesPath);
       /** Отрисовка метки */
       void drawLabel(cv::Mat& img, std::string label, int left, int top);
       /** Предобработка результатов */
       std::vector<cv::Mat> preProcess(cv::Mat& img);
       /** Постобработка результатов */
       cv::Mat postProcess(cv::Mat& img, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_name);
   public:
       NNPreLabeler(const std::string model, const std::string classes, int width, int height);
       std::vector<float> getConfidences(void) { return confidencesSet; }
       std::vector<cv::Rect> getBoxes(void) { return boxesSet; }
       std::vector<int> getClassIds(void) { return classesIdSet; }
       std::vector<std::string> getClasses(void) { return classesSet; }
       float getInference(void) { return inferenceTime; }
       cv::Mat process(cv::Mat& img);
       void writeLabels(const std::string& filename);
   };
   /**
    * @brief функция генерации изображения.
    * 
    * Функция может использоваться для аугментации данных с помощью нейронной сети.
    *
    * @param root - путь к датасету для обучения.
    * @param height - высота сгенерированного изображения.
    * @param width - ширина сгенерированного изображения.
    * @param hDim - размерность скрытого слоя.
    * @param zDim - размерность латентного слоя.
    * @param numEpoch - количество эпох обучения.
    * @param batchSize - размер пакета.
    * @param lrRate - скорость обучения.
    * @return - тензор формата torch::Tensor.
    */
   MRCV_EXPORT torch::Tensor neuralNetworkAugmentationAsTensor(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate);
   /**
    * @brief функция генерации изображения.
    *
    * Функция может использоваться для аугментации данных с помощью нейронной сети.
    *
    * @param root - путь к датасету для обучения.
    * @param height - высота сгенерированного изображения.
    * @param width - ширина сгенерированного изображения.
    * @param hDim - размерность скрытого слоя.
    * @param zDim - размерность латентного слоя.
    * @param numEpoch - количество эпох обучения.
    * @param batchSize - размер пакета.
    * @param lrRate - скорость обучения.
    * @return - изображение формата cv::Mat.
    */
   MRCV_EXPORT cv::Mat neuralNetworkAugmentationAsMat(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate);
   /**
    * @brief функция полуавтоматической разметки.
    *
    * Функция может использоваться для полуавтоматической разметки в формате YOLO.
    *
    * @param inputImage - изображение формата cv::Mat.
    * @param height - высота изображения.
    * @param width - ширина изображения.
    * @param outputPath - полный путь к папке к сохранению результата.
    * @param modelPath -  полный путь к обученной модели.
    * @param classesPath - полный путь к файлу с классами.
    * @return - код результата работы функции. 0 - Success; -1 - Unhandled Exception.
	*/
   MRCV_EXPORT int semiAutomaticLabeler(cv::Mat& inputImage, const int height, const int width, const std::string& outputPath, const std::string& modelPath, const std::string& classesPath);
   /**
      * @brief функция полуавтоматической разметки.
      *
      * Функция может использоваться для полуавтоматической разметки в формате YOLO.
      *
      * @param root - полный путь к изображению.
      * @param height - высота изображения.
      * @param width - ширина изображения.
      * @param outputPath - полный путь к папке к сохранению результата.
      * @param modelPath -  полный путь к обученной модели.
      * @param classesPath - полный путь к файлу с классами.
      * @return - код результата работы функции. 0 - Success; -1 - Unhandled Exception.
      */
   MRCV_EXPORT int semiAutomaticLabeler(const std::string& root, const int height, const int width, const std::string& outputPath, const std::string& modelPath, const std::string& classesPath);
}