#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>
#include <mrcv/mrcv-vae.h>

namespace mrcv
{
    torch::Tensor neuralNetworkAugmentationAsTensor(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate) 
    {
        // Логирование
        std::ostringstream logStream;
        // Количестыо цветов генерируемого изображения
        const int64_t numColor = 1;
        // Расчет размерности входно слоя
        const int64_t inputDim = numColor * height * width;
        // Выбор устройства
        torch::Device device(torch::kCUDA);
        // Загрузка датасета
        auto dataset = CustomDataset(root, numColor).map(torch::data::transforms::Stack<>());

        // Проверка датасета
        if (dataset.size() == 0) 
        {
            logStream << "Dataset is empty!" << std::endl;
            //return ;
        }

        auto dataLoader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(batchSize).workers(2));

        // Создание объекта модели
        VariationalAutoEncoder model(inputDim, hDim, zDim);
        // Перенос модели на устройство
        model.to(device);
        // Оптимизация модели
        torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lrRate));
        logStream << "Dataset size before training: " << dataset.size().value() << std::endl;

        // Обучение модели
        for (int epoch = 0; epoch < numEpoch; ++epoch) 
        {
            model.train();
            for (auto& batch : *dataLoader) 
            {
                auto data = batch.data.to(device).view({ -1, inputDim });
                logStream << "Data size: " << data.sizes() << std::endl;

                if (torch::isnan(data).any().item<bool>() || torch::isinf(data).any().item<bool>()) 
                {
                    logStream << "Data tensor contains NaN or Inf values" << std::endl;
                    continue;
                }

                optimizer.zero_grad();
                auto [output, mu, sigma] = model.forward(data);
                output = torch::sigmoid(output);
                logStream << "Output size: " << output.sizes() << std::endl;

                auto reconstructionLoss = torch::binary_cross_entropy(output, data, {}, torch::Reduction::Sum);
                logStream << "Reconstruction loss: " << reconstructionLoss.item<double>() << std::endl;

                reconstructionLoss.backward();
                optimizer.step();
            }
        }

        logStream << "Dataset size after training: " << dataset.size().value() << std::endl;

        // Коллическтво генерируемых изображений
        int numExamples = 1;

        std::vector<torch::Tensor> images;

        for (size_t i = 0; i < dataset.size(); ++i) 
        {
            auto example = dataset.get_batch(i);
            images.push_back(example.data.to(device));
            if (images.size() == numExamples) break;
        }

        if (images.empty()) 
        {
            logStream << "No images found for digit" << std::endl;
            //return ;
        }

        std::vector<std::pair<torch::Tensor, torch::Tensor>> encodingsDigit;

        // Энкодер
        for (int d = 0; d < numExamples; ++d) 
        {
            torch::NoGradGuard no_grad;
            logStream << "Image tensor size: " << images[d].sizes() << std::endl;

            auto flattenedIimage = images[d].view({ 1, inputDim });
            logStream << "Flattened image tensor size: " << flattenedIimage.sizes() << std::endl;

            auto [mu, sigma] = model.encode(flattenedIimage.to(device));
            logStream << "Mu size: " << mu.sizes() << std::endl;
            logStream << "Sigma size: " << sigma.sizes() << std::endl;
            encodingsDigit.push_back({ mu, sigma });

        }

        // Декодер
        for (int example = 0; example < numExamples; ++example) 
        {
            auto [mu, sigma] = encodingsDigit[0];
            auto sample = model.reparameterize(mu, sigma);
            torch::Tensor tensor = model.decode(sample.to(device));
            tensor = torch::sigmoid(tensor).view({ numColor, width, height });
            return tensor.clone();
        }
        //Сохранение логов
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
    }

    cv::Mat neuralNetworkAugmentationAsMat(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate) 
    {
        // Логирование
        std::ostringstream logStream;
        // Количестыо цветов генерируемого изображения
        const int64_t numColor = 1;
        // 
        torch::Tensor tensor = neuralNetworkAugmentationAsTensor(root, height, width, hDim, zDim, numEpoch, batchSize, lrRate);
        // Проверка, что тензор не пустой
        if (!tensor.defined()) 
        {
            logStream << "Error: Tensor is undefined." << std::endl;
        }
        // Прелбразование тензора в OpenCV Мat
        tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
        tensor = tensor.to(torch::kCPU);
        tensor = tensor.view({ numColor, height, width });

        // Переставляем каналы с {numColor, height, width} на {height, width, numColor} для OpenCV
        tensor = tensor.permute({ 1, 2, 0 });

        // Создаем OpenCV Mat
        cv::Mat image(tensor.size(0), tensor.size(1), CV_8UC1, tensor.data_ptr());
        // Проверка, что изображение не пустое
        if (image.empty()) 
        {
            logStream << "Error: Image is empty." << std::endl;
        }
        //Сохранение логов
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);

        return image.clone();
    }

    
    NNPreLabeler::NNPreLabeler(const std::string model, const std::string classes, int width, int height)
    {
        // Логирование
        std::ostringstream logStream;
        inputWidth = width;
        inputHeight = height;
        if (initNetwork(model, classes)) 
        {
            logStream << "Failed to init neural network!" << std::endl;
        }
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
    }

    int NNPreLabeler::readClasses(const std::string file_path) 
    {
        std::ostringstream logStream;
        std::ifstream classes_file(file_path);
        std::string line;

        if (!classes_file) 
        {
            logStream << "Failed to open classes names!\n";
            return ENOENT;
        }
        while (std::getline(classes_file, line)) 
        {
            classes.push_back(line);
        }
        classes_file.close();
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
        return 0;
    }

    int NNPreLabeler::initNetwork(const std::string model_path, const std::string classes_path) 
    {
        int err = readClasses(classes_path);

        if (err == 0)
        {
            network = cv::dnn::readNetFromONNX(model_path);
            if (network.empty()) 
            {
                return ENETDOWN;
            }
            else 
            {
                network.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        }

        return err;
    }

    int NNPreLabeler::drawLabel(cv::Mat& img, std::string label, int left, int top) 
    {
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, THICKNESS, &baseline);
        top = std::max(top, label_size.height);
        cv::Point tlc = cv::Point(left, top);
        cv::Point brc = cv::Point(left + label_size.width,
            top + label_size.height + baseline);
        cv::rectangle(img, tlc, brc, BLACK, cv::FILLED);
        cv::putText(img, label, cv::Point(left, top + label_size.height), cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS);
        return 0;
    }

    std::vector<cv::Mat> NNPreLabeler::preProcess(cv::Mat& img) 
    {
        std::ostringstream logStream;
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
        network.setInput(blob);
        std::vector<cv::Mat> outputs;
        network.forward(outputs, network.getUnconnectedOutLayersNames());
        if (outputs.empty()) 
        {
            logStream << "Error: Image is empty." << std::endl;
        }
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
        return outputs;
    }

    cv::Mat NNPreLabeler::postProcess(cv::Mat& img, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_name) 
    {
        std::ostringstream logStream;
        cv::Mat ret = img.clone(); 
        classesIdSet.clear();
        confidencesSet.clear();
        boxesSet.clear();
        classesSet.clear();
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float x_factor = img.cols / (float)inputWidth;
        float y_factor = img.rows / (float)inputHeight;

        int dimensions = outputs[0].size[2];
        int rows = outputs[0].size[1];
        float* data = (float*)outputs[0].data;

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
                if (max_class_score > SCORE_THRESHOLD) 
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            
            data += dimensions;
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        for (int i = 0; i < indices.size(); i++) 
        {
            int idx = indices[i];
            cv::Rect box = boxes[idx];

            boxesSet.push_back(box);
            confidencesSet.push_back(confidences[idx]);
            classesIdSet.push_back(class_ids[idx]);
            classesSet.push_back(class_name[class_ids[idx]]);

            int left = box.x;
            int top = box.y;
            int width = box.width;
            int height = box.height;
            cv::rectangle(ret, cv::Point(left, top), cv::Point(left + width, top + height), GREEN, 3 * THICKNESS);
            std::string label = cv::format("%.2f", confidences[idx]);
            label = class_name[class_ids[idx]] + ": " + label;
            drawLabel(ret, label, left, top);
        }
        if (ret.empty()) 
        {
            logStream << "Error: Image is empty." << std::endl;
        }
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
        return ret;
    }

    cv::Mat NNPreLabeler::process(cv::Mat& img) 
    {
        std::ostringstream logStream;
        NNPreLabeler::sourceSize = img.size();

        std::vector<cv::Mat> detections;
        detections = preProcess(img);
        cv::Mat res = postProcess(img, detections, NNPreLabeler::classes);
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency();
        NNPreLabeler::inferenceTime = network.getPerfProfile(layersTimes) / freq;
        if (res.empty()) 
        {
            logStream << "Error: Image is empty." << std::endl;
        }
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
        return res;
    }

    // Функция для записи YOLOv5 лейблов в текстовый файл
    int NNPreLabeler::writeLabels(const std::string& filename) 
    {
        // Логирование
        std::ostringstream logStream;
        // Открываем файл для записи
        std::ofstream outfile(filename);

        // Проверяем, открыт ли файл
        if (!outfile.is_open()) 
        {
            logStream << "Failed to open file: " << filename << std::endl;
            return -1;
        }

        // Записываем каждый лейбл в формате YOLOv5
        for (size_t i = 0; i < NNPreLabeler::classesIdSet.size(); ++i) 
        {
            int class_id = NNPreLabeler::classesIdSet[i];
            const cv::Rect& box = NNPreLabeler::boxesSet[i];

            // Преобразуем координаты и размеры из абсолютных значений в диапазон [0, 1]
            float x_center = (box.x + box.width / 2.0) / NNPreLabeler::sourceSize.width;
            float y_center = (box.y + box.height / 2.0) / NNPreLabeler::sourceSize.height;
            float width = box.width / static_cast<float>(NNPreLabeler::sourceSize.width);
            float height = box.height / static_cast<float>(NNPreLabeler::sourceSize.height);

            // Записываем строку в формате YOLOv5
            outfile << class_id << " " << x_center << " " << y_center << " " << width << " " << height << std::endl;
        }

        // Закрываем файл
        outfile.close();
        //Сохранение логов
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
        return 0;
    }

    int semiAutomaticLabeler(cv::Mat& inputImage, const int64_t height, const int64_t width, const std::string& outputPath, const std::string& modelPath, const std::string& classesPath) 
    {
        // Логирование
        std::ostringstream logStream;
        
        NNPreLabeler labeler(modelPath, classesPath, width, height);
        cv::Mat image = labeler.process(inputImage);
        std::vector<int> class_ids = labeler.getClassIds();
        std::vector<float> confidences = labeler.getConfidences();
        std::vector<cv::Rect> boxes = labeler.getBoxes();
        std::vector<std::string> classes = labeler.getClasses();

        logStream << "class_ids: ";
        for (auto element : class_ids) {
            logStream << element << " ";
        }
        logStream << std::endl;
        logStream << "classes: ";
        for (auto element : classes) {
            logStream << element << " ";
        }
        logStream << std::endl;
        logStream << "confidences: ";
        for (auto element : confidences) {
            logStream << element << " ";
        }
        logStream << std::endl;
        logStream << "inference time: " << labeler.getInference() << std::endl;

        cv::imwrite(outputPath, image);
        labeler.writeLabels(outputPath);
        //Сохранение логов
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);

        return 0;
    }

}