#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>
#include <mrcv/mrcv-vae.h>

namespace mrcv
{
    torch::Tensor neuralNetworkAugmentationAsTensor(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate)
    {
        torch::Tensor tensor;

        // Количестыо цветов генерируемого изображения
        const int64_t numColor = 1;
        // Расчет размерности входно слоя
        const int64_t inputDim = numColor * height * width;
        // Выбор устройства
        torch::Device device(torch::kCUDA);
        // Загрузка датасета
        auto dataset = LoadImageDataset(root, height, width, numColor).map(torch::data::transforms::Stack<>());

        // Проверка датасета
        if (dataset.size() == 0)
        {
            writeLog("No Image loaded!", mrcv::LOGTYPE::ERROR);
        }

        auto dataLoader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(batchSize).workers(2));

        // Создание объекта модели
        VariationalAutoEncoder model(inputDim, hDim, zDim);
        // Перенос модели на устройство
        model.to(device);
        // Оптимизация модели
        torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lrRate));

        writeLog("Training...", mrcv::LOGTYPE::INFO);
        // Обучение модели
        for (int epoch = 0; epoch < numEpoch; ++epoch)
        {
            model.train();
            for (auto& batch : *dataLoader)
            {
                auto data = batch.data.to(device).view({ -1, inputDim });

                if (torch::isnan(data).any().item<bool>() || torch::isinf(data).any().item<bool>())
                {
                    writeLog("Data tensor contains NaN or Inf values!", mrcv::LOGTYPE::ERROR);
                    continue;
                }

                optimizer.zero_grad();
                auto [output, mu, sigma] = model.forward(data);
                output = torch::sigmoid(output);

                auto reconstructionLoss = torch::binary_cross_entropy(output, data, {}, torch::Reduction::Sum);

                reconstructionLoss.backward();
                optimizer.step();
            }
        }
        writeLog("Training is DONE!", mrcv::LOGTYPE::INFO);

        auto numberLoadImages = LoadImageDataset(root, height, width, numColor);
        int num = numberLoadImages.get_num_images();

        // Коллическтво генерируемых изображений
        int numExamples = 1;
        srand((unsigned int)time(NULL));

        std::vector<torch::Tensor> images;

        int randImage = rand() % num;
        auto example = dataset.get_batch(randImage);
        images.push_back(example.data.to(device));

        std::vector<std::pair<torch::Tensor, torch::Tensor>> encodingsDigit;

        writeLog("Encoding...", mrcv::LOGTYPE::INFO);
        // Энкодер
        for (int d = 0; d < numExamples; ++d)
        {
            torch::NoGradGuard no_grad;
            auto flattenedIimage = images[d].view({ 1, inputDim });
            auto [mu, sigma] = model.encode(flattenedIimage.to(device));
            encodingsDigit.push_back({ mu, sigma });
        }
        writeLog("Decoding...", mrcv::LOGTYPE::INFO);
        // Декодер
        for (int example = 0; example < numExamples; ++example)
        {
            auto [mu, sigma] = encodingsDigit[0];
            auto sample = model.reparameterize(mu, sigma);
            tensor = model.decode(sample.to(device));
            tensor = torch::sigmoid(tensor).view({ numColor, width, height });
            return tensor.clone();
        }
        // Сохранение логов
        writeLog("Generated tensor is DONE!", mrcv::LOGTYPE::INFO);

        return tensor.clone();
    }

    cv::Mat neuralNetworkAugmentationAsMat(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate)
    {
        // Количестыо цветов генерируемого изображения
        const int64_t numColor = 1;
        
        torch::Tensor tensor = neuralNetworkAugmentationAsTensor(root, height, width, hDim, zDim, numEpoch, batchSize, lrRate);

        // Проверка, что тензор не пустой
        if (!tensor.defined())
        {
            writeLog("Generated tensor empty!", mrcv::LOGTYPE::ERROR);
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
            writeLog("Generated image empty!", mrcv::LOGTYPE::ERROR);
        }
        image.convertTo(image, -1, 2.3, -300);

        // Сохранение логов
        writeLog("Generated image is DONE!", mrcv::LOGTYPE::INFO);

        return image.clone();
    }

    NNPreLabeler::NNPreLabeler(const std::string model, const std::string classes,
        int width, int height) {
        inputWidth = width;
        inputHeight = height;
        if (initNetwork(model, classes)) {
            writeLog("Failed to init neural network!", mrcv::LOGTYPE::ERROR);
        }
    }

    int NNPreLabeler::readClasses(const std::string filePath) {
        std::ifstream classesFile(filePath);
        std::string line;

        if (!classesFile) {
            writeLog("Failed to open classes names!\n", mrcv::LOGTYPE::ERROR);
            return ENOENT;
        }
        while (std::getline(classesFile, line)) {
            classes.push_back(line);
        }
        classesFile.close();

        return 0;
    }

    int NNPreLabeler::initNetwork(const std::string modelPath, const std::string classesPath) {
        int err = readClasses(classesPath);

        if (err == 0) {
            network = cv::dnn::readNetFromONNX(modelPath);
            if (network.empty()) {
                return ENETDOWN;
            }
            else {
                network.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        }

        return err;
    }

    // Нарисовать прогнозируемый ограничивающий прямоугольник
    void NNPreLabeler::drawLabel(cv::Mat& img, std::string label, int left, int top) {
        // Отобразить метку в верхней части ограничивающей рамки
        int baseline;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
            FONT_SCALE, THICKNESS, &baseline);
        top = std::max(top, labelSize.height);
        // Верхний левый угол
        cv::Point tlc = cv::Point(left, top);
        // Правый нижний угол
        cv::Point brc = cv::Point(left + labelSize.width,
            top + labelSize.height + baseline);
        // Нарисуйте черный прямоугольник
        cv::rectangle(img, tlc, brc, BLACK, cv::FILLED);
        // Наклейте этикетку на черный прямоугольник
        cv::putText(img, label, cv::Point(left, top + labelSize.height),
            cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS);
    }

    std::vector<cv::Mat> NNPreLabeler::preProcess(cv::Mat& img) {
        // Преобразовать в blob
        cv::Mat blob;
        cv::dnn::blobFromImage(img, blob, 1.0 / 255, cv::Size(inputWidth, inputHeight),
            cv::Scalar(), true, false);
        network.setInput(blob);       
        std::vector<cv::Mat> outputs;
        network.forward(outputs, network.getUnconnectedOutLayersNames());
        return outputs;
    }

    cv::Mat NNPreLabeler::postProcess(cv::Mat& img, std::vector<cv::Mat>& outputs, const std::vector<std::string>& className) {
        // Инициализируйте векторы для хранения соответствующих выходных данных при развертывании обнаружений
        cv::Mat ret = img.clone();
        classesIdSet.clear();
        confidencesSet.clear();
        boxesSet.clear();
        classesSet.clear();
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Изменения размера
        float xFactor = img.cols / (float)inputWidth;
        float yFactor = img.rows / (float)inputHeight;

        int dimensions = outputs[0].size[2];
        int rows = outputs[0].size[1];
        float* data = (float*)outputs[0].data;

        // Повтор обнаружения
        for (int i = 0; i < rows; ++i) {
            float confidence = data[4];
            // Отмените неудачные обнаружения и продолжите
            if (confidence >= CONFIDENCE_THRESHOLD) {
                float* classesScores = data + 5;
                // Создайте мат размером 1x85 и сохраните баллы 80 классов
                cv::Mat scores(1, className.size(), CV_32FC1, classesScores);
                // Выполните minMaxLoc и получите индекс наилучшего результата класса
                cv::Point classId;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classId);
                // Продолжайте, если баллы класса превышают пороговое значение
                if (maxClassScore > SCORE_THRESHOLD) {
                    // Сохраните идентификатор класса и уровень уверенности в соответствующих предопределенных векторах
                    confidences.push_back(confidence);
                    classIds.push_back(classId.x);
                    // Центр
                    float cx = data[0];
                    float cy = data[1];
                    // Размеры коробки
                    float w = data[2];
                    float h = data[3];
                    // Координаты ограничивающего прямоугольника
                    int left = int((cx - 0.5 * w) * xFactor);
                    int top = int((cy - 0.5 * h) * yFactor);
                    int width = int(w * xFactor);
                    int height = int(h * yFactor);
                    // Сохраняйте хорошие обнаружения в векторе
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            // Перейти к следующему столбцу
            data += dimensions;
        }

        // Выполнить немаксимальное подавление и составить прогнозы
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
        for (int i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];

            boxesSet.push_back(box);
            confidencesSet.push_back(confidences[idx]);
            classesIdSet.push_back(classIds[idx]);
            classesSet.push_back(className[classIds[idx]]);

            int left = box.x;
            int top = box.y;
            int width = box.width;
            int height = box.height;
            // Нарисуйте ограничивающую рамку
            cv::rectangle(ret, cv::Point(left, top), cv::Point(left + width, top + height), GREEN, 3 * THICKNESS);
            // Получите метку для имени класса и его достоверности
            std::string label = cv::format("%.2f", confidences[idx]);
            label = className[classIds[idx]] + ": " + label;
            // Нарисуйте метки классов
            drawLabel(ret, label, left, top);
        }
        return ret;
    }

    cv::Mat NNPreLabeler::process(cv::Mat& img) {
        NNPreLabeler::sourceSize = img.size();

        std::vector<cv::Mat> detections;
        detections = preProcess(img);
        cv::Mat res = postProcess(img, detections, NNPreLabeler::classes);
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency();
        NNPreLabeler::inferenceTime = network.getPerfProfile(layersTimes) / freq;
        return res;
    }

    // Функция для записи YOLOv5 лейблов в текстовый файл
    void NNPreLabeler::writeLabels(const std::string& filename) {
        // Открываем файл для записи
        std::ofstream outfile(filename);

        // Проверяем, открыт ли файл
        if (!outfile.is_open()) {
            writeLog("Failed to open file: " + filename, mrcv::LOGTYPE::ERROR);
            return;
        }

        // Записываем каждый лейбл в формате YOLOv5
        for (size_t i = 0; i < NNPreLabeler::classesIdSet.size(); ++i) {
            int classId = NNPreLabeler::classesIdSet[i];
            const cv::Rect& box = NNPreLabeler::boxesSet[i];

            // Преобразуем координаты и размеры из абсолютных значений в диапазон [0, 1]
            float xCenter = (box.x + box.width / 2.0) / NNPreLabeler::sourceSize.width;
            float yCenter = (box.y + box.height / 2.0) / NNPreLabeler::sourceSize.height;
            float width = box.width / static_cast<float>(NNPreLabeler::sourceSize.width);
            float height = box.height / static_cast<float>(NNPreLabeler::sourceSize.height);

            // Записываем строку в формате YOLOv5
            outfile << classId << " " << xCenter << " " << yCenter << " " << width << " " << height << std::endl;
        }

        // Закрываем файл
        outfile.close();
    }

    int semiAutomaticLabeler(cv::Mat& inputImage, const int height, const int width, const std::string& outputPath, const std::string& modelPath, const std::string& classesPath) {
        NNPreLabeler labeler(modelPath, classesPath, 640, 640);
        cv::resize(inputImage, inputImage, cv::Size(640, 640));
        cv::Mat img = labeler.process(inputImage);
        std::vector<int> classIds = labeler.getClassIds();
        std::vector<float> confidences = labeler.getConfidences();
        std::vector<cv::Rect> boxes = labeler.getBoxes();
        std::vector<std::string> classes = labeler.getClasses();

        for (auto element : confidences) {
            auto str = std::to_string(element);
            writeLog("confidences: " + str, mrcv::LOGTYPE::INFO);
        }
        auto str = std::to_string(labeler.getInference());
        writeLog("inference time: " + str, mrcv::LOGTYPE::INFO);

        const std::string filename = outputPath + "/result.jpg"; 
        const std::string labels = outputPath + "/result.txt";

        cv::imwrite(filename, img);
        labeler.writeLabels(labels);
        writeLog("Labeling image is DONE!", mrcv::LOGTYPE::INFO);
        return 0;
    }

    int semiAutomaticLabeler(const std::string& root, const int height, const int width, const std::string& outputPath, const std::string& modelPath, const std::string& classesPath) {
        NNPreLabeler labeler(modelPath, classesPath, 640, 640);
        cv::Mat inputImage = cv::imread(root);
        cv::resize(inputImage, inputImage, cv::Size(640, 640));
        cv::Mat img = labeler.process(inputImage);
        std::vector<int> classIds = labeler.getClassIds();
        std::vector<float> confidences = labeler.getConfidences();
        std::vector<cv::Rect> boxes = labeler.getBoxes();
        std::vector<std::string> classes = labeler.getClasses();
        
        for (auto element : confidences) {
            auto str = std::to_string(element);
            writeLog("confidences: " + str, mrcv::LOGTYPE::INFO);
        }
        auto str = std::to_string(labeler.getInference());
        writeLog("inference time: " + str, mrcv::LOGTYPE::INFO);

        const std::string filename = outputPath + "/result.jpg";
        const std::string labels = outputPath + "/result.txt";

        cv::imwrite(filename, img);
        labeler.writeLabels(labels);
        writeLog("Labeling image is DONE!", mrcv::LOGTYPE::INFO);
        return 0;
    }

#ifdef MRCV_CUDA_ENABLED 
    torch::Tensor neuralNetworkAugmentationAsTensorCuda(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate)
    {
        torch::Tensor tensor;

        // Количестыо цветов генерируемого изображения
        const int64_t numColor = 1;
        // Расчет размерности входно слоя
        const int64_t inputDim = numColor * height * width;
        // Выбор устройства
        torch::Device device(torch::kCUDA);
        // Загрузка датасета
        auto dataset = LoadImageDataset(root, height, width, numColor).map(torch::data::transforms::Stack<>());

        // Проверка датасета
        if (dataset.size() == 0)
        {
            writeLog("No Image loaded!", mrcv::LOGTYPE::ERROR);
        }

        auto dataLoader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(batchSize).workers(2));

        // Создание объекта модели
        VariationalAutoEncoder model(inputDim, hDim, zDim);
        // Перенос модели на устройство
        model.to(device);
        // Оптимизация модели
        torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lrRate));

        writeLog("Training...", mrcv::LOGTYPE::INFO);
        // Обучение модели
        for (int epoch = 0; epoch < numEpoch; ++epoch)
        {
            model.train();
            for (auto& batch : *dataLoader)
            {
                auto data = batch.data.to(device).view({ -1, inputDim });

                if (torch::isnan(data).any().item<bool>() || torch::isinf(data).any().item<bool>())
                {
                    writeLog("Data tensor contains NaN or Inf values!", mrcv::LOGTYPE::ERROR);
                    continue;
                }

                optimizer.zero_grad();
                auto [output, mu, sigma] = model.forward(data);
                output = torch::sigmoid(output);

                auto reconstructionLoss = torch::binary_cross_entropy(output, data, {}, torch::Reduction::Sum);

                reconstructionLoss.backward();
                optimizer.step();
            }
        }
        writeLog("Training is DONE!", mrcv::LOGTYPE::INFO);

        auto numberLoadImages = LoadImageDataset(root, height, width, numColor);
        int num = numberLoadImages.get_num_images();

        // Коллическтво генерируемых изображений
        int numExamples = 1;
        srand((unsigned int)time(NULL));

        std::vector<torch::Tensor> images;

        int randImage = rand() % num;
        auto example = dataset.get_batch(randImage);
        images.push_back(example.data.to(device));

        std::vector<std::pair<torch::Tensor, torch::Tensor>> encodingsDigit;

        writeLog("Encoding...", mrcv::LOGTYPE::INFO);
        // Энкодер
        for (int d = 0; d < numExamples; ++d)
        {
            torch::NoGradGuard no_grad;
            auto flattenedIimage = images[d].view({ 1, inputDim });
            auto [mu, sigma] = model.encode(flattenedIimage.to(device));
            encodingsDigit.push_back({ mu, sigma });
        }
        writeLog("Decoding...", mrcv::LOGTYPE::INFO);
        // Декодер
        for (int example = 0; example < numExamples; ++example)
        {
            auto [mu, sigma] = encodingsDigit[0];
            auto sample = model.reparameterize(mu, sigma);
            tensor = model.decode(sample.to(device));
            tensor = torch::sigmoid(tensor).view({ numColor, width, height });
            return tensor.clone();
        }
        // Сохранение логов
        writeLog("Generated tensor is DONE!", mrcv::LOGTYPE::INFO);

        return tensor.clone();
    }

    cv::Mat neuralNetworkAugmentationAsMatCuda(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate)
    {
        // Количестыо цветов генерируемого изображения
        const int64_t numColor = 1;

        torch::Tensor tensor = neuralNetworkAugmentationAsTensorCuda(root, height, width, hDim, zDim, numEpoch, batchSize, lrRate);

        // Проверка, что тензор не пустой
        if (!tensor.defined())
        {
            writeLog("Generated tensor empty!", mrcv::LOGTYPE::ERROR);
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
            writeLog("Generated image empty!", mrcv::LOGTYPE::ERROR);
        }
        image.convertTo(image, -1, 2.3, -300);

        // Сохранение логов
        writeLog("Generated image is DONE!", mrcv::LOGTYPE::INFO);

        return image.clone();
    }

    NNPreLabelerCuda::NNPreLabelerCuda(const std::string& model, const std::string& classes, int width, int height) {
        inputWidth = width;
        inputHeight = height;
        if (initNetworkCuda(model, classes)) {
            writeLog("Failed to init neural network!", mrcv::LOGTYPE::ERROR);
        }
    }

    int NNPreLabelerCuda::readClassesCuda(const std::string& filePath) {
        std::ifstream classesFile(filePath);
        if (!classesFile) {
            writeLog("Failed to open classes names: " + filePath, mrcv::LOGTYPE::ERROR);
            return ENOENT;
        }
        std::string line;
        while (std::getline(classesFile, line)) {
            classes.push_back(line);
        }
        classesFile.close();
        return 0;
    }

    int NNPreLabelerCuda::initNetworkCuda(const std::string& modelPath, const std::string& classesPath) {
        int err = readClassesCuda(classesPath);
        if (err == 0) {
            network = cv::dnn::readNetFromONNX(modelPath);
            if (network.empty()) {
                writeLog("Failed to load ONNX model from " + modelPath, mrcv::LOGTYPE::ERROR);
                return ENETDOWN;
            }
            network.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            network.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        return err;
    }

    std::vector<cv::Mat> NNPreLabelerCuda::preProcessCuda(cv::cuda::GpuMat& img) {
        try {
            writeLog("Pre-processing GpuMat size: " + std::to_string(img.cols) + "x" + std::to_string(img.rows), mrcv::LOGTYPE::INFO);

            // Переносим изображение на CPU
            cv::Mat cpuImg;
            img.download(cpuImg);
            if (cpuImg.empty()) {
                writeLog("Failed to download GpuMat to CPU!", mrcv::LOGTYPE::ERROR);
                throw std::runtime_error("Empty CPU image after download");
            }

            // Преобразуем цветовое пространство на CPU
            cv::cvtColor(cpuImg, cpuImg, cv::COLOR_BGR2RGB);

            // Создаём блоб на CPU
            cv::Mat blob;
            cv::dnn::blobFromImage(cpuImg, blob, 1.0 / 255, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);

            // Устанавливаем вход для сети
            network.setInput(blob);

            // Выполняем прямой проход
            std::vector<cv::Mat> outputs;
            network.forward(outputs, network.getUnconnectedOutLayersNames());

            writeLog("Pre-process outputs size: " + std::to_string(outputs.size()), mrcv::LOGTYPE::INFO);
            return outputs;
        }
        catch (const cv::Exception& e) {
            writeLog("OpenCV exception in preProcessCuda: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            throw;
        }
        catch (const std::exception& e) {
            writeLog("Standard exception in preProcessCuda: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            throw;
        }
    }

    cv::cuda::GpuMat NNPreLabelerCuda::postProcessCuda(cv::cuda::GpuMat& img, std::vector<cv::Mat>& outputs, const std::vector<std::string>& className) {
        cv::cuda::GpuMat ret = img.clone();
        classesIdSet.clear();
        confidencesSet.clear();
        boxesSet.clear();
        classesSet.clear();

        float xFactor = img.cols / (float)inputWidth;
        float yFactor = img.rows / (float)inputHeight;

        int dimensions = outputs[0].size[2];
        int rows = outputs[0].size[1];
        float* data = (float*)outputs[0].data;

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (int i = 0; i < rows; ++i) {
            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD) {
                float* classesScores = data + 5;
                cv::Mat scores(1, className.size(), CV_32FC1, classesScores);
                cv::Point classId;
                double maxClassScore;
                cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classId);
                if (maxClassScore > SCORE_THRESHOLD) {
                    confidences.push_back(confidence);
                    classIds.push_back(classId.x);
                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];
                    int left = int((cx - 0.5 * w) * xFactor);
                    int top = int((cy - 0.5 * h) * yFactor);
                    int width = int(w * xFactor);
                    int height = int(h * yFactor);
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
            data += dimensions;
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        for (int i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            boxesSet.push_back(boxes[idx]);
            confidencesSet.push_back(confidences[idx]);
            classesIdSet.push_back(classIds[idx]);
            classesSet.push_back(className[classIds[idx]]);
        }

        return ret;  // Возвращаем изображение без отрисовки
    }

    cv::cuda::GpuMat NNPreLabelerCuda::processCuda(cv::cuda::GpuMat& img) {
        sourceSize = cv::Size(img.cols, img.rows);
        std::vector<cv::Mat> detections = preProcessCuda(img);
        cv::cuda::GpuMat res = postProcessCuda(img, detections, classes);
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency();
        inferenceTime = network.getPerfProfile(layersTimes) / freq;
        return res;
    }

    void NNPreLabelerCuda::writeLabelsCuda(const std::string& filename) {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            writeLog("Failed to open file: " + filename, mrcv::LOGTYPE::ERROR);
            return;
        }
        for (size_t i = 0; i < classesIdSet.size(); ++i) {
            int classId = classesIdSet[i];
            const cv::Rect& box = boxesSet[i];
            float xCenter = (box.x + box.width / 2.0) / sourceSize.width;
            float yCenter = (box.y + box.height / 2.0) / sourceSize.height;
            float width = box.width / static_cast<float>(sourceSize.width);
            float height = box.height / static_cast<float>(sourceSize.height);
            outfile << classId << " " << xCenter << " " << yCenter << " " << width << " " << height << std::endl;
        }
        outfile.close();
    }

    int semiAutomaticLabelerCuda(cv::Mat& inputImage, const int height, const int width, const std::string& outputPath, const std::string& modelPath, const std::string& classesPath) {
        try {
            if (inputImage.empty()) {
                writeLog("Input image is empty!", mrcv::LOGTYPE::ERROR);
                return -1;
            }

            NNPreLabelerCuda labeler(modelPath, classesPath, 640, 640);
            cv::resize(inputImage, inputImage, cv::Size(640, 640));

            cv::cuda::GpuMat gpuImg;
            gpuImg.upload(inputImage);
            cv::cuda::GpuMat img = labeler.processCuda(gpuImg);

            // Переносим результат на CPU
            cv::Mat cpuImg;
            img.download(cpuImg);
            if (cpuImg.empty()) {
                writeLog("Processed image is empty after download!", mrcv::LOGTYPE::ERROR);
                return -1;
            }

            // Получаем данные после обработки
            std::vector<int> classIds = labeler.getClassIdsCuda();
            std::vector<float> confidences = labeler.getConfidencesCuda();
            std::vector<cv::Rect> boxes = labeler.getBoxesCuda();
            std::vector<std::string> classes = labeler.getClassesCuda();

            // Отрисовка на CPU
            for (size_t i = 0; i < boxes.size(); ++i) {
                int left = boxes[i].x;
                int top = boxes[i].y;
                int width = boxes[i].width;
                int height = boxes[i].height;

                cv::rectangle(cpuImg, cv::Point(left, top), cv::Point(left + width, top + height), GREEN, 3 * THICKNESS);
                std::string label = cv::format("%.2f", confidences[i]);
                label = classes[i] + ": " + label;

                int baseline;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS, &baseline);
                top = std::max(top, labelSize.height);
                cv::Point tlc = cv::Point(left, top);
                cv::Point brc = cv::Point(left + labelSize.width, top + labelSize.height + baseline);
                cv::rectangle(cpuImg, tlc, brc, BLACK, cv::FILLED);
                cv::putText(cpuImg, label, cv::Point(left, top + labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS);
            }

            // Логирование
            for (auto element : confidences) {
                writeLog("confidences: " + std::to_string(element), mrcv::LOGTYPE::INFO);
            }
            writeLog("inference time: " + std::to_string(labeler.getInferenceCuda()), mrcv::LOGTYPE::INFO);

            // Сохранение результата
            const std::string filename = outputPath + "/result.jpg";
            const std::string labels = outputPath + "/result.txt";
            if (!cv::imwrite(filename, cpuImg)) {
                writeLog("Failed to save image to " + filename, mrcv::LOGTYPE::ERROR);
                return -1;
            }
            labeler.writeLabelsCuda(labels);

            writeLog("Labeling image is DONE!", mrcv::LOGTYPE::INFO);
            return 0;
        }
        catch (const cv::Exception& e) {
            writeLog("OpenCV exception: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            return -1;
        }
        catch (const std::exception& e) {
            writeLog("Standard exception: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            return -1;
        }
    }

    int semiAutomaticLabelerCuda(const std::string& root, const int height, const int width, const std::string& outputPath, const std::string& modelPath, const std::string& classesPath) {
        try {
            // Проверка существования входного файла
            std::ifstream fileCheck(root);
            if (!fileCheck.good()) {
                writeLog("Input file does not exist or is inaccessible: " + root, mrcv::LOGTYPE::ERROR);
                return -1;
            }
            fileCheck.close();

            // Загрузка изображения
            cv::Mat inputImage = cv::imread(root);
            if (inputImage.empty()) {
                writeLog("Failed to load image from " + root, mrcv::LOGTYPE::ERROR);
                return -1;
            }

            // Логирование размеров входного изображения
            writeLog("Input image size: " + std::to_string(inputImage.cols) + "x" + std::to_string(inputImage.rows), mrcv::LOGTYPE::INFO);

            NNPreLabelerCuda labeler(modelPath, classesPath, 640, 640);
            cv::resize(inputImage, inputImage, cv::Size(640, 640));

            // Загрузка на GPU
            cv::cuda::GpuMat gpuImg;
            gpuImg.upload(inputImage);

            // Проверка GPU-изображения
            if (gpuImg.empty()) {
                writeLog("Failed to upload image to GPU!", mrcv::LOGTYPE::ERROR);
                return -1;
            }

            // Обработка изображения
            cv::cuda::GpuMat img = labeler.processCuda(gpuImg);

            // Проверка результата обработки
            if (img.empty()) {
                writeLog("Processed GpuMat is empty!", mrcv::LOGTYPE::ERROR);
                return -1;
            }
            writeLog("Processed GpuMat size: " + std::to_string(img.cols) + "x" + std::to_string(img.rows), mrcv::LOGTYPE::INFO);

            // Переносим результат на CPU
            cv::Mat cpuImg;
            img.download(cpuImg);
            if (cpuImg.empty()) {
                writeLog("Processed image is empty after download!", mrcv::LOGTYPE::ERROR);
                return -1;
            }
            writeLog("CPU image size after download: " + std::to_string(cpuImg.cols) + "x" + std::to_string(cpuImg.rows), mrcv::LOGTYPE::INFO);

            // Получаем данные после обработки
            std::vector<int> classIds = labeler.getClassIdsCuda();
            std::vector<float> confidences = labeler.getConfidencesCuda();
            std::vector<cv::Rect> boxes = labeler.getBoxesCuda();
            std::vector<std::string> classes = labeler.getClassesCuda();

            // Логирование
            for (auto element : confidences) {
                writeLog("confidences: " + std::to_string(element), mrcv::LOGTYPE::INFO);
            }
            writeLog("inference time: " + std::to_string(labeler.getInferenceCuda()), mrcv::LOGTYPE::INFO);

            // Сохранение результата
            const std::string filename = outputPath + "/result.jpg";
            const std::string labels = outputPath + "/result.txt";
            if (!cv::imwrite(filename, cpuImg)) {
                writeLog("Failed to save image to " + filename, mrcv::LOGTYPE::ERROR);
                return -1;
            }
            labeler.writeLabelsCuda(labels);

            writeLog("Labeling image is DONE!", mrcv::LOGTYPE::INFO);
            return 0;
        }
        catch (const cv::Exception& e) {
            writeLog("OpenCV exception: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            return -1;
        }
        catch (const std::exception& e) {
            writeLog("Standard exception: " + std::string(e.what()), mrcv::LOGTYPE::ERROR);
            return -1;
        }
        catch (...) {
            writeLog("Unknown exception occurred!", mrcv::LOGTYPE::ERROR);
            return -1;
        }
    }
#endif
}