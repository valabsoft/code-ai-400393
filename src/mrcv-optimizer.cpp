#include "mrcv/mrcv.h"
#include <random>
#include <algorithm> 

static std::pair<float, float> computeRange(const float& baseValue)
{
    float lower = std::max(0.0f, baseValue - baseValue * 0.5f);
    float upper = baseValue + baseValue * 0.5f;
    return std::make_pair(lower, upper);
}

static float normalizeFactor(const std::pair<float, float>& range)
{
    return 1 / std::max(range.first, range.second);
}

namespace mrcv
{
    void Optimizer::generateSyntheticData() 
    {
        float baseDisplacement = std::sqrt(pow(prevCoord.first - nextCoord.first, 2) + pow(prevCoord.second - nextCoord.second, 2));    // Смещение объекта 
        std::vector<std::vector<float>> inputData;
        std::vector<float> targetData;
        // Создаем рандомайзер для создание синтетических даных
        std::random_device rd;
        std::mt19937 gen(rd());

        // Получаем диапазоны для каждого параметра
        std::pair<float, float> displacementRange = computeRange(baseDisplacement);
        std::pair<float, float> objectSizeRange = computeRange(objectSize);
        std::pair<float, float> deviationRange = computeRange(averagePredictionDeviation);

        std::uniform_real_distribution<> displacementDist(displacementRange.first, displacementRange.second);
        std::uniform_real_distribution<> sizeDist(objectSizeRange.first, objectSizeRange.second);
        std::uniform_real_distribution<> deviationDist(deviationRange.first, deviationRange.second);

        for (size_t i = 0; i < sampleSize; ++i) 
        {
            // Генерируем параметры в диапазоне ±50% от базовых значений
            float displacement = displacementDist(gen);
            float objectSize = sizeDist(gen);
            float averagePredictionDeviation = deviationDist(gen);
            // Вычисляем необходимый размер ROI
            float roiSize = (objectSize + averagePredictionDeviation * 2 + displacement)*1;
            // Собираем входные данные
            inputData.push_back({ displacement, objectSize, averagePredictionDeviation });
            // Целевое значение - рассчитанный размер ROI
            targetData.push_back(roiSize);
        }

        float maxRoiSize = 0;

        for (size_t i = 0; i < sampleSize; ++i)
        {
            maxRoiSize = targetData[i] > maxRoiSize ? targetData[i] : maxRoiSize;
        }

        roiSizeNormFactor = 1 / maxRoiSize;

        for (size_t i = 0; i < sampleSize; ++i)
        {
            targetData[i] = targetData[i] * roiSizeNormFactor;
        }
        // Преобразуем данные в тензоры
        std::vector<torch::Tensor> tensorList;
        for (const auto& vec : inputData) 
        {
            tensorList.push_back(torch::tensor({vec[0] * roiSizeNormFactor * 2 - 1, vec[1] * roiSizeNormFactor * 2 - 1, vec[2] * roiSizeNormFactor * 2 - 1 }));
        }
        inputs = torch::stack(tensorList, 0);
        targets = torch::tensor(targetData).unsqueeze(1);


    }

    void Optimizer::trainModel()
    {
        model->train();
        auto optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(0.001));
        for (size_t epoch = 0; epoch < epochs; ++epoch) 
        {
            optimizer.zero_grad();
            at::Tensor output = model->forward(inputs);
            at::Tensor loss = torch::mse_loss(output, targets);
            loss.backward();
            optimizer.step();
        }
    }

    float Optimizer::optimizeRoiSize(const std::pair<float, float>& _prevCoord,
        const std::pair<float, float>& _nextCoord,
        const float& _objectSize,
        const float& _averagePredictionError)
    {
        prevCoord = _prevCoord;
        nextCoord = _nextCoord;
        objectSize = _objectSize;
        averagePredictionDeviation = _averagePredictionError;
        // Создание синтетического набора данных для обучения
        generateSyntheticData();
        // Подготовка входных данных
        float displacement = std::sqrt(pow(prevCoord.first - nextCoord.first, 2) + pow(prevCoord.second - nextCoord.second, 2));
        torch::Tensor input = torch::tensor({ displacement * roiSizeNormFactor * 2 - 1, objectSize * roiSizeNormFactor * 2 - 1, averagePredictionDeviation * roiSizeNormFactor * 2 - 1 });
        input = input.unsqueeze(0); // Добавляем размерность батча
        // Предсказание размера ROI
        model->eval();
        auto predictedSize = model->forward(input).item<float>();
        predictedSize = (predictedSize + 1) / roiSizeNormFactor / 2 ;        
        if (predictedSize >= objectSize * 1.1)
        {
            return predictedSize;
        }
        else
        {
            return 0;
        }
        
    }
}