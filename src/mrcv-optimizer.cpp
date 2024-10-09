#include "mrcv/mrcv.h"
#include <random>
#include <algorithm> 

static std::pair<float, float> computeRange(float baseValue)
{
    float lower = std::max(0.0f, baseValue - baseValue * 0.5f);
    float upper = baseValue + baseValue * 0.5f;
    return std::make_pair(lower, upper);
}

namespace mrcv
{
    void Optimizer::generateSyntheticData() 
    {
        float baseDisplacement = std::sqrt(pow(prevCoord.first - nextCoord.first, 2) + pow(prevCoord.second - nextCoord.second, 2));    // �������� ������� 
        std::vector<std::vector<float>> inputData;
        std::vector<float> targetData;
        // ������� ����������� ��� �������� ������������� �����
        std::random_device rd;
        std::mt19937 gen(rd());

        // �������� ��������� ��� ������� ���������
        std::pair<float, float> displacementRange = computeRange(baseDisplacement);
        std::pair<float, float> objectSizeRange = computeRange(objectSize);
        std::pair<float, float> errorRange = computeRange(averagePredictionError);

        std::uniform_real_distribution<> displacementDist(displacementRange.first, displacementRange.second);
        std::uniform_real_distribution<> sizeDist(objectSizeRange.first, objectSizeRange.second);
        std::uniform_real_distribution<> errorDist(errorRange.first, errorRange.second);

        for (size_t i = 0; i < sampleSize; ++i) 
        {
            // ���������� ��������� � ��������� �50% �� ������� ��������
            float displacement = displacementDist(gen);
            float objectSize = sizeDist(gen);
            float averagePredictionError = errorDist(gen);
            // ��������� ����������� ������ ROI
            float roiSize = displacement + objectSize + 2 * averagePredictionError;
            // �������� ������� ������
            inputData.push_back({ displacement, objectSize, averagePredictionError });
            // ������� �������� - ������������ ������ ROI
            targetData.push_back(roiSize);
        }
        // ����������� ������ � �������
        std::vector<torch::Tensor> tensorList;
        for (const auto& vec : inputData) 
        {
            tensorList.push_back(torch::tensor(vec));
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
        averagePredictionError = _averagePredictionError;
        // �������� �������������� ������ ������ ��� ��������
        generateSyntheticData();
        // ���������� ������� ������
        float displacement = std::sqrt(pow(prevCoord.first - nextCoord.first, 2) + pow(prevCoord.second - nextCoord.second, 2));
        torch::Tensor input = torch::tensor({ displacement, objectSize, averagePredictionError });
        input = input.unsqueeze(0); // ��������� ����������� �����
        // ������������ ������� ROI
        model->eval();
        auto predictedSize = model->forward(input).item<float>();
        // ����������, ��� ROI ����� ���������� ������
        float minRoiSize = objectSize + 2 * averagePredictionError;
        if (predictedSize < minRoiSize) 
        {
            predictedSize = minRoiSize;
        }
        return predictedSize;
    }
}