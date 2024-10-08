#include "mrcv/mrcv.h"

namespace mrcv
{
   void Predictor::trainLSTMNet(const std::vector<std::pair<float, float>> coordinates, bool isTraining) 
   {
        std::vector<std::pair<float, float>> coordinatesNormalized = normilizeInput(coordinates);
        // ��������� trainingData ������ ���� ������������� ����� ����������
        if (!coordinatesNormalized.empty()) 
        {
            trainingData.clear();
            for (const auto& coord : coordinatesNormalized) 
            {
                torch::Tensor input = torch::tensor({ coord.first, coord.second },
                    torch::dtype(torch::kFloat32)).view({ 1, 1, inputSize }); // (seq_len=1, batch_size=1, inputSize)
                trainingData.push_back(input);
            }
        }
        // ���������� ������� ������ � �����������
        auto criterion = torch::nn::MSELoss();
        torch::optim::Adam optimizer(lstm->parameters(), torch::optim::AdamOptions(0.001));
        // ���� ��������
        lstm->train();
        for (size_t epoch = 0; epoch < 50; ++epoch) 
        {
            optimizer.zero_grad();
            // ���������� ������� ������ � �����
            torch::Tensor inputs = torch::cat(trainingData, 0);
            torch::Tensor targets = inputs.clone();           
            // �������������� ������� ���������
            if (!isTraining)    // ��������� ��������� ������� ����� � ��������� � ������ ����������� ��������
            {
                hiddenState = torch::zeros({ numLayers, 1, hiddenSize }, torch::kFloat32);
                cellState = torch::zeros({ numLayers, 1, hiddenSize }, torch::kFloat32);
            }
            // ������ ������
            auto outputs_tuple = lstm->forward(inputs, std::make_tuple(hiddenState, cellState));
            torch::Tensor outputs = std::get<0>(outputs_tuple);
            auto state_tuple = std::get<1>(outputs_tuple);
            hiddenState = std::get<0>(state_tuple).detach();
            cellState = std::get<1>(state_tuple).detach();            
            // ������ ����� �������� ����
            torch::Tensor outputs_linear = linear->forward(outputs.view({ -1, hiddenSize })); // (seq_len * batch_size, inputSize)
            outputs_linear = outputs_linear.view({ -1, 1, inputSize }); // (seq_len, batch_size=1, inputSize)            
            // ���������� ������� ������
            auto loss = criterion(outputs_linear, targets);
            // �������� ��������������� � ��� �����������
            loss.backward();
            torch::nn::utils::clip_grad_norm_(lstm->parameters(), 0.1);
            optimizer.step();
        }
    }

    void Predictor::continueTraining(const std::pair<float, float> coordinate) 
    {
        std::pair<float, float> coordinate_norm = normilizePair(coordinate);
        torch::Tensor input = torch::tensor({ coordinate_norm.first, coordinate_norm.second }).view({ 1, 1, inputSize });
        trainingData.push_back(input);
        // ��������� ������ 100 ��������� ���������
        if (trainingData.size() > 100)
        {
            trainingData.erase(trainingData.begin());
        }
        // �������� �������� ��� ���������� trainingData
        trainLSTMNet({}, true);
    }
    std::pair<float, float> Predictor::predictNextCoordinate() {
        lstm->eval();
        linear->eval();
        torch::Tensor inputs = torch::cat(trainingData, 0);
        auto lstm_out_tuple = lstm->forward(inputs, std::make_tuple(hiddenState, cellState));
        torch::Tensor lstm_out = std::get<0>(lstm_out_tuple);
        torch::Tensor outputs = linear->forward(lstm_out[-1]);
        float x_pred = outputs[0][0].item<float>();
        float y_pred = outputs[0][1].item<float>();
        return denormilizeOutput({ x_pred, y_pred });
    }

    std::pair<float, float> Predictor::normilizePair(std::pair<float, float> coords)
    {
        return  std::make_pair(coords.first / imgWidth * 2 - 1, coords.second / imgHeight * 2 - 1);
    }

    std::vector<std::pair<float, float>> Predictor::normilizeInput(std::vector<std::pair<float, float>> coords) 
    {
        std::vector<std::pair<float, float>> coords_norm;
        for (int i = 0; i < coords.size(); i++) 
        {
            coords_norm.push_back({ coords[i].first / imgWidth, coords[i].second / imgHeight });
            coords_norm.push_back(normilizePair({ coords[i].first, coords[i].second }));
        }
        return coords_norm;
    }
    std::pair<float, float> Predictor::denormilizeOutput(std::pair<float, float> coords)
    {
        return  { (coords.first + 1) * imgWidth / 2, (coords.second + 1) * imgHeight / 2 };
    }
}