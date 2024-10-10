#include "mrcv/mrcv.h"

namespace mrcv
{
   void Predictor::trainLSTMNet(const std::vector<std::pair<float, float>> coordinates, bool isTraining) 
   {
        std::vector<std::pair<float, float>> coordinatesNormalized = normilizeInput(coordinates);
        // Обновляем trainingData только если предоставлены новые координаты
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
        // Определяем функцию потерь и оптимизатор
        auto criterion = torch::nn::MSELoss();
        torch::optim::Adam optimizer(lstm->parameters(), torch::optim::AdamOptions(0.001));
        // Цикл обучения
        lstm->train();
        for (size_t epoch = 0; epoch < 50; ++epoch) 
        {
            optimizer.zero_grad();
            // Подготовка входных данных и целей
            torch::Tensor inputs = torch::cat(trainingData, 0);
            torch::Tensor targets = inputs.clone();           
            // Инициализируем скрытые состояния
            if (!isTraining)    // Отключаем обнуление скрытых слоев и состояний в случае продолжения обучения
            {
                hiddenState = torch::zeros({ numLayers, 1, hiddenSize }, torch::kFloat32);
                cellState = torch::zeros({ numLayers, 1, hiddenSize }, torch::kFloat32);
            }
            // Прямой проход
            auto outputs_tuple = lstm->forward(inputs, std::make_tuple(hiddenState, cellState));
            torch::Tensor outputs = std::get<0>(outputs_tuple);
            auto state_tuple = std::get<1>(outputs_tuple);
            hiddenState = std::get<0>(state_tuple).detach();
            cellState = std::get<1>(state_tuple).detach();            
            // Проход через линейный слой
            torch::Tensor outputs_linear = linear->forward(outputs.view({ -1, hiddenSize })); // (seq_len * batch_size, inputSize)
            outputs_linear = outputs_linear.view({ -1, 1, inputSize }); // (seq_len, batch_size=1, inputSize)            
            // Вычисление функции потерь
            auto loss = criterion(outputs_linear, targets);
            // Обратное распространение и шаг оптимизации
            loss.backward();
            torch::nn::utils::clip_grad_norm_(lstm->parameters(), 0.1);
            optimizer.step();
        }
    }

    void Predictor::continueTraining(const std::pair<float, float> coordinate) 
    {
        coordsReal = coordinate;
        updateDeviations();
        std::pair<float, float> coordinate_norm = normilizePair(coordsReal);
        torch::Tensor input = torch::tensor({ coordinate_norm.first, coordinate_norm.second }).view({ 1, 1, inputSize });
        trainingData.push_back(input);
        // Сохраняем только 100 последних координат
        if (trainingData.size() > 100)
        {
            trainingData.erase(trainingData.begin());
        }
        // Вызываем обучение без обновления trainingData
        trainLSTMNet({}, true);
    }
    std::pair<float, float> Predictor::predictNextCoordinate() 
    {
        lstm->eval();
        linear->eval();
        torch::Tensor inputs = torch::cat(trainingData, 0);
        auto lstm_out_tuple = lstm->forward(inputs, std::make_tuple(hiddenState, cellState));
        torch::Tensor lstm_out = std::get<0>(lstm_out_tuple);
        torch::Tensor outputs = linear->forward(lstm_out[-1]);
        coordsPred = denormilizeOutput({ outputs[0][0].item<float>() , outputs[0][1].item<float>() });
        numPredictions++;
        return coordsPred;
    }

    void Predictor::updateDeviations()
    {
        static float movingAvgPredDevSum = 0;
        static float predDevSum = 0;
        static int successedPredictions = 0;

        predictionDeviation = std::sqrt(std::pow((coordsPred.first - coordsReal.first), 2) + std::pow((coordsPred.second - coordsReal.second), 2));
        lastPredictionDeviations.push_back(predictionDeviation);
        if (numPredictions > movingAvgScale)
        {
            movingAvgPredDevSum -= lastPredictionDeviations.front();
            lastPredictionDeviations.erase(lastPredictionDeviations.begin());
        }
        movingAvgPredDevSum += lastPredictionDeviations.back();
        predDevSum += predictionDeviation;
        averagePredictionDeviation = predDevSum / numPredictions;
        movingAvgPredictionDeviation = numPredictions < movingAvgScale ? averagePredictionDeviation : movingAvgPredDevSum / movingAvgScale;

        successedPredictions = movingAvgPredictionDeviation < failsafeDeviation ? successedPredictions + 1 : 0;
        workState = successedPredictions > failsafeDeviationThreshold;
    }

    float Predictor::getMovingAverageDeviation()
    {
        return movingAvgPredictionDeviation;
    }

    float Predictor::getAverageDeviation()
    {
        return averagePredictionDeviation;
    }

    float Predictor::getLastDeviation()
    {
        return predictionDeviation;
    }

    bool Predictor::isWorkState()
    {
        return workState;
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