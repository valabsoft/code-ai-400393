#include <mrcv/mrcv.h>
#include <mrcv/mrcv-common.h>
#include <mrcv/mrcv-vae.h>

namespace mrcv
{
    torch::Tensor neuralNetworkAugmentationAsTensor(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate) {
        // �����������
        std::ostringstream logStream;
        // ���������� ������ ������������� �����������
        const int64_t numColor = 1;
        // ������ ����������� ������ ����
        const int64_t inputDim = numColor * height * width;
        // ����� ����������
        torch::Device device(torch::kCUDA);
        // �������� ��������
        auto dataset = CustomDataset(root, numColor).map(torch::data::transforms::Stack<>());

        // �������� ��������
        if (dataset.size() == 0) {
            logStream << "Dataset is empty!" << std::endl;
            //return ;
        }

        auto dataLoader = torch::data::make_data_loader(dataset, torch::data::DataLoaderOptions().batch_size(batchSize).workers(2));

        // �������� ������� ������
        VariationalAutoEncoder model(inputDim, hDim, zDim);
        // ������� ������ �� ����������
        model.to(device);
        // ����������� ������
        torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(lrRate));
        logStream << "Dataset size before training: " << dataset.size().value() << std::endl;

        // �������� ������
        for (int epoch = 0; epoch < numEpoch; ++epoch) {
            model.train();
            for (auto& batch : *dataLoader) {
                auto data = batch.data.to(device).view({ -1, inputDim });
                logStream << "Data size: " << data.sizes() << std::endl;

                if (torch::isnan(data).any().item<bool>() || torch::isinf(data).any().item<bool>()) {
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

        // ������������ ������������ �����������
        int numExamples = 1;

        std::vector<torch::Tensor> images;

        for (size_t i = 0; i < dataset.size(); ++i) {
            auto example = dataset.get_batch(i);
            images.push_back(example.data.to(device));
            if (images.size() == numExamples) break;
        }

        if (images.empty()) {
            logStream << "No images found for digit" << std::endl;
            //return ;
        }

        std::vector<std::pair<torch::Tensor, torch::Tensor>> encodingsDigit;

        // �������
        for (int d = 0; d < numExamples; ++d) {
            torch::NoGradGuard no_grad;
            logStream << "Image tensor size: " << images[d].sizes() << std::endl;

            auto flattenedIimage = images[d].view({ 1, inputDim });
            logStream << "Flattened image tensor size: " << flattenedIimage.sizes() << std::endl;

            auto [mu, sigma] = model.encode(flattenedIimage.to(device));
            logStream << "Mu size: " << mu.sizes() << std::endl;
            logStream << "Sigma size: " << sigma.sizes() << std::endl;
            encodingsDigit.push_back({ mu, sigma });

        }

        // �������
        for (int example = 0; example < numExamples; ++example) {
            auto [mu, sigma] = encodingsDigit[0];
            auto sample = model.reparameterize(mu, sigma);
            torch::Tensor tensor = model.decode(sample.to(device));
            tensor = torch::sigmoid(tensor).view({ numColor, width, height });
            return tensor.clone();
        }
        //���������� �����
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);
    }

    cv::Mat neuralNetworkAugmentationAsMat(const std::string& root, const int64_t height, const int64_t width, const int64_t hDim, const int64_t zDim, const int64_t numEpoch, const int64_t batchSize, const double lrRate) {
        // �����������
        std::ostringstream logStream;
        // ���������� ������ ������������� �����������
        const int64_t numColor = 1;
        // 
        torch::Tensor tensor = neuralNetworkAugmentationAsTensor(root, height, width, hDim, zDim, numEpoch, batchSize, lrRate);
        // ��������, ��� ������ �� ������
        if (!tensor.defined()) {
            logStream << "Error: Tensor is undefined." << std::endl;
        }
        // �������������� ������� � OpenCV �at
        tensor = tensor.mul(255).clamp(0, 255).to(torch::kU8);
        tensor = tensor.to(torch::kCPU);
        tensor = tensor.view({ numColor, height, width });

        // ������������ ������ � {numColor, height, width} �� {height, width, numColor} ��� OpenCV
        tensor = tensor.permute({ 1, 2, 0 });

        // ������� OpenCV Mat
        cv::Mat image(tensor.size(0), tensor.size(1), CV_8UC1, tensor.data_ptr());
        // ��������, ��� ����������� �� ������
        if (image.empty()) {
            logStream << "Error: Image is empty." << std::endl;
        }
        //���������� �����
        writeLog(logStream.str(), mrcv::LOGTYPE::INFO);

        return image.clone();
    }

    
    int semiAutomaticLabeler()
    {
        return 0;
    }

}