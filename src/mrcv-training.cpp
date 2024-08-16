#include <iostream>
#include <mrcv/mrcv.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
// TODO: include <utils>

namespace mrcv
{
    int width = 416;
    int height = 416;
    std::vector<std::string> name_list;
    torch::Device device = torch::Device(torch::kCPU);
    YoloBody_tiny detector{nullptr};

    void loadPretrained(std::string pretrained_pth)
    {
        auto net_pretrained = YoloBody_tiny(3, 80);
        torch::load(net_pretrained, pretrained_pth);
        if (this->name_list.size() == 80)
        {
            detector = net_pretrained;
        }

        torch::OrderedDict<std::string, at::Tensor> pretrained_dict =
            net_pretrained->named_parameters();
        torch::OrderedDict<std::string, at::Tensor> model_dict =
            detector->named_parameters();

        for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
        {
            if (strstr((*n).key().c_str(), "yolo_head"))
            {
                continue;
            }
            model_dict[(*n).key()] = (*n).value();
        }

        torch::autograd::GradMode::set_enabled(false);
        auto new_params = model_dict;
        auto params = detector->named_parameters(true);
        auto buffers = detector->named_buffers(true);
        for (auto &val : new_params)
        {
            auto name = val.key();
            auto *t = params.find(name);
            if (t != nullptr)
            {
                t->copy_(val.value());
            }
            else
            {
                t = buffers.find(name);
                if (t != nullptr)
                {
                    t->copy_(val.value());
                }
            }
        }
        torch::autograd::GradMode::set_enabled(true);
    }

    inline bool does_exist(const std::string &name)
    {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

    void trainYOLO(std::string train_val_path, std::string image_type,
                  int num_epochs, int batch_size, float learning_rate,
                  std::string save_path, std::string pretrained_path)
    {
        if (!does_exist(pretrained_path))
        {
            std::cout << "Pretrained path is invalid: " << pretrained_path
                      << "\t random initialzed the model" << std::endl;
        }
        else
        {
            loadPretrained(pretrained_path);
        }

        std::string train_label_path = train_val_path + "/train/labels";
        std::string val_label_path = train_val_path + "/val/labels";

        std::vector<std::string> list_images_train = {};
        std::vector<std::string> list_labels_train = {};
        std::vector<std::string> list_images_val = {};
        std::vector<std::string> list_labels_val = {};

        load_det_data_from_folder(train_label_path, image_type, list_images_train,
                                  list_labels_train);
        load_det_data_from_folder(val_label_path, image_type, list_images_val,
                                  list_labels_val);

        if (list_images_train.size() < batch_size ||
            list_images_val.size() < batch_size)
        {
            std::cout << "Image numbers less than batch size or empty image folder"
                      << std::endl;
            return;
        }
        if (!does_exist(list_images_train[0]))
        {
            std::cout << "Image path is invalid get first train image "
                      << list_images_train[0] << std::endl;
            return;
        }
        auto custom_dataset_train = DetDataset(list_images_train, list_labels_train,
                                              name_list, true, width, height);
        auto custom_dataset_val = DetDataset(list_images_val, list_labels_val,
                                            name_list, false, width, height);
        auto data_loader_train =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(custom_dataset_train), batch_size);
        auto data_loader_val =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                std::move(custom_dataset_val), batch_size);

        float anchor[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
        auto anchors_ =
            torch::from_blob(anchor, {6, 2}, torch::TensorOptions(torch::kFloat32))
                .to(device);
        int image_size[2] = {width, height};

        bool normalize = false;
        auto critia1 = YOLOLossImpl(anchors_, name_list.size(), image_size, 0.01,
                                    device, normalize);
        auto critia2 = YOLOLossImpl(anchors_, name_list.size(), image_size, 0.01,
                                    device, normalize);

        auto pretrained_dict = detector->named_parameters();
        auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();
        for (int epoc_count = 0; epoc_count < num_epochs; epoc_count++)
        {
            float loss_sum = 0;
            int batch_count = 0;
            float loss_train = 0;
            float loss_val = 0;
            float best_loss = 1e10;

            if (epoc_count == int(num_epochs / 2))
            {
                learning_rate /= 10;
            }
            torch::optim::Adam optimizer(detector->parameters(), learning_rate);
            if (epoc_count < int(num_epochs / 10))
            {
                for (auto mm : pretrained_dict)
                {
                    if (strstr(mm.key().c_str(), "yolo_head"))
                    {
                        mm.value().set_requires_grad(true);
                    }
                    else
                    {
                        mm.value().set_requires_grad(false);
                    }
                }
            }
            else
            {
                for (auto mm : pretrained_dict)
                {
                    mm.value().set_requires_grad(true);
                }
            }
            detector->train();
            for (auto &batch : *data_loader_train)
            {
                std::vector<torch::Tensor> images_vec = {};
                std::vector<torch::Tensor> targets_vec = {};
                if (batch.size() < batch_size)
                    continue;
                for (int i = 0; i < batch_size; i++)
                {
                    images_vec.push_back(batch[i].data.to(FloatType));
                    targets_vec.push_back(batch[i].target.to(FloatType));
                }
                auto data = torch::stack(images_vec).div(255.0);

                optimizer.zero_grad();
                auto outputs = detector->forward(data);
                std::vector<torch::Tensor> loss_numpos1 =
                    critia1.forward(outputs[0], targets_vec);
                std::vector<torch::Tensor> loss_numpos2 =
                    critia1.forward(outputs[1], targets_vec);

                auto loss = loss_numpos1[0] + loss_numpos2[0];
                auto num_pos = loss_numpos1[1] + loss_numpos2[1];
                loss = loss / num_pos;
                loss.backward();
                optimizer.step();
                loss_sum += loss.item().toFloat();
                batch_count++;
                loss_train = loss_sum / batch_count;

                std::cout << "Epoch: " << epoc_count << ","
                          << " Training Loss: " << loss_train << "\r";
            }
            std::cout << std::endl;
            detector->eval();
            loss_sum = 0;
            batch_count = 0;
            for (auto &batch : *data_loader_val)
            {
                std::vector<torch::Tensor> images_vec = {};
                std::vector<torch::Tensor> targets_vec = {};
                if (batch.size() < batch_size)
                    continue;
                for (int i = 0; i < batch_size; i++)
                {
                    images_vec.push_back(batch[i].data.to(FloatType));
                    targets_vec.push_back(batch[i].target.to(FloatType));
                }
                auto data = torch::stack(images_vec).div(255.0);

                auto outputs = detector->forward(data);
                std::vector<torch::Tensor> loss_numpos1 =
                    critia1.forward(outputs[1], targets_vec);
                std::vector<torch::Tensor> loss_numpos2 =
                    critia1.forward(outputs[0], targets_vec);
                auto loss = loss_numpos1[0] + loss_numpos2[0];
                auto num_pos = loss_numpos1[1] + loss_numpos2[1];
                loss = loss / num_pos;

                loss_sum += loss.item<float>();
                batch_count++;
                loss_val = loss_sum / batch_count;

                std::cout << "Epoch: " << epoc_count << ","
                          << " Valid Loss: " << loss_val << "\r";
            }
            printf("\n");
            if (best_loss >= loss_val)
            {
                best_loss = loss_val;
                torch::save(detector, save_path);
            }
        }
    }

    void loadWeight(std::string weight_path)
    {
        try
        {
            torch::load(detector, weight_path);
        }
        catch (const std::exception &e)
        {
            std::cout << e.what();
        }
        detector->to(device);
        detector->eval();
        return;
    }
}
