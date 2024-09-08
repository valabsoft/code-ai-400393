#pragma once

#include <filesystem>

#include <torch/script.h>
#include <torch/torch.h>

#include "mrcv-json.hpp"

#ifdef _WIN32
inline char fileSepator() {
    return '\\';
}
#else
inline char fileSepator() {
    return '/';
}
#endif

namespace mrcv
{
    class Backbone : public torch::nn::Module
    {
    public:
        virtual std::vector<torch::Tensor> features(torch::Tensor x, int encoder_depth = 5) = 0;
        virtual torch::Tensor features_at(torch::Tensor x, int stage_num) = 0;
        virtual void load_pretrained(std::string pretrained_path) = 0;
        virtual void make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list) = 0;
        virtual ~Backbone() {}
    };

    inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
        int64_t stride = 1, int64_t padding = 0, int groups = 1, bool with_bias = true, int dilation = 1) {
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
        conv_options.stride(stride);
        conv_options.padding(padding);
        conv_options.bias(with_bias);
        conv_options.groups(groups);
        conv_options.dilation(dilation);
        return conv_options;
    }

    inline torch::nn::UpsampleOptions upsample_options(std::vector<double> scale_size, bool align_corners = true) {
        torch::nn::UpsampleOptions upsample_options = torch::nn::UpsampleOptions();
        upsample_options.scale_factor(scale_size);
        upsample_options.mode(torch::kBilinear).align_corners(align_corners);
        return upsample_options;
    }

    inline torch::nn::Dropout2dOptions dropout_options(float p, bool inplace) {
        torch::nn::Dropout2dOptions dropoutoptions(p);
        dropoutoptions.inplace(inplace);
        return dropoutoptions;
    }

    inline torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride) {
        torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
        maxpool_options.stride(stride);
        return maxpool_options;
    }

    class SegmentationHeadImpl : public torch::nn::Module {
    public:
        SegmentationHeadImpl(int in_channels, int outgoingChannel, int kernel_size = 3, double upsampling = 1);
        torch::Tensor forward(torch::Tensor x);
    private:
        torch::nn::Conv2d convolution2d { nullptr };
        torch::nn::Upsample upsampl { nullptr };
    }; TORCH_MODULE(SegmentationHead);

    std::string replace_all_distinct2(std::string str, const std::string old_value, const std::string new_value);

    void loadDataFromFolder(std::string folder, std::string image_type, std::vector<std::string>& list_images, std::vector<std::string>& list_labels);

    nlohmann::json encoder_params();

    ///////////////////////////////////////////////////////////////////////////    

    class ReLUConv3x3GNImpl : public torch::nn::Module
    {
    public:
        ReLUConv3x3GNImpl(int in_channels, int outgoingChannel, bool upsample = false);
        torch::Tensor forward(torch::Tensor x);
    private:
        bool upsample;
        torch::nn::Sequential block{ nullptr };
    };
    TORCH_MODULE(ReLUConv3x3GN);

    class BlockFPNImpl : public torch::nn::Module
    {
    public:
        BlockFPNImpl(int pyramidChannels, int skipChannels);
        torch::Tensor forward(torch::Tensor x, torch::Tensor skip);
    private:
        torch::nn::Upsample upsample{ nullptr };
        torch::nn::Conv2d skipConvolution{ nullptr };

    };
    TORCH_MODULE(BlockFPN);

    class BlockSegmentationImpl : public torch::nn::Module
    {
    public:
        BlockSegmentationImpl(int in_channels, int outgoingChannel, int n_upsamples = 0);
        torch::Tensor forward(torch::Tensor x);
    private:
        torch::nn::Sequential block{ nullptr };
    }; TORCH_MODULE(BlockSegmentation);

    class BlockMergeImpl : public torch::nn::Module {
    public:
        BlockMergeImpl(std::string policy);
        torch::Tensor forward(std::vector<torch::Tensor> x);
    private:
        std::string _policy;
        std::string policies[2] = { "add","cat" };
    }; TORCH_MODULE(BlockMerge);

    class FPNDecoderImpl : public torch::nn::Module
    {
    public:
        FPNDecoderImpl(std::vector<int> encoderChannels = { 3, 64, 64, 128, 256, 512 }, int depthEncoder = 5, int pyramidChannels = 256,
            int segmentation_channels = 128, float dropout = 0.2, std::string merge_policy = "add");
        torch::Tensor forward(std::vector<torch::Tensor> features);
    private:
        int outgoingChannel;
        torch::nn::Conv2d p5{ nullptr };
        BlockFPN p4{ nullptr };
        BlockFPN p3{ nullptr };
        BlockFPN p2{ nullptr };
        torch::nn::ModuleList seg_blocks{};
        BlockMerge merge{ nullptr };
        torch::nn::Dropout2d dropout{ nullptr };

    }; TORCH_MODULE(FPNDecoder);

    ///////////////////////////////////////////////////////////////////////////

    class FPNImpl : public torch::nn::Module
    {
    public:
        FPNImpl() {}
        ~FPNImpl() {
            //delete encoder;
        }
        FPNImpl(int numClasses, std::string encoder_name = "resnet18", std::string pretrained_path = "", int encoder_depth = 5,
            int decoder_pyramid_channel = 256, int decoder_segmentation_channels = 128, std::string decoder_merge_policy = "add",
            float decoder_dropout = 0.2, double upsampling = 4);
        torch::Tensor forward(torch::Tensor x);
    private:
        Backbone* encoder;
        FPNDecoder decoder{ nullptr };
        SegmentationHead segmentation_head{ nullptr };
        int numClasses = 1;
    }; TORCH_MODULE(FPN);

    ///////////////////////////////////////////////////////////////////////////

    class BlockImpl : public torch::nn::Module {
    public:
        BlockImpl(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
            torch::nn::Sequential downsample_ = nullptr, int groups = 1, int base_width = 64, bool is_basic = true);
        torch::Tensor forward(torch::Tensor x);
        torch::nn::Sequential downsample{ nullptr };
    private:
        bool is_basic = true;
        int64_t stride = 1;
        torch::nn::Conv2d convolution1{ nullptr };
        torch::nn::BatchNorm2d BatchNorm1{ nullptr };
        torch::nn::Conv2d convolution2{ nullptr };
        torch::nn::BatchNorm2d BatchNorm2{ nullptr };
        torch::nn::Conv2d convolution3{ nullptr };
        torch::nn::BatchNorm2d BatchNorm3{ nullptr };
    };
    TORCH_MODULE(Block);

    class ResNetImpl : public Backbone {
    public:
        ResNetImpl(std::vector<int> layers, int numClasses = 1000, std::string model_type = "resnet18",
            int groups = 1, int width_per_group = 64);
        torch::Tensor forward(torch::Tensor x);
        torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1);
        std::vector<torch::nn::Sequential> get_stages();

        std::vector<torch::Tensor> features(torch::Tensor x, int encoder_depth = 5) override;
        torch::Tensor features_at(torch::Tensor x, int stage_num) override;
        void make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list) override;
        void load_pretrained(std::string pretrained_path) override;
    private:
        std::string model_type = "resnet18";
        int expansion = 1; bool is_basic = true;
        int64_t inplanes = 64; int groups = 1; int base_width = 64;
        torch::nn::Conv2d convolution1{ nullptr };
        torch::nn::BatchNorm2d BatchNorm1{ nullptr };
        torch::nn::Sequential layer1{ nullptr };
        torch::nn::Sequential layer2{ nullptr };
        torch::nn::Sequential layer3{ nullptr };
        torch::nn::Sequential layer4{ nullptr };
        torch::nn::Linear fc{ nullptr };
    };
    TORCH_MODULE(ResNet);

    inline std::map<std::string, std::vector<int>> getParams() {
        std::map<std::string, std::vector<int>> name2layers = {};
        name2layers.insert(std::pair<std::string, std::vector<int>>("resnet18", { 2, 2, 2, 2 }));
        name2layers.insert(std::pair<std::string, std::vector<int>>("resnet34", { 3, 4, 6, 3 }));
        name2layers.insert(std::pair<std::string, std::vector<int>>("resnet50", { 3, 4, 6, 3 }));
        name2layers.insert(std::pair<std::string, std::vector<int>>("resnet101", { 3, 4, 23, 3 }));
        name2layers.insert(std::pair<std::string, std::vector<int>>("resnet152", { 3, 8, 36, 3 }));
        name2layers.insert(std::pair<std::string, std::vector<int>>("resnext50_32x4d", { 3, 4, 6, 3 }));
        name2layers.insert(std::pair<std::string, std::vector<int>>("resnext101_32x8d", { 3, 4, 23, 3 }));

        return name2layers;
    }

    ResNet resnet18(int64_t numClasses);
    ResNet resnet34(int64_t numClasses);
    ResNet resnet50(int64_t numClasses);
    ResNet resnet101(int64_t numClasses);

    ResNet pretrained_resnet(int64_t numClasses, std::string model_name, std::string weight_path);
}
