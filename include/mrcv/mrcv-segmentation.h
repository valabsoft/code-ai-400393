#pragma once

#include <filesystem>

#include <torch/script.h>
#include <torch/torch.h>

#include "mrcv-json.hpp"

#ifdef _WIN32
inline char file_sepator() {
	return '\\';
}
#else
inline char file_sepator() {
	return '/';
}
#endif

namespace mrcv
{
	class Backbone : public torch::nn::Module
	{
	public:
		virtual std::vector<torch::Tensor> features(torch::Tensor x, int encoderDepth = 5) = 0;
		virtual torch::Tensor features_at(torch::Tensor x, int stage_num) = 0;
		virtual void load_pretrained(std::string pretrainedPath) = 0;
		virtual void make_dilated(std::vector<int> listStage, std::vector<int> listDilation) = 0;
		virtual ~Backbone() {}
	};

	inline torch::nn::Conv2dOptions conv_options(int64_t inPanes, int64_t outPanes, int64_t kernerSize,
		int64_t stride = 1, int64_t padding = 0, int groups = 1, bool with_bias = true, int dilation = 1) {
		torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(inPanes, outPanes, kernerSize);
		conv_options.stride(stride);
		conv_options.padding(padding);
		conv_options.bias(with_bias);
		conv_options.groups(groups);
		conv_options.dilation(dilation);
		return conv_options;
	}

	inline torch::nn::UpsampleOptions optionsUpsample(std::vector<double> scaleSize, bool alignCorners = true) {
		torch::nn::UpsampleOptions optionsUpsample = torch::nn::UpsampleOptions();
		optionsUpsample.scale_factor(scaleSize);
		optionsUpsample.mode(torch::kBilinear).align_corners(alignCorners);
		return optionsUpsample;
	}

	inline torch::nn::Dropout2dOptions dropoutoptions(float p, bool inplace) {
		torch::nn::Dropout2dOptions dropoutoptions(p);
		dropoutoptions.inplace(inplace);
		return dropoutoptions;
	}

	inline torch::nn::MaxPool2dOptions optionsMaxPool(int kernelSize, int stride) {
		torch::nn::MaxPool2dOptions optionsMaxPool(kernelSize);
		optionsMaxPool.stride(stride);
		return optionsMaxPool;
	}

	class SegmentationHeadImpl : public torch::nn::Module {
	public:
		SegmentationHeadImpl(int inChannels, int outChannels, int kernelSize = 3, double upsampling = 1);
		torch::Tensor forward(torch::Tensor x);
	private:
		torch::nn::Conv2d conv2d{ nullptr };
		torch::nn::Upsample upsampling{ nullptr };
	}; TORCH_MODULE(SegmentationHead);

	std::string replaceAllExtension(std::string str, const std::string oldValue, const std::string newValue);

	void loadDataFromFolder(std::string folder, std::string imageType, std::vector<std::string>& listImages, std::vector<std::string>& listLabels);

	nlohmann::json encoderParameters();

	///////////////////////////////////////////////////////////////////////////    

	class Conv3x3GNReLUImpl : public torch::nn::Module
	{
	public:
		Conv3x3GNReLUImpl(int inChannels, int outChannels, bool upsample = false);
		torch::Tensor forward(torch::Tensor x);
	private:
		bool upsample;
		torch::nn::Sequential block{ nullptr };
	};
	TORCH_MODULE(Conv3x3GNReLU);

	class FPNBlockImpl : public torch::nn::Module
	{
	public:
		FPNBlockImpl(int channelsPyramid, int channelsSkip);
		torch::Tensor forward(torch::Tensor x, torch::Tensor skip);
	private:
		torch::nn::Conv2d skip_conv{ nullptr };
		torch::nn::Upsample upsample{ nullptr };
	};
	TORCH_MODULE(FPNBlock);

	class SegmentationBlockImpl : public torch::nn::Module
	{
	public:
		SegmentationBlockImpl(int inChannels, int outChannels, int upSamplesNam = 0);
		torch::Tensor forward(torch::Tensor x);
	private:
		torch::nn::Sequential block{ nullptr };
	}; TORCH_MODULE(SegmentationBlock);

	class MergeBlockImpl : public torch::nn::Module {
	public:
		MergeBlockImpl(std::string policy);
		torch::Tensor forward(std::vector<torch::Tensor> x);
	private:
		std::string _policy;
		std::string policies[2] = { "add","cat" };
	}; TORCH_MODULE(MergeBlock);

	class FPNDecoderImpl : public torch::nn::Module
	{
	public:
		FPNDecoderImpl(std::vector<int> channelsEncoder = { 3, 64, 64, 128, 256, 512 }, int encoderDepth = 5, int channelsPyramid = 256,
			int channelsSegmentation = 128, float dropout = 0.2, std::string merge_policy = "add");
		torch::Tensor forward(std::vector<torch::Tensor> features);
	private:
		int outChannels;
		torch::nn::Conv2d p5{ nullptr };
		FPNBlock p4{ nullptr };
		FPNBlock p3{ nullptr };
		FPNBlock p2{ nullptr };
		torch::nn::ModuleList seg_blocks{};
		MergeBlock merge{ nullptr };
		torch::nn::Dropout2d dropout{ nullptr };

	}; TORCH_MODULE(FPNDecoder);

	///////////////////////////////////////////////////////////////////////////

	class FPNImpl : public torch::nn::Module
	{
	public:
		FPNImpl() {}
		~FPNImpl() {
		}
		FPNImpl(int numberClasses, std::string encoderName = "resnet18", std::string pretrainedPath = "", int encoderDepth = 5,
			int decoderChannelPyramid = 256, int decoderChannelsSegmentation = 128, std::string decoderMergePolicy = "add",
			float decoder_dropout = 0.2, double upsampling = 4);
		torch::Tensor forward(torch::Tensor x);
	private:
		Backbone* encoder;
		FPNDecoder decoder{ nullptr };
		SegmentationHead segmentation_head{ nullptr };
		int numberClasses = 1;
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
		torch::nn::Conv2d conv1{ nullptr };
		torch::nn::BatchNorm2d bn1{ nullptr };
		torch::nn::Conv2d conv2{ nullptr };
		torch::nn::BatchNorm2d bn2{ nullptr };
		torch::nn::Conv2d conv3{ nullptr };
		torch::nn::BatchNorm2d bn3{ nullptr };
	};
	TORCH_MODULE(Block);

	class ResNetImpl : public Backbone {
	public:
		ResNetImpl(std::vector<int> layers, int numberClasses = 1000, std::string typeModel = "resnet18",
			int groups = 1, int widthGroupPer = 64);
		torch::Tensor forward(torch::Tensor x);
		torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1);
		std::vector<torch::nn::Sequential> get_stages();

		std::vector<torch::Tensor> features(torch::Tensor x, int encoderDepth = 5) override;
		torch::Tensor features_at(torch::Tensor x, int stage_num) override;
		void make_dilated(std::vector<int> listStage, std::vector<int> listDilation) override;
		void load_pretrained(std::string pretrainedPath) override;
	private:
		std::string typeModel = "resnet18";
		int expansion = 1; bool is_basic = true;
		int64_t inplanes = 64; int groups = 1; int base_width = 64;
		torch::nn::Conv2d conv1{ nullptr };
		torch::nn::BatchNorm2d bn1{ nullptr };
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

		return name2layers;
	}

	ResNet resnet18(int64_t numberClasses);
	ResNet resnet34(int64_t numberClasses);
	ResNet resnet50(int64_t numberClasses);
	ResNet resnet101(int64_t numberClasses);

	ResNet pretrained_resnet(int64_t numberClasses, std::string model_name, std::string pathWeight);
}
