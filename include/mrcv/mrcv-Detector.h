#ifndef DETECTOR_H
#define DETECTOR_H
#pragma once

#include"mrcv/mrcv-common.h"
#include"mrcv/mrcv-tinyxml.h"

namespace mrcv
{
	struct DetectionConfig {
		unsigned int freeze_epochs = 0;
		std::vector<unsigned int> decay_epochs = { 0 };
		float horizontal_flip_prob = 0;
		float vertical_flip_prob = 0;
		float scale_rotate_prob = 0;
		float scale_limit = 0.1f;
		float rotate_limit = 45.0f;
		int interpolation = cv::INTER_LINEAR;
		int border_mode = cv::BORDER_CONSTANT;
	};

	struct HyperParams {
		int num_epochs;
		int batch_size;
		float learning_rate;
	};

	struct  BBox
	{
		int xmin = 0;
		int xmax = 0;
		int ymin = 0;
		int ymax = 0;
		std::string name = "";
		int GetH();
		int GetW();
		float CenterX();
		float CenterY();
	};

	struct Data {
		Data(cv::Mat img, std::vector<BBox> boxes) :image(img), bboxes(boxes) {};
		cv::Mat image;
		std::vector<BBox> bboxes;
	};
	
	class Augmentations
	{
	public:
		static Data Resize(Data mData, int width, int height, float probability);
	};

	std::vector<BBox> loadXML(std::string xml_path);

	class BasicConvImpl : public torch::nn::Module {
	public:
		BasicConvImpl(int in_channels, int out_channels, int kernel_size, int stride = 1);
		torch::Tensor forward(torch::Tensor x);
	private:
		torch::nn::Conv2d conv{ nullptr };
		torch::nn::BatchNorm2d bn{ nullptr };
		torch::nn::LeakyReLU activation{ nullptr };
	}; TORCH_MODULE(BasicConv);

	class Resblock_bodyImpl : public torch::nn::Module {
	public:
		Resblock_bodyImpl(int in_channels, int out_channels);
		std::vector<torch::Tensor> forward(torch::Tensor x);
	private:
		int outChannels;
		BasicConv conv1{ nullptr };
		BasicConv conv2{ nullptr };
		BasicConv conv3{ nullptr };
		BasicConv conv4{ nullptr };
		torch::nn::MaxPool2d maxpool{ nullptr };
	}; TORCH_MODULE(Resblock_body);

	class CSPdarknet53_tinyImpl : public torch::nn::Module
	{
	public:
		CSPdarknet53_tinyImpl();
		std::vector<torch::Tensor> forward(torch::Tensor x);
	private:
		BasicConv conv1{ nullptr };
		BasicConv conv2{ nullptr };
		Resblock_body resblock_body1{ nullptr };
		Resblock_body resblock_body2{ nullptr };
		Resblock_body resblock_body3{ nullptr };
		BasicConv conv3{ nullptr };
		int numFeatures = 1;
	}; TORCH_MODULE(CSPdarknet53_tiny);

	class UpsampleImpl : public torch::nn::Module {
	public:
		UpsampleImpl(int in_channels, int out_channels);
		torch::Tensor forward(torch::Tensor x);
	private:
		// Declare layers
		torch::nn::Sequential upsample = torch::nn::Sequential();
	}; TORCH_MODULE(Upsample);

	torch::nn::Sequential yolo_head(std::vector<int> filters_list, int in_filters);

	class YoloBody_tinyImpl : public torch::nn::Module {
	public:
		YoloBody_tinyImpl(int num_anchors, int num_classes);
		std::vector<torch::Tensor> forward(torch::Tensor x);
	private:
		CSPdarknet53_tiny backbone{ nullptr };
		BasicConv conv_for_P5{ nullptr };
		Upsample upsample{ nullptr };
		torch::nn::Sequential yolo_headP5{ nullptr };
		torch::nn::Sequential yolo_headP4{ nullptr };
	}; TORCH_MODULE(YoloBody_tiny);

	struct YOLOLossImpl : public torch::nn::Module {
		YOLOLossImpl(torch::Tensor anchors_, int num_classes, int img_size[], float label_smooth = 0,
			torch::Device device = torch::Device(torch::kCPU), bool normalize = true);
		std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<torch::Tensor> targets);
		std::vector<torch::Tensor> get_target(std::vector<torch::Tensor> targets, torch::Tensor scaled_anchors, int in_w, int in_h, float ignore_threshold);
		std::vector<torch::Tensor> get_ignore(torch::Tensor prediction, std::vector<torch::Tensor> targets, torch::Tensor scaled_anchors, int in_w, int in_h, torch::Tensor noobj_mask);
		torch::Tensor anchors;
		int num_anchors = 3;
		int num_classes = 1;
		int bbox_attrs = 0;
		int image_size[2] = { 416,416 };
		float label_smooth = 0;
		std::vector<int> feature_length = { int(image_size[0] / 32),int(image_size[0] / 16),int(image_size[0] / 8) };

		float ignore_threshold = 0.5;
		float lambda_conf = 1.0;
		float lambda_cls = 1.0;
		float lambda_loc = 1.0;
		torch::Device device = torch::Device(torch::kCPU);
		bool normalize = true;
	}; TORCH_MODULE(YOLOLoss);


	class DetDataset :public torch::data::Dataset<DetDataset> {
	private:
		// Declare 2 std::vectors of tensors for images and labels
		std::vector<std::string> list_images;
		std::vector<std::string> list_labels;
		bool isTrain = true;
		int width = 416; 
		int height = 416;
		float hflipProb = 0; 
		float vflipProb = 0; 
		float noiseProb = 0; 
		float brightProb = 0;
		float noiseMuLimit = 1; 
		float noiseSigmaLimit = 1; 
		float brightContrastLimit = 0.2; 
		float brightnessLimit = 0;
		std::map<std::string, float> name_idx = {};
	public:
		// Constructor
		DetDataset(std::vector<std::string> images, std::vector<std::string> labels, std::vector<std::string> class_names, bool istrain = true,
			int width_ = 416, int height_ = 416, float hflip_prob = 0.5, float vflip_prob = 0)
		{
			list_images = images; list_labels = labels; isTrain = istrain; width = width_; height = height_;
			hflipProb = hflip_prob; vflipProb = vflip_prob;
			for (int i = 0; i < class_names.size(); i++)
			{
				name_idx.insert(std::pair<std::string, float>(class_names[i], float(i)));
			}
		};

		// Override get() function to return tensor at location index
		torch::data::Example<> get(size_t index) override {
			std::string image_path = list_images.at(index);
			std::string annotation_path = list_labels.at(index);

			cv::Mat img = cv::imread(image_path, 1);
			std::vector<BBox> boxes = loadXML(annotation_path);
			Data m_data(img, boxes);

			m_data = Augmentations::Resize(m_data, width, height, 1);
			//Augmentations can be implemented here...

			float width_under1 = 1.0 / m_data.image.cols;//box的长宽归一化用
			float height_under1 = 1.0 / m_data.image.rows;//box的长宽归一化用
			torch::Tensor img_tensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width

			int box_num = m_data.bboxes.size();
			if (m_data.bboxes.size() == 0)
			{
				torch::Tensor label_tensor = torch::ones({ 1 });
				std::cout << annotation_path << std::endl;
				return { img_tensor.clone(), label_tensor.clone() };
			}
			torch::Tensor label_tensor = torch::zeros({ box_num ,5 }).to(torch::kFloat32);
			for (int i = 0; i < box_num; i++)
			{
				label_tensor[i][2] = m_data.bboxes[i].GetW() * width_under1;
				label_tensor[i][3] = m_data.bboxes[i].GetH() * height_under1;
				label_tensor[i][0] = m_data.bboxes[i].xmin * width_under1 + label_tensor[i][2] / 2;
				label_tensor[i][1] = m_data.bboxes[i].ymin * height_under1 + label_tensor[i][3] / 2;
				label_tensor[i][4] = name_idx.at(m_data.bboxes[i].name);
			}
			return { img_tensor.clone(), label_tensor.clone() };
		};

		std::vector<ExampleType> get_batch(c10::ArrayRef<size_t> indices) override {
			std::vector<ExampleType> batch;
			batch.reserve(indices.size());
			for (const auto i : indices) {
				batch.push_back(get(i));
			}
			return batch;
		}

		// Return the length of data
		torch::optional<size_t> size() const override {
			return list_labels.size();
		};
	};

}
#endif // DETECTOR_H