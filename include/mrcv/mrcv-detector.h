#ifndef DETECTOR_H
#define DETECTOR_H
#pragma once

#include"mrcv/mrcv-common.h"
#include"mrcv/tinyxml.h"

namespace mrcv
{
	class BBox
	{
	public:
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

	class DetectorData {
	public:
		cv::Mat image;
		std::vector<BBox> bboxes;
		DetectorData(cv::Mat img, std::vector<BBox> boxes) :image(img), bboxes(boxes) {};
	};
	
	class DetAugmentations
	{
	public:
		static DetectorData Resize(DetectorData mData, int width, int height, float probability);
	};

	std::vector<BBox> loadXML(std::string xmlPath);

	class BasicConvImpl : public torch::nn::Module {
	public:
		BasicConvImpl(int inChannels, int outChannels, int kernelSize, int stride = 1);
		torch::Tensor forward(torch::Tensor x);
	private:
		torch::nn::Conv2d conv{ nullptr };
		torch::nn::BatchNorm2d bn{ nullptr };
		torch::nn::LeakyReLU activation{ nullptr };
	}; TORCH_MODULE(BasicConv);

	class Resblock_bodyImpl : public torch::nn::Module {
	public:
		Resblock_bodyImpl(int inChannels, int outChannels);
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
		Resblock_body resblockBody1{ nullptr };
		Resblock_body resblockBody2{ nullptr };
		Resblock_body resblockBody3{ nullptr };
		BasicConv conv3{ nullptr };
		int numFeatures = 1;
	}; TORCH_MODULE(CSPdarknet53_tiny);

	class UpsampleImpl : public torch::nn::Module {
	public:
		UpsampleImpl(int inChannels, int outChannels);
		torch::Tensor forward(torch::Tensor x);
	private:
		torch::nn::Sequential upsample = torch::nn::Sequential();
	}; TORCH_MODULE(Upsample);

	torch::nn::Sequential yoloHead(std::vector<int> filtersList, int inFilters);

	class YoloBody_tinyImpl : public torch::nn::Module {
	public:
		YoloBody_tinyImpl(int numAnchors, int numClasses);
		std::vector<torch::Tensor> forward(torch::Tensor x);
	private:
		CSPdarknet53_tiny backbone{ nullptr };
		BasicConv convForP5{ nullptr };
		Upsample upsample{ nullptr };
		torch::nn::Sequential yoloHeadP5{ nullptr };
		torch::nn::Sequential yoloHeadP4{ nullptr };
	}; TORCH_MODULE(YoloBody_tiny);

	struct YOLOLossImpl : public torch::nn::Module {
		YOLOLossImpl(torch::Tensor anchors_, int numClasses, int imgSize[], float labelSmooth = 0,
			torch::Device device = torch::Device(torch::kCPU), bool normalize = true);
		std::vector<torch::Tensor> forward(torch::Tensor input, std::vector<torch::Tensor> targets);
		std::vector<torch::Tensor> getTarget(std::vector<torch::Tensor> targets, torch::Tensor scaledAnchors, int inW, int inH, float ignoreThreshold);
		std::vector<torch::Tensor> getIgnore(torch::Tensor prediction, std::vector<torch::Tensor> targets, torch::Tensor scaledAnchors, int inW, int inH, torch::Tensor noobjMask);
		torch::Tensor anchors;
		int numAnchors = 3;
		int numClasses = 1;
		int bboxAttrs = 0;
		int imageSize[2] = { 416,416 };
		float labelSmooth = 0;
		std::vector<int> featureLength = { int(imageSize[0] / 32),int(imageSize[0] / 16),int(imageSize[0] / 8) };

		float ignoreThreshold = 0.5;
		float lambdaConf = 1.0;
		float lambdaCls = 1.0;
		float lambdaLoc = 1.0;
		torch::Device device = torch::Device(torch::kCPU);
		bool normalize = true;
	}; TORCH_MODULE(YOLOLoss);

	class DetDataset :public torch::data::Dataset<DetDataset> {
	private:
		std::vector<std::string> listImages;
		std::vector<std::string> listLabels;
		bool isTrain = true;
		int width = 416; 
		int height = 416;
		float hFlipProb = 0; 
		float vFlipProb = 0; 
		float noiseProb = 0; 
		float brightProb = 0;
		float noiseMuLimit = 1; 
		float noiseSigmaLimit = 1; 
		float brightContrastLimit = 0.2; 
		float brightnessLimit = 0;
		std::map<std::string, float> nameIdx = {};
	public:

		DetDataset(std::vector<std::string> images, std::vector<std::string> labels, std::vector<std::string> classNames, bool isTrain = true,
			int width_ = 416, int height_ = 416, float hFlipProb = 0.5, float vFlipProb = 0)
		{
			listImages = images; listLabels = labels; isTrain = isTrain; width = width_; height = height_;
			hFlipProb = hFlipProb; vFlipProb = vFlipProb;
			for (int i = 0; i < classNames.size(); i++)
			{
				nameIdx.insert(std::pair<std::string, float>(classNames[i], float(i)));
			}
		};
		
		torch::data::Example<> get(size_t index) override {
			std::string imagePath = listImages.at(index);
			std::string annotationPath = listLabels.at(index);

			cv::Mat img = cv::imread(imagePath, 1);
			std::vector<BBox> boxes = loadXML(annotationPath);
			DetectorData m_data(img, boxes);

			m_data = DetAugmentations::Resize(m_data, width, height, 1);

			float widthUnder1 = 1.0 / m_data.image.cols;
			float heightUnder1 = 1.0 / m_data.image.rows;
			torch::Tensor imgTensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 });

			int boxNum = m_data.bboxes.size();
			if (m_data.bboxes.size() == 0)
			{
				torch::Tensor labelTensor = torch::ones({ 1 });
				std::cout << annotationPath << std::endl;
				return { imgTensor.clone(), labelTensor.clone() };
			}
			torch::Tensor labelTensor = torch::zeros({ boxNum ,5 }).to(torch::kFloat32);
			for (int i = 0; i < boxNum; i++)
			{
				labelTensor[i][2] = m_data.bboxes[i].GetW() * widthUnder1;
				labelTensor[i][3] = m_data.bboxes[i].GetH() * heightUnder1;
				labelTensor[i][0] = m_data.bboxes[i].xmin * widthUnder1 + labelTensor[i][2] / 2;
				labelTensor[i][1] = m_data.bboxes[i].ymin * heightUnder1 + labelTensor[i][3] / 2;
				labelTensor[i][4] = nameIdx.at(m_data.bboxes[i].name);
			}
			return { imgTensor.clone(), labelTensor.clone() };
		};

		torch::optional<size_t> size() const override {
			return listLabels.size();
		};
	};
}
#endif // DETECTOR_H