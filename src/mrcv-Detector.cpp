#include <mrcv/mrcv.h>
#include <iostream>

#include<mrcv/mrcv-detector.h>


namespace mrcv
{ 
	template<typename T>
	int vecIndex(std::vector<T> vec, T value) 
	{
		int a = 0;
		for (auto temp : vec)
		{
			if (temp == value)
			{
				return a;
			}
			a++;
		}
		if (a == vec.size())
		{
			std::cout << "No such value in std::vector" << std::endl;
			return a;
		}
	}

	template<typename T>
	bool inVec(std::vector<T> vec, T value) 
	{
		for (auto temp : vec)
		{
			if (temp == value)
			{
				return true;
			}
		}
		return false;
	}

	template<typename T>
	T RandomNum(T _min, T _max)
	{
		T temp;
		if (_min > _max)
		{
			temp = _max;
			_max = _min;
			_min = temp;
		}
		return rand() / (double)RAND_MAX * (_max - _min) + _min;
	}

	torch::nn::Conv2dOptions createConvOptions(int inChannels, int outChannels, int kernelSize, int stride, int padding, int dilation, bool bias) 
	{
		return torch::nn::Conv2dOptions(inChannels, outChannels, kernelSize)
			.stride(stride)
			.padding(padding)
			.dilation(dilation)
			.bias(bias);
	}

	BasicConvImpl::BasicConvImpl(int inChannels, int outChannels, int kernelSize, int stride) :
		conv(createConvOptions(inChannels, outChannels, kernelSize, stride, int(kernelSize / 2), 1, false)),
		bn(torch::nn::BatchNorm2d(outChannels)), activation(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)))
	{
		register_module("conv", conv);
		register_module("bn", bn);
	}

	torch::Tensor BasicConvImpl::forward(torch::Tensor x)
	{
		x = conv->forward(x);
		x = bn->forward(x);
		x = activation(x);
		return x;
	}

	torch::nn::MaxPool2dOptions maxpoolOptions(int kernelSize, int stride) 
	{
		torch::nn::MaxPool2dOptions maxpool_options(kernelSize);
		maxpool_options.stride(stride);
		return maxpool_options;
	}

	Resblock_bodyImpl::Resblock_bodyImpl(int inChannels, int outChannels) 
	{
		this->outChannels = outChannels;
		conv1 = BasicConv(inChannels, outChannels, 3);
		conv2 = BasicConv(outChannels / 2, outChannels / 2, 3);
		conv3 = BasicConv(outChannels / 2, outChannels / 2, 3);
		conv4 = BasicConv(outChannels, outChannels, 1);
		maxpool = torch::nn::MaxPool2d(maxpoolOptions(2, 2));

		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);

	}

	std::vector<torch::Tensor> Resblock_bodyImpl::forward(torch::Tensor x) 
	{
		auto c = outChannels;
		x = conv1->forward(x);
		auto route = x;

		x = torch::split(x, c / 2, 1)[1];
		x = conv2->forward(x);
		auto route1 = x;

		x = conv3->forward(x);
		x = torch::cat({ x, route1 }, 1);
		x = conv4->forward(x);
		auto feat = x;

		x = torch::cat({ route, x }, 1);
		x = maxpool->forward(x);
		return std::vector<torch::Tensor>({ x,feat });
	}

	std::vector<BBox> loadXML(std::string xmlPath)
	{
		std::vector<BBox> objects;

		TiXmlDocument doc;

		if (!doc.LoadFile(xmlPath.c_str()))
		{
			std::cerr << doc.ErrorDesc() << std::endl;
			return objects;
		}

		TiXmlElement* root = doc.FirstChildElement();

		if (root == NULL)
		{
			std::cerr << "Failed to load file: No root element." << std::endl;

			doc.Clear();
			return objects;
		}

		for (TiXmlElement* elem = root->FirstChildElement(); elem != NULL; elem = elem->NextSiblingElement())
		{
			std::string elemName = elem->Value();
			std::string name = "";

			if (strcmp(elemName.data(), "object") == 0)
			{
				for (TiXmlNode* object = elem->FirstChildElement(); object != NULL; object = object->NextSiblingElement())
				{
					if (strcmp(object->Value(), "name") == 0)
					{
						name = object->FirstChild()->Value();
					}

					if (strcmp(object->Value(), "bndbox") == 0)
					{
						BBox obj;
						TiXmlElement* xmin = object->FirstChildElement("xmin");
						TiXmlElement* ymin = object->FirstChildElement("ymin");
						TiXmlElement* xmax = object->FirstChildElement("xmax");
						TiXmlElement* ymax = object->FirstChildElement("ymax");

						obj.xmin = atoi(std::string(xmin->FirstChild()->Value()).c_str());
						obj.xmax = atoi(std::string(xmax->FirstChild()->Value()).c_str());
						obj.ymin = atoi(std::string(ymin->FirstChild()->Value()).c_str());
						obj.ymax = atoi(std::string(ymax->FirstChild()->Value()).c_str());
						obj.name = name;
						objects.push_back(obj);
					}
				}
			}
		}
		doc.Clear();
		return objects;
	}

	CSPdarknet53_tinyImpl::CSPdarknet53_tinyImpl() 
	{
		conv1 = BasicConv(3, 32, 3, 2);
		conv2 = BasicConv(32, 64, 3, 2);
		resblockBody1 = Resblock_body(64, 64);
		resblockBody2 = Resblock_body(128, 128);
		resblockBody3 = Resblock_body(256, 256);
		conv3 = BasicConv(512, 512, 3);

		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("resblock_body1", resblockBody1);
		register_module("resblock_body2", resblockBody2);
		register_module("resblock_body3", resblockBody3);
		register_module("conv3", conv3);
	}
	
	std::vector<torch::Tensor> CSPdarknet53_tinyImpl::forward(torch::Tensor x) 
	{
		x = conv1(x);
		x = conv2(x);

		x = resblockBody1->forward(x)[0];
		x = resblockBody2->forward(x)[0];

		auto resOut = resblockBody3->forward(x);
		x = resOut[0];
		auto feat1 = resOut[1];
		x = conv3->forward(x);
		auto feat2 = x;
		return std::vector<torch::Tensor>({ feat1, feat2 });
	}

	torch::nn::Sequential yoloHead(std::vector<int> filtersList, int inFilters) 
	{
		auto m = torch::nn::Sequential(
			BasicConv(inFilters, filtersList[0], 3),
			torch::nn::Conv2d(createConvOptions(
				filtersList[0],        // in_channels
				filtersList[1],        // out_channels
				1,                      // kernel_size
				1,                      // stride
				0,                      // padding
				1,                      // dilation
				true                    // bias
			))
		);
		return m;
	}

	YoloBody_tinyImpl::YoloBody_tinyImpl(int numAnchors, int numСlasses) 
	{
		backbone = CSPdarknet53_tiny();
		convForP5 = BasicConv(512, 256, 1);
		yoloHeadP5 = yoloHead({ 512, numAnchors * (5 + numСlasses) }, 256);
		upsample = Upsample(256, 128);
		yoloHeadP4 = yoloHead({ 256, numAnchors * (5 + numСlasses) }, 384);

		register_module("backbone", backbone);
		register_module("conv_for_P5", convForP5);
		register_module("yolo_headP5", yoloHeadP5);
		register_module("upsample", upsample);
		register_module("yolo_headP4", yoloHeadP4);
	}
	
	std::vector<torch::Tensor> YoloBody_tinyImpl::forward(torch::Tensor x) 
	{
		auto backboneOut = backbone->forward(x);
		auto feat1 = backboneOut[0];
		auto feat2 = backboneOut[1];
		auto P5 = convForP5->forward(feat2);
		auto out0 = yoloHeadP5->forward(P5);

		auto P5_Upsample = upsample->forward(P5);
		auto P4 = torch::cat({ P5_Upsample, feat1 }, 1);
		auto out1 = yoloHeadP4->forward(P4);
		return std::vector<torch::Tensor>({ out0, out1 });
	}

	YOLOLossImpl::YOLOLossImpl(torch::Tensor anchors_, int numСlasses_, int imgSize_[],
		float label_smooth_, torch::Device device_, bool normalize) 
	{
		this->anchors = anchors_;
		this->numAnchors = anchors_.sizes()[0];
		this->numClasses = numСlasses_;
		this->bboxAttrs = 5 + numClasses;
		memcpy(imageSize, imgSize_, 2 * sizeof(int));
		std::vector<int> featureLength_ = { int(imageSize[0] / 32),int(imageSize[0] / 16),int(imageSize[0] / 8) };
		std::copy(featureLength_.begin(), featureLength_.end(), featureLength.begin());
		this->labelSmooth = label_smooth_;
		this->device = device_;
		this->normalize = normalize;
	}
	
	torch::Tensor jaccard(torch::Tensor _boxA, torch::Tensor _boxB) 
	{
		auto b1X1 = _boxA.select(1, 0) - _boxA.select(1, 2) / 2;
		auto b1X2 = _boxA.select(1, 0) + _boxA.select(1, 2) / 2;
		auto b1Y1 = _boxA.select(1, 1) - _boxA.select(1, 3) / 2;
		auto b1Y2 = _boxA.select(1, 1) + _boxA.select(1, 3) / 2;

		auto b2X1 = _boxB.select(1, 0) - _boxB.select(1, 2) / 2;
		auto b2X2 = _boxB.select(1, 0) + _boxB.select(1, 2) / 2;
		auto b2Y1 = _boxB.select(1, 1) - _boxB.select(1, 3) / 2;
		auto b2Y2 = _boxB.select(1, 1) + _boxB.select(1, 3) / 2;


		auto boxA = torch::zeros_like(_boxA);
		auto boxB = torch::zeros_like(_boxB);

		boxA.select(1, 0) = b1X1;
		boxA.select(1, 1) = b1Y1;
		boxA.select(1, 2) = b1X2;
		boxA.select(1, 3) = b1Y2;

		boxB.select(1, 0) = b2X1;
		boxB.select(1, 1) = b2Y1;
		boxB.select(1, 2) = b2X2;
		boxB.select(1, 3) = b2Y2;

		auto A = boxA.size(0);
		auto B = boxB.size(0);

		auto maxXy = torch::min(boxA.narrow(1, 2, 2).unsqueeze(1).expand({ A, B, 2 }), boxB.narrow(1, 2, 2).unsqueeze(0).expand({ A, B, 2 }));
		auto minXy = torch::max(boxA.narrow(1, 0, 2).unsqueeze(1).expand({ A, B, 2 }), boxB.narrow(1, 0, 2).unsqueeze(0).expand({ A, B, 2 }));

		auto inter = torch::clamp((maxXy - minXy), 0);
		inter = inter.select(2, 0) * inter.select(2, 1);

		auto areaA = ((boxA.select(1, 2) - boxA.select(1, 0)) * (boxA.select(1, 3) - boxA.select(1, 1))).unsqueeze(1).expand_as(inter); 
		auto areaB = ((boxB.select(1, 2) - boxB.select(1, 0)) * (boxB.select(1, 3) - boxB.select(1, 1))).unsqueeze(0).expand_as(inter); 

		auto uni = areaA + areaB - inter;
		return inter / uni;
	}

	torch::Tensor smoothLabel(torch::Tensor yTrue, int labelSmoothing, int numСlasses) 
	{
		return yTrue * (1.0 - labelSmoothing) + labelSmoothing / numСlasses;
	}

	torch::Tensor boxCiou(torch::Tensor b1, torch::Tensor b2)
	{
		auto b1XY = b1.narrow(-1, 0, 2);
		auto b1WH = b1.narrow(-1, 2, 2);
		auto b1WhHalf = b1WH / 2.0;
		auto b1Mins = b1XY - b1WhHalf;
		auto b1Maxes = b1XY + b1WhHalf;

		auto b2XY = b2.narrow(-1, 0, 2);
		auto b2WH = b2.narrow(-1, 2, 2);
		auto b2WhHalf = b2WH / 2.0;
		auto b2Mins = b2XY - b2WhHalf;
		auto b2Maxes = b2XY + b2WhHalf;

		auto intersectmins = torch::max(b1Mins, b2Mins);
		auto intersectMaxes = torch::min(b1Maxes, b2Maxes);
		auto intersectWH = torch::max(intersectMaxes - intersectmins, torch::zeros_like(intersectMaxes));
		auto intersectArea = intersectWH.select(-1, 0) * intersectWH.select(-1, 1);
		auto b1Area = b1WH.select(-1, 0) * b1WH.select(-1, 1);
		auto b2Area = b2WH.select(-1, 0) * b2WH.select(-1, 1);
		auto unionArea = b1Area + b2Area - intersectArea;
		auto iou = intersectArea / torch::clamp(unionArea, 1e-6);

		auto centerDistance = torch::sum(torch::pow((b1XY - b2XY), 2), -1);

		auto enclosemins = torch::min(b1Mins, b2Mins);
		auto enclosemaxes = torch::max(b1Maxes, b2Maxes);
		auto encloseWH = torch::max(enclosemaxes - enclosemins, torch::zeros_like(intersectMaxes));

		auto encloseDiagonal = torch::sum(torch::pow(encloseWH, 2), -1);
		auto ciou = iou - 1.0 * (centerDistance) / (encloseDiagonal + 1e-7);

		auto v = (4 / (Pi * Pi)) * torch::pow((torch::atan(b1WH.select(-1, 0) / b1WH.select(-1, 1)) - torch::atan(b2WH.select(-1, 0) / b2WH.select(-1, 1))), 2);
		auto alpha = v / (1.0 - iou + v);
		ciou = ciou - alpha * v;

		return ciou;
	}

	torch::Tensor clipByTensor(torch::Tensor t, float tmin, float tmax) 
	{
		t = t.to(torch::kFloat32);
		auto result = (t >= tmin).to(torch::kFloat32) * t + (t < tmin).to(torch::kFloat32) * tmin;
		result = (result <= tmax).to(torch::kFloat32) * result + (result > tmax).to(torch::kFloat32) * tmax;
		return result;
	}

	torch::Tensor MSELoss(torch::Tensor pred, torch::Tensor target) 
	{
		return torch::pow((pred - target), 2);
	}

	torch::Tensor BCELoss(torch::Tensor pred, torch::Tensor target) 
	{
		pred = clipByTensor(pred, 1e-7, 1.0 - 1e-7);
		auto output = -target * torch::log(pred) - (1.0 - target) * torch::log(1.0 - pred);
		return output;
	}

	UpsampleImpl::UpsampleImpl(int inChannels, int outChannels)
	{
		upsample = torch::nn::Sequential(BasicConv(inChannels, outChannels, 1));
		register_module("upsample", upsample);
	}

	torch::Tensor UpsampleImpl::forward(torch::Tensor x)
	{
		x = upsample->forward(x);
		x = at::upsample_nearest2d(x, { x.sizes()[2] * 2 , x.sizes()[3] * 2 });
		return x;
	}

	std::vector<torch::Tensor> YOLOLossImpl::forward(torch::Tensor input, std::vector<torch::Tensor> targets)
	{
		auto bs = input.size(0);
		auto inH = input.size(2);
		auto inW = input.size(3);

		auto strideH = imageSize[1] / inH;
		auto strideW = imageSize[0] / inW;

		auto scaledAnchors = anchors.clone();
		scaledAnchors.select(1, 0) = scaledAnchors.select(1, 0) / strideW;
		scaledAnchors.select(1, 1) = scaledAnchors.select(1, 1) / strideH;

		auto prediction = input.view({ bs, int(numAnchors / 2), bboxAttrs, inH, inW }).permute({ 0, 1, 3, 4, 2 }).contiguous();
		
		auto conf = torch::sigmoid(prediction.select(-1, 4));
		auto predcls = torch::sigmoid(prediction.narrow(-1, 5, numClasses)); 

		auto temp = getTarget(targets, scaledAnchors, inW, inH, ignoreThreshold);
		auto BoolType = torch::ones(1).to(torch::kBool).to(device).options();
		auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();
		auto mask = temp[0].to(BoolType);
		auto noobjMask = temp[1].to(device);
		auto tbox = temp[2];
		auto tconf = temp[3];
		auto tcls = temp[4];
		auto boxLossScalex = temp[5];
		auto boxLossScaley = temp[6];

		auto tempCiou = getIgnore(prediction, targets, scaledAnchors, inW, inH, noobjMask);
		noobjMask = tempCiou[0];
		auto predBoxesForCiou = tempCiou[1];

		mask = mask.to(device);
		noobjMask = noobjMask.to(device);
		boxLossScalex = boxLossScalex.to(device);
		boxLossScaley = boxLossScaley.to(device);
		tconf = tconf.to(device);
		tcls = tcls.to(device);
		predBoxesForCiou = predBoxesForCiou.to(device);
		tbox = tbox.to(device);

		auto boxLossScale = 2 - boxLossScalex * boxLossScaley;
		auto ciou = (1 - boxCiou(predBoxesForCiou.index({ mask }), tbox.index({ mask }))) * boxLossScale.index({ mask });
		auto lossLoc = torch::sum(ciou / bs);

		auto lossConf = torch::sum(BCELoss(conf, mask.to(FloatType)) * mask.to(FloatType) / bs) + \
			torch::sum(BCELoss(conf, mask.to(FloatType)) * noobjMask / bs);

		auto lossCls = torch::sum(BCELoss(predcls.index({ mask == 1 }), smoothLabel(tcls.index({ mask == 1 }), labelSmooth, numClasses)) / bs);
		auto loss = lossConf * lambdaConf + lossCls * lambdaCls + lossLoc * lambdaLoc;

		torch::Tensor numPos = torch::tensor({ 0 }).to(device);
		if (normalize) 
		{
			numPos = torch::sum(mask);
			numPos = torch::max(numPos, torch::ones_like(numPos));
		}
		else
			numPos[0] = bs / 2;
		return std::vector<torch::Tensor>({ loss, numPos });
	}

	std::vector<torch::Tensor> YOLOLossImpl::getTarget(std::vector<torch::Tensor> targets, torch::Tensor scaledAnchors, int inW, int inH, float ignoreThreshold)
	{

		int bs = targets.size();
		auto scaledAnchorsType = scaledAnchors.options();

		int index = vecIndex(featureLength, inW);
		std::vector<std::vector<int>> anchorVecInVec = { {3, 4, 5} ,{1, 2, 3} };
		std::vector<int> anchorIndex = anchorVecInVec[index];
		int subtractIndex = 3 * index;

		torch::TensorOptions gradFalse(torch::requires_grad(false));
		auto TensorType = targets[0].options();
		auto mask = torch::zeros({ bs, int(numAnchors / 2), inH, inW }, gradFalse);
		auto noobjMask = torch::ones({ bs, int(numAnchors / 2), inH, inW }, gradFalse);

		auto tx = torch::zeros({ bs, int(numAnchors / 2), inH, inW }, gradFalse);
		auto ty = torch::zeros({ bs, int(numAnchors / 2), inH, inW }, gradFalse);
		auto tw = torch::zeros({ bs, int(numAnchors / 2), inH, inW }, gradFalse);
		auto th = torch::zeros({ bs, int(numAnchors / 2), inH, inW }, gradFalse);
		auto tbox = torch::zeros({ bs, int(numAnchors / 2), inH, inW, 4 }, gradFalse);
		auto tconf = torch::zeros({ bs, int(numAnchors / 2), inH, inW }, gradFalse);
		auto tcls = torch::zeros({ bs, int(numAnchors / 2), inH, inW, numClasses }, gradFalse);

		auto boxLossScalex = torch::zeros({ bs, int(numAnchors / 2), inH, inW }, gradFalse);
		auto boxLossScaley = torch::zeros({ bs, int(numAnchors / 2), inH, inW }, gradFalse);

		for (int b = 0; b < bs; b++)
		{
			if (targets[b].sizes().size() == 1)
			{
				continue;
			}

			auto gxs = targets[b].narrow(-1, 0, 1) * inW;
			auto gys = targets[b].narrow(-1, 1, 1) * inH;

			auto gws = targets[b].narrow(-1, 2, 1) * inW;
			auto ghs = targets[b].narrow(-1, 3, 1) * inH;

			auto gis = torch::floor(gxs);
			auto gjs = torch::floor(gys);


			auto gtBox = torch::Tensor(torch::cat({ torch::zeros_like(gws), torch::zeros_like(ghs), gws, ghs }, 1)).to(torch::kFloat32);

			auto anchorShapes = torch::Tensor(torch::cat({ torch::zeros({ numAnchors, 2 }).to(scaledAnchorsType), torch::Tensor(scaledAnchors) }, 1)).to(TensorType);

			auto anchIous = jaccard(gtBox, anchorShapes);

			auto bestNs = torch::argmax(anchIous, -1);

			for (int i = 0; i < bestNs.sizes()[0]; i++)
			{
				if (!inVec(anchorIndex, bestNs[i].item().toInt()))
				{
					continue;
				}
				auto gi = gis[i].to(torch::kLong).item().toInt();
				auto gj = gjs[i].to(torch::kLong).item().toInt();
				auto gx = gxs[i].item().toFloat();
				auto gy = gys[i].item().toFloat();
				auto gw = gws[i].item().toFloat();
				auto gh = ghs[i].item().toFloat();
				if (gj < inH && gi < inW)
				{
					auto bestN = vecIndex(anchorIndex, bestNs[i].item().toInt());

					noobjMask[b][bestN][gj][gi] = 0;
					mask[b][bestN][gj][gi] = 1;

					tx[b][bestN][gj][gi] = gx;
					ty[b][bestN][gj][gi] = gy;

					tw[b][bestN][gj][gi] = gw;
					th[b][bestN][gj][gi] = gh;

					boxLossScalex[b][bestN][gj][gi] = targets[b][i][2];
					boxLossScaley[b][bestN][gj][gi] = targets[b][i][3];

					tconf[b][bestN][gj][gi] = 1;

					tcls[b][bestN][gj][gi][targets[b][i][4].item().toLong()] = 1;
				}
				else {
					std::cout << gxs << gys << std::endl;
					std::cout << gis << gjs << std::endl;
					std::cout << targets[b];
					std::cout << "Step out of boundary;" << std::endl;

				}
			}
			tbox.select(-1, 0) = tx;
			tbox.select(-1, 1) = ty;
			tbox.select(-1, 2) = tw;
			tbox.select(-1, 3) = th;
			std::vector<torch::Tensor> output = { mask, noobjMask, tbox, tconf, tcls, boxLossScalex, boxLossScaley };
			return output;
		}
	}

	std::vector<torch::Tensor> YOLOLossImpl::getIgnore(torch::Tensor prediction, std::vector<torch::Tensor> targets, torch::Tensor scaledAnchors, int inW, int inH, torch::Tensor noobjMask)
	{
		int bs = targets.size();
		int index = vecIndex(featureLength, inW);
		std::vector<std::vector<int>> anchorVecInVec = { {3, 4, 5}, {0, 1, 2} };
		std::vector<int> anchorIndex = anchorVecInVec[index];

		auto x = torch::sigmoid(prediction.select(-1, 0));
		auto y = torch::sigmoid(prediction.select(-1, 1));

		auto w = prediction.select(-1, 2);  //Width
		auto h = prediction.select(-1, 3);  // Height

		auto FloatType = prediction.options();
		auto LongType = prediction.to(torch::kLong).options();

		auto gridX = torch::linspace(0, inW - 1, inW).repeat({ inH, 1 }).repeat(
			{ int(bs * numAnchors / 2), 1, 1 }).view(x.sizes()).to(FloatType);
		auto gridY = torch::linspace(0, inH - 1, inH).repeat({ inW, 1 }).t().repeat(
			{ int(bs * numAnchors / 2), 1, 1 }).view(y.sizes()).to(FloatType);

		auto anchorW = scaledAnchors.narrow(0, anchorIndex[0], 3).narrow(-1, 0, 1).to(FloatType);
		auto anchorH = scaledAnchors.narrow(0, anchorIndex[0], 3).narrow(-1, 1, 1).to(FloatType);
		anchorW = anchorW.repeat({ bs, 1 }).repeat({ 1, 1, inH * inW }).view(w.sizes());
		anchorH = anchorH.repeat({ bs, 1 }).repeat({ 1, 1, inH * inW }).view(h.sizes());

		auto predBoxes = torch::randn_like(prediction.narrow(-1, 0, 4)).to(FloatType);
		predBoxes.select(-1, 0) = x + gridX;
		predBoxes.select(-1, 1) = y + gridY;

		predBoxes.select(-1, 2) = w.exp() * anchorW;
		predBoxes.select(-1, 3) = h.exp() * anchorH;

		for (int i = 0; i < bs; i++)
		{
			auto predBoxesForIgnore = predBoxes[i];
			predBoxesForIgnore = predBoxesForIgnore.view({ -1, 4 });
			if (targets[i].sizes().size() > 1) 
			{
				auto gx = targets[i].narrow(-1, 0, 1) * inW;
				auto gy = targets[i].narrow(-1, 1, 1) * inH;
				auto gw = targets[i].narrow(-1, 2, 1) * inW;
				auto gh = targets[i].narrow(-1, 3, 1) * inH;
				auto gtBox = torch::cat({ gx, gy, gw, gh }, -1).to(FloatType);

				auto anchIous = jaccard(gtBox, predBoxesForIgnore);
				auto anchIousMaxTuple = torch::max(anchIous, 0);
				auto anchIousMax = std::get<0>(anchIousMaxTuple);

				anchIousMax = anchIousMax.view(predBoxes.sizes().slice(1, 3));
				noobjMask[i] = (anchIousMax <= ignoreThreshold).to(FloatType) * noobjMask[i];
			}

		}

		std::vector<torch::Tensor> output = { noobjMask, predBoxes };
		return output;
	}

	std::vector<int> nmsLibtorch(torch::Tensor bboxes, torch::Tensor scores, float thresh) 
	{
		auto x1 = bboxes.select(-1, 0);
		auto y1 = bboxes.select(-1, 1);
		auto x2 = bboxes.select(-1, 2);
		auto y2 = bboxes.select(-1, 3);
		auto areas = (x2 - x1) * (y2 - y1);
		auto tuple_sorted = scores.sort(0, true);
		auto order = std::get<1>(tuple_sorted);

		std::vector<int>	keep;
		while (order.numel() > 0) 
		{
			if (order.numel() == 1) 
			{
				auto i = order.item();
				keep.push_back(i.toInt());
				break;
			}
			else 
			{
				auto i = order[0].item();
				keep.push_back(i.toInt());
			}

			auto order_mask = order.narrow(0, 1, order.size(-1) - 1);
			x1.index({ order_mask });
			x1.index({ order_mask }).clamp(x1[keep.back()].item().toFloat(), 1e10);
			auto xx1 = x1.index({ order_mask }).clamp(x1[keep.back()].item().toFloat(), 1e10);// [N - 1, ]
			auto yy1 = y1.index({ order_mask }).clamp(y1[keep.back()].item().toFloat(), 1e10);
			auto xx2 = x2.index({ order_mask }).clamp(0, x2[keep.back()].item().toFloat());
			auto yy2 = y2.index({ order_mask }).clamp(0, y2[keep.back()].item().toFloat());
			auto inter = (xx2 - xx1).clamp(0, 1e10) * (yy2 - yy1).clamp(0, 1e10);// [N - 1, ]

			auto iou = inter / (areas[keep.back()] + areas.index({ order.narrow(0,1,order.size(-1) - 1) }) - inter);//[N - 1, ]
			auto idx = (iou <= thresh).nonzero().squeeze();
			if (idx.numel() == 0) 
			{
				break;
			}
			order = order.index({ idx + 1 });
		}
		return keep;
	}

	std::vector<torch::Tensor> nonMaximumSuppression(torch::Tensor prediction, int numСlasses, float confThres, float nmsThres) 
	{
		prediction.select(-1, 0) -= prediction.select(-1, 2) / 2;
		prediction.select(-1, 1) -= prediction.select(-1, 3) / 2;
		prediction.select(-1, 2) += prediction.select(-1, 0);
		prediction.select(-1, 3) += prediction.select(-1, 1);

		std::vector<torch::Tensor> output;
		for (int imageID = 0; imageID < prediction.sizes()[0]; imageID++) 
		{
			auto imagePred = prediction[imageID];
			auto maxOutTuple = torch::max(imagePred.narrow(-1, 5, numСlasses), -1, true);
			auto classConf = std::get<0>(maxOutTuple);
			auto classPred = std::get<1>(maxOutTuple);
			auto confMask = (imagePred.select(-1, 4) * classConf.select(-1, 0) >= confThres).squeeze();
			imagePred = imagePred.index({ confMask }).to(torch::kFloat);
			classConf = classConf.index({ confMask }).to(torch::kFloat);
			classPred = classPred.index({ confMask }).to(torch::kFloat);

			if (!imagePred.sizes()[0]) 
			{
				output.push_back(torch::full({ 1, 7 }, 0));
				continue;
			}

			auto detections = torch::cat({ imagePred.narrow(-1,0,5), classConf, classPred }, 1);
	
			std::vector<torch::Tensor> imgClasses;

			for (int m = 0, len = detections.size(0); m < len; m++)
			{
				bool found = false;
				for (size_t n = 0; n < imgClasses.size(); n++)
				{
					auto ret = (detections[m][6] == imgClasses[n]);
					if (torch::nonzero(ret).size(0) > 0)
					{
						found = true;
						break;
					}
				}
				if (!found) imgClasses.push_back(detections[m][6]);
			}
			std::vector<torch::Tensor> tempClassDetections;
			for (auto c : imgClasses) 
			{
				auto detectionsClass = detections.index({ detections.select(-1,-1) == c });
				auto keep = nmsLibtorch(detectionsClass.narrow(-1, 0, 4), detectionsClass.select(-1, 4) * detectionsClass.select(-1, 5), nmsThres);
				std::vector<torch::Tensor> tempMaxDetections;
				for (auto v : keep) 
				{
					tempMaxDetections.push_back(detectionsClass[v]);
				}
				auto maxDetections = torch::cat(tempMaxDetections, 0);
				tempClassDetections.push_back(maxDetections);
			}
			auto classDetections = torch::cat(tempClassDetections, 0);
			output.push_back(classDetections);
		}
		return output;
	}

	int BBox::GetH()
	{
		return ymax - ymin;
	}
	int BBox::GetW()
	{
		return xmax - xmin;
	}
	float BBox::CenterX()
	{
		return (xmax + xmin) / 2.0;
	}
	float BBox::CenterY()
	{
		return  (ymax + ymin) / 2.0;
	}

	DetectorData DetAugmentations::Resize(DetectorData mData, int width, int height, float probability) 
	{
		float rand_number = RandomNum<float>(0, 1);
		if (rand_number <= probability) 
		{

			float hScale = height * 1.0 / mData.image.rows;
			float wScale = width * 1.0 / mData.image.cols;
			for (int i = 0; i < mData.bboxes.size(); i++)
			{
				mData.bboxes[i].xmin = int(wScale * mData.bboxes[i].xmin);
				mData.bboxes[i].xmax = int(wScale * mData.bboxes[i].xmax);
				mData.bboxes[i].ymin = int(hScale * mData.bboxes[i].ymin);
				mData.bboxes[i].ymax = int(hScale * mData.bboxes[i].ymax);
			}
			cv::resize(mData.image, mData.image, cv::Size(width, height));
		}
		return mData;
	}
	
	bool doesExist(const std::string& name) 
	{
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	}

	// Преобразование строки в нижний регистр
	std::string toLowerCase(const std::string& src) 
	{
		std::string dst = src;
		std::transform(dst.begin(), dst.end(), dst.begin(), ::tolower);
		return dst;
	}

	// Проверка, заканчивается ли строка на заданный суффикс
	bool endsWith(const std::string& src, const std::string& suffix) 
	{
		return src.size() >= suffix.size() && src.compare(src.size() - suffix.size(), suffix.size(), suffix) == 0;
	}

	// Замена всех вхождений подстроки в строке
	std::string replaceAll(const std::string& str, const std::string& from, const std::string& to) 
	{
		std::string result = str;
		size_t startPos = 0;
		while ((startPos = result.find(from, startPos)) != std::string::npos) 
		{
			result.replace(startPos, from.length(), to);
			startPos += to.length();
		}
		return result;
	}

	// Загрузка данных датасета 
	void loadXMLDataFromFolder(const std::string& folder, const std::string& imageType,
		std::vector<std::string>& listImages, std::vector<std::string>& listLabels) 
	{
		for (const auto& entry : std::filesystem::recursive_directory_iterator(folder)) 
		{
			if (entry.is_regular_file()) 
			{
				std::string fullPath = entry.path().string();
				std::string lowerName = toLowerCase(entry.path().filename().string());

				if (endsWith(lowerName, ".xml")) 
				{
					listLabels.push_back(fullPath);

					std::string imagePath = replaceAll(fullPath, ".xml", imageType);
					imagePath = replaceAll(imagePath, "train/labels", "/train/images");
					imagePath = replaceAll(imagePath, "val/labels", "val/images");
					listImages.push_back(imagePath);
				}
			}
		}
	}

	// Функция отрисовки ограничевающего прямоугольника
	void showBbox(cv::Mat image, torch::Tensor bboxes, std::vector<std::string> nameList) 
	{
		int fontFace = cv::FONT_HERSHEY_COMPLEX;
		double fontScale = 0.4;
		int thickness = 1;
		float* bbox = new float[bboxes.size(0)]();
		
		if (bboxes.equal(torch::zeros_like(bboxes))) 
		{
			std::cout << "Boxes not detected" << std::endl;
		} 
		memcpy(bbox, bboxes.cpu().data_ptr(), bboxes.size(0) * sizeof(float));
		for (int i = 0; i < bboxes.size(0); i = i + 7)
		{
			cv::rectangle(image, cv::Rect(bbox[i + 0], bbox[i + 1], bbox[i + 2] - bbox[i + 0], bbox[i + 3] - bbox[i + 1]), cv::Scalar(0, 0, 255));

			cv::Point origin;
			origin.x = bbox[i + 0];
			origin.y = bbox[i + 1] + 8;
			cv::putText(image, nameList[bbox[i + 6]], origin, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness, 1, 0);
		}
		delete bbox;
		cv::imwrite("prediction.jpg", image);
		cv::imshow("Test", image);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	torch::Tensor DecodeBox(torch::Tensor input, torch::Tensor anchors, int numСlasses, int imgSize[])
	{
		int numAnchors = anchors.sizes()[0];
		int bboxAttrs = 5 + numСlasses;
		int batchSize = input.sizes()[0];
		int inputHeight = input.sizes()[2];
		int inputWidth = input.sizes()[3];

		auto strideH = imgSize[1] / inputHeight;
		auto strideW = imgSize[0] / inputWidth;

		auto scaledAnchors = anchors.clone();
		scaledAnchors.select(1, 0) = scaledAnchors.select(1, 0) / strideW;
		scaledAnchors.select(1, 1) = scaledAnchors.select(1, 1) / strideH;

		auto prediction = input.view({ batchSize, numAnchors,bboxAttrs, inputHeight, inputWidth }).permute({ 0, 1, 3, 4, 2 }).contiguous();

		auto x = torch::sigmoid(prediction.select(-1, 0));
		auto y = torch::sigmoid(prediction.select(-1, 1));

		auto w = prediction.select(-1, 2); // Width
		auto h = prediction.select(-1, 3); // Height

		auto conf = torch::sigmoid(prediction.select(-1, 4));

		auto predCls = torch::sigmoid(prediction.narrow(-1, 5, numСlasses));// Cls pred.

		auto LongType = x.clone().to(torch::kLong).options();
		auto FloatType = x.options();

		auto gridX = torch::linspace(0, inputWidth - 1, inputWidth).repeat({ inputHeight, 1 }).repeat(
			{ batchSize * numAnchors, 1, 1 }).view(x.sizes()).to(FloatType);
		auto gridY = torch::linspace(0, inputHeight - 1, inputHeight).repeat({ inputWidth, 1 }).t().repeat(
			{ batchSize * numAnchors, 1, 1 }).view(y.sizes()).to(FloatType);

		auto anchorW = scaledAnchors.to(FloatType).narrow(1, 0, 1);
		auto anchorH = scaledAnchors.to(FloatType).narrow(1, 1, 1);
		anchorW = anchorW.repeat({ batchSize, 1 }).repeat({ 1, 1, inputHeight * inputWidth }).view(w.sizes());
		anchorH = anchorH.repeat({ batchSize, 1 }).repeat({ 1, 1, inputHeight * inputWidth }).view(h.sizes());

		auto predBoxes = torch::randn_like(prediction.narrow(-1, 0, 4)).to(FloatType);
		predBoxes.select(-1, 0) = x + gridX;
		predBoxes.select(-1, 1) = y + gridY;
		predBoxes.select(-1, 2) = w.exp() * anchorW;
		predBoxes.select(-1, 3) = h.exp() * anchorH;

		std::vector<int> scales{ strideW, strideH, strideW, strideH };
		auto _scale = torch::tensor(scales).to(FloatType);

		predBoxes = predBoxes.view({ batchSize, -1, 4 }) * _scale;
		conf = conf.view({ batchSize, -1, 1 });
		predCls = predCls.view({ batchSize, -1, numСlasses });
		auto output = torch::cat({ predBoxes, conf, predCls }, -1);
		return output;
	}

	Detector::Detector()
	{

	}

	void Detector::Initialize(int gpuID, int width, int height,
		std::string nameListPath) 
	{
		if (gpuID >= 0) {
			if (gpuID >= torch::getNumGPUs()) 
			{
				std::cout << "No GPU id " << gpuID << " available" << std::endl;
			}
			device = torch::Device(torch::kCUDA, gpuID);
		}
		else 
		{
			device = torch::Device(torch::kCPU);
		}
		nameList = {};
		std::ifstream ifs;
		ifs.open(nameListPath, std::ios::in);
		if (!ifs.is_open())
		{
			std::cout << "Open " << nameListPath << " file failed.";
			return;
		}
		std::string buf = "";
		while (getline(ifs, buf))
		{
			nameList.push_back(buf);
		}


		int numClasses = nameList.size();
		this->nameList = nameList;

		this->width = width;
		this->height = height;
		if (width % 32 || height % 32) 
		{
			std::cout << "Width or height is not divisible by 32" << std::endl;
			return;
		}

		detector = YoloBody_tiny(3, numClasses);
		detector->to(device);
		return;
	}

	void Detector::LoadPretrained(std::string pretrainedPath) 
	{
		auto netPretrained = YoloBody_tiny(3, 80);
		torch::load(netPretrained, pretrainedPath);
		if (this->nameList.size() == 80)
		{
			detector = netPretrained;
		}

		torch::OrderedDict<std::string, at::Tensor> pretrainedDict = netPretrained->named_parameters();
		torch::OrderedDict<std::string, at::Tensor> modelDict = detector->named_parameters();


		for (auto n = pretrainedDict.begin(); n != pretrainedDict.end(); n++)
		{
			if (strstr((*n).key().c_str(), "yolo_head")) 
			{
				continue;
			}
			modelDict[(*n).key()] = (*n).value();
		}

		torch::autograd::GradMode::set_enabled(false); 
		auto newParams = modelDict;
		auto params = detector->named_parameters(true);
		auto buffers = detector->named_buffers(true);
		for (auto& val : newParams) 
		{
			auto name = val.key();
			auto* t = params.find(name);
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

	int Detector::Train(std::string trainValPath, std::string imageType, int numEpochs, int batchSize,
		float learningRate, std::string savePath, std::string pretrainedPath) 
	{
		if (!doesExist(pretrainedPath))
		{
			std::cout << "Pretrained path is invalid: " << pretrainedPath << "\t random initialzed the model" << std::endl;
			return 1;
		}
		else {
			LoadPretrained(pretrainedPath);
		}

		std::string trainLabelPath = trainValPath + "/train/labels";
		std::string valLabelPath = trainValPath + "/val/labels";

		std::vector<std::string> listImagesTrain = {};
		std::vector<std::string> listLabelsTrain = {};
		std::vector<std::string> listImagesVal = {};
		std::vector<std::string> listLabelsVal = {};

		loadXMLDataFromFolder(trainLabelPath, imageType, listImagesTrain, listLabelsTrain);
		loadXMLDataFromFolder(valLabelPath, imageType, listImagesVal, listLabelsVal);

		if (listImagesTrain.size() < batchSize || listImagesVal.size() < batchSize) 
		{
			std::cout << "Image numbers less than batch size or empty image folder" << std::endl;
			return 2;
		}
		if (!doesExist(listImagesTrain[0]))
		{
			std::cout << "Image path is invalid get first train image " << listImagesTrain[0] << std::endl;
			return 3;
		}
		auto customDatasetTrain = DetDataset(listImagesTrain, listLabelsTrain, nameList, true, width, height);
		auto customDatasetVal = DetDataset(listImagesVal, listLabelsVal, nameList, false, width, height);
		auto dataLoaderTrain = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(customDatasetTrain), batchSize);
		auto dataLoaderVal = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(customDatasetVal), batchSize);

		float anchor[12] = { 10,14,  23,27,  37,58,  81,82,  135,169,  344,319 };
		auto anchors_ = torch::from_blob(anchor, { 6,2 }, torch::TensorOptions(torch::kFloat32)).to(device);
		int imageSize[2] = { width, height };

		bool normalize = false;
		auto critia1 = YOLOLossImpl(anchors_, nameList.size(), imageSize, 0.01, device, normalize);
		auto critia2 = YOLOLossImpl(anchors_, nameList.size(), imageSize, 0.01, device, normalize);

		auto pretrainedDict = detector->named_parameters();
		auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();

		for (int epochCount = 0; epochCount < numEpochs; epochCount++) 
		{
			float lossSum = 0;
			int batchCount = 0;
			float lossTrain = 0;
			float lossVal = 0;
			float bestLoss = 1e10;

			if (epochCount == int(numEpochs / 2)) 
			{ 
				learningRate /= 10; 
			}
			torch::optim::Adam optimizer(detector->parameters(), learningRate);
			if (epochCount < int(numEpochs / 10)) 
			{
				for (auto mm : pretrainedDict)
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
				for (auto mm : pretrainedDict) 
				{
					mm.value().set_requires_grad(true);
				}
			}
			detector->train();
			for (auto& batch : *dataLoaderTrain) 
			{
				std::vector<torch::Tensor> imagesVec = {};
				std::vector<torch::Tensor> targetsVec = {};
				if (batch.size() < batchSize) continue;
				for (int i = 0; i < batchSize; i++)
				{
					imagesVec.push_back(batch[i].data.to(FloatType));
					targetsVec.push_back(batch[i].target.to(FloatType));
				}
				auto data = torch::stack(imagesVec).div(255.0);

				optimizer.zero_grad();
				auto outputs = detector->forward(data);
				std::vector<torch::Tensor> lossNumPos1 = critia1.forward(outputs[0], targetsVec);
				std::vector<torch::Tensor> lossNumPos2 = critia1.forward(outputs[1], targetsVec);

				auto loss = lossNumPos1[0] + lossNumPos2[0];
				auto numPos = lossNumPos1[1] + lossNumPos2[1];
				loss = loss / numPos;
				loss.backward();
				optimizer.step();
				lossSum += loss.item().toFloat();
				batchCount++;
				lossTrain = lossSum / batchCount;

				std::cout << "Epoch: " << epochCount << "," << " Training Loss: " << lossTrain << "\r";
			}
			std::cout << std::endl;
			detector->eval();
			lossSum = 0; batchCount = 0;
			for (auto& batch : *dataLoaderVal) 
			{
				std::vector<torch::Tensor> imagesVec = {};
				std::vector<torch::Tensor> targetsVec = {};
				if (batch.size() < batchSize) continue;
				for (int i = 0; i < batchSize; i++)
				{
					imagesVec.push_back(batch[i].data.to(FloatType));
					targetsVec.push_back(batch[i].target.to(FloatType));
				}
				auto data = torch::stack(imagesVec).div(255.0);

				auto outputs = detector->forward(data);
				std::vector<torch::Tensor> lossNumPos1 = critia1.forward(outputs[1], targetsVec);
				std::vector<torch::Tensor> lossNumPos2 = critia1.forward(outputs[0], targetsVec);
				auto loss = lossNumPos1[0] + lossNumPos2[0];
				auto numPos = lossNumPos1[1] + lossNumPos2[1];
				loss = loss / numPos;

				lossSum += loss.item<float>();
				batchCount++;
				lossVal = lossSum / batchCount;

				std::cout << "Epoch: " << epochCount << "," << " Valid Loss: " << lossVal << "\r";
			}
			printf("\n");
			if (bestLoss >= lossVal) 
			{
				bestLoss = lossVal;
			}
			torch::save(detector, savePath);
		}
		return 0;
	}

	float Detector::Validate(std::string valDataPath, std::string imageType, int batchSize) 
	{
		std::string valLabelPath = valDataPath + "/val/labels/";

		std::vector<std::string> listImagesVal = {};
		std::vector<std::string> listLabelsVal = {};

		loadXMLDataFromFolder(valLabelPath, imageType, listImagesVal, listLabelsVal);

		if (listImagesVal.size() < batchSize) 
		{
			std::cout << "Image numbers less than batch size or empty image folder" << std::endl;
			return 1e10; 
		}

		auto customDatasetVal = DetDataset(listImagesVal, listLabelsVal, nameList, false, width, height);
		auto dataLoaderVal = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(customDatasetVal), batchSize);

		float anchor[12] = { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };
		auto anchors_ = torch::from_blob(anchor, { 6, 2 }, torch::TensorOptions(torch::kFloat32)).to(device);
		int imageSize[2] = { width, height };

		bool normalize = false;
		auto critia1 = YOLOLossImpl(anchors_, nameList.size(), imageSize, 0.01, device, normalize);
		auto critia2 = YOLOLossImpl(anchors_, nameList.size(), imageSize, 0.01, device, normalize);

		detector->eval();  
		float lossSum = 0;
		int batchCount = 0;

		auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();

		for (auto& batch : *dataLoaderVal) 
		{
			std::vector<torch::Tensor> imagesVec = {};
			std::vector<torch::Tensor> targetsVec = {};
			if (batch.size() < batchSize) continue;

			for (int i = 0; i < batchSize; i++) 
			{
				imagesVec.push_back(batch[i].data.to(FloatType));
				targetsVec.push_back(batch[i].target.to(FloatType));
			}

			auto data = torch::stack(imagesVec).div(255.0);
			auto outputs = detector->forward(data);

			std::vector<torch::Tensor> lossNumPos1 = critia1.forward(outputs[1], targetsVec);
			std::vector<torch::Tensor> lossNumPos2 = critia2.forward(outputs[0], targetsVec);

			auto loss = lossNumPos1[0] + lossNumPos2[0];
			auto numPos = lossNumPos1[1] + lossNumPos2[1];
			loss = loss / numPos;

			lossSum += loss.item<float>();
			batchCount++;
		}

		float avgLoss = lossSum / batchCount;
		return avgLoss;
	}

	int Detector::AutoTrain(std::string trainValPath, std::string imageType, std::vector<int> epochsList, std::vector<int> batchSizes,
		std::vector<float> learningRates, std::string savePath, std::string pretrainedPath)
	{
		float bestLoss = 10;
		int bestEpochs = 0;
		int bestBatchSize = 0;
		float bestLearningRate = 0.0;
		
		int statusCode = 0;

		for (int numEpochs : epochsList) 
		{
			for (int batchSize : batchSizes) 
			{
				for (float learningRate : learningRates) 
				{
					std::cout << "Training with epochs: " << numEpochs << ", batch size: " << batchSize
						<< ", learning rate: " << learningRate << std::endl;

					statusCode = Train(trainValPath, imageType, numEpochs, batchSize, learningRate, savePath, pretrainedPath);

					// TODO: Вернуть код работы функции
					// statusCode = LoadPretrained(savePath);
					LoadPretrained(savePath);
					auto valLoss = Validate(trainValPath, imageType, batchSize); 
					
					if (valLoss < bestLoss) 
					{
						bestLoss = valLoss;
						bestEpochs = numEpochs;
						bestBatchSize = batchSize;
						bestLearningRate = learningRate;
					}
					
				}
			}
		}

		std::cout << "Best hyperparameters: " << std::endl;
		std::cout << "  Epochs: " << bestEpochs << std::endl;
		std::cout << "  Batch size: " << bestBatchSize << std::endl;
		std::cout << "  Learning rate: " << bestLearningRate << std::endl;
		std::cout << "Best validation loss: " << bestLoss << std::endl;

		statusCode = Train(trainValPath, imageType, bestEpochs, bestBatchSize, bestLearningRate, savePath, pretrainedPath);
		return statusCode;
	}

	int Detector::LoadWeight(std::string weightPath) 
	{
		try
		{
			torch::load(detector, weightPath);
		}
		catch (const std::exception& e)
		{
			std::cout << e.what();
		}
		detector->to(device);
		detector->eval();
		return EXIT_SUCCESS;
	}

	void Detector::Predict(cv::Mat image, bool show, float confThresh, float nmsThresh) 
	{
		int originWidth = image.cols;
		int originHeight = image.rows;
		cv::resize(image, image, { width,height });
		auto imgTensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte);
		imgTensor = imgTensor.permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat) / 255.0;

		float anchor[12] = { 10,14,  23,27,  37,58,  81,82,  135,169,  344,319 };
		auto anchors_ = torch::from_blob(anchor, { 6,2 }, torch::TensorOptions(torch::kFloat32));
		int imageSize[2] = { width,height };
		imgTensor = imgTensor.to(device);

		auto outputs = detector->forward(imgTensor);
		std::vector<torch::Tensor> outputList = {};
		auto tensorInput = outputs[1];
		auto outputDecoded = DecodeBox(tensorInput, anchors_.narrow(0, 0, 3), nameList.size(), imageSize);
		outputList.push_back(outputDecoded);

		tensorInput = outputs[0];
		outputDecoded = DecodeBox(tensorInput, anchors_.narrow(0, 3, 3), nameList.size(), imageSize);
		outputList.push_back(outputDecoded);

		auto output = torch::cat(outputList, 1);
		auto detection = nonMaximumSuppression(output, nameList.size(), confThresh, nmsThresh);

		float wScale = float(originWidth) / width;
		float hScale = float(originHeight) / height;

		for (int i = 0; i < detection.size(); i++) {
			for (int j = 0; j < detection[i].size(0) / 7; j++)
			{
				detection[i].select(0, 7 * j + 0) *= wScale;
				detection[i].select(0, 7 * j + 1) *= hScale;
				detection[i].select(0, 7 * j + 2) *= wScale;
				detection[i].select(0, 7 * j + 3) *= hScale;
			}
		}

		cv::resize(image, image, { originWidth,originHeight });
		if (show)
			showBbox(image, detection[0], nameList);
		return;
	}
}