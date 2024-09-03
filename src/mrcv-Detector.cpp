#include <mrcv/mrcv.h>
#include <iostream>

#include<mrcv/mrcv-detector.h>


namespace mrcv
{
	torch::nn::Conv2dOptions create_conv_options(int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, bool bias) {
		return torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
			.stride(stride)
			.padding(padding)
			.dilation(dilation)
			.bias(bias);
	}

	BasicConvImpl::BasicConvImpl(int in_channels, int out_channels, int kernel_size, int stride) :
		conv(create_conv_options(in_channels, out_channels, kernel_size, stride, int(kernel_size / 2), 1, false)),
		bn(torch::nn::BatchNorm2d(out_channels)),
		activation(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)))
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

	torch::nn::MaxPool2dOptions maxpoolOptions(int kernel_size, int stride) {
		torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
		maxpool_options.stride(stride);
		return maxpool_options;
	}

	Resblock_bodyImpl::Resblock_bodyImpl(int in_channels, int out_channels) {
		this->outChannels = out_channels;
		conv1 = BasicConv(in_channels, out_channels, 3);
		conv2 = BasicConv(out_channels / 2, out_channels / 2, 3);
		conv3 = BasicConv(out_channels / 2, out_channels / 2, 3);
		conv4 = BasicConv(out_channels, out_channels, 1);
		maxpool = torch::nn::MaxPool2d(maxpoolOptions(2, 2));

		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("conv4", conv4);

	}

	std::vector<torch::Tensor> Resblock_bodyImpl::forward(torch::Tensor x) {
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

	std::vector<BBox> loadXML(std::string xml_path)
	{
		std::vector<BBox> objects;

		TiXmlDocument doc;

		if (!doc.LoadFile(xml_path.c_str()))
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

					//cout << bndbox->Value() << endl;
				}
			}
		}
		//std::cout << xml_path << std::endl;
		doc.Clear();
		return objects;
	}

	CSPdarknet53_tinyImpl::CSPdarknet53_tinyImpl() {
		conv1 = BasicConv(3, 32, 3, 2);
		conv2 = BasicConv(32, 64, 3, 2);
		resblock_body1 = Resblock_body(64, 64);
		resblock_body2 = Resblock_body(128, 128);
		resblock_body3 = Resblock_body(256, 256);
		conv3 = BasicConv(512, 512, 3);

		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("resblock_body1", resblock_body1);
		register_module("resblock_body2", resblock_body2);
		register_module("resblock_body3", resblock_body3);
		register_module("conv3", conv3);
	}
	
	std::vector<torch::Tensor> CSPdarknet53_tinyImpl::forward(torch::Tensor x) {
		// 416, 416, 3 -> 208, 208, 32 -> 104, 104, 64
		x = conv1(x);
		x = conv2(x);

		// 104, 104, 64 -> 52, 52, 128
		x = resblock_body1->forward(x)[0];
		// 52, 52, 128 -> 26, 26, 256
		x = resblock_body2->forward(x)[0];
		// 26, 26, 256->xΪ13, 13, 512
#   //        -> feat1Ϊ26,26,256
		auto res_out = resblock_body3->forward(x);
		x = res_out[0];
		auto feat1 = res_out[1];
		// 13, 13, 512 -> 13, 13, 512
		x = conv3->forward(x);
		auto feat2 = x;
		return std::vector<torch::Tensor>({ feat1, feat2 });
	}

	torch::nn::Sequential yolo_head(std::vector<int> filters_list, int in_filters) {
		auto m = torch::nn::Sequential(
			BasicConv(in_filters, filters_list[0], 3),
			torch::nn::Conv2d(create_conv_options(
				filters_list[0],        // in_channels
				filters_list[1],        // out_channels
				1,                      // kernel_size
				1,                      // stride
				0,                      // padding
				1,                      // dilation
				true                    // bias
			))
		);
		return m;
	}

	YoloBody_tinyImpl::YoloBody_tinyImpl(int num_anchors, int num_classes) {
		backbone = CSPdarknet53_tiny();
		conv_for_P5 = BasicConv(512, 256, 1);
		yolo_headP5 = yolo_head({ 512, num_anchors * (5 + num_classes) }, 256);
		upsample = Upsample(256, 128);
		yolo_headP4 = yolo_head({ 256, num_anchors * (5 + num_classes) }, 384);

		register_module("backbone", backbone);
		register_module("conv_for_P5", conv_for_P5);
		register_module("yolo_headP5", yolo_headP5);
		register_module("upsample", upsample);
		register_module("yolo_headP4", yolo_headP4);
	}
	
	std::vector<torch::Tensor> YoloBody_tinyImpl::forward(torch::Tensor x) {
		//return feat1 with shape of {26,26,256} and feat2 of {13, 13, 512}
		auto backbone_out = backbone->forward(x);
		auto feat1 = backbone_out[0];
		auto feat2 = backbone_out[1];
		//13,13,512 -> 13,13,256
		auto P5 = conv_for_P5->forward(feat2);
		//13, 13, 256 -> 13, 13, 512 -> 13, 13, 255
		auto out0 = yolo_headP5->forward(P5);


		//13,13,256 -> 13,13,128 -> 26,26,128
		auto P5_Upsample = upsample->forward(P5);
		//26, 26, 256 + 26, 26, 128 -> 26, 26, 384
		auto P4 = torch::cat({ P5_Upsample, feat1 }, 1);
		//26, 26, 384 -> 26, 26, 256 -> 26, 26, 255
		auto out1 = yolo_headP4->forward(P4);
		return std::vector<torch::Tensor>({ out0, out1 });
	}

	YOLOLossImpl::YOLOLossImpl(torch::Tensor anchors_, int num_classes_, int img_size_[],
		float label_smooth_, torch::Device device_, bool normalize) {
		this->anchors = anchors_;
		this->num_anchors = anchors_.sizes()[0];
		this->num_classes = num_classes_;
		this->bbox_attrs = 5 + num_classes;
		memcpy(image_size, img_size_, 2 * sizeof(int));
		std::vector<int> feature_length_ = { int(image_size[0] / 32),int(image_size[0] / 16),int(image_size[0] / 8) };
		std::copy(feature_length_.begin(), feature_length_.end(), feature_length.begin());
		this->label_smooth = label_smooth_;
		this->device = device_;
		this->normalize = normalize;
	}
	
	template<typename T>
	
	int vec_index(std::vector<T> vec, T value) {
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
	bool in_vec(std::vector<T> vec, T value) {
		for (auto temp : vec)
		{
			if (temp == value)
			{
				return true;
			}
		}
		return false;
	}


	torch::Tensor jaccard(torch::Tensor _box_a, torch::Tensor _box_b) {
		//auto TensorType = _box_b.options();
		auto b1_x1 = _box_a.select(1, 0) - _box_a.select(1, 2) / 2;
		auto b1_x2 = _box_a.select(1, 0) + _box_a.select(1, 2) / 2;
		auto b1_y1 = _box_a.select(1, 1) - _box_a.select(1, 3) / 2;
		auto b1_y2 = _box_a.select(1, 1) + _box_a.select(1, 3) / 2;

		auto b2_x1 = _box_b.select(1, 0) - _box_b.select(1, 2) / 2;
		auto b2_x2 = _box_b.select(1, 0) + _box_b.select(1, 2) / 2;
		auto b2_y1 = _box_b.select(1, 1) - _box_b.select(1, 3) / 2;
		auto b2_y2 = _box_b.select(1, 1) + _box_b.select(1, 3) / 2;


		auto box_a = torch::zeros_like(_box_a);
		auto box_b = torch::zeros_like(_box_b);

		box_a.select(1, 0) = b1_x1;
		box_a.select(1, 1) = b1_y1;
		box_a.select(1, 2) = b1_x2;
		box_a.select(1, 3) = b1_y2;

		box_b.select(1, 0) = b2_x1;
		box_b.select(1, 1) = b2_y1;
		box_b.select(1, 2) = b2_x2;
		box_b.select(1, 3) = b2_y2;

		auto A = box_a.size(0);
		auto B = box_b.size(0);

		//try
		//{
		//	auto max_xy = torch::min(box_a.narrow(1, 2, 2).unsqueeze(1).expand({ A, B, 2 }), box_b.narrow(1, 2, 2).unsqueeze(0).expand({ A, B, 2 }));
		//	auto min_xy = torch::max(box_a.narrow(1, 0, 2).unsqueeze(1).expand({ A, B, 2 }), box_b.narrow(1, 0, 2).unsqueeze(0).expand({ A, B, 2 }));
		//}
		//catch (const std::exception&e)
		//{
		//	cout << e.what() << endl;
		//}
		auto max_xy = torch::min(box_a.narrow(1, 2, 2).unsqueeze(1).expand({ A, B, 2 }), box_b.narrow(1, 2, 2).unsqueeze(0).expand({ A, B, 2 }));
		auto min_xy = torch::max(box_a.narrow(1, 0, 2).unsqueeze(1).expand({ A, B, 2 }), box_b.narrow(1, 0, 2).unsqueeze(0).expand({ A, B, 2 }));

		auto inter = torch::clamp((max_xy - min_xy), 0);
		inter = inter.select(2, 0) * inter.select(2, 1);

		//������������ʵ����Ե����
		auto area_a = ((box_a.select(1, 2) - box_a.select(1, 0)) * (box_a.select(1, 3) - box_a.select(1, 1))).unsqueeze(1).expand_as(inter); // [A, B]
		auto area_b = ((box_b.select(1, 2) - box_b.select(1, 0)) * (box_b.select(1, 3) - box_b.select(1, 1))).unsqueeze(0).expand_as(inter);  // [A, B]

		//��IOU
		auto uni = area_a + area_b - inter;
		return inter / uni;  // [A, B]
	}

	torch::Tensor smooth_label(torch::Tensor y_true, int label_smoothing, int num_classes) {
		return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes;
	}

	torch::Tensor box_ciou(torch::Tensor b1, torch::Tensor b2)
	{
	
		auto b1_xy = b1.narrow(-1, 0, 2);
		auto b1_wh = b1.narrow(-1, 2, 2);
		auto b1_wh_half = b1_wh / 2.0;
		auto b1_mins = b1_xy - b1_wh_half;
		auto b1_maxes = b1_xy + b1_wh_half;

		auto b2_xy = b2.narrow(-1, 0, 2);
		auto b2_wh = b2.narrow(-1, 2, 2);
		auto b2_wh_half = b2_wh / 2.0;
		auto b2_mins = b2_xy - b2_wh_half;
		auto b2_maxes = b2_xy + b2_wh_half;

		auto intersect_mins = torch::max(b1_mins, b2_mins);
		auto intersect_maxes = torch::min(b1_maxes, b2_maxes);
		auto intersect_wh = torch::max(intersect_maxes - intersect_mins, torch::zeros_like(intersect_maxes));
		auto intersect_area = intersect_wh.select(-1, 0) * intersect_wh.select(-1, 1);
		auto b1_area = b1_wh.select(-1, 0) * b1_wh.select(-1, 1);
		auto b2_area = b2_wh.select(-1, 0) * b2_wh.select(-1, 1);
		auto union_area = b1_area + b2_area - intersect_area;
		auto iou = intersect_area / torch::clamp(union_area, 1e-6);


		auto center_distance = torch::sum(torch::pow((b1_xy - b2_xy), 2), -1);

		auto enclose_mins = torch::min(b1_mins, b2_mins);
		auto enclose_maxes = torch::max(b1_maxes, b2_maxes);
		auto enclose_wh = torch::max(enclose_maxes - enclose_mins, torch::zeros_like(intersect_maxes));


		auto enclose_diagonal = torch::sum(torch::pow(enclose_wh, 2), -1);
		auto ciou = iou - 1.0 * (center_distance) / (enclose_diagonal + 1e-7);

		auto v = (4 / (Pi * Pi)) * torch::pow((torch::atan(b1_wh.select(-1, 0) / b1_wh.select(-1, 1)) - torch::atan(b2_wh.select(-1, 0) / b2_wh.select(-1, 1))), 2);
		auto alpha = v / (1.0 - iou + v);
		ciou = ciou - alpha * v;

		return ciou;
	}

	torch::Tensor clip_by_tensor(torch::Tensor t, float t_min, float t_max) {
		t = t.to(torch::kFloat32);
		auto result = (t >= t_min).to(torch::kFloat32) * t + (t < t_min).to(torch::kFloat32) * t_min;
		result = (result <= t_max).to(torch::kFloat32) * result + (result > t_max).to(torch::kFloat32) * t_max;
		return result;
	}

	torch::Tensor MSELoss(torch::Tensor pred, torch::Tensor target) {
		return torch::pow((pred - target), 2);
	}

	torch::Tensor BCELoss(torch::Tensor pred, torch::Tensor target) {
		pred = clip_by_tensor(pred, 1e-7, 1.0 - 1e-7);
		auto output = -target * torch::log(pred) - (1.0 - target) * torch::log(1.0 - pred);
		return output;
	}

	UpsampleImpl::UpsampleImpl(int in_channels, int out_channels)
	{
		upsample = torch::nn::Sequential(
			BasicConv(in_channels, out_channels, 1)
		);
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
		auto in_h = input.size(2);
		auto in_w = input.size(3);

		auto stride_h = image_size[1] / in_h;
		auto stride_w = image_size[0] / in_w;

		auto scaled_anchors = anchors.clone();
		scaled_anchors.select(1, 0) = scaled_anchors.select(1, 0) / stride_w;
		scaled_anchors.select(1, 1) = scaled_anchors.select(1, 1) / stride_h;

		auto prediction = input.view({ bs, int(num_anchors / 2), bbox_attrs, in_h, in_w }).permute({ 0, 1, 3, 4, 2 }).contiguous();
		
		auto conf = torch::sigmoid(prediction.select(-1, 4));
		auto pred_cls = torch::sigmoid(prediction.narrow(-1, 5, num_classes)); 

		auto temp = get_target(targets, scaled_anchors, in_w, in_h, ignore_threshold);
		auto BoolType = torch::ones(1).to(torch::kBool).to(device).options();
		auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();
		auto mask = temp[0].to(BoolType);
		auto noobj_mask = temp[1].to(device);
		auto t_box = temp[2];
		auto tconf = temp[3];
		auto tcls = temp[4];
		auto box_loss_scale_x = temp[5];
		auto box_loss_scale_y = temp[6];

		auto temp_ciou = get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask);
		noobj_mask = temp_ciou[0];
		auto pred_boxes_for_ciou = temp_ciou[1];


		mask = mask.to(device);
		noobj_mask = noobj_mask.to(device);
		box_loss_scale_x = box_loss_scale_x.to(device);
		box_loss_scale_y = box_loss_scale_y.to(device);
		tconf = tconf.to(device);
		tcls = tcls.to(device);
		pred_boxes_for_ciou = pred_boxes_for_ciou.to(device);
		t_box = t_box.to(device);


		auto box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y;
		auto ciou = (1 - box_ciou(pred_boxes_for_ciou.index({ mask }), t_box.index({ mask }))) * box_loss_scale.index({ mask });
		auto loss_loc = torch::sum(ciou / bs);

		auto loss_conf = torch::sum(BCELoss(conf, mask.to(FloatType)) * mask.to(FloatType) / bs) + \
			torch::sum(BCELoss(conf, mask.to(FloatType)) * noobj_mask / bs);

		auto loss_cls = torch::sum(BCELoss(pred_cls.index({ mask == 1 }), smooth_label(tcls.index({ mask == 1 }), label_smooth, num_classes)) / bs);
		auto loss = loss_conf * lambda_conf + loss_cls * lambda_cls + loss_loc * lambda_loc;

		//std::cout << mask.sum();
		//std::cout << loss.item()<< std::endl<< loss_conf<< loss_cls<< loss_loc << std::endl;
		torch::Tensor num_pos = torch::tensor({ 0 }).to(device);
		if (normalize) {
			num_pos = torch::sum(mask);
			num_pos = torch::max(num_pos, torch::ones_like(num_pos));
		}
		else
			num_pos[0] = bs / 2;
		return std::vector<torch::Tensor>({ loss, num_pos });
	}

	std::vector<torch::Tensor> YOLOLossImpl::get_target(std::vector<torch::Tensor> targets, torch::Tensor scaled_anchors, int in_w, int in_h, float ignore_threshold)
	{

		int bs = targets.size();
		auto scaled_anchorsType = scaled_anchors.options();
		
		int index = vec_index(feature_length, in_w);
		std::vector<std::vector<int>> anchor_vec_in_vec = { {3, 4, 5} ,{1, 2, 3} };
		std::vector<int> anchor_index = anchor_vec_in_vec[index];
		int subtract_index = 3 * index;

		torch::TensorOptions grad_false(torch::requires_grad(false));
		auto TensorType = targets[0].options();
		auto mask = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
		auto noobj_mask = torch::ones({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);

		auto tx = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
		auto ty = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
		auto tw = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
		auto th = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
		auto t_box = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w, 4 }, grad_false);
		auto tconf = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
		auto tcls = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w, num_classes }, grad_false);

		auto box_loss_scale_x = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
		auto box_loss_scale_y = torch::zeros({ bs, int(num_anchors / 2), in_h, in_w }, grad_false);
		for (int b = 0; b < bs; b++)
		{
			if (targets[b].sizes().size() == 1)
			{
				continue;
			}

			auto gxs = targets[b].narrow(-1, 0, 1) * in_w;
			auto gys = targets[b].narrow(-1, 1, 1) * in_h;

			auto gws = targets[b].narrow(-1, 2, 1) * in_w;
			auto ghs = targets[b].narrow(-1, 3, 1) * in_h;

			auto gis = torch::floor(gxs);
			auto gjs = torch::floor(gys);

		
			auto gt_box = torch::Tensor(torch::cat({ torch::zeros_like(gws), torch::zeros_like(ghs), gws, ghs }, 1)).to(torch::kFloat32);

			auto anchor_shapes = torch::Tensor(torch::cat({ torch::zeros({ num_anchors, 2 }).to(scaled_anchorsType), torch::Tensor(scaled_anchors) }, 1)).to(TensorType);
			
			auto anch_ious = jaccard(gt_box, anchor_shapes);

			//Find the best matching anchor box
			auto best_ns = torch::argmax(anch_ious, -1);

			for (int i = 0; i < best_ns.sizes()[0]; i++)
			{
				if (!in_vec(anchor_index, best_ns[i].item().toInt()))
				{
					continue;
				}
				auto gi = gis[i].to(torch::kLong).item().toInt();
				auto gj = gjs[i].to(torch::kLong).item().toInt();
				auto gx = gxs[i].item().toFloat();
				auto gy = gys[i].item().toFloat();
				auto gw = gws[i].item().toFloat();
				auto gh = ghs[i].item().toFloat();
				if (gj < in_h && gi < in_w) {
					auto best_n = vec_index(anchor_index, best_ns[i].item().toInt());// (best_ns[i] - subtract_index).item().toInt();

					noobj_mask[b][best_n][gj][gi] = 0;
					mask[b][best_n][gj][gi] = 1;
					
					tx[b][best_n][gj][gi] = gx;
					ty[b][best_n][gj][gi] = gy;
					
					tw[b][best_n][gj][gi] = gw;
					th[b][best_n][gj][gi] = gh;
					
					box_loss_scale_x[b][best_n][gj][gi] = targets[b][i][2];
					box_loss_scale_y[b][best_n][gj][gi] = targets[b][i][3];
					
					tconf[b][best_n][gj][gi] = 1;
					
					tcls[b][best_n][gj][gi][targets[b][i][4].item().toLong()] = 1;
				}
				else {
					std::cout << gxs << gys << std::endl;
					std::cout << gis << gjs << std::endl;
					std::cout << targets[b];
					std::cout << "Step out of boundary;" << std::endl;
				}

			}
		}
		t_box.select(-1, 0) = tx;
		t_box.select(-1, 1) = ty;
		t_box.select(-1, 2) = tw;
		t_box.select(-1, 3) = th;
		std::vector<torch::Tensor> output = { mask, noobj_mask, t_box, tconf, tcls, box_loss_scale_x, box_loss_scale_y };
		return output;
	}

	std::vector<torch::Tensor> YOLOLossImpl::get_ignore(torch::Tensor prediction, std::vector<torch::Tensor> targets, torch::Tensor scaled_anchors, int in_w, int in_h, torch::Tensor noobj_mask)
	{
		int bs = targets.size();
		int index = vec_index(feature_length, in_w);
		std::vector<std::vector<int>> anchor_vec_in_vec = { {3, 4, 5}, {0, 1, 2} };
		std::vector<int> anchor_index = anchor_vec_in_vec[index];

		auto x = torch::sigmoid(prediction.select(-1, 0));
		auto y = torch::sigmoid(prediction.select(-1, 1));

		auto w = prediction.select(-1, 2);  //Width
		auto h = prediction.select(-1, 3);  // Height

		auto FloatType = prediction.options();
		auto LongType = prediction.to(torch::kLong).options();

		auto grid_x = torch::linspace(0, in_w - 1, in_w).repeat({ in_h, 1 }).repeat(
			{ int(bs * num_anchors / 2), 1, 1 }).view(x.sizes()).to(FloatType);
		auto grid_y = torch::linspace(0, in_h - 1, in_h).repeat({ in_w, 1 }).t().repeat(
			{ int(bs * num_anchors / 2), 1, 1 }).view(y.sizes()).to(FloatType);

		auto anchor_w = scaled_anchors.narrow(0, anchor_index[0], 3).narrow(-1, 0, 1).to(FloatType);
		auto anchor_h = scaled_anchors.narrow(0, anchor_index[0], 3).narrow(-1, 1, 1).to(FloatType);
		anchor_w = anchor_w.repeat({ bs, 1 }).repeat({ 1, 1, in_h * in_w }).view(w.sizes());
		anchor_h = anchor_h.repeat({ bs, 1 }).repeat({ 1, 1, in_h * in_w }).view(h.sizes());

		auto pred_boxes = torch::randn_like(prediction.narrow(-1, 0, 4)).to(FloatType);
		pred_boxes.select(-1, 0) = x + grid_x;
		pred_boxes.select(-1, 1) = y + grid_y;

		pred_boxes.select(-1, 2) = w.exp() * anchor_w;
		pred_boxes.select(-1, 3) = h.exp() * anchor_h;

		for (int i = 0; i < bs; i++)
		{
			auto pred_boxes_for_ignore = pred_boxes[i];
			pred_boxes_for_ignore = pred_boxes_for_ignore.view({ -1, 4 });
			if (targets[i].sizes().size() > 1) {
				auto gx = targets[i].narrow(-1, 0, 1) * in_w;
				auto gy = targets[i].narrow(-1, 1, 1) * in_h;
				auto gw = targets[i].narrow(-1, 2, 1) * in_w;
				auto gh = targets[i].narrow(-1, 3, 1) * in_h;
				auto gt_box = torch::cat({ gx, gy, gw, gh }, -1).to(FloatType);

				auto anch_ious = jaccard(gt_box, pred_boxes_for_ignore);
				auto anch_ious_max_tuple = torch::max(anch_ious, 0);
				auto anch_ious_max = std::get<0>(anch_ious_max_tuple);

				anch_ious_max = anch_ious_max.view(pred_boxes.sizes().slice(1, 3));
				noobj_mask[i] = (anch_ious_max <= ignore_threshold).to(FloatType) * noobj_mask[i];
			}

		}

		std::vector<torch::Tensor> output = { noobj_mask, pred_boxes };
		return output;
	}

	std::vector<int> nms_libtorch(torch::Tensor bboxes, torch::Tensor scores, float thresh) {
		auto x1 = bboxes.select(-1, 0);
		auto y1 = bboxes.select(-1, 1);
		auto x2 = bboxes.select(-1, 2);
		auto y2 = bboxes.select(-1, 3);
		auto areas = (x2 - x1) * (y2 - y1);
		auto tuple_sorted = scores.sort(0, true);
		auto order = std::get<1>(tuple_sorted);

		std::vector<int>	keep;
		while (order.numel() > 0) {
			if (order.numel() == 1) {
				auto i = order.item();
				keep.push_back(i.toInt());
				break;
			}
			else {
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
			if (idx.numel() == 0) {
				break;
			}
			order = order.index({ idx + 1 });
		}
		return keep;
	}

	std::vector<torch::Tensor> non_maximum_suppression(torch::Tensor prediction, int num_classes, float conf_thres, float nms_thres) {

		prediction.select(-1, 0) -= prediction.select(-1, 2) / 2;
		prediction.select(-1, 1) -= prediction.select(-1, 3) / 2;
		prediction.select(-1, 2) += prediction.select(-1, 0);
		prediction.select(-1, 3) += prediction.select(-1, 1);

		std::vector<torch::Tensor> output;
		for (int image_id = 0; image_id < prediction.sizes()[0]; image_id++) {
			auto image_pred = prediction[image_id];
			auto max_out_tuple = torch::max(image_pred.narrow(-1, 5, num_classes), -1, true);
			auto class_conf = std::get<0>(max_out_tuple);
			auto class_pred = std::get<1>(max_out_tuple);
			auto conf_mask = (image_pred.select(-1, 4) * class_conf.select(-1, 0) >= conf_thres).squeeze();
			image_pred = image_pred.index({ conf_mask }).to(torch::kFloat);
			class_conf = class_conf.index({ conf_mask }).to(torch::kFloat);
			class_pred = class_pred.index({ conf_mask }).to(torch::kFloat);

			if (!image_pred.sizes()[0]) {
				output.push_back(torch::full({ 1, 7 }, 0));
				continue;
			}

			auto detections = torch::cat({ image_pred.narrow(-1,0,5), class_conf, class_pred }, 1);
	
			std::vector<torch::Tensor> img_classes;

			for (int m = 0, len = detections.size(0); m < len; m++)
			{
				bool found = false;
				for (size_t n = 0; n < img_classes.size(); n++)
				{
					auto ret = (detections[m][6] == img_classes[n]);
					if (torch::nonzero(ret).size(0) > 0)
					{
						found = true;
						break;
					}
				}
				if (!found) img_classes.push_back(detections[m][6]);
			}
			std::vector<torch::Tensor> temp_class_detections;
			for (auto c : img_classes) {
				auto detections_class = detections.index({ detections.select(-1,-1) == c });
				auto keep = nms_libtorch(detections_class.narrow(-1, 0, 4), detections_class.select(-1, 4) * detections_class.select(-1, 5), nms_thres);
				std::vector<torch::Tensor> temp_max_detections;
				for (auto v : keep) {
					temp_max_detections.push_back(detections_class[v]);
				}
				auto max_detections = torch::cat(temp_max_detections, 0);
				temp_class_detections.push_back(max_detections);
			}
			auto class_detections = torch::cat(temp_class_detections, 0);
			output.push_back(class_detections);
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

	Data Augmentations::Resize(Data mData, int width, int height, float probability) {
		float rand_number = RandomNum<float>(0, 1);
		if (rand_number <= probability) {

			float h_scale = height * 1.0 / mData.image.rows;
			float w_scale = width * 1.0 / mData.image.cols;
			for (int i = 0; i < mData.bboxes.size(); i++)
			{
				mData.bboxes[i].xmin = int(w_scale * mData.bboxes[i].xmin);
				mData.bboxes[i].xmax = int(w_scale * mData.bboxes[i].xmax);
				mData.bboxes[i].ymin = int(h_scale * mData.bboxes[i].ymin);
				mData.bboxes[i].ymax = int(h_scale * mData.bboxes[i].ymax);
			}

			cv::resize(mData.image, mData.image, cv::Size(width, height));

		}
		return mData;
	}

	Detector::Detector()
	{

	}

	void Detector::Initialize(int gpu_id, int width, int height,
		std::string name_list_path) {
		if (gpu_id >= 0) {
			if (gpu_id >= torch::getNumGPUs()) {
				std::cout << "No GPU id " << gpu_id << " available" << std::endl;
			}
			device = torch::Device(torch::kCUDA, gpu_id);
		}
		else {
			device = torch::Device(torch::kCPU);
		}
		name_list = {};
		std::ifstream ifs;
		ifs.open(name_list_path, std::ios::in);
		if (!ifs.is_open())
		{
			std::cout << "Open " << name_list_path << " file failed.";
			return;
		}
		std::string buf = "";
		while (getline(ifs, buf))
		{
			name_list.push_back(buf);
		}


		int num_classes = name_list.size();
		this->name_list = name_list;

		this->width = width;
		this->height = height;
		if (width % 32 || height % 32) {
			std::cout << "Width or height is not divisible by 32" << std::endl;
			return;
		}

		detector = YoloBody_tiny(3, num_classes);
		detector->to(device);
		return;
	}

	void Detector::loadPretrained(std::string pretrained_pth) {
		auto net_pretrained = YoloBody_tiny(3, 80);
		torch::load(net_pretrained, pretrained_pth);
		if (this->name_list.size() == 80)
		{
			detector = net_pretrained;
		}

		torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
		torch::OrderedDict<std::string, at::Tensor> model_dict = detector->named_parameters();


		for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
		{
			if (strstr((*n).key().c_str(), "yolo_head")) {
				continue;
			}
			model_dict[(*n).key()] = (*n).value();
		}

		torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
		auto new_params = model_dict; // implement this
		auto params = detector->named_parameters(true /*recurse*/);
		auto buffers = detector->named_buffers(true /*recurse*/);
		for (auto& val : new_params) {
			auto name = val.key();
			auto* t = params.find(name);
			if (t != nullptr) {
				t->copy_(val.value());
			}
			else {
				t = buffers.find(name);
				if (t != nullptr) {
					t->copy_(val.value());
				}
			}
		}
		torch::autograd::GradMode::set_enabled(true);
	}

	bool doesExist(const std::string& name) {
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	}

	// Преобразование строки в нижний регистр
	std::string to_lowercase(const std::string& src) {
		std::string dst = src;
		std::transform(dst.begin(), dst.end(), dst.begin(), ::tolower);
		return dst;
	}

	// Проверка, заканчивается ли строка на заданный суффикс
	bool ends_with(const std::string& src, const std::string& suffix) {
		return src.size() >= suffix.size() && src.compare(src.size() - suffix.size(), suffix.size(), suffix) == 0;
	}

	// Замена всех вхождений подстроки в строке
	std::string replace_all(const std::string& str, const std::string& from, const std::string& to) {
		std::string result = str;
		size_t start_pos = 0;
		while ((start_pos = result.find(from, start_pos)) != std::string::npos) {
			result.replace(start_pos, from.length(), to);
			start_pos += to.length();
		}
		return result;
	}

	// Функция для загрузки данных сегментации из папки
	void load_data_from_folder(const std::string& folder, const std::string& image_type,
		std::vector<std::string>& list_images, std::vector<std::string>& list_labels) {

		for (const auto& entry : std::filesystem::recursive_directory_iterator(folder)) {
			if (entry.is_regular_file()) {
				std::string full_path = entry.path().string();
				std::string lower_name = to_lowercase(entry.path().filename().string());

				if (ends_with(lower_name, ".json")) {
					list_labels.push_back(full_path);

					std::string image_path = replace_all(full_path, ".json", image_type);
					list_images.push_back(image_path);
				}
			}
		}
	}

	void Detector::Train(std::string train_val_path, std::string image_type, int num_epochs, int batch_size,
		float learning_rate, std::string save_path, std::string pretrained_path) {
		if (!doesExist(pretrained_path))
		{
			std::cout << "Pretrained path is invalid: " << pretrained_path << "\t random initialzed the model" << std::endl;
		}
		else {
			loadPretrained(pretrained_path);
		}

		//the datasets must be the required structure
		std::string train_label_path = train_val_path + "/train/labels";
		std::string val_label_path = train_val_path + "/val/labels";

		std::vector<std::string> list_images_train = {};
		std::vector<std::string> list_labels_train = {};
		std::vector<std::string> list_images_val = {};
		std::vector<std::string> list_labels_val = {};

		load_data_from_folder(train_label_path, image_type, list_images_train, list_labels_train);
		load_data_from_folder(val_label_path, image_type, list_images_val, list_labels_val);

		if (list_images_train.size() < batch_size || list_images_val.size() < batch_size) {
			std::cout << "Image numbers less than batch size or empty image folder" << std::endl;
			return;
		}
		if (!doesExist(list_images_train[0]))
		{
			std::cout << "Image path is invalid get first train image " << list_images_train[0] << std::endl;
			return;
		}
		auto custom_dataset_train = DetDataset(list_images_train, list_labels_train, name_list, true,
			width, height);
		auto custom_dataset_val = DetDataset(list_images_val, list_labels_val, name_list, false, width, height);
		auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), batch_size);
		auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), batch_size);

		float anchor[12] = { 10,14,  23,27,  37,58,  81,82,  135,169,  344,319 };
		auto anchors_ = torch::from_blob(anchor, { 6,2 }, torch::TensorOptions(torch::kFloat32)).to(device);
		int image_size[2] = { width, height };

		bool normalize = false;
		auto critia1 = YOLOLossImpl(anchors_, name_list.size(), image_size, 0.01, device, normalize);
		auto critia2 = YOLOLossImpl(anchors_, name_list.size(), image_size, 0.01, device, normalize);

		auto pretrained_dict = detector->named_parameters();
		auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();
		for (int epoc_count = 0; epoc_count < num_epochs; epoc_count++) {
			float loss_sum = 0;
			int batch_count = 0;
			float loss_train = 0;
			float loss_val = 0;
			float best_loss = 1e10;

			if (epoc_count == int(num_epochs / 2)) { learning_rate /= 10; }
			torch::optim::Adam optimizer(detector->parameters(), learning_rate); // Learning Rate
			if (epoc_count < int(num_epochs / 10)) {
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
			else {
				for (auto mm : pretrained_dict) {
					mm.value().set_requires_grad(true);
				}
			}
			detector->train();
			for (auto& batch : *data_loader_train) {
				std::vector<torch::Tensor> images_vec = {};
				std::vector<torch::Tensor> targets_vec = {};
				if (batch.size() < batch_size) continue;
				for (int i = 0; i < batch_size; i++)
				{
					images_vec.push_back(batch[i].data.to(FloatType));
					targets_vec.push_back(batch[i].target.to(FloatType));
				}
				auto data = torch::stack(images_vec).div(255.0);

				optimizer.zero_grad();
				auto outputs = detector->forward(data);
				std::vector<torch::Tensor> loss_numpos1 = critia1.forward(outputs[0], targets_vec);
				std::vector<torch::Tensor> loss_numpos2 = critia1.forward(outputs[1], targets_vec);

				auto loss = loss_numpos1[0] + loss_numpos2[0];
				auto num_pos = loss_numpos1[1] + loss_numpos2[1];
				loss = loss / num_pos;
				loss.backward();
				optimizer.step();
				loss_sum += loss.item().toFloat();
				batch_count++;
				loss_train = loss_sum / batch_count;

				std::cout << "Epoch: " << epoc_count << "," << " Training Loss: " << loss_train << "\r";
			}
			std::cout << std::endl;
			detector->eval();
			loss_sum = 0; batch_count = 0;
			for (auto& batch : *data_loader_val) {
				std::vector<torch::Tensor> images_vec = {};
				std::vector<torch::Tensor> targets_vec = {};
				if (batch.size() < batch_size) continue;
				for (int i = 0; i < batch_size; i++)
				{
					images_vec.push_back(batch[i].data.to(FloatType));
					targets_vec.push_back(batch[i].target.to(FloatType));
				}
				auto data = torch::stack(images_vec).div(255.0);

				auto outputs = detector->forward(data);
				std::vector<torch::Tensor> loss_numpos1 = critia1.forward(outputs[1], targets_vec);
				std::vector<torch::Tensor> loss_numpos2 = critia1.forward(outputs[0], targets_vec);
				auto loss = loss_numpos1[0] + loss_numpos2[0];
				auto num_pos = loss_numpos1[1] + loss_numpos2[1];
				loss = loss / num_pos;

				loss_sum += loss.item<float>();
				batch_count++;
				loss_val = loss_sum / batch_count;

				std::cout << "Epoch: " << epoc_count << "," << " Valid Loss: " << loss_val << "\r";
			}
			printf("\n");
			if (best_loss >= loss_val) {
				best_loss = loss_val;
				torch::save(detector, save_path);
			}
		}
	}

	float Detector::Validate(std::string val_data_path, std::string image_type, int batch_size) {
		std::string val_label_path = val_data_path + "/val/labels";

		std::vector<std::string> list_images_val = {};
		std::vector<std::string> list_labels_val = {};

		load_data_from_folder(val_label_path, image_type, list_images_val, list_labels_val);

		if (list_images_val.size() < batch_size) {
			std::cout << "Image numbers less than batch size or empty image folder" << std::endl;
			return 1e10; // Âîçâðàùàåì áîëüøîå çíà÷åíèå ëîññà, åñëè äàííûõ íåäîñòàòî÷íî
		}

		// Ñîçäàåì âàëèäàöèîííûé äàòàñåò è çàãðóç÷èê äàííûõ
		auto custom_dataset_val = DetDataset(list_images_val, list_labels_val, name_list, false, width, height);
		auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), batch_size);

		// Ïàðàìåòðû äëÿ ôóíêöèè ïîòåðü
		float anchor[12] = { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };
		auto anchors_ = torch::from_blob(anchor, { 6, 2 }, torch::TensorOptions(torch::kFloat32)).to(device);
		int image_size[2] = { width, height };

		bool normalize = false;
		auto critia1 = YOLOLossImpl(anchors_, name_list.size(), image_size, 0.01, device, normalize);
		auto critia2 = YOLOLossImpl(anchors_, name_list.size(), image_size, 0.01, device, normalize);

		// Îöåíêà ìîäåëè íà âàëèäàöèîííîì íàáîðå
		detector->eval();  // Óñòàíàâëèâàåì ìîäåëü â ðåæèì îöåíêè
		float loss_sum = 0;
		int batch_count = 0;

		auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();

		for (auto& batch : *data_loader_val) {
			std::vector<torch::Tensor> images_vec = {};
			std::vector<torch::Tensor> targets_vec = {};
			if (batch.size() < batch_size) continue;

			// Ïîäãîòîâêà äàííûõ
			for (int i = 0; i < batch_size; i++) {
				images_vec.push_back(batch[i].data.to(FloatType));
				targets_vec.push_back(batch[i].target.to(FloatType));
			}

			auto data = torch::stack(images_vec).div(255.0);
			auto outputs = detector->forward(data);

			// Âû÷èñëåíèå ëîññà
			std::vector<torch::Tensor> loss_numpos1 = critia1.forward(outputs[1], targets_vec);
			std::vector<torch::Tensor> loss_numpos2 = critia2.forward(outputs[0], targets_vec);

			auto loss = loss_numpos1[0] + loss_numpos2[0];
			auto num_pos = loss_numpos1[1] + loss_numpos2[1];
			loss = loss / num_pos;

			loss_sum += loss.item<float>();
			batch_count++;
		}

		float avg_loss = loss_sum / batch_count;
		std::cout << "Validation Loss: " << avg_loss << std::endl;
		return avg_loss;  // Âîçâðàùàåì ñðåäíåå çíà÷åíèå ëîññà
	}

	void Detector::AutoTrain(std::string train_val_path, std::string image_type,
		std::vector<int> epochs_list, std::vector<int> batch_sizes,
		std::vector<float> learning_rates, std::string save_path,
		std::string pretrained_path) {
		float best_loss = 1e10;
		int best_epochs = 0;
		int best_batch_size = 0;
		float best_learning_rate = 0.0;

		
		for (int num_epochs : epochs_list) {
			for (int batch_size : batch_sizes) {
				for (float learning_rate : learning_rates) {
					std::cout << "Training with epochs: " << num_epochs << ", batch size: " << batch_size
						<< ", learning rate: " << learning_rate << std::endl;

					// Train
					this->Train(train_val_path, image_type, num_epochs, batch_size, learning_rate, save_path, pretrained_path);

					this->loadPretrained(save_path);
					auto val_loss = this->Validate(train_val_path, image_type, batch_size); // Äîáàâüòå ìåòîä Validate

					if (val_loss < best_loss) {
						best_loss = val_loss;
						best_epochs = num_epochs;
						best_batch_size = batch_size;
						best_learning_rate = learning_rate;
					}
				}
			}
		}

		std::cout << "Best hyperparameters: " << std::endl;
		std::cout << "  Epochs: " << best_epochs << std::endl;
		std::cout << "  Batch size: " << best_batch_size << std::endl;
		std::cout << "  Learning rate: " << best_learning_rate << std::endl;
		std::cout << "Best validation loss: " << best_loss << std::endl;

		this->Train(train_val_path, image_type, best_epochs, best_batch_size, best_learning_rate, save_path, pretrained_path);
	}

	void Detector::LoadWeight(std::string weight_path) {
		try
		{
			torch::load(detector, weight_path);
		}
		catch (const std::exception& e)
		{
			std::cout << e.what();
		}
		detector->to(device);
		detector->eval();
		return;
	}

	void showBbox(cv::Mat image, torch::Tensor bboxes, std::vector<std::string> name_list) {

		int font_face = cv::FONT_HERSHEY_COMPLEX;
		double font_scale = 0.4;
		int thickness = 1;
		float* bbox = new float[bboxes.size(0)]();
		std::cout << bboxes << std::endl;
		if (bboxes.equal(torch::zeros_like(bboxes))) return;
		memcpy(bbox, bboxes.cpu().data_ptr(), bboxes.size(0) * sizeof(float));
		for (int i = 0; i < bboxes.size(0); i = i + 7)
		{
			cv::rectangle(image, cv::Rect(bbox[i + 0], bbox[i + 1], bbox[i + 2] - bbox[i + 0], bbox[i + 3] - bbox[i + 1]), cv::Scalar(0, 0, 255));

			cv::Point origin;
			origin.x = bbox[i + 0];
			origin.y = bbox[i + 1] + 8;
			cv::putText(image, name_list[bbox[i + 6]], origin, font_face, font_scale, cv::Scalar(0, 0, 255), thickness, 1, 0);
		}
		delete bbox;
		cv::imwrite("prediction.jpg", image);
		cv::imshow("test", image);
		cv::waitKey(0);
		cv::destroyAllWindows();
	}

	torch::Tensor DecodeBox(torch::Tensor input, torch::Tensor anchors, int num_classes, int img_size[])
	{
		int num_anchors = anchors.sizes()[0];
		int bbox_attrs = 5 + num_classes;
		int batch_size = input.sizes()[0];
		int input_height = input.sizes()[2];
		int input_width = input.sizes()[3];

		//416 / 13 = 32
		auto stride_h = img_size[1] / input_height;
		auto stride_w = img_size[0] / input_width;

		auto scaled_anchors = anchors.clone();
		scaled_anchors.select(1, 0) = scaled_anchors.select(1, 0) / stride_w;
		scaled_anchors.select(1, 1) = scaled_anchors.select(1, 1) / stride_h;

		auto prediction = input.view({ batch_size, num_anchors,bbox_attrs, input_height, input_width }).permute({ 0, 1, 3, 4, 2 }).contiguous();

		auto x = torch::sigmoid(prediction.select(-1, 0));
		auto y = torch::sigmoid(prediction.select(-1, 1));

		auto w = prediction.select(-1, 2); // Width
		auto h = prediction.select(-1, 3); // Height

		auto conf = torch::sigmoid(prediction.select(-1, 4));

		auto pred_cls = torch::sigmoid(prediction.narrow(-1, 5, num_classes));// Cls pred.

		auto LongType = x.clone().to(torch::kLong).options();
		auto FloatType = x.options();

		auto grid_x = torch::linspace(0, input_width - 1, input_width).repeat({ input_height, 1 }).repeat(
			{ batch_size * num_anchors, 1, 1 }).view(x.sizes()).to(FloatType);
		auto grid_y = torch::linspace(0, input_height - 1, input_height).repeat({ input_width, 1 }).t().repeat(
			{ batch_size * num_anchors, 1, 1 }).view(y.sizes()).to(FloatType);

		auto anchor_w = scaled_anchors.to(FloatType).narrow(1, 0, 1);
		auto anchor_h = scaled_anchors.to(FloatType).narrow(1, 1, 1);
		anchor_w = anchor_w.repeat({ batch_size, 1 }).repeat({ 1, 1, input_height * input_width }).view(w.sizes());
		anchor_h = anchor_h.repeat({ batch_size, 1 }).repeat({ 1, 1, input_height * input_width }).view(h.sizes());

		auto pred_boxes = torch::randn_like(prediction.narrow(-1, 0, 4)).to(FloatType);
		pred_boxes.select(-1, 0) = x + grid_x;
		pred_boxes.select(-1, 1) = y + grid_y;
		pred_boxes.select(-1, 2) = w.exp() * anchor_w;
		pred_boxes.select(-1, 3) = h.exp() * anchor_h;

		std::vector<int> scales{ stride_w, stride_h, stride_w, stride_h };
		auto _scale = torch::tensor(scales).to(FloatType);

		pred_boxes = pred_boxes.view({ batch_size, -1, 4 }) * _scale;
		conf = conf.view({ batch_size, -1, 1 });
		pred_cls = pred_cls.view({ batch_size, -1, num_classes });
		auto output = torch::cat({ pred_boxes, conf, pred_cls }, -1);
		return output;
	}

	void Detector::Predict(cv::Mat image, bool show, float conf_thresh, float nms_thresh) {
		int origin_width = image.cols;
		int origin_height = image.rows;
		cv::resize(image, image, { width,height });
		auto img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte);
		img_tensor = img_tensor.permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat) / 255.0;

		float anchor[12] = { 10,14,  23,27,  37,58,  81,82,  135,169,  344,319 };
		auto anchors_ = torch::from_blob(anchor, { 6,2 }, torch::TensorOptions(torch::kFloat32));
		int image_size[2] = { width,height };
		img_tensor = img_tensor.to(device);

		auto outputs = detector->forward(img_tensor);
		std::vector<torch::Tensor> output_list = {};
		auto tensor_input = outputs[1];
		auto output_decoded = DecodeBox(tensor_input, anchors_.narrow(0, 0, 3), name_list.size(), image_size);
		output_list.push_back(output_decoded);

		tensor_input = outputs[0];
		output_decoded = DecodeBox(tensor_input, anchors_.narrow(0, 3, 3), name_list.size(), image_size);
		output_list.push_back(output_decoded);

		auto output = torch::cat(output_list, 1);
		auto detection = non_maximum_suppression(output, name_list.size(), conf_thresh, nms_thresh);

		float w_scale = float(origin_width) / width;
		float h_scale = float(origin_height) / height;
		for (int i = 0; i < detection.size(); i++) {
			for (int j = 0; j < detection[i].size(0) / 7; j++)
			{
				detection[i].select(0, 7 * j + 0) *= w_scale;
				detection[i].select(0, 7 * j + 1) *= h_scale;
				detection[i].select(0, 7 * j + 2) *= w_scale;
				detection[i].select(0, 7 * j + 3) *= h_scale;
			}
		}

		cv::resize(image, image, { origin_width,origin_height });
		if (show)
			showBbox(image, detection[0], name_list);
		return;
	}
}