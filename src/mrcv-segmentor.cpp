#include <mrcv/mrcv.h>
#include <iostream>
#include<torch/torch.h>
#include<opencv2/opencv.hpp>


namespace mrcv
{




class SegDataset :public torch::data::Dataset<SegDataset>
{
public:
    SegDataset(int resize_width, int resize_height, std::vector<std::string> list_images,
               std::vector<std::string> list_labels, std::vector<std::string> name_list,
			   trainTricks tricks, bool isTrain = false);
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override;
    // Return the length of data
    torch::optional<size_t> size() const override {
        return list_labels.size();
    };
private:
    void draw_mask(std::string json_path, cv::Mat &mask);
	int resize_width = 512; int resize_height = 512; bool isTrain = false;
    std::vector<std::string> name_list = {};
    std::map<std::string, int> name2index = {};
    std::map<std::string, cv::Scalar> name2color = {};
    std::vector<std::string> list_images;
    std::vector<std::string> list_labels;
	trainTricks tricks;
};

//prediction [NCHW], a tensor after softmax activation at C dim
//target [N1HW], a tensor refer to label
//num_class: int, equal to C, refer to class numbers, including background
torch::Tensor DiceLoss(torch::Tensor prediction, torch::Tensor target, int num_class) {
	auto target_onehot = torch::zeros_like(prediction); // N x C x H x W
	target_onehot.scatter_(1, target, 1);

	auto prediction_roi = prediction.slice(1, 1, num_class, 1);
	auto target_roi = target_onehot.slice(1, 1, num_class, 1);
	auto intersection = (prediction_roi*target_roi).sum();
	auto union_ = prediction_roi.sum() + target_roi.sum() - intersection;
	auto dice = (intersection + 0.0001) / (union_ + 0.0001);
	//cout << "prediction_roi: " << prediction_roi.sizes() << "\t" << "target roi: " << target_roi.sizes() << endl;
	//cout << "intersection: " << intersection << "\t" << "union: " << union_ << endl;
	//target_onehot.scatter()
	return 1 - dice;
}

//prediction [NCHW], target [NHW]
torch::Tensor CELoss(torch::Tensor prediction, torch::Tensor target) {
	return torch::nll_loss2d(torch::log_softmax(prediction, /*dim=*/1), target);
}




void Segmentor::Initialize(int gpu_id, int _width, int _height, std::vector<std::string>&& _name_list,
	std::string encoder_name, std::string pretrained_path) {
	width = _width;
	height = _height;
	name_list = _name_list;
	//std::cout << pretrained_path << std::endl;
	//struct stat s {};
	//lstat(pretrained_path.c_str(), &s);
#ifdef _WIN32
	if ((_access(pretrained_path.data(), 0)) == -1)
	{
		std::cout<< "Pretrained path is invalid";
	}
#else
	if (access(pretrained_path.data(), F_OK) != 0)
	{
		std::cout<< "Pretrained path is invalid";
	}
#endif
	if (name_list.size() < 2) std::cout<<  "Class num is less than 1";
	int gpu_num = (int)torch::getNumGPUs();
	if (gpu_id >= gpu_num) std::cout<< "GPU id exceeds max number of gpus";
	if (gpu_id >= 0) device = torch::Device(torch::kCUDA, gpu_id);

	fpn = FPN(name_list.size(), encoder_name, pretrained_path);
	//    fpn = FPN(name_list.size(),encoder_name,pretrained_path);
	fpn->to(device);
}


void Segmentor::SetTrainTricks(trainTricks &tricks) {
	this->tricks = tricks;
	return;
}


void Segmentor::Train(float learning_rate, unsigned int epochs, int batch_size,
	std::string train_val_path, std::string image_type, std::string save_path) {

	std::string train_dir = train_val_path+file_sepator()+"train";
	//std::cout << train_dir <<  std::endl;
	std::string val_dir = train_val_path+file_sepator()+"test";
	//std::cout << val_dir <<  std::endl;
	
	std::vector<std::string> list_images_train = {};
	std::vector<std::string> list_labels_train = {};
	std::vector<std::string> list_images_val = {};
	std::vector<std::string> list_labels_val = {};

	loadDataFromFolder(train_dir, image_type, list_images_train, list_labels_train);
	loadDataFromFolder(val_dir, image_type, list_images_val, list_labels_val);
	
	//load_seg_data_from_folder(train_dir, image_type, list_images_train, list_labels_train);
	//load_seg_data_from_folder(val_dir, image_type, list_images_val, list_labels_val);
	

	auto custom_dataset_train = SegDataset(width, height, list_images_train, list_labels_train, \
										   name_list, tricks, true).map(torch::data::transforms::Stack<>());
	auto custom_dataset_val = SegDataset(width, height, list_images_val, list_labels_val, \
		                                 name_list, tricks, false).map(torch::data::transforms::Stack<>());
	auto options = torch::data::DataLoaderOptions();
	options.drop_last(true);
	options.batch_size(batch_size);
	auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), options);
	auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), options);
	
	float best_loss = 1e10;
	for (unsigned int epoch = 0; epoch < epochs; epoch++) {
		float loss_sum = 0;
		int batch_count = 0;
		float loss_train = 0;
		float dice_coef_sum = 0;

		for (auto decay_epoch : tricks.decay_epochs) {
			if(decay_epoch-1 == epoch)
				learning_rate /= 10;
		}
		torch::optim::Adam optimizer(fpn->parameters(), learning_rate);
		if (epoch < tricks.freeze_epochs) {
			for (auto mm : fpn->named_parameters())
			{
				if (strstr(mm.key().data(), "encoder"))
				{
					mm.value().set_requires_grad(false);
				}
				else
				{
					mm.value().set_requires_grad(true);
				}
			}
		}
		else {
			for (auto mm : fpn->named_parameters())
			{
				mm.value().set_requires_grad(true);
			}
		}
		fpn->train();
		for (auto& batch : *data_loader_train) {
			auto data = batch.data;
			auto target = batch.target;
			data = data.to(torch::kF32).to(device).div(255.0);
			target = target.to(torch::kLong).to(device).squeeze(1);//.clamp_max(1);//if you choose clamp, all classes will be set to only one

			optimizer.zero_grad();
			// Execute the fpn
			torch::Tensor prediction = fpn->forward(data);
			// Compute loss value
			torch::Tensor ce_loss = CELoss(prediction, target);
			torch::Tensor dice_loss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), (int)name_list.size());
			auto loss = dice_loss * tricks.dice_ce_ratio + ce_loss * (1 - tricks.dice_ce_ratio);
			// Compute gradients
			loss.backward();
			// Update the parameters
			optimizer.step();
			loss_sum += loss.item().toFloat();
			dice_coef_sum += (1- dice_loss).item().toFloat();
			batch_count++;
			loss_train = loss_sum / batch_count / batch_size;
			auto dice_coef = dice_coef_sum / batch_count;

			std::cout << "Epoch: " << epoch << "," << " Training Loss: " << loss_train << \
											   "," << " Dice coefficient: " << dice_coef << "\r";
		}
		std::cout << std::endl;
		// validation part
		fpn->eval();
		loss_sum = 0; batch_count = 0; dice_coef_sum = 0;
		float loss_val = 0;
		for (auto& batch : *data_loader_val) {
			auto data = batch.data;
			auto target = batch.target;
			data = data.to(torch::kF32).to(device).div(255.0);
			target = target.to(torch::kLong).to(device).squeeze(1);//.clamp_max(1);

			// Execute the fpn
			torch::Tensor prediction = fpn->forward(data);

			// Compute loss value
			torch::Tensor ce_loss = CELoss(prediction, target);
			torch::Tensor dice_loss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), (int)name_list.size());
			auto loss = dice_loss * tricks.dice_ce_ratio + ce_loss * (1 - tricks.dice_ce_ratio);
			loss_sum += loss.template item<float>();
			dice_coef_sum += (1 - dice_loss).item().toFloat();
			batch_count++;
			loss_val = loss_sum / batch_count / batch_size;
			auto dice_coef = dice_coef_sum / batch_count;

			std::cout << "Epoch: " << epoch << "," << " Validation Loss: " << loss_val << \
											   "," << " Dice coefficient: " << dice_coef << "\r";
		}
		std::cout << std::endl;
		if (loss_val < best_loss) {
			torch::save(fpn, save_path);
			best_loss = loss_val;
		}
	}
	return;
}


void Segmentor::LoadWeight(std::string weight_path) {
	torch::load(fpn, weight_path);
	fpn->to(device);
	fpn->eval();
	return;
}


void Segmentor::Predict(cv::Mat& image, const std::string& which_class) {
	cv::Mat srcImg = image.clone();
	int which_class_index = -1;
	for (int i = 0; i < name_list.size(); i++) {
		if (name_list[i] == which_class) {
			which_class_index = i;
			break;
		}
	}
	if (which_class_index == -1) std::cout<< which_class + "not in the name list";
	int image_width = image.cols;
	int image_height = image.rows;
	cv::resize(image, image, cv::Size(width, height));
	torch::Tensor tensor_image = torch::from_blob(image.data, { 1, height, width,3 }, torch::kByte);
	tensor_image = tensor_image.to(device);
	tensor_image = tensor_image.permute({ 0,3,1,2 });
	tensor_image = tensor_image.to(torch::kFloat);
	tensor_image = tensor_image.div(255.0);

	try
	{
		at::Tensor output = fpn->forward({ tensor_image });

	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}
	at::Tensor output = fpn->forward({ tensor_image });
	output = torch::softmax(output, 1).mul(255.0).toType(torch::kByte);

	image = cv::Mat::ones(cv::Size(width, height), CV_8UC1);

	at::Tensor re = output[0][which_class_index].to(at::kCPU).detach();
	memcpy(image.data, re.data_ptr(), width * height * sizeof(unsigned char));
	cv::resize(image, image, cv::Size(image_width, image_height));

	// draw the prediction
	cv::imwrite("prediction.jpg", image);
	cv::imshow("prediction", image);
	cv::imshow("srcImage", srcImg);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return;
}



//contains mask and source image
struct Data {
	Data(cv::Mat img, cv::Mat _mask) :image(img), mask(_mask) {};
	cv::Mat image;
	cv::Mat mask;
};

class Augmentations
{
public:
	static Data Resize(Data mData, int width, int height, float probability);

};

float RandomNum(float _min, float _max)
{
	float temp;
	if (_min > _max)
	{
		temp = _max;
		_max = _min;
		_min = temp;
	}
	return rand() / (float)RAND_MAX *(_max - _min) + _min;
}


Data Augmentations::Resize(Data mData, int width, int height, float probability) {
	float rand_number = RandomNum(0, 1);
	if (rand_number <= probability) {
		// масштаб (не задействован)
		//float h_scale = height * 1.0 / mData.image.rows;
		//float w_scale = width * 1.0 / mData.image.cols;

		cv::resize(mData.image, mData.image, cv::Size(width, height));
		cv::resize(mData.mask, mData.mask, cv::Size(width, height));
	}
	return mData;
}


std::vector<cv::Scalar> get_color_list(){
    std::vector<cv::Scalar> color_list = {
        cv::Scalar(0, 0, 0),
        cv::Scalar(128, 0, 0),
        cv::Scalar(0, 128, 0),
        cv::Scalar(128, 128, 0),
        cv::Scalar(0, 0, 128),
        cv::Scalar(128, 0, 128),
        cv::Scalar(0, 128, 128),
        cv::Scalar(128, 128, 128),
        cv::Scalar(64, 0, 0),
        cv::Scalar(192, 0, 0),
        cv::Scalar(64, 128, 0),
        cv::Scalar(192, 128, 0),
        cv::Scalar(64, 0, 128),
        cv::Scalar(192, 0, 128),
        cv::Scalar(64, 128, 128),
        cv::Scalar(192, 128, 128),
        cv::Scalar(0, 64, 0),
        cv::Scalar(128, 64, 0),
        cv::Scalar(0, 192, 0),
        cv::Scalar(128, 192, 0),
        cv::Scalar(0, 64, 128),
    };
    return color_list;
}



void SegDataset::draw_mask(std::string json_path, cv::Mat &mask){
    std::ifstream jfile(json_path);
    nlohmann::json j;
    jfile >> j;
    size_t num_blobs = j["shapes"].size();


    for (int i = 0; i < num_blobs; i++)
    {
        std::string name = j["shapes"][i]["label"];
        size_t points_len = j["shapes"][i]["points"].size();
//        std::cout << name << std::endl;
        std::vector<cv::Point> contour = {};
        for (int t = 0; t < points_len; t++)
        {
            int x = (int)round(double(j["shapes"][i]["points"][t][0]));
            int y = (int)round(double(j["shapes"][i]["points"][t][1]));
//            std::cout << x << "\t" << y << std::endl;
            contour.push_back(cv::Point(x, y));
        }
        const cv::Point* ppt[1] = { contour.data() };
        int npt[] = { int(contour.size()) };
        cv::fillPoly(mask, ppt, npt, 1, name2color[name]);
    }
}

SegDataset::SegDataset(int resize_width, int resize_height, std::vector<std::string> list_images,
                       std::vector<std::string> list_labels, std::vector<std::string> name_list,
					   trainTricks tricks, bool isTrain)
{
	this->tricks = tricks;
	this->name_list = name_list;
	this->resize_width = resize_width;
	this->resize_height = resize_height;
	this->list_images = list_images;
	this->list_labels = list_labels;
	this->isTrain = isTrain;
    for(int i=0; i<name_list.size(); i++){
        name2index.insert(std::pair<std::string, int>(name_list[i], i));
    }
    std::vector<cv::Scalar> color_list = get_color_list();
    if(name_list.size()>color_list.size()){
        std::cout<< "Количество классов превышает определенный список цветов, пожалуйста, добавьте цвет в список цветов";
    }
    for(int i = 0; i<name_list.size(); i++){
        name2color.insert(std::pair<std::string, cv::Scalar>(name_list[i],color_list[i]));
    }
}

torch::data::Example<> SegDataset::get(size_t index) {
    std::string image_path = list_images.at(index);
    std::string label_path = list_labels.at(index);
    cv::Mat image = cv::imread(image_path);
    cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    draw_mask(label_path,mask);

    //Data augmentation like flip or rotate could be implemented here...
	auto m_data = Data(image, mask);
	if (isTrain) {
		m_data = Augmentations::Resize(m_data, resize_width, resize_height, 1);
	}
	else {
		m_data = Augmentations::Resize(m_data, resize_width, resize_height, 1);
	}
    torch::Tensor img_tensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
    torch::Tensor colorful_label_tensor = torch::from_blob(m_data.mask.data, { m_data.mask.rows, m_data.mask.cols, 3 }, torch::kByte);
    torch::Tensor label_tensor = torch::zeros({ m_data.image.rows, m_data.image.cols});

    //encode "colorful" tensor to class_index meaning tensor, [w,h,3]->[w,h], pixel value is the index of a class
    for(int i = 0; i<name_list.size(); i++){
        cv::Scalar color = name2color[name_list[i]];
        torch::Tensor color_tensor = torch::tensor({color.val[0],color.val[1],color.val[2]});
        label_tensor = label_tensor + torch::all(colorful_label_tensor==color_tensor,-1)*i;
    }
    label_tensor = label_tensor.unsqueeze(0);
    return { img_tensor.clone(), label_tensor.clone() };
}
}
