#include <mrcv/mrcv.h>
#include <iostream>

#include<mrcv/mrcv-common.h>
#include<mrcv/mrcv-Detector.h>
#include<mrcv/mrcv-yolotiny.h>
#include<mrcv/mrcv-util.h>


namespace mrcv
{
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


	void load_det_data_from_folder(std::string folder, std::string image_type,
		std::vector<std::string>& list_images, std::vector<std::string>& list_labels)
	{
		for_each_file(folder,
			[&](const char* path, const char* name) {
				auto full_path = std::string(path).append({ file_sepator() }).append(name);
				std::string lower_name = tolower1(name);

				if (end_with(lower_name, ".xml")) {
					list_labels.push_back(full_path);
					std::string image_path = replace_all_distinct(full_path, ".xml", image_type);
					image_path = replace_all_distinct(image_path, "/train/labels", "/train/images");
					image_path = replace_all_distinct(image_path, "/val/labels", "/val/images");
					list_images.push_back(image_path);
				}
				return false;
			}
			, true
				);
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

	inline bool does_exist(const std::string& name) {
		struct stat buffer;
		return (stat(name.c_str(), &buffer) == 0);
	}

	void Detector::Train(std::string train_val_path, std::string image_type, int num_epochs, int batch_size,
		float learning_rate, std::string save_path, std::string pretrained_path) {
		if (!does_exist(pretrained_path))
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

		load_det_data_from_folder(train_label_path, image_type, list_images_train, list_labels_train);
		load_det_data_from_folder(val_label_path, image_type, list_images_val, list_labels_val);

		if (list_images_train.size() < batch_size || list_images_val.size() < batch_size) {
			std::cout << "Image numbers less than batch size or empty image folder" << std::endl;
			return;
		}
		if (!does_exist(list_images_train[0]))
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
		// Путь к папке с валидационными метками
		std::string val_label_path = val_data_path + "/val/labels";

		// Загружаем данные валидации
		std::vector<std::string> list_images_val = {};
		std::vector<std::string> list_labels_val = {};

		load_det_data_from_folder(val_label_path, image_type, list_images_val, list_labels_val);

		if (list_images_val.size() < batch_size) {
			std::cout << "Image numbers less than batch size or empty image folder" << std::endl;
			return 1e10; // Возвращаем большое значение лосса, если данных недостаточно
		}

		// Создаем валидационный датасет и загрузчик данных
		auto custom_dataset_val = DetDataset(list_images_val, list_labels_val, name_list, false, width, height);
		auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_val), batch_size);

		// Параметры для функции потерь
		float anchor[12] = { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 };
		auto anchors_ = torch::from_blob(anchor, { 6, 2 }, torch::TensorOptions(torch::kFloat32)).to(device);
		int image_size[2] = { width, height };

		bool normalize = false;
		auto critia1 = YOLOLossImpl(anchors_, name_list.size(), image_size, 0.01, device, normalize);
		auto critia2 = YOLOLossImpl(anchors_, name_list.size(), image_size, 0.01, device, normalize);

		// Оценка модели на валидационном наборе
		detector->eval();  // Устанавливаем модель в режим оценки
		float loss_sum = 0;
		int batch_count = 0;

		auto FloatType = torch::ones(1).to(torch::kFloat).to(device).options();

		for (auto& batch : *data_loader_val) {
			std::vector<torch::Tensor> images_vec = {};
			std::vector<torch::Tensor> targets_vec = {};
			if (batch.size() < batch_size) continue;

			// Подготовка данных
			for (int i = 0; i < batch_size; i++) {
				images_vec.push_back(batch[i].data.to(FloatType));
				targets_vec.push_back(batch[i].target.to(FloatType));
			}

			auto data = torch::stack(images_vec).div(255.0);
			auto outputs = detector->forward(data);

			// Вычисление лосса
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
		return avg_loss;  // Возвращаем среднее значение лосса
	}


	void Detector::AutoTrain(std::string train_val_path, std::string image_type,
		std::vector<int> epochs_list, std::vector<int> batch_sizes,
		std::vector<float> learning_rates, std::string save_path,
		std::string pretrained_path) {
		float best_loss = 1e10;
		int best_epochs = 0;
		int best_batch_size = 0;
		float best_learning_rate = 0.0;

		// Перебор всех комбинаций гиперпараметров
		for (int num_epochs : epochs_list) {
			for (int batch_size : batch_sizes) {
				for (float learning_rate : learning_rates) {
					std::cout << "Training with epochs: " << num_epochs << ", batch size: " << batch_size
						<< ", learning rate: " << learning_rate << std::endl;

					// Вызов существующего метода Train
					this->Train(train_val_path, image_type, num_epochs, batch_size, learning_rate, save_path, pretrained_path);

					// Оценка сохраненной модели
					// Загрузка лучшей модели и вычисление ошибки на валидационной выборке
					this->loadPretrained(save_path);
					auto val_loss = this->Validate(train_val_path, image_type, batch_size); // Добавьте метод Validate

					if (val_loss < best_loss) {
						best_loss = val_loss;
						best_epochs = num_epochs;
						best_batch_size = batch_size;
						best_learning_rate = learning_rate;
					}
				}
			}
		}

		// Выводим лучшие найденные гиперпараметры
		std::cout << "Best hyperparameters: " << std::endl;
		std::cout << "  Epochs: " << best_epochs << std::endl;
		std::cout << "  Batch size: " << best_batch_size << std::endl;
		std::cout << "  Learning rate: " << best_learning_rate << std::endl;
		std::cout << "Best validation loss: " << best_loss << std::endl;

		// Повторно тренируем модель с лучшими гиперпараметрами и сохраняем её
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

	void show_bbox(cv::Mat image, torch::Tensor bboxes, std::vector<std::string> name_list) {

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

		//std::cout << tensor_input << anchors_.narrow(0, 3, 3);

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
			show_bbox(image, detection[0], name_list);
		return;
	}

}
