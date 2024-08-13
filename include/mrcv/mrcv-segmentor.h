#ifndef SEGMENTOR_H
#define SEGMENTOR_H
#pragma once
#include "mrcv-FPN.h"
#include<opencv2/opencv.hpp>
#include <sys/stat.h>
#if _WIN32
#include <io.h>
#else
#include<unistd.h>
#endif


namespace mrcv
{


	struct trainTricks {
		unsigned int freeze_epochs = 0; //freeze_epochs (unsigned int): замораживает магистраль нейронной сети во время первых freeze_epochs, по умолчанию 0;
		std::vector<unsigned int> decay_epochs = { 0 }; //decay_epochs (std::vector<unsigned int>): при каждом decay_epochs скорость обучения будет снижаться на 90 процентов, по умолчанию 0;
		float dice_ce_ratio = 0.5; //dice_ce_ratio (float): вес выпадения кубиков в общем проигрыше, по умолчанию 0,5;

		float horizontal_flip_prob = 0;//horizontal_flip_prob (float): вероятность увеличения поворота по горизонтали, по умолчанию 0;
		float vertical_flip_prob = 0; //vertical_flip_prob (float): вероятность увеличения поворота по вертикали, по умолчанию 0;
		float scale_rotate_prob = 0; //scale_rotate_prob (float): вероятность выполнения поворота и увеличения масштаба, по умолчанию 0;

		float scale_limit = (float)0.1;
		float rotate_limit = (float)45;
		int interpolation = cv::INTER_LINEAR;
		int border_mode = cv::BORDER_CONSTANT;
	};


class Segmentor
	{
	public:
		Segmentor();
		~Segmentor() {};
		void Initialize(int gpu_id, int width, int height, std::vector<std::string>&& name_list,
			std::string encoder_name, std::string pretrained_path);
		void SetTrainTricks(trainTricks &tricks);
		void Train(float learning_rate, unsigned int epochs, int batch_size,
			std::string train_val_path, std::string image_type, std::string save_path);
		void LoadWeight(std::string weight_path);
		void Predict(cv::Mat& image, const std::string& which_class);
	private:
		int width = 512;
		int height = 512;
		std::vector<std::string> name_list;
		torch::Device device = torch::Device(torch::kCPU);
		trainTricks tricks;
		//    FPN fpn{nullptr};
		//    UNet unet{nullptr};
		FPN fpn{ nullptr };
	};

	Segmentor::Segmentor()
	{
	};


}

#endif // SEGMENTOR_H
