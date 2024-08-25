#ifndef CSPDARKNET53_H
#define CSPDARKNET53_H
#pragma once

#include"mrcv/mrcv-common.h"


class BasicConvImpl : public torch::nn::Module {
public:
	BasicConvImpl(int in_channels, int out_channels, int kernel_size, int stride = 1);
	torch::Tensor forward(torch::Tensor x);
private:
	// Declare layers
	torch::nn::Conv2d conv{ nullptr };
	torch::nn::BatchNorm2d bn{ nullptr };
	torch::nn::LeakyReLU acitivation{ nullptr };
}; TORCH_MODULE(BasicConv);


/*
					input
					  |
				  BasicConv
					  -----------------------
					  |                     |
				 route_group              route
					  |                     |
				  BasicConv                 |
					  |                     |
	-------------------                     |
	|                 |                     |
 route_1          BasicConv                 |
	|                 |                     |
	-----------------cat                    |
					  |                     |
		----      BasicConv                 |
		|             |                     |
	  feat           cat---------------------
					  |
				 MaxPooling2D
*/
class Resblock_bodyImpl : public torch::nn::Module {
public:
	Resblock_bodyImpl(int in_channels, int out_channels);
	std::vector<torch::Tensor> forward(torch::Tensor x);
private:
	int out_channels;
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
	int num_features = 1;
}; TORCH_MODULE(CSPdarknet53_tiny);

#endif // CSPDARKNET53_H