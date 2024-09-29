#include "mrcv/mrcv-segmentation.h"
#include "mrcv/mrcv-common.h"
#include "mrcv/mrcv.h"

namespace mrcv
{
  SegmentationHeadImpl::SegmentationHeadImpl(int inChannels, int outChannels, int kernelSize, double _upsampling) {
    conv2d = torch::nn::Conv2d(conv_options(inChannels, outChannels, kernelSize, 1, kernelSize / 2));
    upsampling = torch::nn::Upsample(upsample_options(std::vector<double>{_upsampling, _upsampling}));
    register_module("conv2d", conv2d);
  }

  torch::Tensor SegmentationHeadImpl::forward(torch::Tensor x) {
    x = conv2d->forward(x);
    x = upsampling->forward(x);
    return x;
  }

  std::string replaceAll(std::string str, const std::string oldValue, const std::string newValue)
  {
    std::filesystem::path path{ str };
    return path.replace_extension(newValue).string();
  }

  void loadDataFromFolder(std::string folder, std::string imageType,
    std::vector<std::string>& listImages, std::vector<std::string>& listLabels)
  {
    for (auto& p : std::filesystem::directory_iterator(folder))
    {
      if (std::filesystem::path(p).extension() == imageType)
      {
        listImages.push_back(p.path().string());
        listLabels.push_back(replaceAll(p.path().string(), imageType, ".json"));

      }
    }
  }

  nlohmann::json encoderParameters() {
    nlohmann::json params = {
    {"resnet18", {
      {"class_type", "resnet"},
      {"outChannels", {3, 64, 64, 128, 256, 512}},
      {"layers" , {2, 2, 2, 2}},
      },
    },
    {"resnet34", {
      {"class_type", "resnet"},
      {"outChannels", {3, 64, 64, 128, 256, 512}},
      {"layers" , {3, 4, 6, 3}},
      },
    },
    {"resnet50", {
      {"class_type", "resnet"},
      {"outChannels", {3, 64, 256, 512, 1024, 2048}},
      {"layers" , {3, 4, 6, 3}},
      },
    },

    };
    return params;
  }

  ///////////////////////////////////////////////////////////////////////////

  FPNImpl::FPNImpl(int _numberClasses, std::string encoderName, std::string pretrainedPath, int encoderDepth,
    int decoderChannelPyramid, int decoderChannelsSegmentation, std::string decoderMergePolicy,
    float decoder_dropout, double upsampling) {
    numberClasses = _numberClasses;
    auto encoderParameter = encoderParameters();
    std::vector<int> channelsEncoder = encoderParameter[encoderName]["outChannels"];

    if (encoderParameter[encoderName]["class_type"] == "resnet")
      encoder = new ResNetImpl(encoderParameter[encoderName]["layers"], 1000, encoderName);
    else std::cout << "unknown error in backbone initialization";

    encoder->load_pretrained(pretrainedPath);
    decoder = FPNDecoder(channelsEncoder, encoderDepth, decoderChannelPyramid,
      decoderChannelsSegmentation, decoder_dropout, decoderMergePolicy);
    segmentHeader = SegmentationHead(decoderChannelsSegmentation, numberClasses, 1, upsampling);

    register_module("encoder", std::shared_ptr<Backbone>(encoder));
    register_module("decoder", decoder);
    register_module("segmentHeader", segmentHeader);
  }

  torch::Tensor FPNImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> features = encoder->features(x);
    x = decoder->forward(features);
    x = segmentHeader->forward(x);
    return x;
  }

  ///////////////////////////////////////////////////////////////////////////

  Conv3x3GNReLUImpl::Conv3x3GNReLUImpl(int _inChannels, int _outChannels, bool _upsample) {
    upsample = _upsample;
    block = torch::nn::Sequential(torch::nn::Conv2d(conv_options(_inChannels, _outChannels, 3, 1, 1, 1, false)),
      torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, _outChannels)),
      torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    register_module("block", block);
  }

  torch::Tensor Conv3x3GNReLUImpl::forward(torch::Tensor x) {
    x = block->forward(x);
    if (upsample) {
      x = torch::nn::Upsample(upsample_options(std::vector<double>{2, 2}))->forward(x);
    }
    return x;
  }

  FPNBlockImpl::FPNBlockImpl(int channelsPyramid, int channelsSkip)
  {
    skip_conv = torch::nn::Conv2d(conv_options(channelsSkip, channelsPyramid, 1));
    upsample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2,2 })).mode(torch::kNearest));

    register_module("skip_conv", skip_conv);
  }

  torch::Tensor FPNBlockImpl::forward(torch::Tensor x, torch::Tensor skip) {
    x = upsample->forward(x);
    skip = skip_conv(skip);
    x = x + skip;
    return x;
  }

  SegmentationBlockImpl::SegmentationBlockImpl(int inChannels, int outChannels, int upSamplesNam)
  {
    block = torch::nn::Sequential();
    block->push_back(Conv3x3GNReLU(inChannels, outChannels, bool(upSamplesNam)));
    if (upSamplesNam > 1) {
      for (int i = 1; i < upSamplesNam; i++) {
        block->push_back(Conv3x3GNReLU(outChannels, outChannels, true));
      }
    }
    register_module("block", block);
  }

  torch::Tensor SegmentationBlockImpl::forward(torch::Tensor x) {
    x = block->forward(x);
    return x;
  }

  template<typename T>
  T sumTensor(std::vector<T> listX) {

    T re = listX[0];
    for (int i = 1; i < listX.size(); i++) {
      re += listX[i];
    }
    return re;
  }

  MergeBlockImpl::MergeBlockImpl(std::string policy) {
    if (policy != policies[0] && policy != policies[1]) {
      std::cout << "policy должен быть add или cat";
    }
    _policy = policy;
  }

  torch::Tensor MergeBlockImpl::forward(std::vector<torch::Tensor> x) {
    if (_policy == "add") return sumTensor(x);
    else if (_policy == "cat") return torch::cat(x, 1);
    else
    {
      std::cout << "policy должен быть add или cat";
      return torch::cat(x, 1);
    }
  }

  FPNDecoderImpl::FPNDecoderImpl(std::vector<int> channelsEncoder, int encoderDepth, int channelsPyramid, int channelsSegmentation,
    float dropout_, std::string merge_policy)
  {
    outChannels = merge_policy == "add" ? channelsSegmentation : channelsSegmentation * 4;
    if (encoderDepth < 3) std::cout << "Encoder depth for FPN decoder cannot be less than 3";
    std::reverse(std::begin(channelsEncoder), std::end(channelsEncoder));
    channelsEncoder = std::vector<int>(channelsEncoder.begin(), channelsEncoder.begin() + encoderDepth + 1);
    p5 = torch::nn::Conv2d(conv_options(channelsEncoder[0], channelsPyramid, 1));
    p4 = FPNBlock(channelsPyramid, channelsEncoder[1]);
    p3 = FPNBlock(channelsPyramid, channelsEncoder[2]);
    p2 = FPNBlock(channelsPyramid, channelsEncoder[3]);
    for (int i = 3; i >= 0; i--) {
      seg_blocks->push_back(SegmentationBlock(channelsPyramid, channelsSegmentation, i));
    }
    merge = MergeBlock(merge_policy);
    dropout = torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(dropout_).inplace(true));

    register_module("p5", p5);
    register_module("p4", p4);
    register_module("p3", p3);
    register_module("p2", p2);
    register_module("seg_blocks", seg_blocks);
    register_module("merge", merge);
  }

  torch::Tensor FPNDecoderImpl::forward(std::vector<torch::Tensor> features) {
    int lengthFeatures = (int)features.size();
    auto _p5 = p5->forward(features[lengthFeatures - 1]);
    auto _p4 = p4->forward(_p5, features[lengthFeatures - 2]);
    auto _p3 = p3->forward(_p4, features[lengthFeatures - 3]);
    auto _p2 = p2->forward(_p3, features[lengthFeatures - 4]);
    _p5 = seg_blocks[0]->as<SegmentationBlock>()->forward(_p5);
    _p4 = seg_blocks[1]->as<SegmentationBlock>()->forward(_p4);
    _p3 = seg_blocks[2]->as<SegmentationBlock>()->forward(_p3);
    _p2 = seg_blocks[3]->as<SegmentationBlock>()->forward(_p2);

    auto x = merge->forward(std::vector<torch::Tensor>{_p5, _p4, _p3, _p2});
    x = dropout->forward(x);
    return x;
  }

  ///////////////////////////////////////////////////////////////////////////

  BlockImpl::BlockImpl(int64_t inplanes, int64_t planes, int64_t stride_,
    torch::nn::Sequential downsample_, int groups, int base_width, bool _is_basic)
  {
    downsample = downsample_;
    stride = stride_;
    int width = int(planes * (base_width / 64.)) * groups;

    conv1 = torch::nn::Conv2d(conv_options(inplanes, width, 3, stride_, 1, groups, false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
    conv2 = torch::nn::Conv2d(conv_options(width, width, 3, 1, 1, groups, false));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
    is_basic = _is_basic;
    if (!is_basic) {
      conv1 = torch::nn::Conv2d(conv_options(inplanes, width, 1, 1, 0, 1, false));
      conv2 = torch::nn::Conv2d(conv_options(width, width, 3, stride_, 1, groups, false));
      conv3 = torch::nn::Conv2d(conv_options(width, planes * 4, 1, 1, 0, 1, false));
      bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * 4));
    }

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    if (!is_basic) {
      register_module("conv3", conv3);
      register_module("bn3", bn3);
    }

    if (!downsample->is_empty()) {
      register_module("downsample", downsample);
    }
  }

  torch::Tensor BlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);

    if (!is_basic) {
      x = torch::relu(x);
      x = conv3->forward(x);
      x = bn3->forward(x);
    }

    if (!downsample->is_empty()) {
      residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
  }

  ResNetImpl::ResNetImpl(std::vector<int> layers, int numberClasses, std::string _typeModel, int _groups, int _widthGroupPer)
  {
    typeModel = _typeModel;
    if (typeModel != "resnet18" && typeModel != "resnet34")
    {
      expansion = 4;
      is_basic = false;
    }
    if (typeModel == "resnext50_32x4d") {
      groups = 32; base_width = 4;
    }
    if (typeModel == "resnext101_32x8d") {
      groups = 32; base_width = 8;
    }
    conv1 = torch::nn::Conv2d(conv_options(3, 64, 7, 2, 3, 1, false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
    layer1 = torch::nn::Sequential(_make_layer(64, layers[0]));
    layer2 = torch::nn::Sequential(_make_layer(128, layers[1], 2));
    layer3 = torch::nn::Sequential(_make_layer(256, layers[2], 2));
    layer4 = torch::nn::Sequential(_make_layer(512, layers[3], 2));

    fc = torch::nn::Linear(512 * expansion, numberClasses);
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
  }

  torch::Tensor  ResNetImpl::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = torch::avg_pool2d(x, 7, 1);
    x = x.view({ x.sizes()[0], -1 });
    x = fc->forward(x);

    return torch::log_softmax(x, 1);
  }

  std::vector<torch::nn::Sequential> ResNetImpl::get_stages() {
    std::vector<torch::nn::Sequential> ans;
    ans.push_back(this->layer1);
    ans.push_back(this->layer2);
    ans.push_back(this->layer3);
    ans.push_back(this->layer4);
    return ans;
  }

  std::vector<torch::Tensor> ResNetImpl::features(torch::Tensor x, int encoderDepth) {
    std::vector<torch::Tensor> features;
    features.push_back(x);
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    features.push_back(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    std::vector<torch::nn::Sequential> stages = get_stages();
    for (int i = 0; i < encoderDepth - 1; i++) {
      x = stages[i]->as<torch::nn::Sequential>()->forward(x);
      features.push_back(x);
    }
    return features;
  }

  torch::Tensor ResNetImpl::features_at(torch::Tensor x, int stage_num) {
    assert(stage_num > 0 && "the stage number must in range(1,5)");
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    if (stage_num == 1) return x;
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    if (stage_num == 2) return x;
    x = layer2->forward(x);
    if (stage_num == 3) return x;
    x = layer3->forward(x);
    if (stage_num == 4) return x;
    x = layer4->forward(x);
    if (stage_num == 5) return x;
    return x;
  }

  void ResNetImpl::load_pretrained(std::string pretrainedPath) {
    std::map<std::string, std::vector<int>> name2layers = getParams();
    ResNet net_pretrained = ResNet(name2layers[typeModel], 1000, typeModel, groups, base_width);
    torch::load(net_pretrained, pretrainedPath);

    torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
    torch::OrderedDict<std::string, at::Tensor> model_dict = this->named_parameters();

    for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
    {
      if (strstr((*n).key().data(), "fc.")) {
        continue;
      }
      model_dict[(*n).key()] = (*n).value();
    }

    torch::autograd::GradMode::set_enabled(false);
    auto new_params = model_dict;
    auto params = this->named_parameters(true);
    auto buffers = this->named_buffers(true);
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
    return;
  }

  torch::nn::Sequential ResNetImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {

    torch::nn::Sequential downsample;
    if (stride != 1 || inplanes != planes * expansion) {
      downsample = torch::nn::Sequential(
        torch::nn::Conv2d(conv_options(inplanes, planes * expansion, 1, stride, 0, 1, false)),
        torch::nn::BatchNorm2d(planes * expansion)
      );
    }
    torch::nn::Sequential layers;
    layers->push_back(Block(inplanes, planes, stride, downsample, groups, base_width, is_basic));
    inplanes = planes * expansion;
    for (int64_t i = 1; i < blocks; i++) {
      layers->push_back(Block(inplanes, planes, 1, torch::nn::Sequential(), groups, base_width, is_basic));
    }

    return layers;
  }

  void ResNetImpl::make_dilated(std::vector<int> listStage, std::vector<int> listDilation) {
    if (listStage.size() != listDilation.size()) {
      std::cout << "make sure stage list len equal to dilation list len";
      return;
    }
    std::map<int, torch::nn::Sequential> stage_dict = {};
    stage_dict.insert(std::pair<int, torch::nn::Sequential>(5, this->layer4));
    stage_dict.insert(std::pair<int, torch::nn::Sequential>(4, this->layer3));
    stage_dict.insert(std::pair<int, torch::nn::Sequential>(3, this->layer2));
    stage_dict.insert(std::pair<int, torch::nn::Sequential>(2, this->layer1));
    for (int i = 0; i < listStage.size(); i++) {
      int rateDilation = listDilation[i];
      for (auto m : stage_dict[listStage[i]]->modules()) {
        if (m->name() == "torch::nn::Conv2dImpl") {
          m->as<torch::nn::Conv2d>()->options.stride(1);
          m->as<torch::nn::Conv2d>()->options.dilation(rateDilation);
          int kernelSize = (int)(m->as<torch::nn::Conv2d>()->options.kernel_size()->at(0));
          m->as<torch::nn::Conv2d>()->options.padding((kernelSize / 2) * rateDilation);
        }
      }
    }
    return;
  }

  ResNet resnet18(int64_t numberClasses) {
    std::vector<int> layers = { 2, 2, 2, 2 };
    ResNet model(layers, numberClasses, "resnet18");
    return model;
  }

  ResNet resnet34(int64_t numberClasses) {
    std::vector<int> layers = { 3, 4, 6, 3 };
    ResNet model(layers, numberClasses, "resnet34");
    return model;
  }

  ResNet resnet50(int64_t numberClasses) {
    std::vector<int> layers = { 3, 4, 6, 3 };
    ResNet model(layers, numberClasses, "resnet50");
    return model;
  }

  ResNet resnet101(int64_t numberClasses) {
    std::vector<int> layers = { 3, 4, 23, 3 };
    ResNet model(layers, numberClasses, "resnet101");
    return model;
  }

  ResNet pretrained_resnet(int64_t numberClasses, std::string model_name, std::string weight_path) {
    std::map<std::string, std::vector<int>> name2layers = getParams();
    int groups = 1;
    int widthGroupPer = 64;
    if (model_name == "resnext50_32x4d") {
      groups = 32; widthGroupPer = 4;
    }
    if (model_name == "resnext101_32x8d") {
      groups = 32; widthGroupPer = 8;
    }
    ResNet net_pretrained = ResNet(name2layers[model_name], 1000, model_name, groups, widthGroupPer);
    torch::load(net_pretrained, weight_path);
    if (numberClasses == 1000) return net_pretrained;
    ResNet module = ResNet(name2layers[model_name], numberClasses, model_name);

    torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
    torch::OrderedDict<std::string, at::Tensor> model_dict = module->named_parameters();

    for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
    {
      if (strstr((*n).key().data(), "fc.")) {
        continue;
      }
      model_dict[(*n).key()] = (*n).value();
    }

    torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
    auto new_params = model_dict; // implement this
    auto params = module->named_parameters(true /*recurse*/);
    auto buffers = module->named_buffers(true /*recurse*/);
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
    return module;
  }

  ///////////////////////////////////////////////////////////////////////////

  class SegDataset :public torch::data::Dataset<SegDataset>
  {
  public:
    SegDataset(int resize_width, int resize_height, std::vector<std::string> listImages,
      std::vector<std::string> listLabels, std::vector<std::string> name_list,
      trainTricks tricks, bool isTrain = false);
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override;
    // Return the length of data
    torch::optional<size_t> size() const override {
      return listLabels.size();
    };
  private:
    void draw_mask(std::string json_path, cv::Mat& mask);
    int resize_width = 512; int resize_height = 512; bool isTrain = false;
    std::vector<std::string> name_list = {};
    std::map<std::string, int> name2index = {};
    std::map<std::string, cv::Scalar> name2color = {};
    std::vector<std::string> listImages;
    std::vector<std::string> listLabels;
    trainTricks tricks;
  };

  torch::Tensor DiceLoss(torch::Tensor prediction, torch::Tensor target, int num_class) {
    auto target_onehot = torch::zeros_like(prediction); // N x C x H x W
    target_onehot.scatter_(1, target, 1);

    auto prediction_roi = prediction.slice(1, 1, num_class, 1);
    auto target_roi = target_onehot.slice(1, 1, num_class, 1);
    auto intersection = (prediction_roi * target_roi).sum();
    auto union_ = prediction_roi.sum() + target_roi.sum() - intersection;
    auto dice = (intersection + 0.0001) / (union_ + 0.0001);
    return 1 - dice;
  }

  // prediction [NCHW], target [NHW]
  torch::Tensor CELoss(torch::Tensor prediction, torch::Tensor target) {
    return torch::nll_loss2d(torch::log_softmax(prediction, /*dim=*/1), target);
  }

  void Segmentor::Initialize(int gpu_id, int _width, int _height, std::vector<std::string>&& _name_list,
    std::string encoderName, std::string pretrainedPath) {
    width = _width;
    height = _height;
    name_list = _name_list;
    //struct stat s {};
    //lstat(pretrainedPath.c_str(), &s);
    // TODO: Здесь функция должна выводить ошибку на верхний уровень
#ifdef _WIN32
    if ((_access(pretrainedPath.data(), 0)) == -1)
    {
      std::cout << "Pretrained path is invalid";
    }
#else
    if (access(pretrainedPath.data(), F_OK) != 0)
    {
      std::cout << "Pretrained path is invalid";
    }
#endif

    if (name_list.size() < 2) std::cout << "Class num is less than 1";
    int gpu_num = (int)torch::getNumGPUs();
    if (gpu_id >= gpu_num) std::cout << "GPU id exceeds max number of gpus";
    if (gpu_id >= 0) device = torch::Device(torch::kCUDA, gpu_id);

    fpn = FPN(name_list.size(), encoderName, pretrainedPath);
    //  fpn = FPN(name_list.size(),encoderName,pretrainedPath);
    fpn->to(device);
  }

  void Segmentor::SetTrainTricks(trainTricks& tricks) {
    this->tricks = tricks;
    return;
  }

  void Segmentor::Train(float learning_rate, unsigned int epochs, int batch_size,
    std::string train_val_path, std::string imageType, std::string save_path) {

    // TODO: Переписать код с использованием filesystem
    std::string train_dir = train_val_path + file_sepator() + "train";
    std::string val_dir = train_val_path + file_sepator() + "test";

    std::vector<std::string> listImages_train = {};
    std::vector<std::string> listLabels_train = {};
    std::vector<std::string> listImages_val = {};
    std::vector<std::string> listLabels_val = {};

    loadDataFromFolder(train_dir, imageType, listImages_train, listLabels_train);
    loadDataFromFolder(val_dir, imageType, listImages_val, listLabels_val);

    auto custom_dataset_train = SegDataset(width, height, listImages_train, listLabels_train, \
      name_list, tricks, true).map(torch::data::transforms::Stack<>());
    auto custom_dataset_val = SegDataset(width, height, listImages_val, listLabels_val, \
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
        if (decay_epoch - 1 == epoch)
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
        target = target.to(torch::kLong).to(device).squeeze(1);

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
        dice_coef_sum += (1 - dice_loss).item().toFloat();
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

        // TODO: Эта информация должна выводиться в диагностический лог-файл
        // std::cout << "Epoch: " << epoch << "," << " Validation Loss: " << loss_val << \
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
    if (which_class_index == -1) std::cout << which_class + "not in the name list";
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

  // contains mask and source image
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
    return rand() / (float)RAND_MAX * (_max - _min) + _min;
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

  std::vector<cv::Scalar> get_color_list() {
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

  void SegDataset::draw_mask(std::string json_path, cv::Mat& mask) {
    std::ifstream jfile(json_path);
    nlohmann::json j;
    jfile >> j;
    size_t num_blobs = j["shapes"].size();


    for (int i = 0; i < num_blobs; i++)
    {
      std::string name = j["shapes"][i]["label"];
      size_t points_len = j["shapes"][i]["points"].size();
      std::vector<cv::Point> contour = {};
      for (int t = 0; t < points_len; t++)
      {
        int x = (int)round(double(j["shapes"][i]["points"][t][0]));
        int y = (int)round(double(j["shapes"][i]["points"][t][1]));
        contour.push_back(cv::Point(x, y));
      }
      const cv::Point* ppt[1] = { contour.data() };
      int npt[] = { int(contour.size()) };
      cv::fillPoly(mask, ppt, npt, 1, name2color[name]);
    }
  }

  SegDataset::SegDataset(int resize_width, int resize_height, std::vector<std::string> listImages,
    std::vector<std::string> listLabels, std::vector<std::string> name_list,
    trainTricks tricks, bool isTrain)
  {
    this->tricks = tricks;
    this->name_list = name_list;
    this->resize_width = resize_width;
    this->resize_height = resize_height;
    this->listImages = listImages;
    this->listLabels = listLabels;
    this->isTrain = isTrain;
    for (int i = 0; i < name_list.size(); i++) {
      name2index.insert(std::pair<std::string, int>(name_list[i], i));
    }
    std::vector<cv::Scalar> color_list = get_color_list();
    // TODO: Информация должна выводиться в диагностический лог-файл
    if (name_list.size() > color_list.size()) {
      std::cout << "Количество классов превышает определенный список цветов, пожалуйста, добавьте цвет в список цветов";
    }
    for (int i = 0; i < name_list.size(); i++) {
      name2color.insert(std::pair<std::string, cv::Scalar>(name_list[i], color_list[i]));
    }
  }

  torch::data::Example<> SegDataset::get(size_t index) {
    std::string image_path = listImages.at(index);
    std::string label_path = listLabels.at(index);
    cv::Mat image = cv::imread(image_path);
    cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    draw_mask(label_path, mask);

    // Data augmentation like flip or rotate could be implemented here...
    auto m_data = Data(image, mask);
    if (isTrain) {
      m_data = Augmentations::Resize(m_data, resize_width, resize_height, 1);
    }
    else {
      m_data = Augmentations::Resize(m_data, resize_width, resize_height, 1);
    }
    torch::Tensor img_tensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
    torch::Tensor colorful_label_tensor = torch::from_blob(m_data.mask.data, { m_data.mask.rows, m_data.mask.cols, 3 }, torch::kByte);
    torch::Tensor label_tensor = torch::zeros({ m_data.image.rows, m_data.image.cols });

    // encode "colorful" tensor to class_index meaning tensor, [w,h,3]->[w,h], pixel value is the index of a class
    for (int i = 0; i < name_list.size(); i++) {
      cv::Scalar color = name2color[name_list[i]];
      torch::Tensor color_tensor = torch::tensor({ color.val[0],color.val[1],color.val[2] });
      label_tensor = label_tensor + torch::all(colorful_label_tensor == color_tensor, -1) * i;
    }
    label_tensor = label_tensor.unsqueeze(0);
    return { img_tensor.clone(), label_tensor.clone() };
  }

}
