#include "mrcv/mrcv-segmentation.h"
#include "mrcv/mrcv-common.h"
#include "mrcv/mrcv.h"

namespace mrcv
{
  SegmentationHeadImpl::SegmentationHeadImpl(int inChannels, int outChannels, int kernelSize, double _upsampling) {
    conv2d = torch::nn::Conv2d(conv_options(inChannels, outChannels, kernelSize, 1, kernelSize / 2));
    upsampling = torch::nn::Upsample(optionsUpsample(std::vector<double>{_upsampling, _upsampling}));
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
    else writeLog("unknown error in backbone initialization");

    encoder->load_pretrained(pretrainedPath);
    decoder = FPNDecoder(channelsEncoder, encoderDepth, decoderChannelPyramid,
      decoderChannelsSegmentation, decoder_dropout, decoderMergePolicy);
    segmentation_head = SegmentationHead(decoderChannelsSegmentation, numberClasses, 1, upsampling);

    register_module("encoder", std::shared_ptr<Backbone>(encoder));
    register_module("decoder", decoder);
    register_module("segmentation_head", segmentation_head);
  }

  torch::Tensor FPNImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> features = encoder->features(x);
    x = decoder->forward(features);
    x = segmentation_head->forward(x);
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
      x = torch::nn::Upsample(optionsUpsample(std::vector<double>{2, 2}))->forward(x);
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

    T rezult = listX[0];
    for (int i = 1; i < listX.size(); i++) {
      rezult += listX[i];
    }
    return rezult;
  }

  MergeBlockImpl::MergeBlockImpl(std::string policy) {
    if (policy != policies[0] && policy != policies[1]) {
      writeLog("policy должен быть add или cat");
    }
    _policy = policy;
  }

  torch::Tensor MergeBlockImpl::forward(std::vector<torch::Tensor> x) {
    if (_policy == "add") return sumTensor(x);
    else if (_policy == "cat") return torch::cat(x, 1);
    else
    {
      writeLog("policy должен быть add или cat");
      return torch::cat(x, 1);
    }
  }

  FPNDecoderImpl::FPNDecoderImpl(std::vector<int> channelsEncoder, int encoderDepth, int channelsPyramid, int channelsSegmentation,
    float dropout_, std::string merge_policy)
  {
    outChannels = merge_policy == "add" ? channelsSegmentation : channelsSegmentation * 4;
    if (encoderDepth < 3) writeLog("Encoder depth for FPN decoder cannot be less than 3");
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

  torch::Tensor ResNetImpl::features_at(torch::Tensor x, int numStage) {
    assert(numStage > 0 && "the stage number must in range(1,5)");
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    if (numStage == 1) return x;
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    if (numStage == 2) return x;
    x = layer2->forward(x);
    if (numStage == 3) return x;
    x = layer3->forward(x);
    if (numStage == 4) return x;
    x = layer4->forward(x);
    if (numStage == 5) return x;
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
     // std::cout << "make sure stage list len equal to dilation list len";
      writeLog("make sure stage list len equal to dilation list len");
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


  ResNet pretrained_resnet(int64_t numberClasses, std::string nameModel, std::string pathWeight) {
    std::map<std::string, std::vector<int>> name2layers = getParams();
    int groups = 1;
    int widthGroupPer = 64;
    if (nameModel == "resnext50_32x4d") {
      groups = 32; widthGroupPer = 4;
    }
    if (nameModel == "resnext101_32x8d") {
      groups = 32; widthGroupPer = 8;
    }
    ResNet net_pretrained = ResNet(name2layers[nameModel], 1000, nameModel, groups, widthGroupPer);
    torch::load(net_pretrained, pathWeight);
    if (numberClasses == 1000) return net_pretrained;
    ResNet module = ResNet(name2layers[nameModel], numberClasses, nameModel);

    torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
    torch::OrderedDict<std::string, at::Tensor> model_dict = module->named_parameters();

    for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
    {
      if (strstr((*n).key().data(), "fc.")) {
        continue;
      }
      model_dict[(*n).key()] = (*n).value();
    }

    torch::autograd::GradMode::set_enabled(false);
    auto new_params = model_dict;
    auto params = module->named_parameters(true);
    auto buffers = module->named_buffers(true);
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
    SegDataset(int widthResize, int heightResize, std::vector<std::string> listImages,
      std::vector<std::string> listLabels, std::vector<std::string> listName,
      trainTricks tricks, bool isTrain = false);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override {
      return listLabels.size();
    };
  private:
    void draw_mask(std::string pathJson, cv::Mat& mask);
    int widthResize = 512; int heightResize = 512; bool isTrain = false;
    std::vector<std::string> listName = {};
    std::map<std::string, int> name2index = {};
    std::map<std::string, cv::Scalar> name2color = {};
    std::vector<std::string> listImages;
    std::vector<std::string> listLabels;
    trainTricks tricks;
  };

  torch::Tensor DiceLoss(torch::Tensor prediction, torch::Tensor target, int classNum) {
    auto target_onehot = torch::zeros_like(prediction);
    target_onehot.scatter_(1, target, 1);

    auto roiPrediction = prediction.slice(1, 1, classNum, 1);
    auto target_roi = target_onehot.slice(1, 1, classNum, 1);
    auto intersection = (roiPrediction * target_roi).sum();
    auto union_ = roiPrediction.sum() + target_roi.sum() - intersection;
    auto dice = (intersection + 0.0001) / (union_ + 0.0001);
    return 1 - dice;
  }

  torch::Tensor CELoss(torch::Tensor prediction, torch::Tensor target) {
    return torch::nll_loss2d(torch::log_softmax(prediction, 1), target);
  }

  void Segmentor::Initialize(int gpu_id, int _width, int _height, std::vector<std::string>&& _listName,
    std::string encoderName, std::string pretrainedPath) {
    width = _width;
    height = _height;
    listName = _listName;


    if (listName.size() < 2) writeLog("Class num is less than 1");
    int gpu_num = (int)torch::getNumGPUs();
    if (gpu_id >= gpu_num) writeLog("GPU id exceeds max number of gpus");
    if (gpu_id >= 0) device = torch::Device(torch::kCUDA, gpu_id);

    fpn = FPN(listName.size(), encoderName, pretrainedPath);
    //  fpn = FPN(listName.size(),encoderName,pretrainedPath);
    fpn->to(device);
  }

  void Segmentor::SetTrainTricks(trainTricks& tricks) {
    this->tricks = tricks;
    return;
  }

  void Segmentor::Train(float learning_rate, unsigned int epochs, int batch_size,
    std::string train_val_path, std::string imageType, std::string save_path) {

    std::string train_dir = train_val_path + file_sepator() + "train";
    std::string val_dir = train_val_path + file_sepator() + "test";

    std::vector<std::string> listImagesTrain = {};
    std::vector<std::string> listLabelsTrain = {};
    std::vector<std::string> listImagesVal = {};
    std::vector<std::string> listLabelsVal = {};

    loadDataFromFolder(train_dir, imageType, listImagesTrain, listLabelsTrain);
    loadDataFromFolder(val_dir, imageType, listImagesVal, listLabelsVal);

    auto customTrainDataset = SegDataset(width, height, listImagesTrain, listLabelsTrain, \
      listName, tricks, true).map(torch::data::transforms::Stack<>());
    auto customValidDataset = SegDataset(width, height, listImagesVal, listLabelsVal, \
      listName, tricks, false).map(torch::data::transforms::Stack<>());
    auto options = torch::data::DataLoaderOptions();
    options.drop_last(true);
    options.batch_size(batch_size);
    auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(customTrainDataset), options);
    auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(customValidDataset), options);

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
        torch::Tensor prediction = fpn->forward(data);
        torch::Tensor ce_loss = CELoss(prediction, target);
        torch::Tensor dice_loss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), (int)listName.size());
        auto loss = dice_loss * tricks.dice_ce_ratio + ce_loss * (1 - tricks.dice_ce_ratio);
        loss.backward();
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
      fpn->eval();
      loss_sum = 0; batch_count = 0; dice_coef_sum = 0;
      float loss_val = 0;
      for (auto& batch : *data_loader_val) {
        auto data = batch.data;
        auto target = batch.target;
        data = data.to(torch::kF32).to(device).div(255.0);
        target = target.to(torch::kLong).to(device).squeeze(1);

        torch::Tensor prediction = fpn->forward(data);

        torch::Tensor ce_loss = CELoss(prediction, target);
        torch::Tensor dice_loss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), (int)listName.size());
        auto loss = dice_loss * tricks.dice_ce_ratio + ce_loss * (1 - tricks.dice_ce_ratio);
        loss_sum += loss.template item<float>();
        dice_coef_sum += (1 - dice_loss).item().toFloat();
        batch_count++;
        loss_val = loss_sum / batch_count / batch_size;
        auto dice_coef = dice_coef_sum / batch_count;
      }
      std::cout << std::endl;
      if (loss_val < best_loss) {
        torch::save(fpn, save_path);
        best_loss = loss_val;
      }
    }
    return;
  }

  void Segmentor::LoadWeight(std::string pathWeight) {
    torch::load(fpn, pathWeight);
    fpn->to(device);
    fpn->eval();
    return;
  }

  void Segmentor::Predict(cv::Mat& image, const std::string& which_class) {
    cv::Mat srcImg = image.clone();
    int which_class_index = -1;
    for (int i = 0; i < listName.size(); i++) {
      if (listName[i] == which_class) {
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
      cv::resize(mData.image, mData.image, cv::Size(width, height));
      cv::resize(mData.mask, mData.mask, cv::Size(width, height));
    }
    return mData;
  }

  std::vector<cv::Scalar> getListColor() {
    std::vector<cv::Scalar> listColor = {
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
    return listColor;
  }

  void SegDataset::draw_mask(std::string pathJson, cv::Mat& mask) {
    std::ifstream jfile(pathJson);
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

  SegDataset::SegDataset(int widthResize, int heightResize, std::vector<std::string> listImages,
    std::vector<std::string> listLabels, std::vector<std::string> listName,
    trainTricks tricks, bool isTrain)
  {
    this->tricks = tricks;
    this->listName = listName;
    this->widthResize = widthResize;
    this->heightResize = heightResize;
    this->listImages = listImages;
    this->listLabels = listLabels;
    this->isTrain = isTrain;
    for (int i = 0; i < listName.size(); i++) {
      name2index.insert(std::pair<std::string, int>(listName[i], i));
    }
    std::vector<cv::Scalar> listColor = getListColor();

    if (listName.size() > listColor.size()) {
      //std::cout << "Количество классов превышает определенный список цветов, пожалуйста, добавьте цвет в список цветов";
      writeLog("Количество классов превышает определенный список цветов, пожалуйста, добавьте цвет в список цветов");
    }
    for (int i = 0; i < listName.size(); i++) {
      name2color.insert(std::pair<std::string, cv::Scalar>(listName[i], listColor[i]));
    }
  }

  torch::data::Example<> SegDataset::get(size_t index) {
    std::string image_path = listImages.at(index);
    std::string label_path = listLabels.at(index);
    cv::Mat image = cv::imread(image_path);
    cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    draw_mask(label_path, mask);

    auto m_data = Data(image, mask);
    if (isTrain) {
      m_data = Augmentations::Resize(m_data, widthResize, heightResize, 1);
    }
    else {
      m_data = Augmentations::Resize(m_data, widthResize, heightResize, 1);
    }
    torch::Tensor img_tensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 });
    torch::Tensor colorful_label_tensor = torch::from_blob(m_data.mask.data, { m_data.mask.rows, m_data.mask.cols, 3 }, torch::kByte);
    torch::Tensor label_tensor = torch::zeros({ m_data.image.rows, m_data.image.cols });

    for (int i = 0; i < listName.size(); i++) {
      cv::Scalar color = name2color[listName[i]];
      torch::Tensor color_tensor = torch::tensor({ color.val[0],color.val[1],color.val[2] });
      label_tensor = label_tensor + torch::all(colorful_label_tensor == color_tensor, -1) * i;
    }
    label_tensor = label_tensor.unsqueeze(0);
    return { img_tensor.clone(), label_tensor.clone() };
  }

}
