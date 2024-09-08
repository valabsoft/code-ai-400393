#include "mrcv/mrcv-segmentation.h"
#include "mrcv/mrcv-common.h"
#include "mrcv/mrcv.h"

namespace mrcv
{
  SegmentationHeadImpl::SegmentationHeadImpl(int incomingChannel, int outgoingChannel, int coreSize, double _upsampl) {
    convolution2d = torch::nn::Conv2d(conv_options(incomingChannel, outgoingChannel, coreSize, 1, coreSize / 2));
    upsampl = torch::nn::Upsample(upsample_options(std::vector<double>{_upsampl, _upsampl}));
    register_module("convolution2d", convolution2d);
  }

  torch::Tensor SegmentationHeadImpl::forward(torch::Tensor x) {
    x = convolution2d->forward(x);
    x = upsampl->forward(x);
    return x;
  }

  std::string replaceDistinct(std::string str, const std::string newExtension)
  {
    std::filesystem::path path{ str };
    return path.replace_extension(newExtension).string();
  }

  void loadDataFromFolder(std::string folder, std::string imageType,
    std::vector<std::string>& listImages, std::vector<std::string>& listLabels)
  {
    for (auto& p : std::filesystem::directory_iterator(folder))
    {
      if (std::filesystem::path(p).extension() == imageType)
      {
        listImages.push_back(p.path().string());
        listLabels.push_back(replaceDistinct(p.path().string(), ".json"));
      }
    }
  }

  nlohmann::json encoderParams() {
    nlohmann::json params = {
    {"resnet18", {
      {"class_type", "resnet"},
      {"outgoingChannel", {3, 64, 64, 128, 256, 512}},
      {"layers" , {2, 2, 2, 2}},
      },
    },
    {"resnet34", {
      {"class_type", "resnet"},
      {"outgoingChannel", {3, 64, 64, 128, 256, 512}},
      {"layers" , {3, 4, 6, 3}},
      },
    },
    };
    return params;
  }

  ///////////////////////////////////////////////////////////////////////////

  FPNImpl::FPNImpl(int _numClasses, std::string nameEncoder, std::string pathPretrained, int depthEncoder,
    int decoderChannelPyramid, int decoderChannelsSegmentation, std::string decoderMergePolicy,
    float dropoutDecoder, double upsampl) {
    numClasses = _numClasses;
    auto encoderParam = encoderParams();
    std::vector<int> encoderChannels = encoderParam[nameEncoder]["outgoingChannel"];
    if (!encoderParam.contains(nameEncoder))
      std::cout << "encoder name must in {resnet18, resnet34, resnet50, resnet101, resnet150, \
				resnext50_32x4d, resnext101_32x8d, vgg11, vgg11_bn, vgg13, vgg13_bn, \
				vgg16, vgg16_bn, vgg19, vgg19_bn,}";
    if (encoderParam[nameEncoder]["class_type"] == "resnet")
      encoder = new ResNetImpl(encoderParam[nameEncoder]["layers"], 1000, nameEncoder);
    //else if (encoderParam[nameEncoder]["class_type"] == "vgg")
    //	encoder = new VGGImpl(encoderParam[nameEncoder]["cfg"], 1000, encoderParam[nameEncoder]["batch_norm"]);
    else std::cout << "unknown error in backbone initialization";

    encoder->load_pretrained(pathPretrained);
    decoder = FPNDecoder(encoderChannels, depthEncoder, decoderChannelPyramid,
      decoderChannelsSegmentation, dropoutDecoder, decoderMergePolicy);
    segmentation_head = SegmentationHead(decoderChannelsSegmentation, numClasses, 1, upsampl);

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

  ReLUConv3x3GNImpl::ReLUConv3x3GNImpl(int _incomingChannel, int _outgoingChannel, bool _upsample) {
    upsample = _upsample;
    block = torch::nn::Sequential(torch::nn::Conv2d(conv_options(_incomingChannel, _outgoingChannel, 3, 1, 1, 1, false)),
      torch::nn::GroupNorm(torch::nn::GroupNormOptions(32, _outgoingChannel)),
      torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    register_module("block", block);
  }

  torch::Tensor ReLUConv3x3GNImpl::forward(torch::Tensor x) {
    x = block->forward(x);
    if (upsample) {
      x = torch::nn::Upsample(upsample_options(std::vector<double>{2, 2}))->forward(x);
    }
    return x;
  }

  BlockFPNImpl::BlockFPNImpl(int pyramidChannels, int skipChannels)
  {
    skipConvolution = torch::nn::Conv2d(conv_options(skipChannels, pyramidChannels, 1));
    upsample = torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({ 2,2 })).mode(torch::kNearest));

    register_module("skipConvolution", skipConvolution);
  }

  torch::Tensor BlockFPNImpl::forward(torch::Tensor x, torch::Tensor skip) {
    x = upsample->forward(x);
    skip = skipConvolution(skip);
    x = x + skip;
    return x;
  }

  BlockSegmentationImpl::BlockSegmentationImpl(int incomingChannel, int outgoingChannel, int n_upsamples)
  {
    block = torch::nn::Sequential();
    block->push_back(ReLUConv3x3GN(incomingChannel, outgoingChannel, bool(n_upsamples)));
    if (n_upsamples > 1) {
      for (int i = 1; i < n_upsamples; i++) {
        block->push_back(ReLUConv3x3GN(outgoingChannel, outgoingChannel, true));
      }
    }
    register_module("block", block);
  }

  torch::Tensor BlockSegmentationImpl::forward(torch::Tensor x) {
    x = block->forward(x);
    return x;
  }

  template<typename T>
  T sumTensor(std::vector<T> x_list) {
    if (x_list.empty()) std::cout << "sumTensor only accept non-empty list";
    T re = x_list[0];
    for (int i = 1; i < x_list.size(); i++) {
      re += x_list[i];
    }
    return re;
  }

  BlockMergeImpl::BlockMergeImpl(std::string policy) {
    if (policy != policies[0] && policy != policies[1]) {
      std::cout << "`merge_policy` must be one of: ['add', 'cat'], got " + policy;
    }
    _policy = policy;
  }

  torch::Tensor BlockMergeImpl::forward(std::vector<torch::Tensor> x) {
    if (_policy == "add") return sumTensor(x);
    else if (_policy == "cat") return torch::cat(x, 1);
    else
    {
      std::cout << "`merge_policy` must be one of: ['add', 'cat'], got " + _policy;
      return torch::cat(x, 1);
    }
  }

  FPNDecoderImpl::FPNDecoderImpl(std::vector<int> encoderChannels, int depthEncoder, int pyramidChannels, int segmentation_channels,
    float dropout_, std::string merge_policy)
  {
    outgoingChannel = merge_policy == "add" ? segmentation_channels : segmentation_channels * 4;
    if (depthEncoder < 3) std::cout << "Encoder depth for FPN decoder cannot be less than 3";
    std::reverse(std::begin(encoderChannels), std::end(encoderChannels));
    encoderChannels = std::vector<int>(encoderChannels.begin(), encoderChannels.begin() + depthEncoder + 1);
    p5 = torch::nn::Conv2d(conv_options(encoderChannels[0], pyramidChannels, 1));
    p4 = BlockFPN(pyramidChannels, encoderChannels[1]);
    p3 = BlockFPN(pyramidChannels, encoderChannels[2]);
    p2 = BlockFPN(pyramidChannels, encoderChannels[3]);
    for (int i = 3; i >= 0; i--) {
      seg_blocks->push_back(BlockSegmentation(pyramidChannels, segmentation_channels, i));
    }
    merge = BlockMerge(merge_policy);
    dropout = torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(dropout_).inplace(true));

    register_module("p5", p5);
    register_module("p4", p4);
    register_module("p3", p3);
    register_module("p2", p2);
    register_module("seg_blocks", seg_blocks);
    register_module("merge", merge);
  }

  torch::Tensor FPNDecoderImpl::forward(std::vector<torch::Tensor> features) {
    int features_len = (int)features.size();
    auto _p5 = p5->forward(features[features_len - 1]);
    auto _p4 = p4->forward(_p5, features[features_len - 2]);
    auto _p3 = p3->forward(_p4, features[features_len - 3]);
    auto _p2 = p2->forward(_p3, features[features_len - 4]);
    _p5 = seg_blocks[0]->as<BlockSegmentation>()->forward(_p5);
    _p4 = seg_blocks[1]->as<BlockSegmentation>()->forward(_p4);
    _p3 = seg_blocks[2]->as<BlockSegmentation>()->forward(_p3);
    _p2 = seg_blocks[3]->as<BlockSegmentation>()->forward(_p2);

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

    convolution1 = torch::nn::Conv2d(conv_options(inplanes, width, 3, stride_, 1, groups, false));
    BatchNorm1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
    convolution2 = torch::nn::Conv2d(conv_options(width, width, 3, 1, 1, groups, false));
    BatchNorm2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
    is_basic = _is_basic;
    if (!is_basic) {
      convolution1 = torch::nn::Conv2d(conv_options(inplanes, width, 1, 1, 0, 1, false));
      convolution2 = torch::nn::Conv2d(conv_options(width, width, 3, stride_, 1, groups, false));
      convolution3 = torch::nn::Conv2d(conv_options(width, planes * 4, 1, 1, 0, 1, false));
      BatchNorm3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * 4));
    }

    register_module("convolution1", convolution1);
    register_module("BatchNorm1", BatchNorm1);
    register_module("convolution2", convolution2);
    register_module("BatchNorm2", BatchNorm2);
    if (!is_basic) {
      register_module("convolution3", convolution3);
      register_module("BatchNorm3", BatchNorm3);
    }

    if (!downsample->is_empty()) {
      register_module("downsample", downsample);
    }
  }

  torch::Tensor BlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();

    x = convolution1->forward(x);
    x = BatchNorm1->forward(x);
    x = torch::relu(x);

    x = convolution2->forward(x);
    x = BatchNorm2->forward(x);

    if (!is_basic) {
      x = torch::relu(x);
      x = convolution3->forward(x);
      x = BatchNorm3->forward(x);
    }

    if (!downsample->is_empty()) {
      residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
  }

  ResNetImpl::ResNetImpl(std::vector<int> layers, int numClasses, std::string _model_type, int _groups, int _width_per_group)
  {
    model_type = _model_type;
    if (model_type != "resnet18" && model_type != "resnet34")
    {
      expansion = 4;
      is_basic = false;
    }
    if (model_type == "resnext50_32x4d") {
      groups = 32; base_width = 4;
    }
    if (model_type == "resnext101_32x8d") {
      groups = 32; base_width = 8;
    }
    convolution1 = torch::nn::Conv2d(conv_options(3, 64, 7, 2, 3, 1, false));
    BatchNorm1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
    layer1 = torch::nn::Sequential(_make_layer(64, layers[0]));
    layer2 = torch::nn::Sequential(_make_layer(128, layers[1], 2));
    layer3 = torch::nn::Sequential(_make_layer(256, layers[2], 2));
    layer4 = torch::nn::Sequential(_make_layer(512, layers[3], 2));

    fc = torch::nn::Linear(512 * expansion, numClasses);
    register_module("convolution1", convolution1);
    register_module("BatchNorm1", BatchNorm1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
  }

  torch::Tensor  ResNetImpl::forward(torch::Tensor x) {
    x = convolution1->forward(x);
    x = BatchNorm1->forward(x);
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

  std::vector<torch::Tensor> ResNetImpl::features(torch::Tensor x, int depthEncoder) {
    std::vector<torch::Tensor> features;
    features.push_back(x);
    x = convolution1->forward(x);
    x = BatchNorm1->forward(x);
    x = torch::relu(x);
    features.push_back(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    std::vector<torch::nn::Sequential> stages = get_stages();
    for (int i = 0; i < depthEncoder - 1; i++) {
      x = stages[i]->as<torch::nn::Sequential>()->forward(x);
      features.push_back(x);
    }
    return features;
  }

  torch::Tensor ResNetImpl::features_at(torch::Tensor x, int stage_num) {
    assert(stage_num > 0 && "the stage number must in range(1,5)");
    x = convolution1->forward(x);
    x = BatchNorm1->forward(x);
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

  void ResNetImpl::load_pretrained(std::string pathPretrained) {
    std::map<std::string, std::vector<int>> name2layers = getParams();
    ResNet net_pretrained = ResNet(name2layers[model_type], 1000, model_type, groups, base_width);
    torch::load(net_pretrained, pathPretrained);

    torch::OrderedDict<std::string, at::Tensor> pretrained_dict = net_pretrained->named_parameters();
    torch::OrderedDict<std::string, at::Tensor> model_dict = this->named_parameters();

    for (auto n = pretrained_dict.begin(); n != pretrained_dict.end(); n++)
    {
      if (strstr((*n).key().data(), "fc.")) {
        continue;
      }
      model_dict[(*n).key()] = (*n).value();
    }

    torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
    auto new_params = model_dict; // implement this
    auto params = this->named_parameters(true /*recurse*/);
    auto buffers = this->named_buffers(true /*recurse*/);
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

  void ResNetImpl::make_dilated(std::vector<int> stage_list, std::vector<int> dilation_list) {
    if (stage_list.size() != dilation_list.size()) {
      std::cout << "make sure stage list len equal to dilation list len";
      return;
    }
    std::map<int, torch::nn::Sequential> stage_dict = {};
    stage_dict.insert(std::pair<int, torch::nn::Sequential>(5, this->layer4));
    stage_dict.insert(std::pair<int, torch::nn::Sequential>(4, this->layer3));
    stage_dict.insert(std::pair<int, torch::nn::Sequential>(3, this->layer2));
    stage_dict.insert(std::pair<int, torch::nn::Sequential>(2, this->layer1));
    for (int i = 0; i < stage_list.size(); i++) {
      int dilation_rate = dilation_list[i];
      for (auto m : stage_dict[stage_list[i]]->modules()) {
        if (m->name() == "torch::nn::Conv2dImpl") {
          m->as<torch::nn::Conv2d>()->options.stride(1);
          m->as<torch::nn::Conv2d>()->options.dilation(dilation_rate);
          int coreSize = (int)(m->as<torch::nn::Conv2d>()->options.kernel_size()->at(0));
          m->as<torch::nn::Conv2d>()->options.padding((coreSize / 2) * dilation_rate);
        }
      }
    }
    return;
  }

  ResNet resnet18(int64_t numClasses) {
    std::vector<int> layers = { 2, 2, 2, 2 };
    ResNet model(layers, numClasses, "resnet18");
    return model;
  }

  ResNet resnet34(int64_t numClasses) {
    std::vector<int> layers = { 3, 4, 6, 3 };
    ResNet model(layers, numClasses, "resnet34");
    return model;
  }

  ResNet resnet50(int64_t numClasses) {
    std::vector<int> layers = { 3, 4, 6, 3 };
    ResNet model(layers, numClasses, "resnet50");
    return model;
  }

  ResNet resnet101(int64_t numClasses) {
    std::vector<int> layers = { 3, 4, 23, 3 };
    ResNet model(layers, numClasses, "resnet101");
    return model;
  }

  ResNet pretrained_resnet(int64_t numClasses, std::string model_name, std::string weight_path) {
    std::map<std::string, std::vector<int>> name2layers = getParams();
    int groups = 1;
    int width_per_group = 64;
    if (model_name == "resnext50_32x4d") {
      groups = 32; width_per_group = 4;
    }
    if (model_name == "resnext101_32x8d") {
      groups = 32; width_per_group = 8;
    }
    ResNet net_pretrained = ResNet(name2layers[model_name], 1000, model_name, groups, width_per_group);
    torch::load(net_pretrained, weight_path);
    if (numClasses == 1000) return net_pretrained;
    ResNet module = ResNet(name2layers[model_name], numClasses, model_name);

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
    SegDataset(int widthResize, int heightResize, std::vector<std::string> listImages,
      std::vector<std::string> listLabels, std::vector<std::string> listName,
      trainTricks tricks, bool isTrain = false);
    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override;
    // Return the length of data
    torch::optional<size_t> size() const override {
      return listLabels.size();
    };
  private:
    void draw_mask(std::string jsonPath, cv::Mat& mask);
    int widthResize = 512; int heightResize = 512; bool isTrain = false;
    std::vector<std::string> listName = {};
    std::map<std::string, int> name2index = {};
    std::map<std::string, cv::Scalar> name2color = {};
    std::vector<std::string> listImages;
    std::vector<std::string> listLabels;
    trainTricks tricks;
  };

  // prediction [NCHW], a tensor after softmax activation at C dim
  // target [N1HW], a tensor refer to label
  // num_class: int, equal to C, refer to class numbers, including background
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

  int Segmentor::Initialize(int gpu_id, int _width, int _height, std::vector<std::string>&& _listName,
    std::string nameEncoder, std::string pathPretrained) {
    width = _width;
    height = _height;
    listName = _listName;
    //struct stat s {};
    //lstat(pathPretrained.c_str(), &s);
    // TODO: Здесь функция должна выводить ошибку на верхний уровень
#ifdef _WIN32
    if ((_access(pathPretrained.data(), 0)) == -1)
    {
      std::cout << "Pretrained path is invalid";
      return -1;
    }
#else
    if (access(pathPretrained.data(), F_OK) != 0)
    {
      std::cout << "Pretrained path is invalid";
      return -1;
    }
#endif

    if (listName.size() < 2)
    {
      std::cout << "Class num is less than 1";
      return -1;
    }
    int gpu_num = (int)torch::getNumGPUs();
    if (gpu_id >= gpu_num) std::cout << "GPU id exceeds max number of gpus";
    if (gpu_id >= 0) device = torch::Device(torch::kCUDA, gpu_id);

    fpn = FPN(listName.size(), nameEncoder, pathPretrained);
    //  fpn = FPN(listName.size(),nameEncoder,pathPretrained);
    fpn->to(device);
    return 0;
  }

  void Segmentor::SetTrainTricks(trainTricks& tricks) {
    this->tricks = tricks;
    return;
  }

  void Segmentor::Train(float learning_rate, unsigned int epochs, int batchSize,
    std::string pathImages, std::string imageType, std::string savePath) {

    // TODO: Переписать код с использованием filesystem
    std::string pathTrain = pathImages + fileSepator() + "train";
    std::string pathVal = pathImages + fileSepator() + "test";

    std::vector<std::string> trainListImages = {};
    std::vector<std::string> trainListLabels = {};
    std::vector<std::string> valListImages = {};
    std::vector<std::string> valListLabels = {};

    loadDataFromFolder(pathTrain, imageType, trainListImages, trainListLabels);
    loadDataFromFolder(pathVal, imageType, valListImages, valListLabels);

    auto customTrainDataset = SegDataset(width, height, trainListImages, trainListLabels, \
      listName, tricks, true).map(torch::data::transforms::Stack<>());
    auto customValidationDataset = SegDataset(width, height, valListImages, valListLabels, \
      listName, tricks, false).map(torch::data::transforms::Stack<>());
    auto options = torch::data::DataLoaderOptions();
    options.drop_last(true);
    options.batch_size(batchSize);
    auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(customTrainDataset), options);
    auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(customValidationDataset), options);

    float best_loss = 1e10;
    for (unsigned int epoch = 0; epoch < epochs; epoch++) {
      float sumLoss = 0;
      int batchCount = 0;
      float trainLoss = 0;
      float diceCoefficientSum = 0;

      for (auto decay_epoch : tricks.decayEpochs) {
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
        torch::Tensor crossEntropyLoss = CELoss(prediction, target);
        torch::Tensor diceLoss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), (int)listName.size());
        auto loss = diceLoss * tricks.diceCrossEntropyRatio + crossEntropyLoss * (1 - tricks.diceCrossEntropyRatio);
        loss.backward();
        optimizer.step();
        sumLoss += loss.item().toFloat();
        diceCoefficientSum += (1 - diceLoss).item().toFloat();
        batchCount++;
        trainLoss = sumLoss / batchCount / batchSize;
        auto diceCoefficient = diceCoefficientSum / batchCount;

        std::cout << "Epoch: " << epoch << "," << " Training Loss: " << trainLoss << \
          "," << " Dice coefficient: " << diceCoefficient << "\r";
      }
      std::cout << std::endl;
      // validation part
      fpn->eval();
      sumLoss = 0; batchCount = 0; diceCoefficientSum = 0;
      float loss_val = 0;
      for (auto& batch : *data_loader_val) {
        auto data = batch.data;
        auto target = batch.target;
        data = data.to(torch::kF32).to(device).div(255.0);
        target = target.to(torch::kLong).to(device).squeeze(1);//.clamp_max(1);

        torch::Tensor prediction = fpn->forward(data);

        torch::Tensor crossEntropyLoss = CELoss(prediction, target);
        torch::Tensor diceLoss = DiceLoss(torch::softmax(prediction, 1), target.unsqueeze(1), (int)listName.size());
        auto loss = diceLoss * tricks.diceCrossEntropyRatio + crossEntropyLoss * (1 - tricks.diceCrossEntropyRatio);
        sumLoss += loss.template item<float>();
        diceCoefficientSum += (1 - diceLoss).item().toFloat();
        batchCount++;
        loss_val = sumLoss / batchCount / batchSize;
        auto diceCoefficient = diceCoefficientSum / batchCount;

        // TODO: Эта информация должна выводиться в диагностический лог-файл
        // std::cout << "Epoch: " << epoch << "," << " Validation Loss: " << loss_val << \
					"," << " Dice coefficient: " << diceCoefficient << "\r";
      }
      std::cout << std::endl;
      if (loss_val < best_loss) {
        torch::save(fpn, savePath);
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

  void SegDataset::draw_mask(std::string jsonPath, cv::Mat& mask) {
    std::ifstream jfile(jsonPath);
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
    std::vector<cv::Scalar> color_list = get_color_list();
    // TODO: Информация должна выводиться в диагностический лог-файл
    if (listName.size() > color_list.size()) {
      std::cout << "Количество классов превышает определенный список цветов, пожалуйста, добавьте цвет в список цветов";
    }
    for (int i = 0; i < listName.size(); i++) {
      name2color.insert(std::pair<std::string, cv::Scalar>(listName[i], color_list[i]));
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
      m_data = Augmentations::Resize(m_data, widthResize, heightResize, 1);
    }
    else {
      m_data = Augmentations::Resize(m_data, widthResize, heightResize, 1);
    }
    torch::Tensor img_tensor = torch::from_blob(m_data.image.data, { m_data.image.rows, m_data.image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }); // Channels x Height x Width
    torch::Tensor colorful_label_tensor = torch::from_blob(m_data.mask.data, { m_data.mask.rows, m_data.mask.cols, 3 }, torch::kByte);
    torch::Tensor label_tensor = torch::zeros({ m_data.image.rows, m_data.image.cols });

    // encode "colorful" tensor to class_index meaning tensor, [w,h,3]->[w,h], pixel value is the index of a class
    for (int i = 0; i < listName.size(); i++) {
      cv::Scalar color = name2color[listName[i]];
      torch::Tensor color_tensor = torch::tensor({ color.val[0],color.val[1],color.val[2] });
      label_tensor = label_tensor + torch::all(colorful_label_tensor == color_tensor, -1) * i;
    }
    label_tensor = label_tensor.unsqueeze(0);
    return { img_tensor.clone(), label_tensor.clone() };
  }

}
