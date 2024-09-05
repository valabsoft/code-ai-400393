#pragma once

#include <torch/torch.h>

using namespace torch;
using namespace torch::nn;
using namespace torch::indexing;

namespace mrcv
{
struct Conv : Module
{
    Conv(int c1, int c2, int k = 1, int s = 1, int p = -1, int g = 1, int d = 1,
         bool act = true);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d conv{nullptr};
    BatchNorm2d bn{nullptr};
    SiLU default_act{nullptr};
    int i;         // attach index
    int f;         // 'from' index
    std::string t; // type, default class name
    int np;        // number params
};

struct Bottleneck : Module
{
    // ch_in, ch_out, shortcut, groups, expansion
    Bottleneck(int c1, int c2, bool shortcut = true, int g = 1, float e = 0.5);

    torch::Tensor forward(torch::Tensor x);

    ModuleHolder<Conv> cv1{nullptr}, cv2{nullptr};
    bool add = false;
};

struct C3 : Module
{
    // CSP Bottleneck with 3 convolutions
    C3(int c1, int c2, int n = 1, bool shortcut = true, int g = 1,
       float e = 0.5); // ch_in, ch_out, number, shortcut, groups, expansion

    torch::Tensor forward(torch::Tensor x);

    ModuleHolder<Conv> cv1{nullptr}, cv2{nullptr}, cv3{nullptr};
    Sequential m;
    int i;         // attach index
    int f;         // 'from' index
    std::string t; // type, default class name
    int np;        // number params
};

struct SPPF : Module
{
    SPPF(int c1, int c2, int k = 5);

    torch::Tensor forward(torch::Tensor x);

    ModuleHolder<Conv> cv1{nullptr}, cv2{nullptr};
    MaxPool2d m{nullptr};
    int i;         // attach index
    int f;         // 'from' index
    std::string t; // type, default class name
    int np;        // number params
};

struct Concat : Module
{
    Concat(int dimension = 1);

    torch::Tensor forward(std::vector<Tensor> x);

    int d;
    int i;              // attach index
    std::vector<int> f; // 'from' index
    std::string t;      // type, default class name
    int np;             // number params
};

struct Proto : Module
{
    Proto(int c1, int c_ = 256, int c2 = 32);

    torch::Tensor forward(torch::Tensor x);

    ModuleHolder<Conv> cv1{nullptr}, cv2{nullptr}, cv3{nullptr};
    Upsample upsample{nullptr};
};

struct Detect : torch::nn::Module
{
    Detect(int nc_, std::vector<std::vector<int>> anchors_,
           std::vector<int> ch_, bool inplace_);

    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);

    std::tuple<torch::Tensor, torch::Tensor> _make_grid(int nx = 20,
                                                        int ny = 20, int i = 0);

    int nc;                  // number of classes
    int no;                  // number of outputs per anchor
    int nl;                  // number of detection layers
    int na;                  // number of anchors
    torch::Tensor anchors;   // shape(nl,na,2)
    torch::nn::ModuleList m; // output conv
    bool inplace;            // use inplace ops (e.g. slice assignment)
    torch::Tensor stride;    // default [8., 16., 32.]
    bool training;
    torch::Tensor t_test;

    int i;              // attach index
    std::vector<int> f; // 'from' index
    std::string t;      // type, default class name
    int np;             // number params
};

struct DetectionModel : Module
{
    DetectionModel(std::string cfg = "", int ch = 3);

    std::vector<torch::Tensor> forward(torch::Tensor x);

    std::vector<torch::Tensor> _forward_once(torch::Tensor x);

    void train(bool on = true);

    torch::nn::ModuleList module_list;
    std::map<std::string, float> hyp;
};

struct FocalLoss : Module
{
    FocalLoss(BCEWithLogitsLoss loss_fcn_, float gamma_ = 1.5,
              float alpha_ = 0.25);

    torch::Tensor forward(torch::Tensor pred_, torch::Tensor true_);

    BCEWithLogitsLoss loss_fcn; // must be nn.BCEWithLogitsLoss()
    float gamma;
    float alpha;
    at::Reduction::Reduction reduction;
};

struct ComputeLoss : Module
{
    torch::Device device;
    std::map<std::string, float> hyp;
    BCEWithLogitsLoss BCEcls, BCEobj;
    float cp, cn;
    std::vector<float> balance;
    int ssi;
    float gr;
    int na;
    int nc;
    int nl;
    torch::Tensor anchors;
    bool sort_obj_iou;
    bool autobalance;
    ComputeLoss(ModuleHolder<DetectionModel> model, bool autobalance_ = false);

    std::vector<torch::Tensor> operator()(std::vector<torch::Tensor> p,
                                          torch::Tensor targets);

    std::tuple<std::vector<Tensor>, std::vector<Tensor>,
               std::vector<std::vector<Tensor>>, std::vector<Tensor>>
    build_targets(std::vector<torch::Tensor> p, torch::Tensor targets);
};
} // namespace mrcv
