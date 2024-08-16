#include <mrcv/mrcv-yolov5.h>
#include <torch/script.h>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>

namespace mrcv
{
static int make_divisible(float x, int divisor)
{
    return ceil(x / divisor) * divisor;
}

static int autopad(int k, int p = -1, int d = 1)
{
    // Pad to 'same' shape outputs
    if (d > 1)
        k = d * (k - 1) + 1;
    if (p == -1)
        p = static_cast<int>(k / 2);
    return p;
}

static torch::nn::ModuleList parse_model(std::string f = "yolov5s.yaml",
                                         int img_channels = 3)
{
    torch::nn::ModuleList module_list;
    YAML::Node config = YAML::LoadFile(f);
    std::vector<std::vector<int>> anchors;
    int nc = config["nc"].as<int>();
    float gd = config["depth_multiple"].as<float>();
    float gw = config["width_multiple"].as<float>();

    anchors = config["anchors"].as<std::vector<std::vector<int>>>();

    int na = anchors[0].size() / 2; // number of anchors
    int no = na * (nc + 5); // number of outputs = anchors * (classes + 5)

    std::vector<int> ch{img_channels};

    auto backbone = config["backbone"];
    for (std::size_t i = 0; i < backbone.size(); i++)
    {
        int c2;
        int from = backbone[i][0].as<int>();
        int number = backbone[i][1].as<int>();
        if (number > 1)
        {
            number = round(number * gd);
            number = std::max(number, 1);
        }

        std::string mdtype = backbone[i][2].as<std::string>();
        std::vector<int> args = backbone[i][3].as<std::vector<int>>();
        if (mdtype == "Conv")
        {
            int c1 = ch.back();
            c2 = args[0];
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            std::vector<int> args_{c1, c2};
            args_.insert(args_.end(), args.begin() + 1, args.end());
            {
                int c1;
                int c2;
                int k = 1;
                int s = 1;
                int p = -1;
                int g = 1;
                int d = 1;
                bool act = true;

                int n = args_.size();
                switch (n)
                {
                case 5: {
                    p = args_[4];
                }
                case 4: {
                    s = args_[3];
                }
                case 3: {
                    k = args_[2];
                }
                case 2: {
                    c2 = args_[1];
                }
                case 1: {
                    c1 = args_[0];
                }
                }
                if (number > 1)
                {
                    Sequential m;
                    for (size_t j = 0; j < number; j++)
                    {
                        m->push_back(torch::nn::ModuleHolder<Conv>(
                            c1, c2, k, s, p, g, d, act));
                    }
                    module_list->push_back(m);
                }
                else
                {
                    auto layer = torch::nn::ModuleHolder<Conv>(c1, c2, k, s, p,
                                                               g, d, act);
                    layer->i = i;
                    layer->f = from;
                    layer->t = layer->name();
                    layer->np = 0;
                    auto np = layer->parameters();
                    for (const auto &p : np)
                    {
                        layer->np += p.numel();
                    }
                    module_list->push_back(layer);
                }
            }
        }
        else if (mdtype == "C3")
        {
            int c1 = ch.back();
            c2 = args[0];
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            std::vector<int> args_{c1, c2};
            args_.insert(args_.end(), args.begin() + 1, args.end());
            args_.insert(args_.begin() + 2, number);
            {
                int c1;
                int c2;
                int n = 1;
                bool shortcut = true;
                int g = 1;
                float e = 0.5;
                switch (args_.size())
                {
                case 3: {
                    n = args_[2];
                }
                case 2: {
                    c2 = args_[1];
                }
                case 1: {
                    c1 = args_[0];
                }
                }
                auto layer =
                    torch::nn::ModuleHolder<C3>(c1, c2, n, shortcut, g, e);
                layer->i = i;
                layer->f = from;
                layer->t = layer->name();
                layer->np = 0;
                auto np = layer->parameters();
                for (const auto &p : np)
                {
                    layer->np += p.numel();
                }
                module_list->push_back(layer);
            }
        }
        else if (mdtype == "SPPF")
        {
            int c1 = ch.back();
            c2 = args[0];
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            std::vector<int> args_{c1, c2};
            args_.insert(args_.end(), args.begin() + 1, args.end());
            {
                int c1;
                int c2;
                int k = 5;
                switch (args_.size())
                {
                case 3: {
                    k = args_[2];
                }
                case 2: {
                    c2 = args_[1];
                }
                case 1: {
                    c1 = args_[0];
                }
                }
                auto layer = torch::nn::ModuleHolder<SPPF>(c1, c2, k);
                layer->i = i;
                layer->f = from;
                layer->t = layer->name();
                layer->np = 0;
                auto np = layer->parameters();
                for (const auto &p : np)
                {
                    layer->np += p.numel();
                }
                module_list->push_back(layer);
            }
        }

        if (i == 0)
            ch.clear();
        ch.push_back(c2);
    }

    auto head = config["head"];
    for (std::size_t i = 0; i < head.size(); i++)
    {
        int number = head[i][1].as<int>();
        if (number > 1)
        {
            number = round(number * gd);
            number = std::max(number, 1);
        }

        std::string mdtype = head[i][2].as<std::string>();
        if (mdtype == "Conv")
        {
            int c1 = ch.back();
            std::vector<int> args = head[i][3].as<std::vector<int>>();
            int c2 = args[0];
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            ch.push_back(c2);
            std::vector<int> args_{c1, c2};
            args_.insert(args_.end(), args.begin() + 1, args.end());
            {
                int c1;
                int c2;
                int k = 1;
                int s = 1;
                int p = -1;
                int g = 1;
                int d = 1;
                bool act = true;

                int n = args_.size();
                switch (n)
                {
                case 5: {
                    p = args_[4];
                }
                case 4: {
                    s = args_[3];
                }
                case 3: {
                    k = args_[2];
                }
                case 2: {
                    c2 = args_[1];
                }
                case 1: {
                    c1 = args_[0];
                }
                }
                if (number > 1)
                {
                    Sequential m;
                    for (size_t j = 0; j < number; j++)
                    {
                        m->push_back(torch::nn::ModuleHolder<Conv>(
                            c1, c2, k, s, p, g, d, act));
                    }
                    module_list->push_back(m);
                }
                else
                {
                    int from = head[i][0].as<int>();
                    auto layer = torch::nn::ModuleHolder<Conv>(c1, c2, k, s, p,
                                                               g, d, act);
                    layer->i = i + backbone.size();
                    layer->f = from;
                    layer->t = layer->name();
                    layer->np = 0;
                    auto np = layer->parameters();
                    for (const auto &p : np)
                    {
                        layer->np += p.numel();
                    }
                    module_list->push_back(layer);
                }
            }
        }
        else if (mdtype == "nn.Upsample")
        {
            int c2 = ch.back();
            ch.push_back(c2);
            YAML::Node args = head[i][3];
            int scale_factor = args[1].as<int>();
            if (number > 1)
            {
                Sequential m;
                for (size_t j = 0; j < number; j++)
                {
                    m->push_back(Upsample(UpsampleOptions()
                                              .scale_factor(std::vector<double>(
                                                  {double(scale_factor)}))
                                              .mode(torch::kNearest)));
                }
                module_list->push_back(m);
            }
            else
            {
                int from = head[i][0].as<int>();
                auto layer = Upsample(
                    UpsampleOptions()
                        .scale_factor(std::vector<double>(
                            {double(scale_factor), double(scale_factor)}))
                        .mode(torch::kNearest));
                module_list->push_back(layer);
            }
        }
        else if (mdtype == "Concat")
        {
            int c2 = 0;
            std::vector<int> from = head[i][0].as<std::vector<int>>();
            for (size_t i = 0; i < from.size(); i++)
            {
                if (from[i] == -1)
                    c2 += ch.back();
                else
                    c2 += ch[from[i]];
            }
            ch.push_back(c2);
            YAML::Node args = head[i][3];
            int d = args[0].as<int>();
            if (number > 1)
            {
                Sequential m;
                for (size_t j = 0; j < number; j++)
                {
                    m->push_back(torch::nn::ModuleHolder<Concat>(d));
                }
                module_list->push_back(m);
            }
            else
            {
                auto layer = torch::nn::ModuleHolder<Concat>(d);
                layer->i = i + backbone.size();
                layer->f = from;
                layer->t = layer->name();
                layer->np = 0;
                auto np = layer->parameters();
                for (const auto &p : np)
                {
                    layer->np += p.numel();
                }
                module_list->push_back(layer);
            }
        }
        else if (mdtype == "C3")
        {
            int c1 = ch.back();
            YAML::Node args = head[i][3];
            int c2 = args[0].as<int>();
            if (c2 != no)
            {
                c2 = make_divisible(c2 * gw, 8);
            }
            ch.push_back(c2);
            std::vector<int> args_{c1, c2};
            args_.insert(args_.begin() + 2, number);
            {
                int c1;
                int c2;
                int n = 1;
                bool shortcut = false;
                int g = 1;
                float e = 0.5;
                switch (args_.size())
                {
                case 3: {
                    n = args_[2];
                }
                case 2: {
                    c2 = args_[1];
                }
                case 1: {
                    c1 = args_[0];
                }
                }
                int from = head[i][0].as<int>();
                auto layer =
                    torch::nn::ModuleHolder<C3>(c1, c2, n, shortcut, g, e);
                layer->i = i + backbone.size();
                layer->f = from;
                layer->t = layer->name();
                layer->np = 0;
                auto np = layer->parameters();
                for (const auto &p : np)
                {
                    layer->np += p.numel();
                }
                module_list->push_back(layer);
            }
        }
        else if (mdtype == "Detect")
        {
            std::vector<int> ch_;
            std::vector<int> from = head[i][0].as<std::vector<int>>();
            for (size_t i = 0; i < from.size(); i++)
            {
                ch_.push_back(ch[from[i]]);
            }
            auto layer =
                torch::nn::ModuleHolder<Detect>(nc, anchors, ch_, true);
            layer->i = i + backbone.size();
            layer->f = from;
            layer->t = layer->name();
            layer->np = 0;
            auto np = layer->parameters();
            for (const auto &p : np)
            {
                layer->np += p.numel();
            }
            module_list->push_back(layer);
        }
    }

    return module_list;
}

static Tensor bbox_iou(Tensor box1, Tensor box2, bool xywh = true,
                       bool GIoU = false, bool DIoU = false, bool CIoU = false,
                       float eps = 1e-7)
{
    Tensor x1, y1, w1, h1, x2, y2, w2, h2;
    Tensor b1_x1, b1_x2, b1_y1, b1_y2, b2_x1, b2_x2, b2_y1, b2_y2;
    if (xywh)
    {
        auto vec0 = box1.chunk(4, -1);
        x1 = vec0[0];
        y1 = vec0[1];
        w1 = vec0[2];
        h1 = vec0[3];
        auto vec1 = box2.chunk(4, -1);
        x2 = vec1[0];
        y2 = vec1[1];
        w2 = vec1[2];
        h2 = vec1[3];

        auto w1_ = w1 / 2;
        auto h1_ = h1 / 2;
        auto w2_ = w2 / 2;
        auto h2_ = h2 / 2;

        b1_x1 = x1 - w1_;
        b1_x2 = x1 + w1_;
        b1_y1 = y1 - h1_;
        b1_y2 = y1 + h1_;

        b2_x1 = x2 - w2_;
        b2_x2 = x2 + w2_;
        b2_y1 = y2 - h2_;
        b2_y2 = y2 + h2_;
    }
    else
    {
        auto vec0 = box1.chunk(4, -1);
        b1_x1 = vec0[0];
        b1_y1 = vec0[1];
        b1_x2 = vec0[2];
        b1_y2 = vec0[3];
        auto vec1 = box2.chunk(4, -1);
        b2_x1 = vec1[0];
        b2_y1 = vec1[1];
        b2_x2 = vec1[2];
        b2_y2 = vec1[3];
        w1 = b1_x2 - b1_x1;
        h1 = b1_y2 - b1_y1 + eps;
        w2 = b2_x2 - b2_x1;
        h2 = b2_y2 - b2_y1 + eps;
    }

    auto t0_w = torch::min(b1_x2, b2_x2) - torch::max(b1_x1, b2_x1);
    t0_w = t0_w.clamp(0);
    auto t0_h = torch::min(b1_y2, b2_y2) - torch::max(b1_y1, b2_y1);
    t0_h = t0_h.clamp(0);

    auto inter = t0_w * t0_h;
    auto unions = w1 * h1 + w2 * h2 - inter + eps;

    auto iou = inter / unions;
    if (CIoU || DIoU || GIoU)
    {
        auto cw = torch::max(b1_x2, b2_x2) - torch::min(b1_x1, b2_x1);
        auto ch = torch::max(b1_y2, b2_y2) - torch::min(b1_y1, b2_y1);
        if (CIoU || DIoU)
        {
            auto c2 = cw.pow(2) + ch.pow(2) + eps;
            auto rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
                         (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) /
                        4;
            if (CIoU)
            {
                auto v =
                    (4 / (M_PI * M_PI)) *
                    torch::pow(torch::atan(w2 / h2) - torch::atan(w1 / h1), 2);
                auto alpha = v / (v - iou + (1 + eps));
                return iou - (rho2 / c2 + v * alpha);
            }
            return iou - rho2 / c2;
        }
        auto c_area = cw * ch + eps;
        return iou - (c_area - unions) / c_area;
    }
    return iou;
}

static std::tuple<float, float> smooth_BCE(float eps = 0.1)
{
    return std::tuple<float, float>(1.0 - 0.5 * eps, 0.5 * eps);
}

Conv::Conv(int c1, int c2, int k, int s, int p, int g, int d, bool act)
{
    conv =
        register_module("conv", torch::nn::Conv2d(Conv2dOptions(c1, c2, k)
                                                      .stride(s)
                                                      .padding(autopad(k, p, d))
                                                      .bias(false)
                                                      .groups(g)
                                                      .dilation(d)));

    bn = register_module("bn", BatchNorm2d(BatchNorm2dOptions(c2)));
    default_act = register_module("act", SiLU());
}

torch::Tensor Conv::forward(torch::Tensor x)
{
    return default_act(bn(conv(x)));
}

Bottleneck::Bottleneck(int c1, int c2, bool shortcut, int g, float e)
{
    int c_ = static_cast<int>(c2 * e);
    cv1 = ModuleHolder<Conv>(c1, c_, 1, 1);
    cv2 = ModuleHolder<Conv>(c_, c2, 3, 1, -1, g);

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    add = shortcut && (c1 == c2);
}

torch::Tensor Bottleneck::forward(torch::Tensor x)
{
    if (add)
    {
        return x + cv2(cv1(x));
    }
    else
    {
        return cv2(cv1(x));
    }
}

C3::C3(int c1, int c2, int n, bool shortcut, int g, float e)
{
    int c_ = static_cast<int>(c2 * e);
    cv1 = ModuleHolder<Conv>(c1, c_, 1, 1);
    cv2 = ModuleHolder<Conv>(c1, c_, 1, 1);
    cv3 = ModuleHolder<Conv>(2 * c_, c2, 1);
    for (size_t i = 0; i < n; i++)
    {
        m->push_back(ModuleHolder<Bottleneck>(c_, c_, shortcut, g, 1.0));
    }
    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
    register_module("m", m);
}

torch::Tensor C3::forward(torch::Tensor x)
{
    return cv3(torch::cat({m->forward(cv1(x)), cv2(x)}, 1));
}

SPPF::SPPF(int c1, int c2, int k)
{
    int c_ = static_cast<int>(c1 / 2);
    cv1 = ModuleHolder<Conv>(c1, c_, 1, 1);
    cv2 = ModuleHolder<Conv>(c_ * 4, c2, 1, 1);
    m = MaxPool2d(
        MaxPool2dOptions(k).stride(1).padding(static_cast<int>(k / 2)));

    register_module("cv1", cv1);
    register_module("cv2", cv2);
    register_module("m", m);
}

torch::Tensor SPPF::forward(torch::Tensor x)
{
    auto x_ = cv1(x);
    auto y1 = m(x_);
    auto y2 = m(y1);
    return cv2(torch::cat({x_, y1, y2, m(y2)}, 1));
}

Concat::Concat(int dimension)
{
    d = dimension;
}

torch::Tensor Concat::forward(std::vector<Tensor> x)
{
    return torch::cat(x, d);
}

Proto::Proto(int c1, int c_, int c2)
{
    cv1 = ModuleHolder<Conv>(c1, c_, 3);
    upsample =
        Upsample(UpsampleOptions()
                     .scale_factor(std::vector<double>({double(2), double(2)}))
                     .mode(torch::kNearest));
    cv2 = ModuleHolder<Conv>(c_, c_, 3);
    cv3 = ModuleHolder<Conv>(c_, c2);

    register_module("cv1", cv1);
    register_module("upsample", upsample);
    register_module("cv2", cv2);
    register_module("cv3", cv3);
}

torch::Tensor Proto::forward(torch::Tensor x)
{
    return cv3(cv2(upsample(cv1(x))));
}

Detect::Detect(int nc_, std::vector<std::vector<int>> anchors_,
               std::vector<int> ch_, bool inplace_)
{
    nc = nc_;
    no = nc + 5;
    nl = anchors_.size();
    na = anchors_[0].size() / 2;
    std::vector<int> v_;

    for (size_t i = 0; i < nl; i++)
    {
        v_.insert(v_.end(), anchors_[i].begin(), anchors_[i].end());
    }

    auto a = torch::from_blob(&v_[0], {int64_t(v_.size())}, torch::kInt32);
    anchors = a.to(torch::kFloat).view({nl, -1, 2}).clone();

    for (size_t i = 0; i < ch_.size(); i++)
    {
        m->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(ch_[i], no * na, 1).bias(true)));
    }
    register_module("m", m);
    inplace = inplace_;
    stride = torch::tensor({8, 16, 32});
    training = true;
}

std::vector<torch::Tensor> Detect::forward(std::vector<torch::Tensor> x)
{
    std::vector<torch::Tensor> z;
    for (size_t i = 0; i < nl; i++)
    {
        auto module = m->ptr<Module>(i);
        auto m = module->as<torch::nn::Conv2dImpl>();
        int bs = x[i].size(0);
        int ny = x[i].size(2);
        int nx = x[i].size(3);
        x[i] = m->forward(x[i]);
        x[i] = x[i].view({bs, na, no, ny, nx})
                   .permute({0, 1, 3, 4, 2})
                   .contiguous();

        if (!training) // inference
        {
            auto [grid, anchor_grid] = _make_grid(nx, ny, i);
            auto vec = x[i].sigmoid().split({2, 2, nc + 1}, 4);
            auto xy = vec[0], wh = vec[1], conf = vec[2];
            xy = (xy * 2 + grid) * stride[i];

            wh = wh.mul(2).pow(2) * anchor_grid;
            auto y = torch::cat({xy, wh, conf}, 4);
            z.push_back(y.view({bs, na * nx * ny, no}));
        }
    }
    if (training)
    {
        return x;
    }
    else
    {
        std::vector<torch::Tensor> o;
        o.push_back(torch::cat(z, 1));
        return o;
    }
}

std::tuple<torch::Tensor, torch::Tensor> Detect::_make_grid(int nx, int ny,
                                                            int i)
{
    auto device = parameters()[0].device();
    anchors = anchors.to(device);
    stride = stride.to(device);
    auto d = anchors[i].options();
    std::vector<int64_t> shape{1, na, ny, nx, 2};
    auto y = torch::arange(ny, d).to(device);
    auto x = torch::arange(nx, d).to(device);
    std::vector<torch::Tensor> vec = torch::meshgrid({y, x});
    torch::Tensor yv = vec[0];
    torch::Tensor xv = vec[1];
    auto grid = torch::stack({xv, yv}, 2).expand(shape) - 0.5;
    auto temp = anchors[i] * stride[i];
    auto anchor_grid = temp.view({1, na, 1, 1, 2}).expand(shape);
    return std::tuple<torch::Tensor, torch::Tensor>(grid, anchor_grid);
}

DetectionModel::DetectionModel(std::string cfg, int ch)
{
    module_list = parse_model(cfg, ch);
    register_module("model", module_list);
    int s = 256;
    auto inputs = torch::zeros({1, ch, s, s});
    auto o = forward(inputs);
    std::vector<int> stride;
    for (size_t i = 0; i < o.size(); i++)
    {
        int v = s / o[i].size(3);
        stride.push_back(v);
    }
    auto module = module_list->ptr<Module>(module_list->size() - 1);
    auto m = module->as<Detect>();
    m->stride = torch::tensor(stride, torch::kFloat32);
    m->anchors = m->anchors / m->stride.view({-1, 1, 1});
}

std::vector<torch::Tensor> DetectionModel::forward(torch::Tensor x)
{
    return _forward_once(x);
}

std::vector<torch::Tensor> DetectionModel::_forward_once(torch::Tensor x)
{
    std::vector<torch::Tensor> outputs{x};
    torch::Tensor O;
    for (size_t i = 0; i < module_list->size(); i++)
    {
        auto module = module_list->ptr<Module>(i);
        auto nm = module->name();
        if (nm == "Conv")
        {
            auto m = module->as<Conv>();
            if (m)
            {
                auto inputs = outputs.back();
                O = m->forward(inputs);
            }
        }
        else if (nm == "C3")
        {
            auto m = module->as<C3>();
            if (m)
            {
                auto inputs = outputs.back();
                O = m->forward(inputs);
            }
        }
        else if (nm == "SPPF")
        {
            auto m = module->as<SPPF>();
            if (m)
            {
                auto inputs = outputs.back();
                O = m->forward(inputs);
            }
        }
        else if (nm == "Concat")
        {
            std::vector<torch::Tensor> inputs;
            auto m = module->as<Concat>();
            for (size_t i = 0; i < m->f.size(); i++)
            {
                if (m->f[i] == -1)
                {
                    auto inp = outputs.back();
                    inputs.push_back(inp);
                }
                else
                {
                    auto inp = outputs[m->f[i]];
                    inputs.push_back(inp);
                }
            }
            O = m->forward(inputs);
        }
        else if (nm == "torch::nn::UpsampleImpl")
        {
            auto m = module->as<Upsample>();
            if (m)
            {
                auto inputs = outputs.back();
                O = m->forward(inputs);
            }
        }
        else if (nm == "Detect")
        {
            std::vector<torch::Tensor> inputs;
            auto m = module->as<Detect>();
            for (size_t i = 0; i < m->f.size(); i++)
            {
                if (m->f[i] == -1)
                {
                    auto inp = outputs.back();
                    inputs.push_back(inp);
                }
                else
                {
                    auto inp = outputs[m->f[i]];
                    inputs.push_back(inp);
                }
            }
            auto outp = m->forward(inputs);
            return outp;
        }

        if (i == 0)
            outputs.clear();
        outputs.emplace_back(O);
    }
}

void DetectionModel::train(bool on)
{
    auto module = module_list->ptr<Module>(module_list->size() - 1);
    auto m = module->as<Detect>();
    m->training = on;

    Module::train(on);
}

FocalLoss::FocalLoss(BCEWithLogitsLoss loss_fcn_, float gamma_, float alpha_)
{
    loss_fcn = loss_fcn_;
    gamma = gamma_;
    alpha = alpha_;
    auto r = loss_fcn->options.reduction();
    reduction = torch::enumtype::reduction_get_enum<
        torch::nn::BCEWithLogitsLossOptions::reduction_t>(r);
    loss_fcn->options.reduction(torch::kNone);
}

torch::Tensor FocalLoss::forward(torch::Tensor pred_, torch::Tensor true_)
{
    auto loss = loss_fcn(pred_, true_);
    auto pred_prob = pred_.sigmoid();
    auto p_t = true_ * pred_prob + (1 - true_) * (1 - pred_prob);
    auto alpha_factor = true_ * alpha + (1 - true_) * (1 - alpha);
    auto modulating_factor = 1.0 - p_t;
    modulating_factor = modulating_factor.pow(gamma);
    loss *= alpha_factor * modulating_factor;

    torch::Tensor x;
    if (reduction == at::Reduction::Reduction::Mean)
    {
        x = loss.mean();
    }
    else if (reduction == at::Reduction::Reduction::Sum)
    {
        x = loss.sum();
    }
    else
    {
        x = loss;
    }

    return x;
}

ComputeLoss::ComputeLoss(ModuleHolder<DetectionModel> model, bool autobalance_)
    : device(torch::kCPU), sort_obj_iou(false), autobalance(autobalance_)
{
    device = model->parameters()[0].device();
    hyp = model->hyp;
    auto weight = torch::tensor({hyp["cls_pw"]});
    BCEcls = BCEWithLogitsLoss(BCEWithLogitsLossOptions().pos_weight(weight));
    BCEcls->to(device);
    weight = torch::tensor({hyp["obj_pw"]});
    BCEobj = BCEWithLogitsLoss(BCEWithLogitsLossOptions().pos_weight(weight));
    BCEobj->to(device);
    float eps = 0.0;

    if (hyp.find("label_smoothing") != hyp.end())
        eps = hyp["label_smoothing"];
    auto s = smooth_BCE(eps);
    // std::cout << "eps = " << eps << std::endl;
    cp = std::get<0>(s);
    // std::cout << "cp = " << cp << std::endl;
    cn = std::get<1>(s);
    // std::cout << "cn = " << cn << std::endl;

    // Focal loss
    float g = hyp["fl_gamma"];
    if (g > 0.0)
    {
        // BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
    }
    auto module =
        model->module_list->ptr<Module>(model->module_list->size() - 1);
    auto m = module->as<Detect>();

    if (m->nl == 3)
    {
        balance = std::vector<float>({4.0, 1.0, 0.4});
    }
    else
    {
        balance = std::vector<float>({4.0, 1.0, 0.25, 0.06, 0.02});
    }

    if (autobalance)
    {
        ssi = 1;
    }
    else
    {
        ssi = 0;
    }
    gr = 1.0;
    na = m->na;
    nc = m->nc;
    nl = m->nl;
    anchors = m->anchors;
}

std::vector<torch::Tensor> ComputeLoss::operator()(std::vector<torch::Tensor> p,
                                                   torch::Tensor targets)
{
    auto lcls = torch::zeros({1}, device);
    auto lbox = torch::zeros({1}, device);
    auto lobj = torch::zeros({1}, device);
    auto tobj = torch::zeros({1}, device);

    auto [tcls, tbox, indices, anchors] = build_targets(p, targets);
    // std::cout << "build_targets-------- " << std::endl;
    // Losses
    for (size_t i = 0; i < p.size(); i++)
    {
        auto pi = p[i]; // layer index, layer predictions
        // image, anchor, gridy, gridx
        auto b = indices[i][0];
        auto a = indices[i][1];
        auto gj = indices[i][2];
        auto gi = indices[i][3];
        tobj = torch::zeros({pi.size(0), pi.size(1), pi.size(2), pi.size(3)})
                   .to(device);
        int n = b.size(0);
        if (n)
        {
            auto vec0 =
                pi.index({b, a, gj, gi})
                    .split({2, 2, 1, nc}, 1); // target-subset of predictions
            // std::cout << vec0.size() << std::endl;
            auto pxy = vec0[0];
            auto pwh = vec0[1];
            auto pcls = vec0[3];

            // Regression
            pxy = pxy.sigmoid() * 2 - 0.5;
            pwh = pwh.sigmoid().mul(2).pow(2) * anchors[i];
            auto pbox = torch::cat({pxy, pwh}, 1); // predicted box
            auto iou =
                bbox_iou(pbox, tbox[i], true, false, false, true).squeeze();
            lbox += (1.0 - iou).mean();
            // std::cout << "iou= " << iou.sizes() << std::endl;
            // std::cout << "lbox= " << lbox << std::endl;

            // Objectness
            iou = iou.detach().clamp(0);
            if (sort_obj_iou)
            {
                auto j = iou.argsort();
                b = b[j];
                a = a[j];
                gj = gj[j];
                gi = gi[j];
                iou = iou[j];
            }
            if (gr < 1)
            {
                iou = (1.0 - gr) + gr * iou;
            }
            // tobj.index_put_({b, a, gj, gi}, iou);
            // std::cout << "tobj=" << tobj.index({b,a,gj,gi}) << std::endl;
            for (size_t i = 0; i < b.numel(); i++)
            {
                /* code */
                tobj.index({b[i], a[i], gj[i], gi[i]}) = iou[i];
            }

            // Classification
            if (nc > 1)
            {
                auto t = torch::full_like(pcls, cn);
                // std::cout << "t1=" << t.sizes() << std::endl;
                t.index_put_({torch::arange(n), tcls[i]}, cp);
                // std::cout << "t2=" << t.sizes() << std::endl;
                lcls += BCEcls(pcls, t);
                // std::cout << "lcls=" << lcls << std::endl;
            }
        }
        auto obji = BCEobj(pi.index({"...", 4}), tobj);
        lobj += obji * balance[i];

        if (autobalance)
        {
            balance[i] =
                balance[i] * 0.9999 + 0.0001 / obji.detach().item<float>();
        }
    }

    if (autobalance)
    {
        for (size_t i = 0; i < balance.size(); i++)
        {
            balance[i] = balance[i] / balance[ssi];
        }
    }
    lbox *= hyp["box"];
    lobj *= hyp["obj"];
    lcls *= hyp["cls"];
    auto bs = tobj.size(0);
    std::vector<torch::Tensor> outps;
    outps.push_back((lbox + lobj + lcls) * bs);
    outps.push_back(torch::cat({lbox, lobj, lcls}).detach());
    return outps;
}

std::tuple<std::vector<Tensor>, std::vector<Tensor>,
           std::vector<std::vector<Tensor>>, std::vector<Tensor>>
ComputeLoss::build_targets(std::vector<torch::Tensor> p, torch::Tensor targets)
{
    // std::cout << "------------------" << std::endl;
    std::vector<std::vector<Tensor>> oupts_indices;
    std::vector<Tensor> oupts_tbox;
    std::vector<Tensor> oupts_anch;
    std::vector<Tensor> oupts_tcls;
    int nt = targets.size(0);             // number of targets
    auto gain = torch::ones({7}, device); // normalized to gridspace gain
    auto ai = torch::arange(na, device)
                  .to(torch::kFloat)
                  .view({na, 1})
                  .repeat({1, nt});
    targets = torch::cat({targets.repeat({na, 1, 1}), ai.unsqueeze(ai.dim())},
                         2); // append anchor indices
    float g = 0.5;
    auto off = torch::tensor({{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}}, device)
                   .to(torch::kFloat) *
               g;
    for (size_t i = 0; i < nl; i++)
    {
        auto anch = anchors[i].to(device);
        // std::cout << "anch = " << anch.device() << std::endl;
        auto shape = p[i].sizes();
        gain.index({Slice(2, 6)}) =
            torch::tensor(shape).to(device).index_select(
                0, torch::tensor({3, 2, 3, 2}).to(device)); // xyxy gain
        // std::cout << gain << std::endl;
        // Match targets to anchors
        auto t = targets * gain; // shape(3,n,7)
        if (nt)
        {
            // Matches
            auto r =
                t.index({"...", Slice(4, 6)}) / anch.index({Slice(), None});
            auto [j, k] = torch::max(r, 1 / r).max(2);
            j = j < hyp["anchor_t"]; // compare
            t = t.index({j});        // filter
            // std::cout << t.sizes() << std::endl;

            // Offsets
            auto gxy = t.index({Slice(), Slice(2, 4)}); // grid xy
            auto gxi =
                gain.index_select(0, torch::tensor({2, 3}).to(device)) - gxy;
            auto Temp = (gxy % 1 < g) & (gxy > 1);
            Temp = Temp.t();
            j = Temp.index({0, Slice()});
            k = Temp.index({1, Slice()});
            // std::cout << "j =" << j.sizes() << std::endl;
            // std::cout << "k =" <<k.sizes() << std::endl;
            Temp = (gxi % 1 < g) & (gxi > 1);
            Temp = Temp.t();
            auto l = Temp.index({0, Slice()});
            auto m = Temp.index({1, Slice()});
            j = torch::stack({torch::ones_like(j), j, k, l, m});
            t = t.repeat({5, 1, 1}).index({j});
            auto offsets = torch::zeros_like(gxy).index({None}) +
                           off.index({Slice(), None});
            offsets = offsets.index({j});
            // std::cout << "offsets" << offsets.sizes() << std::endl;
            auto vec =
                t.chunk(4, 1); //(image, class), grid xy, grid wh, anchors
            auto &bc = vec[0];
            gxy = vec[1];
            auto &gwh = vec[2];
            auto &a = vec[3];
            a = a.to(torch::kInt32).view({-1}); // anchors
            // std::cout << "a= " << a.device() << std::endl;
            bc = bc.to(torch::kInt32).t();
            // std::cout << "bc= " << bc.device() << std::endl;
            auto b = bc.index({0, Slice()});
            // std::cout << "b= " << b.device() << std::endl;
            auto c = bc.index({1, Slice()});
            // std::cout << "c= " << c.device() << std::endl;
            auto gij = gxy - offsets;
            gij = gij.to(torch::kInt32);
            auto gij_ = gij.t();
            // std::cout << "gij= " << gij.device() << std::endl;
            auto gi = gij_.index({0, Slice()});
            // std::cout << "gi= " << gi.device() << std::endl;
            auto gj = gij_.index({1, Slice()});
            // std::cout << "gj= " << gj.device() << std::endl;

            // std::cout << "anchors[a]= " << anch.index({a}).device() <<
            // std::endl;
            oupts_indices.push_back(std::vector<Tensor>(
                {b, a, gj.clamp_(0, shape[2] - 1),
                 gi.clamp_(0, shape[3] - 1)})); // image, anchor, grid
            oupts_tbox.push_back(torch::cat({gxy - gij, gwh}, 1)); // box
            oupts_anch.push_back(anch.index({a}));                 // anchors
            oupts_tcls.push_back(c);                               // class
        }
        else
        {
            t = targets[0];
            int offsets = 0;

            auto vec =
                t.chunk(4, 1); //(image, class), grid xy, grid wh, anchors
            auto &bc = vec[0];
            auto gxy = vec[1];
            auto &gwh = vec[2];
            auto &a = vec[3];
            a = a.to(torch::kInt32).view({-1}); // anchors
            // std::cout << "a= " << a.sizes() << std::endl;
            bc = bc.to(torch::kInt32).t();
            // std::cout << "bc= " << bc.sizes() << std::endl;
            auto b = bc.index({0, Slice()});
            // std::cout << "b= " << b.sizes() << std::endl;
            auto c = bc.index({1, Slice()});
            // std::cout << "c= " << c.sizes() << std::endl;
            auto gij = gxy - offsets;
            auto gij_ = gij.t();
            // std::cout << "gij= " << gij.sizes() << std::endl;
            auto gi = gij_.index({0, Slice()});
            // std::cout << "gi= " << gi.sizes() << std::endl;
            auto gj = gij_.index({1, Slice()});
            // std::cout << "gj= " << gj.sizes() << std::endl;

            // std::cout << "anchors[a]= " << anch.index({a}).sizes() <<
            // std::endl;
            oupts_indices.push_back(std::vector<Tensor>(
                {b, a, gj.clamp_(0, shape[2] - 1),
                 gi.clamp_(0, shape[3] - 1)})); // image, anchor, grid
            oupts_tbox.push_back(torch::cat({gxy - gij, gwh}, 1)); // box
            oupts_anch.push_back(anch.index({a}));                 // anchors
            oupts_tcls.push_back(c);                               // class
        }
    }
    return std::tuple<std::vector<Tensor>, std::vector<Tensor>,
                      std::vector<std::vector<Tensor>>, std::vector<Tensor>>(
        oupts_tcls, oupts_tbox, oupts_indices, oupts_anch);
}

template <typename DataLoader>

void train(torch::nn::ModuleHolder<DetectionModel> &network, DataLoader &loader,
           torch::optim::Optimizer &optimizer)
{
    auto device = network->parameters()[0].device();
    ComputeLoss compute_loss(network);
    network->train(true);

    for (auto &batch : loader)
    {
        auto inputs = batch.data.to(device);
        // std::cout << "inputs= " << inputs.sizes() << std::endl;
        auto targets = batch.target.to(device); //.reshape({-1, 6})
        // std::cout << "targets= " << targets.device() << std::endl;
        auto preds = network->forward(inputs);
        // std::cout << "preds= " << preds[0].device() << std::endl;
        auto vec_loss = compute_loss(preds, targets);
        // std::cout << "vec_loss= " << vec_loss.size() << std::endl;
        torch::Tensor loss = vec_loss[0];
        torch::Tensor loss2 = vec_loss[1];

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        std::cout << "loss = " << loss.item<float>() << " "
                  << "box_loss = " << loss2[0].item<float>() << " "
                  << "obj_loss= " << loss2[1].item<float>() << " "
                  << "cls_loss= " << loss2[2].item<float>() << std::endl;
    }
}
} // namespace mrcv
