import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import math
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BBox:
    def __init__(self):
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0
        self.name = ""

    def get_h(self):
        return self.ymax - self.ymin

    def get_w(self):
        return self.xmax - self.xmin

    def center_x(self):
        return (self.xmax + self.xmin) / 2.0

    def center_y(self):
        return (self.ymax + self.ymin) / 2.0

class DetectorData:
    def __init__(self, image, bboxes):
        self.image = image
        self.bboxes = bboxes

class DetAugmentations:
    @staticmethod
    def resize(m_data, width, height, probability):
        if random.random() <= probability:
            h_scale = height / m_data.image.shape[0]
            w_scale = width / m_data.image.shape[1]
            for bbox in m_data.bboxes:
                bbox.xmin = int(w_scale * bbox.xmin)
                bbox.xmax = int(w_scale * bbox.xmax)
                bbox.ymin = int(h_scale * bbox.ymin)
                bbox.ymax = int(h_scale * bbox.ymax)
            m_data.image = cv2.resize(m_data.image, (width, height))
        return m_data

def load_xml(xml_path):
    objects = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for elem in root.findall('object'):
            name = elem.find('name').text
            bbox_elem = elem.find('bndbox')
            obj = BBox()
            obj.xmin = int(bbox_elem.find('xmin').text)
            obj.xmax = int(bbox_elem.find('xmax').text)
            obj.ymin = int(bbox_elem.find('ymin').text)
            obj.ymax = int(bbox_elem.find('ymax').text)
            obj.name = name
            objects.append(obj)
    except Exception as e:
        logger.error(f"Error loading XML file {xml_path}: {str(e)}")
    return objects

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ResblockBody(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = BasicConv(in_channels, out_channels, 3)
        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        c = self.out_channels
        x = self.conv1(x)
        route = x
        x = torch.split(x, c // 2, dim=1)[1]
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)
        x = self.maxpool(x)
        return [x, feat]

class CSPDarknet53Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = BasicConv(3, 32, 3, 2)
        self.conv2 = BasicConv(32, 64, 3, 2)
        self.resblock_body1 = ResblockBody(64, 64)
        self.resblock_body2 = ResblockBody(128, 128)
        self.resblock_body3 = ResblockBody(256, 256)
        self.conv3 = BasicConv(512, 512, 3)
        self.num_features = 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.resblock_body1(x)[0]
        x = self.resblock_body2(x)[0]
        res_out = self.resblock_body3(x)
        x = res_out[0]
        feat1 = res_out[1]
        x = self.conv3(x)
        feat2 = x
        return [feat1, feat2]

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1)
        )

    def forward(self, x):
        x = self.upsample(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return x

def yolo_head(filters_list, in_filters):
    return nn.Sequential(
        BasicConv(in_filters, filters_list[0], 3),
        nn.Conv2d(filters_list[0], filters_list[1], 1, bias=True)
    )

class YoloBodyTiny(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super().__init__()
        self.backbone = CSPDarknet53Tiny()
        self.conv_for_p5 = BasicConv(512, 256, 1)
        self.yolo_head_p5 = yolo_head([512, num_anchors * (5 + num_classes)], 256)
        self.upsample = Upsample(256, 128)
        self.yolo_head_p4 = yolo_head([256, num_anchors * (5 + num_classes)], 384)

    def forward(self, x):
        feat1, feat2 = self.backbone(x)
        p5 = self.conv_for_p5(feat2)
        out0 = self.yolo_head_p5(p5)
        p5_upsample = self.upsample(p5)
        p4 = torch.cat([p5_upsample, feat1], dim=1)
        out1 = self.yolo_head_p4(p4)
        return [out0, out1]

def jaccard(box_a, box_b):
    b1_x1 = box_a[:, 0] - box_a[:, 2] / 2
    b1_x2 = box_a[:, 0] + box_a[:, 2] / 2
    b1_y1 = box_a[:, 1] - box_a[:, 3] / 2
    b1_y2 = box_a[:, 1] + box_a[:, 3] / 2

    b2_x1 = box_b[:, 0] - box_b[:, 2] / 2
    b2_x2 = box_b[:, 0] + box_b[:, 2] / 2
    b2_y1 = box_b[:, 1] - box_b[:, 3] / 2
    b2_y2 = box_b[:, 1] + box_b[:, 3] / 2

    box_a = torch.zeros_like(box_a)
    box_b = torch.zeros_like(box_b)

    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

    A, B = box_a.size(0), box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union

def smooth_label(y_true, label_smoothing, num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes

def box_ciou(b1, b2):
    b1_xy, b1_wh = b1[..., :2], b1[..., 2:4]
    b2_xy, b2_wh = b2[..., :2], b2[..., 2:4]
    b1_wh_half, b2_wh_half = b1_wh / 2., b2_wh / 2.
    b1_mins, b1_maxes = b1_xy - b1_wh_half, b1_xy + b1_wh_half
    b2_mins, b2_maxes = b2_xy - b2_wh_half, b2_xy + b2_wh_half

    intersect_mins = torch.max(b1_mins, b2_mins)
    intersect_maxes = torch.min(b1_maxes, b2_maxes)
    intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / torch.clamp(union_area, min=1e-6)

    center_distance = torch.sum(torch.pow(b1_xy - b2_xy, 2), dim=-1)
    enclose_mins = torch.min(b1_mins, b2_mins)
    enclose_maxes = torch.max(b1_maxes, b2_maxes)
    enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
    enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), dim=-1)
    ciou = iou - 1.0 * center_distance / (enclose_diagonal + 1e-7)

    v = (4 / (math.pi ** 2)) * torch.pow(
        torch.atan(b1_wh[..., 0] / b1_wh[..., 1]) - torch.atan(b2_wh[..., 0] / b2_wh[..., 1]), 2)
    alpha = v / (1.0 - iou + v)
    ciou = ciou - alpha * v
    return ciou

def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def mse_loss(pred, target):
    return torch.pow(pred - target, 2)

def bce_loss(pred, target):
    pred = clip_by_tensor(pred, 1e-7, 1.0 - 1e-7)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, label_smooth=0, device=torch.device('cpu'), normalize=True):
        super().__init__()
        self.anchors = anchors
        self.num_anchors = anchors.size(0)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_size = img_size
        self.feature_length = [img_size[0] // s for s in [32, 16, 8]]
        self.label_smooth = label_smooth
        self.device = device
        self.normalize = normalize
        self.ignore_threshold = 0.5
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_loc = 1.0

    def get_target(self, targets, scaled_anchors, in_w, in_h, ignore_threshold):
        bs = len(targets)
        index = self.feature_length.index(in_w)
        anchor_vec = [[3, 4, 5], [1, 2, 3]]
        anchor_index = anchor_vec[index]
        subtract_index = 3 * index

        mask = torch.zeros((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)
        noobj_mask = torch.ones((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)
        tx = torch.zeros((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)
        ty = torch.zeros((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)
        tw = torch.zeros((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)
        th = torch.zeros((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)
        tbox = torch.zeros((bs, self.num_anchors // 2, in_h, in_w, 4), requires_grad=False)
        tconf = torch.zeros((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)
        tcls = torch.zeros((bs, self.num_anchors // 2, in_h, in_w, self.num_classes), requires_grad=False)
        box_loss_scale_x = torch.zeros((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)
        box_loss_scale_y = torch.zeros((bs, self.num_anchors // 2, in_h, in_w), requires_grad=False)

        for b in range(bs):
            if targets[b].size(0) == 0:
                continue
            gxs = targets[b][:, 0:1] * in_w
            gys = targets[b][:, 1:2] * in_h
            gws = targets[b][:, 2:3] * in_w
            ghs = targets[b][:, 3:4] * in_h
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)

            gt_box = torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], dim=1).float()
            anchor_shapes = torch.cat([torch.zeros((self.num_anchors, 2)), scaled_anchors], dim=1).type_as(targets[b])
            anch_ious = jaccard(gt_box, anchor_shapes)
            best_ns = torch.argmax(anch_ious, dim=-1)

            for i in range(len(best_ns)):
                if best_ns[i].item() not in anchor_index:
                    continue
                gi = int(gis[i].item())
                gj = int(gjs[i].item())
                gx, gy = gxs[i].item(), gys[i].item()
                gw, gh = gws[i].item(), ghs[i].item()
                if gj < in_h and gi < in_w:
                    best_n = anchor_index.index(best_ns[i].item())
                    noobj_mask[b, best_n, gj, gi] = 0
                    mask[b, best_n, gj, gi] = 1
                    tx[b, best_n, gj, gi] = gx
                    ty[b, best_n, gj, gi] = gy
                    tw[b, best_n, gj, gi] = gw
                    th[b, best_n, gj, gi] = gh
                    box_loss_scale_x[b, best_n, gj, gi] = targets[b][i, 2]
                    box_loss_scale_y[b, best_n, gj, gi] = targets[b][i, 3]
                    tconf[b, best_n, gj, gi] = 1
                    tcls[b, best_n, gj, gi, int(targets[b][i, 4].item())] = 1
                else:
                    logger.warning(f"Step out of boundary: {gxs} {gys} {gis} {gjs} {targets[b]}")

        tbox[..., 0] = tx
        tbox[..., 1] = ty
        tbox[..., 2] = tw
        tbox[..., 3] = th
        return [mask, noobj_mask, tbox, tconf, tcls, box_loss_scale_x, box_loss_scale_y]

    def get_ignore(self, prediction, targets, scaled_anchors, in_w, in_h, noobj_mask):
        bs = len(targets)
        index = self.feature_length.index(in_w)
        anchor_vec = [[3, 4, 5], [0, 1, 2]]
        anchor_index = anchor_vec[index]

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]

        FloatType = prediction.dtype
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * self.num_anchors // 2), 1, 1).view(x.shape).type(FloatType)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * self.num_anchors // 2), 1, 1).view(y.shape).type(FloatType)

        anchor_w = scaled_anchors[anchor_index, 0:1].type(FloatType)
        anchor_h = scaled_anchors[anchor_index, 1:2].type(FloatType)
        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        pred_boxes = torch.zeros_like(prediction[..., :4]).type(FloatType)
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h

        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i].view(-1, 4)
            if targets[i].size(0) > 0:
                gx = targets[i][:, 0:1] * in_w
                gy = targets[i][:, 1:2] * in_h
                gw = targets[i][:, 2:3] * in_w
                gh = targets[i][:, 3:4] * in_h
                gt_box = torch.cat([gx, gy, gw, gh], dim=-1).type(FloatType)
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                # Reshape to [num_anchors, height, width]
                anch_ious_max = anch_ious_max.view(self.num_anchors // 2, in_h, in_w)
                noobj_mask[i] = (anch_ious_max <= self.ignore_threshold).float()

        return [noobj_mask, pred_boxes]

    def forward(self, input, targets):
        bs, in_h, in_w = input.size(0), input.size(2), input.size(3)
        stride_h = self.image_size[1] / in_h
        stride_w = self.image_size[0] / in_w

        scaled_anchors = self.anchors.clone()
        scaled_anchors[:, 0] /= stride_w
        scaled_anchors[:, 1] /= stride_h

        prediction = input.view(bs, self.num_anchors // 2, self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                2).contiguous()
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        temp = self.get_target(targets, scaled_anchors, in_w, in_h, self.ignore_threshold)
        mask, noobj_mask, tbox, tconf, tcls, box_loss_scale_x, box_loss_scale_y = temp

        temp_ciou = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, noobj_mask)
        noobj_mask, pred_boxes_for_ciou = temp_ciou

        mask = mask.to(self.device)
        noobj_mask = noobj_mask.to(self.device)
        box_loss_scale_x = box_loss_scale_x.to(self.device)
        box_loss_scale_y = box_loss_scale_y.to(self.device)
        tconf = tconf.to(self.device)
        tcls = tcls.to(self.device)
        pred_boxes_for_ciou = pred_boxes_for_ciou.to(self.device)
        tbox = tbox.to(self.device)

        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y
        mask_flat = mask.bool()
        if mask_flat.sum() > 0:
            ciou = (1 - box_ciou(pred_boxes_for_ciou[mask_flat], tbox[mask_flat])) * box_loss_scale[mask_flat]
            loss_loc = torch.sum(ciou) / bs
        else:
            loss_loc = torch.tensor(0.0).to(self.device)

        loss_conf = (torch.sum(bce_loss(conf, mask.float()) * mask.float()) / bs +
                     torch.sum(bce_loss(conf, mask.float()) * noobj_mask) / bs)

        loss_cls = torch.sum(bce_loss(pred_cls[mask == 1],
                                      smooth_label(tcls[mask == 1], self.label_smooth, self.num_classes))) / bs

        loss = loss_conf * self.lambda_conf + loss_cls * self.lambda_cls + loss_loc * self.lambda_loc

        num_pos = torch.tensor(0).to(self.device)
        if self.normalize:
            num_pos = torch.sum(mask)
            num_pos = torch.max(num_pos, torch.ones_like(num_pos))
        else:
            num_pos = bs / 2
        return [loss, num_pos]

def nms_libtorch(bboxes, scores, thresh):
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        order_mask = order[1:]
        xx1 = x1[order_mask].clamp(min=x1[i].item())
        yy1 = y1[order_mask].clamp(min=y1[i].item())
        xx2 = x2[order_mask].clamp(max=x2[i].item())
        yy2 = y2[order_mask].clamp(max=y2[i].item())
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[order_mask] - inter)
        idx = (iou <= thresh).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    return keep

def non_maximum_suppression(prediction, num_classes, conf_thres, nms_thres):
    prediction[..., 0] -= prediction[..., 2] / 2
    prediction[..., 1] -= prediction[..., 3] / 2
    prediction[..., 2] += prediction[..., 0]
    prediction[..., 3] += prediction[..., 1]

    output = []
    for image_id in range(prediction.size(0)):
        image_pred = prediction[image_id]
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask].float()
        class_conf = class_conf[conf_mask].float()
        class_pred = class_pred[conf_mask].float()

        if not image_pred.size(0):
            output.append(torch.zeros((1, 7)))
            continue

        detections = torch.cat([image_pred[:, :5], class_conf, class_pred], 1)
        img_classes = []
        for m in range(detections.size(0)):
            if detections[m, 6] not in img_classes:
                img_classes.append(detections[m, 6])

        temp_class_detections = []
        for c in img_classes:
            detections_class = detections[detections[:, -1] == c]
            keep = nms_libtorch(detections_class[:, :4],
                                detections_class[:, 4] * detections_class[:, 5],
                                nms_thres)
            temp_max_detections = [detections_class[v] for v in keep]
            if temp_max_detections:
                max_detections = torch.cat(temp_max_detections, dim=0)
                temp_class_detections.append(max_detections)

        if temp_class_detections:
            class_detections = torch.cat(temp_class_detections, dim=0)
            output.append(class_detections)
        else:
            output.append(torch.zeros((1, 7)))

    return output

def decode_box(input, anchors, num_classes, img_size):
    num_anchors = anchors.size(0)
    bbox_attrs = 5 + num_classes
    batch_size, input_height, input_width = input.size(0), input.size(2), input.size(3)

    stride_h = img_size[1] / input_height
    stride_w = img_size[0] / input_width

    scaled_anchors = anchors.clone()
    scaled_anchors[:, 0] /= stride_w
    scaled_anchors[:, 1] /= stride_h

    prediction = input.view(batch_size, num_anchors, bbox_attrs, input_height, input_width).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()

    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    w = prediction[..., 2]
    h = prediction[..., 3]
    conf = torch.sigmoid(prediction[..., 4])
    pred_cls = torch.sigmoid(prediction[..., 5:])

    FloatType = x.dtype
    grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
        batch_size * num_anchors, 1, 1).view(x.shape).type(FloatType)
    grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
        batch_size * num_anchors, 1, 1).view(y.shape).type(FloatType)

    anchor_w = scaled_anchors[:, 0:1].type(FloatType)
    anchor_h = scaled_anchors[:, 1:2].type(FloatType)
    anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
    anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

    pred_boxes = torch.zeros_like(prediction[..., :4]).type(FloatType)
    pred_boxes[..., 0] = x + grid_x
    pred_boxes[..., 1] = y + grid_y
    pred_boxes[..., 2] = torch.exp(w) * anchor_w
    pred_boxes[..., 3] = torch.exp(h) * anchor_h

    scales = torch.tensor([stride_w, stride_h, stride_w, stride_h]).type(FloatType)
    pred_boxes = pred_boxes.view(batch_size, -1, 4) * scales
    conf = conf.view(batch_size, -1, 1)
    pred_cls = pred_cls.view(batch_size, -1, num_classes)
    output = torch.cat([pred_boxes, conf, pred_cls], dim=-1)
    return output

def load_xml_data_from_folder(folder, image_type, list_images, list_labels):
    for entry in Path(folder).rglob('*.xml'):
        full_path = str(entry)
        list_labels.append(full_path)
        image_path = full_path.replace('labels', 'images').replace('.xml', image_type)
        if os.path.exists(image_path):
            list_images.append(image_path)
        else:
            logger.warning(f"Image file does not exist: {image_path}")

def show_bbox(image, bboxes, confidence, name_list):
    font_face = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.4
    thickness = 1

    if torch.all(bboxes == 0):
        logger.info("Boxes not detected")

    bboxes = bboxes.cpu().numpy()
    for i in range(0, bboxes.shape[0], 7):
        x, y, w, h = bboxes[i:i + 4]
        cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0, 0, 255), 1)

        origin = (int(x), int(y + 8))
        label = name_list[int(bboxes[i + 6])]
        conf = bboxes[i + 4] * 100
        conf_text = f"{label} {conf:.2f}%"

        cv2.putText(image, conf_text, origin, font_face, font_scale, (0, 255, 255), thickness)

    cv2.imwrite("prediction.jpg", image)
    cv2.imshow("Test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def custom_collate_fn(batch):
    images = []
    targets = []
    for img, lbl in batch:
        images.append(img)
        if lbl.size(0) == 0:
            targets.append(torch.zeros((0, 5), dtype=torch.float32))
        else:
            targets.append(lbl)
    images = torch.stack(images, dim=0)
    return images, targets

class DetDataset(Dataset):
    def __init__(self, images, labels, class_names, is_train=True, width=416, height=416, h_flip_prob=0.5,
                 v_flip_prob=0):
        self.list_images = []
        self.list_labels = []
        for img_path, lbl_path in zip(images, labels):
            if os.path.exists(img_path):
                self.list_images.append(img_path)
                self.list_labels.append(lbl_path)
            else:
                logger.warning(f"Skipping invalid image path: {img_path}")
        self.is_train = is_train
        self.width = width
        self.height = height
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.name_idx = {name: float(i) for i, name in enumerate(class_names)}

    def __len__(self):
        return len(self.list_labels)

    def __getitem__(self, index):
        image_path = self.list_images[index]
        annotation_path = self.list_labels[index]

        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            dummy_img = torch.zeros((3, self.height, self.width), dtype=torch.uint8)
            dummy_label = torch.zeros((0, 5), dtype=torch.float32)
            return dummy_img, dummy_label

        boxes = load_xml(annotation_path)
        m_data = DetectorData(img, boxes)
        m_data = DetAugmentations.resize(m_data, self.width, self.height, 1)

        width_under1 = 1.0 / m_data.image.shape[1]
        height_under1 = 1.0 / m_data.image.shape[0]
        img_tensor = torch.from_numpy(m_data.image).permute(2, 0, 1)

        box_num = len(m_data.bboxes)
        if box_num == 0:
            label_tensor = torch.zeros((0, 5), dtype=torch.float32)
            return img_tensor.clone(), label_tensor.clone()

        label_tensor = torch.zeros((box_num, 5), dtype=torch.float32)
        for i in range(box_num):
            label_tensor[i, 2] = m_data.bboxes[i].get_w() * width_under1
            label_tensor[i, 3] = m_data.bboxes[i].get_h() * width_under1
            label_tensor[i, 0] = m_data.bboxes[i].xmin * width_under1 + label_tensor[i, 2] / 2
            label_tensor[i, 1] = m_data.bboxes[i].ymin * height_under1 + label_tensor[i, 3] / 2
            label_tensor[i, 4] = self.name_idx[m_data.bboxes[i].name]

        return img_tensor.clone(), label_tensor.clone()

class Detector:
    def __init__(self):
        self.width = 416
        self.height = 416
        self.name_list = []
        self.device = torch.device('cpu')
        self.detector = None

    def initialize(self, width, height, name_list_path):
        self.device = torch.device('cpu')
        logger.info("Device is CPU")

        with open(name_list_path, 'r') as f:
            self.name_list = [line.strip() for line in f]

        num_classes = len(self.name_list)
        self.width = width
        self.height = height

        if width % 32 or height % 32:
            logger.error("Width or height is not divisible by 32")
            return

        self.detector = YoloBodyTiny(3, num_classes).to(self.device)
        logger.info("Model initialization complete!")

    def load_pretrained(self, pretrained_path):
        if not os.path.exists(pretrained_path):
            logger.error(f"Pretrained path does not exist: {pretrained_path}")
            return 1

        try:
            net_pretrained = torch.jit.load(pretrained_path, map_location=self.device)
            if len(self.name_list) == 80:
                self.detector = net_pretrained
                return 0
        except Exception as e:
            logger.error(f"Error loading pretrained model: {str(e)}\n{traceback.format_exc()}")
            return 1

        try:
            pretrained_dict = net_pretrained.state_dict()
            model_dict = self.detector.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'yolo_head' not in k}
            model_dict.update(pretrained_dict)
            self.detector.load_state_dict(model_dict)
            return 0
        except Exception as e:
            logger.error(f"Error transferring pretrained weights: {str(e)}\n{traceback.format_exc()}")
            return 1

    def train(self, train_val_path, image_type, num_epochs=30, batch_size=4, learning_rate=0.0003,
              save_path="detector.pt", pretrained_path="detector.pt"):
        if not os.path.exists(pretrained_path):
            logger.warning(f"Pretrained path is invalid: {pretrained_path}. Randomly initialized model.")
            return 1

        if self.load_pretrained(pretrained_path) != 0:
            logger.error("Failed to load pretrained model")
            return 1

        train_label_path = os.path.join(train_val_path, "train", "labels")
        val_label_path = os.path.join(train_val_path, "val", "labels")

        list_images_train, list_labels_train = [], []
        list_images_val, list_labels_val = [], []

        load_xml_data_from_folder(train_label_path, image_type, list_images_train, list_labels_train)
        load_xml_data_from_folder(val_label_path, image_type, list_images_val, list_labels_val)

        if len(list_images_train) < batch_size or len(list_images_val) < batch_size:
            logger.error("Image numbers less than batch size or empty image folder")
            return 2

        if not list_images_train or not os.path.exists(list_images_train[0]):
            logger.error(f"Invalid image path: {list_images_train[0] if list_images_train else 'Empty list'}")
            return 3

        train_dataset = DetDataset(list_images_train, list_labels_train, self.name_list, True, self.width, self.height)
        val_dataset = DetDataset(list_images_val, list_labels_val, self.name_list, False, self.width, self.height)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

        anchors = torch.tensor([[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]],
                               dtype=torch.float32).to(self.device)
        image_size = [self.width, self.height]
        normalize = False
        criterion1 = YOLOLoss(anchors, len(self.name_list), image_size, 0.01, self.device, normalize)
        criterion2 = YOLOLoss(anchors, len(self.name_list), image_size, 0.01, self.device, normalize)

        FloatType = torch.float32
        best_loss = float('inf')

        for epoch in range(num_epochs):
            loss_sum = 0
            batch_count = 0
            loss_train = 0
            loss_val = 0

            if epoch == num_epochs // 2:
                learning_rate /= 10

            optimizer = optim.Adam(self.detector.parameters(), lr=learning_rate)

            if epoch < num_epochs // 10:
                for name, param in self.detector.named_parameters():
                    param.requires_grad = 'yolo_head' in name
            else:
                for param in self.detector.parameters():
                    param.requires_grad = True

            self.detector.train()
            for batch in train_loader:
                images, targets = batch
                images = images.to(self.device, dtype=FloatType).div(255.0)
                targets = [t.to(self.device, dtype=FloatType) for t in targets]

                optimizer.zero_grad()
                outputs = self.detector(images)
                loss_num_pos1 = criterion1(outputs[0], targets)
                loss_num_pos2 = criterion2(outputs[1], targets)

                loss = loss_num_pos1[0] + loss_num_pos2[0]
                num_pos = loss_num_pos1[1] + loss_num_pos2[1]
                loss = loss / num_pos

                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                batch_count += 1
                loss_train = loss_sum / batch_count
                print(f"\rEpoch: {epoch}, Training Loss: {loss_train:.4f}", end="")

            print()

            self.detector.eval()
            loss_sum = 0
            batch_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    images, targets = batch
                    images = images.to(self.device, dtype=FloatType).div(255.0)
                    targets = [t.to(self.device, dtype=FloatType) for t in targets]

                    outputs = self.detector(images)
                    loss_num_pos1 = criterion1(outputs[1], targets)
                    loss_num_pos2 = criterion2(outputs[0], targets)

                    loss = loss_num_pos1[0] + loss_num_pos2[0]
                    num_pos = loss_num_pos1[1] + loss_num_pos2[1]
                    loss = loss / num_pos

                    loss_sum += loss.item()
                    batch_count += 1
                    loss_val = loss_sum / batch_count
                    print(f"\rEpoch: {epoch}, Valid Loss: {loss_val:.4f}", end="")

            print()

            if loss_val < best_loss:
                best_loss = loss_val
                torch.save(self.detector.state_dict(), save_path)

        return 0

    def validate(self, val_data_path, image_type, batch_size):
        val_label_path = os.path.join(val_data_path, "val", "labels")
        list_images_val, list_labels_val = [], []
        load_xml_data_from_folder(val_label_path, image_type, list_images_val, list_labels_val)

        if len(list_images_val) < batch_size:
            logger.error("Image numbers less than batch size or empty image folder")
            return float('inf')

        val_dataset = DetDataset(list_images_val, list_labels_val, self.name_list, False, self.width, self.height)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

        anchors = torch.tensor([[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]],
                               dtype=torch.float32).to(self.device)
        image_size = [self.width, self.height]
        normalize = False
        criterion1 = YOLOLoss(anchors, len(self.name_list), image_size, 0.01, self.device, normalize)
        criterion2 = YOLOLoss(anchors, len(self.name_list), image_size, 0.01, self.device, normalize)

        self.detector.eval()
        loss_sum = 0
        batch_count = 0
        FloatType = torch.float32

        with torch.no_grad():
            for batch in val_loader:
                images, targets = batch
                images = images.to(self.device, dtype=FloatType).div(255.0)
                targets = [t.to(self.device, dtype=FloatType) for t in targets]

                outputs = self.detector(images)
                loss_num_pos1 = criterion1(outputs[1], targets)
                loss_num_pos2 = criterion2(outputs[0], targets)

                loss = loss_num_pos1[0] + loss_num_pos2[0]
                num_pos = loss_num_pos1[1] + loss_num_pos2[1]
                loss = loss / num_pos

                loss_sum += loss.item()
                batch_count += 1

        return loss_sum / batch_count if batch_count > 0 else float('inf')

    def auto_train(self, train_val_path, image_type, epochs_list=[10, 30, 50], batch_sizes=[4, 8, 10],
                   learning_rates=[0.1, 0.01], pretrained_path="detector.pt", save_path="detector.pt"):
        best_loss = float('inf')
        best_epochs = 0
        best_batch_size = 0
        best_learning_rate = 0
        status_code = 0

        for num_epochs in epochs_list:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    print(
                        f"Training with epochs: {num_epochs}, batch size: {batch_size}, learning rate: {learning_rate}")
                    status_code = self.train(train_val_path, image_type, num_epochs, batch_size, learning_rate,
                                             save_path, pretrained_path)
                    if status_code != 0:
                        logger.error(
                            f"Training failed with status code {status_code} for epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
                        continue
                    if self.load_weight(save_path) != 0:
                        logger.error(f"Failed to load weights for validation after training")
                        continue
                    val_loss = self.validate(train_val_path, image_type, batch_size)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_epochs = num_epochs
                        best_batch_size = batch_size
                        best_learning_rate = learning_rate

        print(
            f"Best hyperparameters:\n  Epochs: {best_epochs}\n  Batch size: {best_batch_size}\n  Learning rate: {best_learning_rate}")
        print(f"Best validation loss: {best_loss}")

        if best_epochs == 0:
            logger.error("No successful training runs completed")
            return status_code

        status_code = self.train(train_val_path, image_type, best_epochs, best_batch_size, best_learning_rate,
                                 save_path, pretrained_path)
        return status_code

    def load_weight(self, weight_path):
        if not os.path.exists(weight_path):
            logger.error(f"Weight path does not exist: {weight_path}")
            return 1
        try:
            self.detector.load_state_dict(torch.load(weight_path, map_location=self.device))
            self.detector.to(self.device)
            self.detector.eval()
            return 0
        except Exception as e:
            logger.error(f"Error loading weights: {str(e)}\n{traceback.format_exc()}")
            return 1

    def predict(self, image, show=True, conf_thresh=0.3, nms_thresh=0.3):
        logger.info("Starting detector...")
        origin_width, origin_height = image.shape[1], image.shape[0]
        image = cv2.resize(image, (self.width, self.height))
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(self.device)

        anchors = torch.tensor([[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]], dtype=torch.float32)
        image_size = [self.width, self.height]

        with torch.no_grad():
            outputs = self.detector(img_tensor)

        output_list = []
        for i, tensor_input in enumerate(outputs):
            output_decoded = decode_box(tensor_input, anchors[3 * i:3 * (i + 1)], len(self.name_list), image_size)
            output_list.append(output_decoded)

        output = torch.cat(output_list, dim=1)
        detection = non_maximum_suppression(output, len(self.name_list), conf_thresh, nms_thresh)

        w_scale = origin_width / self.width
        h_scale = origin_height / self.height

        counter = 0
        for i in range(len(detection)):
            for j in range(detection[i].size(0) // 7):
                detection[i][7 * j:7 * j + 4:2] *= w_scale
                detection[i][7 * j + 1:7 * j + 4:2] *= h_scale
                confidence = detection[i][7 * j + 4].item()
                logger.info(f"Boxes detected: {counter} with {confidence:.4f} confidence")
                counter += 1

        logger.info("Detector is complete!")

        image = cv2.resize(image, (origin_width, origin_height))
        if show:
            show_bbox(image, detection[0], confidence, self.name_list)

        return 0