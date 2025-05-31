
import cv2
import numpy as np
import logging
from enum import Enum
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
def write_log(message, log_type=None):
    if log_type == "ERROR":
        logging.error(message)
    else:
        logging.info(message)

# Helper function to create an error image
def get_error_image(message):
    img = np.zeros((100, 500, 3), dtype=np.uint8)
    cv2.putText(img, message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

# Define disparity method enum
class METOD_DISPARITY(Enum):
    MODE_NONE = 0
    MODE_BM = 1
    MODE_SGBM = 2
    MODE_SGBM_3WAY = 3
    MODE_HH = 4
    MODE_HH4 = 5

# Constants for neural network
CONFIDENCE_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
FONT_SCALE = 0.5
THICKNESS = 1
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)

def making_stereo_pair(input_image_camera01, input_image_camera02):
    if input_image_camera01.size == 0 or input_image_camera02.size == 0:
        return get_error_image("makingStereoPair:: Image is Empty"), 1
    max_v = max(input_image_camera01.shape[0], input_image_camera02.shape[0])
    max_u = input_image_camera01.shape[1] + input_image_camera02.shape[1]
    image_pair = np.zeros((max_v, max_u, 3), dtype=np.uint8)
    image_pair[:input_image_camera01.shape[0], :input_image_camera01.shape[1]] = input_image_camera01
    image_pair[:input_image_camera02.shape[0], input_image_camera01.shape[1]:] = input_image_camera02
    return image_pair, 0

def show_image(input_image, window_name, coef_show_window=1.0):
    if input_image.size == 0:
        input_image = get_error_image("showImage:: Image is Empty")
        write_log("showImage:: Image is Empty, status = 1", "ERROR")
        return 1
    output_image = cv2.resize(input_image, (int(input_image.shape[1] * coef_show_window), int(input_image.shape[0] * coef_show_window)))
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, output_image)
    cv2.waitKey(10)
    return 0

def read_camera_stereo_parameters_from_file(path_to_file):
    fs = cv2.FileStorage(path_to_file, cv2.FileStorage_READ)
    if not fs.isOpened():
        fs.release()
        return None, 1
    camera_parameters = {}
    camera_parameters['M1'] = fs.getNode('M1').mat()
    camera_parameters['D1'] = fs.getNode('D1').mat()
    camera_parameters['M2'] = fs.getNode('M2').mat()
    camera_parameters['D2'] = fs.getNode('D2').mat()
    camera_parameters['E'] = fs.getNode('E').mat()
    camera_parameters['F'] = fs.getNode('F').mat()
    camera_parameters['R'] = fs.getNode('R').mat()
    camera_parameters['T'] = fs.getNode('T').mat()
    camera_parameters['R1'] = fs.getNode('R1').mat()
    camera_parameters['R2'] = fs.getNode('R2').mat()
    camera_parameters['P1'] = fs.getNode('P1').mat()
    camera_parameters['P2'] = fs.getNode('P2').mat()
    camera_parameters['Q'] = fs.getNode('Q').mat()
    camera_parameters['imageSize'] = (int(fs.getNode('imageSize').at(0).real()), int(fs.getNode('imageSize').at(1).real()))
    camera_parameters['rms'] = fs.getNode('rms').real()
    camera_parameters['avgErr'] = fs.getNode('avgErr').real()
    fs.release()

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_parameters['M1'], camera_parameters['D1'],
        camera_parameters['M2'], camera_parameters['D2'],
        camera_parameters['imageSize'],
        camera_parameters['R'], camera_parameters['T'],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1
    )
    camera_parameters['R1'] = R1
    camera_parameters['R2'] = R2
    camera_parameters['P1'] = P1
    camera_parameters['P2'] = P2
    camera_parameters['Q'] = Q

    M1n = P1[:, :3]
    M2n = P2[:, :3]
    camera_parameters['map11'], camera_parameters['map12'] = cv2.initUndistortRectifyMap(
        camera_parameters['M1'], camera_parameters['D1'], R1, M1n, camera_parameters['imageSize'], cv2.CV_16SC2
    )
    camera_parameters['map21'], camera_parameters['map22'] = cv2.initUndistortRectifyMap(
        camera_parameters['M2'], camera_parameters['D2'], R2, M2n, camera_parameters['imageSize'], cv2.CV_16SC2
    )
    return camera_parameters, 0

def converting_undistort_rectify(image_input, map11, map12):
    if image_input.size == 0:
        return get_error_image("convetingUndistortRectify:: Image is Empty"), 1
    image_output = cv2.remap(image_input, map11, map12, cv2.INTER_LINEAR)
    return image_output, 0

def find_3d_points_ads(input_image_camera01, input_image_camera02, settings_metod_disparity, camera_parameters, limit_3d_points, limits_outlier_area):
    points_3d = {
        'vu0': [], 'xyz0': [], 'rgb0': [], 'numPoints0': 0,
        'vu': [], 'xyz': [], 'rgb': [], 'segment': [], 'numPoints': 0,
        'numSegments': 0, 'pointsInSegments': [], 'numPointsInSegments': [], 'center2dSegments': [], 'center3dSegments': []
    }
    if input_image_camera01.size == 0 or input_image_camera02.size == 0:
        return points_3d, 1
    if settings_metod_disparity['metodDisparity'] == METOD_DISPARITY.MODE_NONE:
        return points_3d, 2
    if camera_parameters['imageSize'][0] == 0 or camera_parameters['imageSize'][1] == 0:
        return points_3d, 3

    image01_gray = cv2.cvtColor(input_image_camera01, cv2.COLOR_BGR2GRAY)
    image02_gray = cv2.cvtColor(input_image_camera02, cv2.COLOR_BGR2GRAY)

    if settings_metod_disparity['metodDisparity'] == METOD_DISPARITY.MODE_BM:
        stereo = cv2.StereoBM_create(
            numDisparities=settings_metod_disparity.get('smbNumDisparities', 16),
            blockSize=settings_metod_disparity.get('smbBlockSize', 15)
        )
        stereo.setPreFilterType(cv2.StereoBM_PREFILTER_XSOBEL)
        stereo.setPreFilterSize(7)
        stereo.setPreFilterCap(settings_metod_disparity.get('smbPreFilterCap', 31))
        stereo.setMinDisparity(settings_metod_disparity.get('smbMinDisparity', 0))
        stereo.setTextureThreshold(settings_metod_disparity.get('smbTextureThreshold', 10))
        stereo.setUniquenessRatio(settings_metod_disparity.get('smbUniquenessRatio', 15))
        stereo.setSpeckleWindowSize(settings_metod_disparity.get('smbSpeckleWindowSize', 100))
        stereo.setSpeckleRange(settings_metod_disparity.get('smbSpeckleRange', 32))
        stereo.setDisp12MaxDiff(settings_metod_disparity.get('smbDisp12MaxDiff', 1))
        disparity_map = stereo.compute(image01_gray, image02_gray)
    else:
        sgbm_win_size = settings_metod_disparity.get('smbBlockSize', 15)
        mode_map = {
            METOD_DISPARITY.MODE_SGBM: cv2.StereoSGBM_MODE_SGBM,
            METOD_DISPARITY.MODE_HH: cv2.StereoSGBM_MODE_HH,
            METOD_DISPARITY.MODE_SGBM_3WAY: cv2.StereoSGBM_MODE_SGBM_3WAY,
            METOD_DISPARITY.MODE_HH4: cv2.StereoSGBM_MODE_HH4
        }
        stereo = cv2.StereoSGBM_create(
            minDisparity=settings_metod_disparity.get('smbMinDisparity', 64),
            numDisparities=settings_metod_disparity.get('smbNumDisparities', 128),
            blockSize=sgbm_win_size,
            mode=mode_map.get(settings_metod_disparity['metodDisparity'], cv2.StereoSGBM_MODE_SGBM)
        )
        stereo.setPreFilterCap(settings_metod_disparity.get('smbPreFilterCap', 0))
        cn = image01_gray.shape[0]
        stereo.setP1(8 * cn * sgbm_win_size * sgbm_win_size)
        stereo.setP2(32 * cn * sgbm_win_size * sgbm_win_size)
        stereo.setUniquenessRatio(settings_metod_disparity.get('smbUniquenessRatio', 40))
        stereo.setSpeckleWindowSize(settings_metod_disparity.get('smbSpeckleWindowSize', 200))
        stereo.setSpeckleRange(settings_metod_disparity.get('smbSpeckleRange', 2))
        stereo.setDisp12MaxDiff(settings_metod_disparity.get('smbDisp12MaxDiff', 0))
        disparity_map = stereo.compute(image01_gray, image02_gray)

    disparity_map = disparity_map.astype(np.float32) / 16.0
    xyz_all_points = cv2.reprojectImageTo3D(disparity_map, camera_parameters['Q'], handleMissingValues=True)

    for v in range(xyz_all_points.shape[0]):
        for u in range(xyz_all_points.shape[1]):
            point = xyz_all_points[v, u]
            if np.any(np.isnan(point)):
                continue
            if (point[0] < limits_outlier_area[0] or point[0] > limits_outlier_area[3] or
                point[1] < limits_outlier_area[1] or point[1] > limits_outlier_area[4] or
                point[2] < limits_outlier_area[2] or point[2] > limits_outlier_area[5]):
                continue
            points_3d['vu0'].append([v, u])
            points_3d['xyz0'].append(point.tolist())
            rgb_value = input_image_camera01[v, u]
            points_3d['rgb0'].append(rgb_value.tolist())

    points_3d['numPoints0'] = len(points_3d['vu0'])
    churn = max(1, points_3d['numPoints0'] // limit_3d_points)
    for qi in range(0, points_3d['numPoints0'], churn):
        points_3d['vu'].append(points_3d['vu0'][qi])
        points_3d['xyz'].append(points_3d['xyz0'][qi])
        points_3d['rgb'].append(points_3d['rgb0'][qi])
        points_3d['segment'].append(-1)
    points_3d['numPoints'] = len(points_3d['vu'])

    return points_3d, disparity_map, 0

class NeuralNetSegmentator:
    def __init__(self, model, classes):
        self.input_width = 640
        self.input_height = 640
        self.classes = []
        self.masks_colors_set = []
        self.network = None
        self.time_inference = 0
        self.processed_image = None
        self.masks_set = []
        self.classes_id_set = []
        self.confidences_set = []
        self.boxes_set = []
        self.classes_set = []
        # Initialize mask_params
        self.mask_params = {
            'segChannels': 32,
            'netWidth': self.input_width,
            'netHeight': self.input_height,
            'segWidth': 160,  # Typical for YOLOv5-seg
            'segHeight': 160,  # Typical for YOLOv5-seg
            'maskThreshold': 0.5,
            'params': None,  # Will be set in process
            'srcImgShape': None  # Will be set in process
        }
        if not self.initialization_network(model, classes):
            write_log("Neural network been inited!")
            write_log(f"  Input width: {self.input_width}; Input height: {self.input_height}")
        else:
            write_log("Failed to init neural network!", "ERROR")

    def read_classes(self, file_path):
        with open(file_path, 'r') as f:
            self.classes = [line.strip() for line in f]
        self.masks_colors_set = [np.random.randint(0, 256, 3).tolist() for _ in self.classes]
        return 0

    def initialization_network(self, model_path, classes_path):
        err = self.read_classes(classes_path)
        if err == 0:
            try:
                self.network = cv2.dnn.readNetFromONNX(model_path)
                if self.network:
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        write_log("CUDA backend successfully initialized.")
                    else:
                        write_log("CUDA unavailable, using CPU.", "WARNING")
                        self.network.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                        self.network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                else:
                    write_log("Failed to load ONNX model!", "ERROR")
                    return -1
            except cv2.error as e:
                write_log(f"Network initialization error: {e}", "ERROR")
                return -1
        return err

    def letter_box(self, img, new_shape=(640, 640), auto_shape=False, scale_fill=False, scale_up=True, stride=32):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scale_up:
            r = min(r, 1.0)
        ratio = (r, r)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto_shape:
            dw, dh = dw % stride, dh % stride
        elif scale_fill:
            dw, dh = 0, 0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (new_shape[1] / shape[1], new_shape[0] / shape[0])
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
        params = (ratio[0], ratio[1], left, top)
        return img, params

    def pre_process(self, img):
        input_img, params = self.letter_box(img)
        blob = cv2.dnn.blobFromImage(input_img, 1.0 / 255.0, (self.input_width, self.input_height), swapRB=True, crop=False)
        self.network.setInput(blob)
        output_layer_names = ["output0", "output1"]
        outputs = self.network.forward(output_layer_names)
        return outputs, params

    def get_mask(self, mask_proposals, mask_protos, box, mask_params):
        seg_channels = mask_params['segChannels']
        net_width = mask_params['netWidth']
        seg_width = mask_params['segWidth']
        net_height = mask_params['netHeight']
        seg_height = mask_params['segHeight']
        mask_threshold = mask_params['maskThreshold']
        params = mask_params['matrix']
        src_img_shape = mask_params['srcImgShape']
        temp_rect = box

        rang_x = int((temp_rect[0] * params[0] + params[2]) / net_width * seg_width)
        rang_y = int((temp_rect[1] * params[1] + params[3]) / net_height * seg_height)
        rang_w = int(np.ceil(((temp_rect[0] + temp_rect[2]) * params[0] + params[2]) / net_width * seg_width)) - rang_x
        rang_h = int(np.ceil(((temp_rect[1] + temp_rect[3]) * params[1] + params[3]) / net_height * seg_height)) - rang_y

        rang_w = max(rang_w, 1)
        rang_h = max(rang_h, 1)
        if rang_x + rang_w > seg_width:
            if seg_width - rang_x > 0:
                rang_w = seg_width - rang_x
            else:
                rang_x -= 1
        if rang_y + rang_h > seg_height:
            if seg_height - rang_y > 0:
                rang_h = seg_height - rang_y
            else:
                rang_y -= 1

        temp_mask_protos = mask_protos[0, :, rang_y:rang_y + rang_h, rang_x:rang_x + rang_w]
        protos = temp_mask_protos.reshape(seg_channels, -1)
        matmul_res = np.dot(mask_proposals, protos).T
        masks_feature = matmul_res.reshape(rang_h, rang_w)
        dest = 1.0 / (1.0 + np.exp(-masks_feature))

        # Изменение размера до точных размеров bounding box
        box_width, box_height = temp_rect[2], temp_rect[3]
        mask = cv2.resize(dest, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
        mask = (mask > mask_threshold).astype(np.uint8)

        return mask

    def draw_label(self, img, label, left, top):
        baseline = 0
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
        top = max(top, label_size[1])
        tlc = (left, top - label_size[1])
        brc = (left + label_size[0], top + baseline)
        cv2.rectangle(img, tlc, brc, BLACK, cv2.FILLED)
        cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, YELLOW, THICKNESS)

    def draw_result(self, img, result, class_name):
        mask = img.copy()
        for res in result:
            box = res['box']
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), GREEN, 1)
            mask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]][res['boxMask']] = self.masks_colors_set[res['id']]
            label = f"{class_name[res['id']]}: {res['confidence']:.2f}"
            self.draw_label(img, label, box[0], box[1])
        cv2.addWeighted(img, 0.5, mask, 0.5, 0, img)

    def post_process(self, img, outputs, class_name, params):
        self.classes_id_set = []
        self.confidences_set = []
        self.boxes_set = []
        self.classes_set = []
        self.masks_set = []

        # Update mask_params with current parameters
        self.mask_params['matrix'] = params
        self.mask_params['srcImgShape'] = img.shape[:2]

        data = outputs[0].flatten()
        dimensions = len(class_name) + 5 + 32
        rows = 25200
        boxes = []
        confidences = []
        class_ids = []
        picked_proposals = []
        for i in range(rows):
            row = data[i * dimensions:(i + 1) * dimensions]
            confidence = row[4]
            if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = row[5:5 + len(class_name)]
                class_id = np.argmax(classes_scores)
                max_class_score = classes_scores[class_id]
                if max_class_score > SCORE_THRESHOLD:
                    x, y, w, h = row[0], row[1], row[2], row[3]
                    left = max(int((x - w / 2 - params[2]) / params[0]), 0)
                    top = max(int((y - h / 2 - params[3]) / params[1]), 0)
                    width = int(w / params[0])
                    height = int(h / params[1])
                    # Clip bounding box to image boundaries
                    right = min(left + width, img.shape[1])
                    bottom = min(top + height, img.shape[0])
                    width = right - left
                    height = bottom - top
                    if width > 0 and height > 0:  # Only append valid boxes
                        boxes.append([left, top, width, height])
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        picked_proposals.append(row[5 + len(class_name):])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD)
        output = []
        for i in indices:
            box = boxes[i]
            self.boxes_set.append(box)
            self.confidences_set.append(confidences[i])
            self.classes_id_set.append(class_ids[i])
            self.classes_set.append(class_name[class_ids[i]])
            result = {'id': class_ids[i], 'confidence': confidences[i], 'box': box}
            mask = self.get_mask(picked_proposals[i], outputs[1], box, self.mask_params)
            result['boxMask'] = mask
            output.append(result)
            canvas = np.zeros(img.shape[:2], dtype=np.uint8)
            try:
                canvas[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = mask
            except ValueError as e:
                write_log(f"Mask assignment failed: {e}", "ERROR")
                continue
            self.masks_set.append(canvas)

        self.draw_result(img, output, class_name)
        return img

    def process(self, img):
        outputs, params = self.pre_process(img)
        res = self.post_process(img, outputs, self.classes, params)
        self.processed_image = res
        # Timing inference
        start = time.time()
        self.network.forward()
        self.time_inference = time.time() - start
        return res

    def get_image(self):
        return self.processed_image

    def get_masks(self):
        return self.masks_set

    def get_class_ids(self):
        return self.classes_id_set

    def get_confidences(self):
        return self.confidences_set

    def get_boxes(self):
        return self.boxes_set

    def get_classes(self):
        return self.classes_set

    def get_inference(self):
        return self.time_inference

def detecting_segments_neural_net(image_input, file_path_to_model, file_path_to_classes):
    if image_input.size == 0:
        image_output = get_error_image("detectingSegmentsNeuralNet:: Image is Empty")
        return image_output, [], 1
    segmentator = NeuralNetSegmentator(file_path_to_model, file_path_to_classes)
    segmentator.process(image_input)
    reply_masks = segmentator.get_masks()
    image_output = segmentator.get_image()
    return image_output, reply_masks, 0

def match_segments_with_3d_points(points_3d, reply_masks):
    if points_3d['numPoints0'] < 1 or points_3d['numPoints'] < 1:
        return 1
    if not reply_masks:
        return 2
    points_3d['numSegments'] = len(reply_masks)
    points_3d['pointsInSegments'] = [[] for _ in range(points_3d['numSegments'])]
    points_3d['numPointsInSegments'] = [-1] * points_3d['numSegments']
    points_3d['center2dSegments'] = [None] * points_3d['numSegments']
    points_3d['center3dSegments'] = [None] * points_3d['numSegments']

    for qs in range(points_3d['numSegments']):
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        reply_masks[qs] = cv2.morphologyEx(reply_masks[qs], cv2.MORPH_CLOSE, se1)
        for qp in range(points_3d['numPoints']):
            v, u = points_3d['vu'][qp]
            if reply_masks[qs][v, u] != 0:
                points_3d['segment'][qp] = qs
                points_3d['pointsInSegments'][qs].append(qp)
        points_3d['numPointsInSegments'][qs] = len(points_3d['pointsInSegments'][qs])

    for qs in range(points_3d['numSegments']):
        if points_3d['numPointsInSegments'][qs] > 0:
            mean_v = np.mean([points_3d['vu'][qp][0] for qp in points_3d['pointsInSegments'][qs]])
            mean_u = np.mean([points_3d['vu'][qp][1] for qp in points_3d['pointsInSegments'][qs]])
            mean_x = np.mean([points_3d['xyz'][qp][0] for qp in points_3d['pointsInSegments'][qs]])
            mean_y = np.mean([points_3d['xyz'][qp][1] for qp in points_3d['pointsInSegments'][qs]])
            mean_z = np.mean([points_3d['xyz'][qp][2] for qp in points_3d['pointsInSegments'][qs]])
            points_3d['center2dSegments'][qs] = (int(mean_u), int(mean_v))
            points_3d['center3dSegments'][qs] = (mean_x, mean_y, mean_z)
    return 0

def add_to_image_center_3d_segments(input_image, points_3d):
    if input_image.size == 0:
        return input_image, 1
    if points_3d['numSegments'] < 1:
        return input_image, 3
    output_image = input_image.copy()
    for qs in range(points_3d['numSegments']):
        if points_3d['center3dSegments'][qs] is not None:
            label = f"z = {points_3d['center3dSegments'][qs][2]:.2f}"
            left, top = points_3d['center2dSegments'][qs]
            baseline = 0
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
            top = max(top, label_size[1])
            tlc = (left, top - label_size[1])
            brc = (left + label_size[0], top + baseline)
            cv2.rectangle(output_image, tlc, brc, YELLOW, cv2.FILLED)
            cv2.putText(output_image, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, RED, THICKNESS)
    return output_image, 0

def show_disparity_map(disparity_map, window_name, coef_show_window=1.0):
    if disparity_map.size == 0:
        disparity_map = get_error_image("showDispsarityMap:: Image is Empty")
        return 1
    min_val, max_val, _, _ = cv2.minMaxLoc(disparity_map)
    out_disparity = ((disparity_map - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    out_disparity = cv2.applyColorMap(out_disparity, cv2.COLORMAP_TURBO)
    out_disparity = cv2.resize(out_disparity, (int(out_disparity.shape[1] * coef_show_window), int(out_disparity.shape[0] * coef_show_window)))
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, out_disparity)
    cv2.waitKey(10)
    return 0

def get_image_3d_scene(points_3d, parameters_3d_scene, camera_parameters):
    if points_3d['numPoints0'] < 1 or points_3d['numPoints'] < 1:
        return None, 1
    img_size = camera_parameters['imageSize']
    output_image_3d_scene = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    ang_x, ang_y, ang_z = parameters_3d_scene['angX'], parameters_3d_scene['angY'], parameters_3d_scene['angZ']
    t_x, t_y, t_z = parameters_3d_scene['tX'], parameters_3d_scene['tY'], parameters_3d_scene['tZ']
    scale = parameters_3d_scene.get('scale', 1.0)
    d_z = parameters_3d_scene['dZ']

    coef_pi = np.pi / 180
    ang_x *= coef_pi
    ang_y *= coef_pi
    ang_z *= coef_pi

    Rx = np.array([[1, 0, 0], [0, np.cos(ang_x), -np.sin(ang_x)], [0, np.sin(ang_x), np.cos(ang_x)]])
    Ry = np.array([[np.cos(ang_y), 0, np.sin(ang_y)], [0, 1, 0], [-np.sin(ang_y), 0, np.cos(ang_y)]])
    Rz = np.array([[np.cos(ang_z), -np.sin(ang_z), 0], [np.sin(ang_z), np.cos(ang_z), 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz
    T = np.array([t_x, t_y, t_z])

    P = np.hstack((np.eye(3), np.array([[0], [0], [-d_z]])))
    P = camera_parameters['M1'] @ P

    for qp in range(points_3d['numPoints0']):
        xyz = np.array(points_3d['xyz0'][qp]) * scale
        xyz = R @ (xyz + T)
        xyz1 = np.append(xyz, 1)
        uv1 = P @ xyz1
        uv = uv1[:2] / uv1[2]
        r, c = int(round(uv[1])), int(round(uv[0]))
        if 0 <= r < img_size[1] and 0 <= c < img_size[0]:
            v, u = points_3d['vu0'][qp]
            output_image_3d_scene[r, c] = points_3d['rgb0'][qp]

    return output_image_3d_scene, 0

def save_in_file_3d_points_in_objects_segments(points_3d, path_to_file):
    if points_3d['numPoints0'] < 1 or points_3d['numPoints'] < 1:
        return 1
    with open(path_to_file, 'w') as f:
        for qp in range(points_3d['numPoints']):
            if points_3d['segment'][qp] != -1:
                f.write(f"{points_3d['vu'][qp][0]}\t{points_3d['vu'][qp][1]}\t"
                        f"{points_3d['xyz'][qp][0]}\t{points_3d['xyz'][qp][1]}\t{points_3d['xyz'][qp][2]}\t"
                        f"{points_3d['rgb'][qp][0]}\t{points_3d['rgb'][qp][1]}\t{points_3d['rgb'][qp][2]}\t"
                        f"{points_3d['segment'][qp]}\n")
    return 0

def find_3d_points_in_objects_segments(input_image_camera01, input_image_camera02, camera_parameters,
                                       settings_metod_disparity, limit_out_points, limits_outlier_area,
                                       file_path_to_model, file_path_to_classes, parameters_3d_scene):
    input_image_camera01_remap, state = converting_undistort_rectify(input_image_camera01, camera_parameters['map11'], camera_parameters['map12'])
    if state != 0:
        write_log(f"convetingUndistortRectify 01, status = {state}", "ERROR")
    input_image_camera02_remap, state = converting_undistort_rectify(input_image_camera02, camera_parameters['map21'], camera_parameters['map22'])
    if state != 0:
        write_log(f"convetingUndistortRectify 02, status = {state}", "ERROR")

    points_3d, disparity_map, state = find_3d_points_ads(
        input_image_camera01_remap, input_image_camera02_remap,
        settings_metod_disparity, camera_parameters, limit_out_points, limits_outlier_area
    )
    if state == 0:
        write_log("A2. 3D point cloud found (success)")
        write_log(f"points3D['numPoints0'] = {points_3d['numPoints0']}")
        write_log(f"points3D['numPoints'] = {points_3d['numPoints']}")
        write_log(f"points3D['numSegments'] = {points_3d['numSegments']}")
    else:
        write_log(f"find3dPointsADS, status = {state}", "ERROR")

    output_image, reply_masks, state = detecting_segments_neural_net(
        input_image_camera01_remap, file_path_to_model, file_path_to_classes
    )
    if state == 0:
        write_log("A3. Image segmentation (success)")
        write_log(f"    neural network model path: {file_path_to_model}")
        write_log(f"    replyMasks.size() = {len(reply_masks)}")
    else:
        write_log(f"detectingSegmentsNeuralNet, status = {state}", "ERROR")

    state = match_segments_with_3d_points(points_3d, reply_masks)
    if state == 0:
        write_log("A4. Matching coordinates and segments (success)")
        write_log(f"    points3D['numSegments'] = {points_3d['numSegments']}")
        for qs in range(points_3d['numSegments']):
            write_log(f"    points in segment {qs} = {points_3d['numPointsInSegments'][qs]}; 3D center: {points_3d['center3dSegments'][qs]}")
    else:
        write_log(f"matchSegmentsWith3dPoints, status = {state}", "ERROR")

    output_image, state = add_to_image_center_3d_segments(output_image, points_3d)
    if state == 0:
        write_log("A5. Adding 3D segment centers to image (success)")
    else:
        write_log(f"addToImageCenter3dSegments, status = {state}", "ERROR")

    output_image_3d_scene, state = get_image_3d_scene(points_3d, parameters_3d_scene, camera_parameters)
    if state == 0:
        write_log("A6. Projecting 3D scene to 2D image (success)")
    else:
        write_log(f"getImage3dSceene, status = {state}", "ERROR")

    return output_image, output_image_3d_scene, points_3d, reply_masks, disparity_map, 0
