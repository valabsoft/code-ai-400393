from mrcv import yolov5_labeler_processing, yolov5_generate_config, yolov5_generate_hyperparameters, YOLOv5Model

yolov5_labeler_processing("path/to/input/dir", "path/to/output/dir")
yolov5_generate_config(YOLOv5Model.YOLOv5s, "config.yaml", 10)
yolov5_generate_hyperparameters(YOLOv5Model.YOLOv5s, 640, 640, "hyperparameters.yaml", 10)