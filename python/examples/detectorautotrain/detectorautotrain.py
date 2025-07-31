from mrcv import Detector

voc_classes_path = "../../../examples/detectorautotrain/files/onwater/voc_classes.txt"
dataset_path = "../../../examples/detectorautotrain/files/onwater/"
pretrained_model_path = "../../../examples/detectorautotrain/files/onwater_autodetector.pt"
model_save_path = "../../../examples/detectorautotrain/files/yolo4_tiny.pt"

detector = Detector()
detector.initialize(416, 416, voc_classes_path)
detector.auto_train(
    dataset_path,
    ".jpg",
    epochs_list=[10, 15, 30],
    batch_sizes=[4, 8],
    learning_rates=[0.001, 1e-4],
    pretrained_path=pretrained_model_path,
    save_path=model_save_path
)