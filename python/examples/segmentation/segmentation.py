from mrcv import Segmentor
import cv2

if __name__ == "__main__":
    image_path = "segmentation/file/images/test/43.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        exit(1)

    segmentor = Segmentor()
    segmentor.initialize(-1, 512, 320, ["background", "ship"], "resnet34", "segmentation/file/weights/resnet34.pt")
    segmentor.load_weight("segmentation/file/weights/segmentor.pt")
    segmentor.predict(image, "ship")