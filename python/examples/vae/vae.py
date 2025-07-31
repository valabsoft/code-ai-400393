import os
import cv2
from mrcv import neural_network_augmentation_as_mat, semi_automatic_labeler_image

images_path = "../../../examples/vae/files/images"
result_path = "../../../examples/vae/files/result"
model_path = "../../../examples/vae/files/ship.onnx"
class_path = "../../../examples/vae/files/ship.names"

height = 640
width = 640

genImage = neural_network_augmentation_as_mat(images_path, height, width, 200, 2, 2, 16, 3E-4)

colorGenImage = cv2.cvtColor(genImage, cv2.COLOR_GRAY2BGR)

output_path = os.path.join(result_path, "generated.jpg")
cv2.imwrite(output_path, colorGenImage)

semi_automatic_labeler_image(colorGenImage, 640, 640, result_path, model_path, class_path)