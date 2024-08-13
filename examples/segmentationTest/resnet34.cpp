#include <mrcv/mrcv.h>

#include <iostream>

int main() {

  auto image_path = "../images/test/43.jpg";

  cv::Mat image = cv::imread(image_path);

  mrcv::Segmentor segmentor;

  segmentor.Initialize(-1, 512, 320, {"background","ship"}, "resnet34", "../weights/resnet34.pt");
  segmentor.LoadWeight("../weights/segmentor.pt");
  segmentor.Predict(image, "ship");

  return 0;
}
