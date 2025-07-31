import cv2
from mrcv import read_camera_stereo_parameters_from_file,\
    write_log, \
    METOD_DISPARITY, \
    find_3d_points_in_objects_segments,\
    show_image, \
    save_in_file_3d_points_in_objects_segments, \
    show_disparity_map, \
    converting_undistort_rectify, \
    making_stereo_pair


def main():
    write_log("=== NEW RUN ===")

    # Load images
    input_image_camera01 = cv2.imread("../../../examples/3dscene/files/L1000.bmp")
    input_image_camera02 = cv2.imread("../../../examples/3dscene/files/R1000.bmp")
    if input_image_camera01 is None or input_image_camera02 is None:
        write_log("Failed to load images", "ERROR")
        return
    write_log("1. Loading images from file (success)")
    write_log(
        f"    loaded image: ./files/L1000.bmp :: {input_image_camera01.shape[1]}x{input_image_camera01.shape[0]}x{input_image_camera01.shape[2]}")
    write_log(
        f"    loaded image: ./files/R1000.bmp :: {input_image_camera02.shape[1]}x{input_image_camera02.shape[0]}x{input_image_camera02.shape[2]}")

    # Load camera parameters
    camera_parameters, state = read_camera_stereo_parameters_from_file(
        "../../../examples/3dscene/files/(66a)_(960p)_NewCamStereoModule_Air.xml")
    if state == 0:
        write_log("2. Loading stereo camera parameters from file (success)")
    else:
        write_log(f"readCameraStereoParametrsFromFile, status = {state}", "ERROR")

    # Initialize parameters
    settings_metod_disparity = {'metodDisparity': METOD_DISPARITY.MODE_SGBM}
    limit_out_points = 8000
    limits_outlier_area = [-4.0e3, -4.0e3, 450, 4.0e3, 4.0e3, 3.0e3]
    file_path_model_yolo_neural_net = "../../../examples/3dscene/files/NeuralNet/yolov5n-seg.onnx"
    file_path_classes = "../../../examples/3dscene/files/NeuralNet/yolov5.names"
    parameters_3d_scene = {
        'angX': 25, 'angY': 45, 'angZ': 35,
        'tX': -200, 'tY': 200, 'tZ': -600,
        'dZ': -1000, 'scale': 1.0
    }

    # Process images
    output_image, output_image_3d_scene, points_3d, reply_masks, disparity_map, state = find_3d_points_in_objects_segments(
        input_image_camera01, input_image_camera02, camera_parameters,
        settings_metod_disparity, limit_out_points, limits_outlier_area,
        file_path_model_yolo_neural_net, file_path_classes, parameters_3d_scene
    )

    # Display results
    foto_experimental_stand = cv2.imread("../../../examples/3dscene/files/experimantalStand.jpg")
    show_image(foto_experimental_stand, "fotoExperimantalStand")

    output_stereo_pair, state = making_stereo_pair(input_image_camera01, input_image_camera02)
    if state == 0:
        show_image(output_stereo_pair, "SourceStereoImage")
        write_log("4.2 Displaying source image (success)")
    else:
        write_log(f"makingStereoPair (outputStereoPair) status = {state}", "ERROR")

    input_image_camera01_remap, _ = converting_undistort_rectify(input_image_camera01, camera_parameters['map11'],
                                                                 camera_parameters['map12'])
    input_image_camera02_remap, _ = converting_undistort_rectify(input_image_camera02, camera_parameters['map21'],
                                                                 camera_parameters['map22'])
    output_stereo_pair_remap, state = making_stereo_pair(input_image_camera01_remap, input_image_camera02_remap)
    show_image(output_stereo_pair_remap, "outputStereoPairRemap")
    if state == 0:
        write_log("4.3 Displaying rectified stereo pair (success)")
    else:
        write_log(f"4.3 Displaying rectified stereo pair, status = {state}", "ERROR")

    show_disparity_map(disparity_map, "disparityMap")
    write_log("4.4 Displaying disparity map (success)")

    for qs, mask in enumerate(reply_masks):
        show_image(mask, f"replyMasks {qs}", 0.5)
    write_log("4.5 Displaying binary segment images (success)")

    path_to_file = "../../../examples/3dscene/files/3DPointsInObjectsSegments.txt"
    state = save_in_file_3d_points_in_objects_segments(points_3d, path_to_file)
    if state == 0:
        write_log("4.6 Saving 3D points to text file (success)")
        write_log(f"    file path: {path_to_file}")
    else:
        write_log(f"4.6 Saving 3D points to text file, status = {state}", "ERROR")

    show_image(output_image, "outputImage", 1.0)
    write_log("4.7 Displaying image with 3D segment centers (success)")

    show_image(output_image_3d_scene, "outputImage3dSceene", 1.0)
    write_log("4.8 Displaying 3D scene projection (success)")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()