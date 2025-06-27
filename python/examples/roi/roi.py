import cv2
import numpy as np
import time
from mrcv import Predictor, Optimizer, generate_coordinates, extract_roi, to_point

# Основная программа
if __name__ == "__main__":
    # Инициализация переменных
    img_size = (1440, 1080)
    predictor_train_points_num = 50
    total_points_num = 500
    object_size = 100
    max_error = 200
    draw_obj = True
    gen_type = 1
    R = 300
    time_fracture = 100
    hidden_size = 20
    layers_num = 1
    predictor = Predictor(hidden_size, layers_num, predictor_train_points_num, img_size, max_error)
    epochs = 50000
    sample_size = 1000
    optimizer = Optimizer(sample_size, epochs)

    # Генерация начальных координат для обучения
    coordinates = []
    for i in range(1, predictor_train_points_num + 1):
        real_coordinate = generate_coordinates(i, gen_type, R, time_fracture, img_size)
        coordinates.append(real_coordinate)

    # Обучение предиктора
    predictor.train_lstm_net(coordinates)

    # Основной цикл
    start_time = time.time()
    tmp_real_coordinate = real_coordinate
    predicted_coordinate = None
    tmp_predicted_coordinate = real_coordinate
    img_r = np.full((img_size[1], img_size[0], 3), 255, dtype=np.uint8)
    roi = np.zeros((object_size, object_size, 3), dtype=np.uint8)
    image_full = np.zeros((img_r.shape[0], img_r.shape[1] + roi.shape[1], 3), dtype=np.uint8)
    image_full[:roi.shape[0], :roi.shape[1]] = roi
    image_full[:, roi.shape[1]:] = img_r
    roi_size_acquired = False
    roi_size = 0
    wk = 1
    roi_tries = 0

    for i in range(predictor_train_points_num + 1, total_points_num + 1):
        tmp_predicted_coordinate = predicted_coordinate
        predicted_coordinate = predictor.predict_next_coordinate()
        if tmp_predicted_coordinate:
            cv2.line(img_r, to_point(tmp_predicted_coordinate), to_point(predicted_coordinate), (255, 0, 0), 1)
        if draw_obj:
            img_r = np.full((img_size[1], img_size[0], 3), 255, dtype=np.uint8)
            center = to_point(real_coordinate)
            cv2.circle(img_r, center, object_size // 2, (0, 124, 0), 1)

        if predictor.is_work_state() and not roi_size_acquired:
            roi_tries += 1
            roi_size = optimizer.optimize_roi_size(tmp_real_coordinate, real_coordinate,
                                                   object_size, predictor.get_moving_average_deviation() / 2)
            if roi_size:
                print(f"Оптимизированный размер ROI: {roi_size}")
                roi_size_acquired = True
                print(f"Для корректного получения ROI потребовалось {roi_tries} попыток")

        if predictor.is_work_state() and roi_size_acquired:
            roi = extract_roi(img_r, to_point(predicted_coordinate), (int(roi_size), int(roi_size)))

        image_full = np.zeros((img_r.shape[0], img_r.shape[1] + roi.shape[1], 3), dtype=np.uint8)
        image_full[:roi.shape[0], :roi.shape[1]] = roi
        image_full[:, roi.shape[1]:] = img_r

        if predictor.is_work_state() and roi_size_acquired:
            cv2.putText(image_full, "ROI acting", (10, 600), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 124, 0), 2)
        else:
            cv2.putText(image_full, "ROI training", (10, 600), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 124, 124), 2)

        cv2.imshow("image", image_full)
        cv2.waitKey(wk)

        tmp_real_coordinate = real_coordinate
        real_coordinate = generate_coordinates(i, gen_type, R, time_fracture, img_size)
        predictor.continue_training(real_coordinate)
        cv2.line(img_r, to_point(tmp_real_coordinate), to_point(real_coordinate), (0, 0, 255), 1)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Прошедшее время: {time_elapsed}")
    print(f"FPS: {(total_points_num - predictor_train_points_num) / time_elapsed}")
    print(f"Среднее отклонение: {predictor.get_average_deviation()}")

    cv2.imshow("image", image_full)
    cv2.waitKey(0)
    cv2.destroyAllWindows()