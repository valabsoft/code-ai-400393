import torch
import torch.nn as nn
import math

# Функция генерации координат
def generate_coordinates(time, gen_type=0, R=500, time_fracture=1, img_size=(1400, 1080)):
    dt = time / time_fracture
    if gen_type == 0:
        x = time + 200
        y = math.sin(dt) * R + R + (img_size[1] / 2 - R)
        return (x, y)
    elif gen_type == 1:
        x = math.sin(dt) * R + img_size[0] / 2
        y = math.cos(dt) * R + img_size[1] / 2
        return (x, y)
    return (0, 0)


# Функция преобразования пары в точку
def to_point(pair):
    return (int(pair[0]), int(pair[1]))


# Класс Predictor для предсказания координат
class Predictor:
    def __init__(self, hidden_size, layers_num, train_points_num, img_size, max_error):
        self.hidden_size = hidden_size
        self.layers_num = layers_num
        self.input_size = 2  # x и y координаты
        self.img_width, self.img_height = img_size
        self.max_error = max_error
        self.moving_avg_scale = 10
        self.failsafe_deviation = max_error
        self.failsafe_deviation_threshold = 5
        self.lstm = nn.LSTM(self.input_size, hidden_size, layers_num, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.input_size)
        self.training_data = []
        self.hidden_state = None
        self.cell_state = None
        self.coords_real = None
        self.coords_pred = None
        self.num_predictions = 0
        self.prediction_deviation = 0
        self.last_prediction_deviations = []
        self.average_prediction_deviation = 0
        self.moving_avg_prediction_deviation = 0
        self.work_state = False
        self.successed_predictions = 0

    def normalize_pair(self, coords):
        return (coords[0] / self.img_width * 2 - 1, coords[1] / self.img_height * 2 - 1)

    def denormalize_output(self, coords):
        return ((coords[0] + 1) * self.img_width / 2, (coords[1] + 1) * self.img_height / 2)

    def normalize_input(self, coords):
        return [self.normalize_pair(coord) for coord in coords]

    def train_lstm_net(self, coordinates, is_training=False):
        coordinates_normalized = self.normalize_input(coordinates)
        if coordinates_normalized:
            self.training_data = [torch.tensor([coord], dtype=torch.float32).view(1, 1, self.input_size)
                                  for coord in coordinates_normalized]
        if not self.training_data:  # Проверка на пустой training_data
            return
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.lstm.parameters()) + list(self.linear.parameters()), lr=0.001)
        self.lstm.train()
        for epoch in range(50):
            optimizer.zero_grad()
            inputs = torch.cat(self.training_data, dim=0)
            targets = inputs.clone()
            batch_size = inputs.size(0)  # Динамически определяем размер пакета
            # Проверяем и обновляем скрытые состояния, если их размер не соответствует
            if self.hidden_state is None or self.hidden_state.size(1) != batch_size:
                self.hidden_state = torch.zeros(self.layers_num, batch_size, self.hidden_size)
                self.cell_state = torch.zeros(self.layers_num, batch_size, self.hidden_size)
            # Отсоединяем скрытые состояния перед новым проходом, чтобы избежать старых графов
            self.hidden_state = self.hidden_state.detach()
            self.cell_state = self.cell_state.detach()
            outputs, (self.hidden_state, self.cell_state) = self.lstm(inputs, (self.hidden_state, self.cell_state))
            outputs = self.linear(outputs.view(-1, self.hidden_size))
            outputs = outputs.view(-1, 1, self.input_size)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)  # Сохраняем граф для следующей итерации
            nn.utils.clip_grad_norm_(self.lstm.parameters(), 0.1)
            optimizer.step()
        # Финально отсоединяем состояния, чтобы очистить граф
        self.hidden_state = self.hidden_state.detach()
        self.cell_state = self.cell_state.detach()

    def continue_training(self, coordinate):
        self.coords_real = coordinate
        self.update_deviations()
        coordinate_norm = self.normalize_pair(coordinate)
        input_tensor = torch.tensor([coordinate_norm], dtype=torch.float32).view(1, 1, self.input_size)
        self.training_data.append(input_tensor)
        if len(self.training_data) > 100:
            self.training_data.pop(0)
        self.train_lstm_net([], True)

    def predict_next_coordinate(self):
        self.lstm.eval()
        self.linear.eval()
        with torch.no_grad():
            inputs = torch.cat(self.training_data, dim=0)
            batch_size = inputs.size(0)
            if self.hidden_state is None or self.hidden_state.size(1) != batch_size:
                self.hidden_state = torch.zeros(self.layers_num, batch_size, self.hidden_size)
                self.cell_state = torch.zeros(self.layers_num, batch_size, self.hidden_size)
            outputs, (self.hidden_state, self.cell_state) = self.lstm(inputs, (self.hidden_state, self.cell_state))
            outputs = self.linear(outputs[-1])
            coords_pred_norm = outputs[0].tolist()
            self.coords_pred = self.denormalize_output(coords_pred_norm)
            self.num_predictions += 1
            return self.coords_pred

    def update_deviations(self):
        if self.coords_pred and self.coords_real:
            pred_dev = math.sqrt((self.coords_pred[0] - self.coords_real[0]) ** 2 +
                                 (self.coords_pred[1] - self.coords_real[1]) ** 2)
            self.prediction_deviation = pred_dev
            self.last_prediction_deviations.append(pred_dev)
            if len(self.last_prediction_deviations) > self.moving_avg_scale:
                self.last_prediction_deviations.pop(0)
            moving_avg_pred_dev_sum = sum(self.last_prediction_deviations)
            pred_dev_sum = sum(self.last_prediction_deviations)
            self.average_prediction_deviation = pred_dev_sum / self.num_predictions if self.num_predictions > 0 else 0
            self.moving_avg_prediction_deviation = (moving_avg_pred_dev_sum / len(self.last_prediction_deviations)
                                                    if self.last_prediction_deviations else 0)
            if self.moving_avg_prediction_deviation < self.failsafe_deviation:
                self.successed_predictions += 1
            else:
                self.successed_predictions = 0
            self.work_state = self.successed_predictions > self.failsafe_deviation_threshold

    def get_moving_average_deviation(self):
        return self.moving_avg_prediction_deviation

    def get_average_deviation(self):
        return self.average_prediction_deviation

    def get_last_deviation(self):
        return self.prediction_deviation

    def is_work_state(self):
        return self.work_state


# Класс Optimizer (упрощённая версия)
class Optimizer:
    def __init__(self, sample_size, epochs):
        self.sample_size = sample_size
        self.epochs = epochs

    def optimize_roi_size(self, prev_coord, curr_coord, object_size, deviation):
        if deviation < object_size / 2:
            return object_size
        return 0


# Функция извлечения ROI
def extract_roi(image, center, roi_size):
    left_upper_corner_x = center[0] - roi_size[0] // 2
    left_upper_corner_y = center[1] - roi_size[1] // 2
    left_upper_corner_x = max(0, left_upper_corner_x)
    left_upper_corner_y = max(0, left_upper_corner_y)
    width = roi_size[0]
    height = roi_size[1]
    if left_upper_corner_x + width > image.shape[1]:
        width = image.shape[1] - left_upper_corner_x
    if left_upper_corner_y + height > image.shape[0]:
        height = image.shape[0] - left_upper_corner_y
    roi = image[left_upper_corner_y:left_upper_corner_y + height,
          left_upper_corner_x:left_upper_corner_x + width].copy()
    return roi