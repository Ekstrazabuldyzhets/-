import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

'''
BatteryDataset пользовательский класс набора данных
'''
# Общий базовый класс для CNN LSTM и TCN
class BatteryDataset():
    def __init__(self, data_tensor, labels_tensor, sequence_length=50, filenames=None, times=None):
        self.sequence_length = sequence_length
        self.features = data_tensor
        self.labels = labels_tensor
        self.filenames = filenames
        self.times = times

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        # Определение конца последовательности
        idx_end = idx + self.sequence_length
        # Извлечение последовательности признаков
        sequence = self.features[idx:idx_end]
        # Извлечение метки (SOC на последнем шаге)
        label = self.labels[idx_end - 1]
        # Дополнительная информация
        filename = self.filenames[idx_end - 1]
        time = self.times[idx_end - 1]

        # Создание копий для безопасности
        sequence = sequence.clone().detach()
        label = label.clone().detach()

        return sequence, label, filename, time

'''
LSTM
'''
# Специализированный класс LSTM
class BatteryDatasetLSTM(BatteryDataset):
    # Можно добавить LSTM-специфичную логику
    pass

# SoCLSTM Model Функция возвращает тензор предсказанных значений SOC для каждой входной последовательности в батче.
class SoCLSTM(nn.Module):
    # input_size=5 - 5 входных признаков (напряжение, ток, температура, мощность, емкость)
    def __init__(self, input_size, hidden_size, num_layers):
        super(SoCLSTM, self).__init__()
        self.hidden_size = hidden_size # hidden_size - размер скрытого состояния (настраивается Optuna)
        self.num_layers = num_layers # num_layers - количество LSTM слоев (настраивается Optuna)

        # Инициализация
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # LSTM - рекуррентная архитектура
        self.fc = nn.Linear(hidden_size, 1) # преобразует в одно число SOC = w₁·h₁ + w₂·h₂ + ... + w₅₀·h₅₀ + b

    def forward(self, x):
        # x - входной тензор с размерностью: [batch_size, sequence_length, input_size]
        # Инициализация скрытых состояний h0 (hidden state) - краткосрочная память:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        # Инициализация скрытых состояний c0 (cell state) - долгосрочная память:
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)

        # моделим РАЗОБРАТЬСЯ В LSTM
        out, _ = self.lstm(x, (h0, c0)) # Берем out, игнорируем hn, cn
        out = self.fc(out[:, -1, :])
        return out

'''
CNN - сверточная нейронная сеть
'''
class BatteryDatasetCNN(BatteryDataset):
    # Можно добавить CNN-специфичную логику
    pass

class SoCCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3, dropout=0.2):
        super(SoCCNN, self).__init__()
        self.input_size = input_size  # 5 признаков (напряжение, ток, ...)
        self.hidden_size = hidden_size  # размер скрытого представления
        self.num_layers = num_layers  # количество CNN блоков
        self.kernel_size = kernel_size  # размер ядра
        self.dropout_rate = dropout  # защита от переобучения

        # Создание CNN блоков
        self.cnn_blocks = nn.ModuleList()
        in_channels = input_size

        for i in range(num_layers):
            out_channels = hidden_size * (2 ** i)  # Увеличиваем каналы в 2 раза каждый слой

            cnn_block = CNNBlock(
                in_channels,
                out_channels,
                kernel_size,
                dropout=dropout
            )
            self.cnn_blocks.append(cnn_block)
            in_channels = out_channels

        # Глобальный пуллинг и финальные слои
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # сводим всю последовательность к 1 значению
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x - входной тензор с размерностью: [batch_size, sequence_length, input_size]
        batch_size, seq_len, input_size = x.shape

        # меняем оси для свертки: [128, 5, 20]
        x = x.transpose(1, 2)

        # применяем все CNN блоки
        for cnn_block in self.cnn_blocks:
            x = cnn_block(x)  # Каждый блок извлекает признаки

        # глобальный пуллинг: [128, channels, seq_len] → [128, channels, 1]
        x = self.global_avg_pool(x)  # Усредняем по всей последовательности
        x = x.squeeze(-1)  # [128, channels, 1] → [128, channels]

        # полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Финальный прогноз SOC

        return x

class CNNBlock(nn.Module):
    # in_channels - количество исходных признаков/измерений
    # out_channels - количество ядер
    # kernel_size - размер ядра (ядро - это матрица весов размером [kernel_size × in_channels])
    # dropout - вероятность отключения нейронов для регуляризации
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super(CNNBlock, self).__init__()

        # Основные сверточные слои
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        # Pooling для уменьшения длины последовательности (опционально)
        self.pool = nn.MaxPool1d(2) if in_channels != out_channels else None

    def forward(self, x):
        residual = x  # сохраняем оригинальный вход

        # первая свертка + активация через relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # вторая свертка + активация
        out = self.conv2(out)
        out = self.bn2(out)

        # Остаточное соединение
        if self.downsample is not None:
            residual = self.downsample(residual)

        # Складываем и активируем
        out = out + residual
        out = F.relu(out)
        out = self.dropout(out)

        # уменьшаем длину последовательности если нужно
        if self.pool is not None:
            out = self.pool(out)

        return out

'''
Classic KAL - фильтр Калмана
'''

# Класс фильтра Калмана для SOC estimation
class KalmanFilterSOC:
    def __init__(self, process_noise=0.1, measurement_noise=0.1, initial_soc=1.0):
        self.Q = process_noise  # (насколько неточна наша модель)
        self.R = measurement_noise  # (насколько неточны датчики)
        self.soc = initial_soc  # оценка SOC
        self.P = 1.0  # (насколько мы доверяем текущей оценке)
        self.dt = 1.0  # временной шаг

    def set_time_step(self, time_step):
        # Обновляем временной шаг из реальных данных
        self.dt = time_step

    # 2) prediction step с использованием кумулятивного заряда
    def predict(self, current, cumulative_charge, total_capacity):
        # 2.1) используем кумулятивный заряд из данных (более точный) ΔSOC = ΔCumulative_Charge / Total_Capacity
        delta_soc = cumulative_charge / total_capacity
        # 2.2) обновляем SOC
        self.soc = self.soc - delta_soc
        # 2.3) обновление ковариации ошибки P = P + Q - увеличиваем неопределенность после предсказания
        self.P = self.P + self.Q

        return self.soc

    # 3) коррекия
    def update(self, voltage_measurement, predicted_voltage, power=None):
        # 3.1) ошибка между измеренным и предсказанным напряжением
        y = voltage_measurement - predicted_voltage

        # 3.2) коэффициент усиления Калмана
        # Innovation covariance (ковариация инноваций)
        S = self.P + self.R  # S = P + R (суммарная неопределенность)
        # Kalman gain (коэффициент усиления Калмана)
        K = self.P / S  # K = P / S (насколько доверять измерениям)

        # адаптивная чувствительность на основе мощности
        if power is not None and abs(power) > 10:  # Высокая мощность
            voltage_to_soc_sensitivity = 0.02  # Большая чувствительность
        else:  # Низкая мощность
            voltage_to_soc_sensitivity = 0.01  # Стандартная чувствительность

        # обновление оценки SOC
        self.soc =  self.soc + K * y * voltage_to_soc_sensitivity
        # обновление ковариации ошибки
        self.P = (1 - K) * self.P

        return self.soc

    # 1) цикл фильтра Калмана: prediction + update
    def estimate_soc(self, current, cumulative_charge, total_capacity, voltage_measurement, predicted_voltage,
                     time_step=None, power=None):
        # 1.0) # временной шаг если предоставлен
        self.set_time_step(time_step)
        # 1.1) # шаг предсказания
        self.predict(current, cumulative_charge, total_capacity)
        # 1.2) # шаг коррекции
        soc_estimate = self.update(voltage_measurement, predicted_voltage, power)
        return soc_estimate

# Класс модели на основе фильтра Калмана
class KalmanSOCModel:
    def __init__(self, process_noise=0.01, measurement_noise=0.1, capacity=3.0):
        self.process_noise = process_noise # шум процесса для фильтра
        self.measurement_noise = measurement_noise # шум измерений для фильтра
        self.capacity = capacity # емкость батареи в Ah
        self.kf = None # экземпляр фильтра
    # 2) создаем экземпляр филтра
    def initialize_filter(self, initial_soc=1.0):
        self.kf = KalmanFilterSOC(process_noise=self.process_noise,
                                  measurement_noise=self.measurement_noise,
                                  initial_soc=initial_soc)

    # 3) предсказываем напряжение
    def predict_voltage(self, current, temperature, soc, power=None, cumulative_charge=None):
        # OCV кривая - зависимость напряжения холостого хода от SOC
        if soc > 0.8:
            open_circuit_voltage = 4.1 + 0.1 * (soc - 0.8)  # Высокий SOC
        elif soc > 0.2:
            open_circuit_voltage = 3.7 + 0.5 * (soc - 0.2)  # Средний SOC
        else:
            open_circuit_voltage = 3.0 + 3.5 * soc  # Низкий SOC

        # 3.1) базовые компоненты внутреннего сопротивления
        base_resistance = 0.05  # минимальное сопротивление новой батареи
        soc_dependent_resistance = 0.02 * (1 - soc)  # сопротивление растет при разряде
        temp_dependent_resistance = 0.01 * (25 - temperature) / 25  # сопротивление растет при охлаждении

        # 3.2) коэффициент мощности - учет нагрева при высокой мощности нагрев изменяет ионную проводимость электролита.
        power_factor = 1.0
        if power is not None:
            if abs(power) > 15:  # Высокая мощность > 15W
                power_factor = 1.2  # +20% сопротивление из-за нагрева
            elif abs(power) < 5:   # Низкая мощность < 5W
                power_factor = 0.9  # -10% сопротивление

        # 3.3) коэффициент старения - сопротивление растет с "пробегом" батареи
        # цикл заряда-разряда необратимо повреждает электроды.
        aging_factor = 1.0
        if cumulative_charge is not None:
            # Увеличиваем сопротивление на 0.01% за каждый А·ч пропущенного заряда
            aging_factor = 1.0 + 0.0001 * abs(cumulative_charge)

        # 3.4) Итоговое внутреннее сопротивление
        internal_resistance = (base_resistance + soc_dependent_resistance +
                             temp_dependent_resistance) * power_factor * aging_factor

        # 3.5) Напряжение под нагрузкой: V = OCV - I×R
        voltage = open_circuit_voltage - current * internal_resistance

        return voltage

    # 1) оценим SOC для полной последовательности данных
    def estimate(self, data_sequence, time_steps=None):
        # 1.1) создаем фильтр
        if self.kf is None:
            self.initialize_filter()

        # 1.2) список для хранения оценок SOC
        soc_estimates = []

        # 1.3) обрабатываем каждую "строчку" с признаками в последовательности data_sequence = file_data['features']
        for i in range(len(data_sequence)):
            # 1.3.1) выделяем 5 признаков
            voltage = data_sequence[i, 0] # напряжение [V]
            current = data_sequence[i, 1] # ток [A]
            temperature = data_sequence[i, 2] # температура [C]
            power = data_sequence[i, 3] # мощность [W]
            cumulative_charge = data_sequence[i, 4] # кумулятивный заряд [Ah]
            time_step = time_steps[i] if time_steps is not None else 1.0 # временной шаг

            # 1.3.2) подсчет текущего soc, для первого шага = 1
            current_soc = soc_estimates[-1] if i > 0 else 1.0

            # 1.3.3) напряжение на основе текущего SOC и всех признаков
            predicted_voltage = self.predict_voltage(current=current, temperature=temperature,
                                                     soc=current_soc, power=power, cumulative_charge=cumulative_charge)

            # 1.3.4) обновление SOC с использованием всех признаков
            current_soc = self.kf.estimate_soc(current=current, cumulative_charge=cumulative_charge,
                                               total_capacity=self.capacity, voltage_measurement=voltage,
                                               predicted_voltage=predicted_voltage, time_step=time_step, power=power)

            # 1.3.5) сохраняем оценку soc, ограничиваемая чтоб она не перезаряжалась и не переразряжалась
            current_soc = np.clip(current_soc, 0.0, 1.0)
            soc_estimates.append(current_soc)

        return {'soc_estimates': np.array(soc_estimates), 'time_steps_used': np.array(time_steps)}

'''
KAL - фильтр Калмана
'''
# # Более простой датасет для фильтра Калмана
# class BatteryDatasetKalman():
#     '''
#     В отличие от последовательных моделей (LSTM), здесь обрабатываются отдельные временные точки, а не последовательности.
#     '''
#     def __init__(self, features, targets, source_files, time_data):
#         self.features = features
#         self.targets = targets
#         self.source_files = source_files
#         self.time_data = time_data
#
#     def __len__(self):
#         return len(self.features)
#
#     def __getitem__(self, idx):
#         # Для Калмана нам нужны отдельные временные шаги, а не последовательности
#         features = self.features[idx]
#         target = self.targets[idx]
#         source_file = self.source_files[idx]
#         time_point = self.time_data[idx]
#
#         return features, target, source_file, time_point
#
# # Класс фильтра Калмана для SOC estimation важный вопрос у нас есть
# # уже расчитанный SOC для предсказания в данных нужно ли считать заново и вообще в каом виде все сюда поджпется расписать
# class KalmanFilterSOC:
#     def __init__(self, process_noise=0.1, measurement_noise=0.1, initial_soc=1.0):
#         self.process_noise = process_noise  # Шум процесса (неопределенность модели)
#         self.measurement_noise = measurement_noise  # Шум измерений (погрешность датчиков)
#         self.soc = initial_soc  # Начальное значение SOC (100%)
#         self.error_covariance = 1.0  # Начальная неопределенность оценки
#         self.initial_capacity = 3.0  # Емкость батареи в А·ч
#
#     def predict(self, current, dt=1):
#         '''
#         Шаг прогнозирования - обновление SOC на основе текущей интеграции
#         Кулоновский подсчет: ΔSOC = - (I * Δt) / Capacity
#         '''
#         delta_soc = - (current * dt) / (self.initial_capacity * 3600)  # Convert seconds to hours
#         self.soc = np.clip(self.soc + delta_soc, 0, 1)  # SOC between 0 and 1
#         self.error_covariance += self.process_noise
#         return self.soc
#
#     def update(self, measured_soc=None, voltage=None, temperature=None):
#         """Update step - correct SOC based on measurements"""
#         if measured_soc is not None:
#             # If we have direct SOC measurement
#             kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
#             self.soc += kalman_gain * (measured_soc - self.soc)
#             self.error_covariance *= (1 - kalman_gain)
#         elif voltage is not None and temperature is not None:
#             # If we need to estimate SOC from voltage/temperature
#             estimated_soc = self._soc_from_voltage(voltage, temperature)
#             kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
#             self.soc += kalman_gain * (estimated_soc - self.soc)
#             self.error_covariance *= (1 - kalman_gain)
#
#         self.soc = np.clip(self.soc, 0, 1)
#         return self.soc
#
#     def _soc_from_voltage(self, voltage, temperature):
#         """Simple voltage-SOC relationship model"""
#         # Simplified model - in practice should be battery-specific
#         # Linear approximation: 3.0V = 0% SOC, 4.2V = 100% SOC
#         soc_from_voltage = (voltage - 3.0) / (4.2 - 3.0)
#         soc_from_voltage = np.clip(soc_from_voltage, 0, 1)
#         return soc_from_voltage
#
#     def reset(self, initial_soc=1.0):
#         """Reset filter to initial state"""
#         self.soc = initial_soc
#         self.error_covariance = 1.0
#
# # Класс модели на основе фильтра Калмана
# class KalmanSOCModel(nn.Module):
#     def __init__(self, input_size, hidden_size=50, num_layers=2):
#         super(KalmanSOCModel, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         # Feature extraction layers
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#         )
#
#         # SOC estimation head
#         self.soc_predictor = nn.Linear(hidden_size, 1)
#         self.soc_activation = nn.Sigmoid()  # SOC between 0 and 1
#
#     def forward(self, x):
#         # x shape: [batch_size, input_size]
#         features = self.feature_extractor(x)
#         soc_pred = self.soc_predictor(features)
#         soc_pred = self.soc_activation(soc_pred)
#         return soc_pred

'''
TCN
'''
# Специализированный класс TCN
class BatteryDatasetTCN(BatteryDataset):
    # Можно добавить TCN-специфичную логику
    pass

# Новая модель: Temporal Convolutional Network (TCN)
class SoCTCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size=3, dropout=0.2):
        super(SoCTCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout

        # Инициализация
        # Создание TCN блоков
        self.tcn_blocks = nn.ModuleList()
        in_channels = input_size

        for i in range(num_layers):
            dilation = 2 ** i  # Экспоненциальное увеличение dilation для захвата долгосрочных зависимостей
            out_channels = hidden_size

            tcn_block = TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                dropout=dropout
            )
            self.tcn_blocks.append(tcn_block)
            in_channels = out_channels

        # Финальный слой
        self.fc = nn.Linear(hidden_size, 1)

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_size]
        batch_size, seq_len, input_size = x.shape

        # Транспонируем для Conv1d: [batch_size, input_size, seq_len]
        x = x.transpose(1, 2)

        # Применяем TCN блоки
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)

        # Берем последний временной шаг
        x = x[:, :, -1]  # [batch_size, hidden_size]

        # Финальный прогноз
        output = self.fc(x)

        return output

# Базовый блок TCN с residual connection
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation  # Поддержание длины последовательности

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        residual = x

        # Первая свертка
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Вторая свертка
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Обрезаем padding чтобы вернуть исходную длину
        out = out[:, :, :-self.conv1.padding[0]] if self.conv1.padding[0] > 0 else out

        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)

        residual = residual[:, :, -out.size(2):]  # Обрезаем residual до размера out

        out += residual
        return out