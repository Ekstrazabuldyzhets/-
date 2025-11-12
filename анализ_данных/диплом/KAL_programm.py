# Импорты и настройка библиотек
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import optuna
import time
import json

from sklearn.metrics import mean_squared_error, mean_absolute_error
from optuna.visualization import plot_optimization_history, plot_param_importances

# глобальные переменные
features_cols = ["Voltage [V]", "Current [A]", "Temperature [degC]", "Power [W]", "Capacity [Ah]"]
target_variable = "SOC [-]"
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0001
SEQUENCE_LENGTH = 20

# возможность обучения ускоренного модели
# 1. Проверяем CUDA (NVIDIA GPU)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
# 2. Проверяем MPS (Apple Silicon GPU)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
# 3. Fallback на CPU
else:
    device = torch.device("cpu")


# Класс фильтра Калмана для SOC estimation
class KalmanFilterSOC:
    def __init__(self, process_noise=0.1, measurement_noise=0.1, initial_soc=1.0):
        """
        Фильтр Калмана для оценки SOC батареи
        Parameters:
        - process_noise: шум процесса (Q)
        - measurement_noise: шум измерений (R)
        - initial_soc: начальное значение SOC
        """
        self.Q = process_noise  # Шум процесса
        self.R = measurement_noise  # Шум измерений
        self.soc = initial_soc  # Текущая оценка SOC
        self.P = 1.0  # Ковариация ошибки оценки
        self.dt = 1.0  # Временной шаг (предполагаем 1 секунду)

    def predict(self, current, capacity):
        """
        Prediction step - предсказание SOC на основе модели
        """
        # SOC обновляется на основе тока (кулоновский счет)
        delta_soc = (current * self.dt) / (3600 * capacity)  # dSOC = I*dt/Capacity
        self.soc = self.soc - delta_soc  # Обновление SOC

        # Обновление ковариации ошибки
        self.P += self.Q

        return self.soc

    def update(self, voltage_measurement, predicted_voltage):
        """
        Update step - коррекция на основе измерения напряжения
        """
        # Innovation (ошибка между измеренным и предсказанным напряжением)
        y = voltage_measurement - predicted_voltage

        # Innovation covariance
        S = self.P + self.R

        # Kalman gain
        K = self.P / S

        # Update SOC estimate
        self.soc += K * (y / 10.0)  # Эвристика: предполагаем, что 1V изменение ≈ 0.1 SOC изменения

        # Update error covariance
        self.P = (1 - K) * self.P

        return self.soc

    def estimate_soc(self, current, capacity, voltage_measurement, predicted_voltage):
        """
        Полный цикл фильтра Калмана: prediction + update
        """
        self.predict(current, capacity)
        soc_estimate = self.update(voltage_measurement, predicted_voltage)
        return soc_estimate

# Класс модели на основе фильтра Калмана
class KalmanSOCModel:
    def __init__(self, process_noise=0.01, measurement_noise=0.1, capacity=3.0):
        """
        Модель SOC на основе фильтра Калмана

        Parameters:
        - process_noise: шум процесса
        - measurement_noise: шум измерений
        - capacity: емкость батареи в Ah
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.capacity = capacity
        self.kf = None

    def initialize_filter(self, initial_soc=1.0):
        """Инициализация фильтра Калмана"""
        self.kf = KalmanFilterSOC(
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            initial_soc=initial_soc
        )

    def predict_voltage(self, current, temperature, soc):
        """
        Простая модель напряжения батареи на основе SOC
        Это упрощенная модель, которую можно заменить на более сложную
        """
        # Базовая модель напряжения литиевой батареи
        open_circuit_voltage = 3.0 + 1.2 * soc  # OCV кривая
        internal_resistance = 0.05 + 0.01 * (1 - soc)  # Внутреннее сопротивление
        voltage = open_circuit_voltage - current * internal_resistance

        # Коррекция на температуру
        temperature_effect = 0.001 * (temperature - 25)
        voltage += temperature_effect

        return voltage

    def estimate(self, data_sequence):
        """
        Оценка SOC для последовательности данных
        """
        if self.kf is None:
            self.initialize_filter()

        soc_estimates = []
        current_soc = 1.0

        for i, row in enumerate(data_sequence):
            current = row[1]  # Current [A]
            voltage = row[0]  # Voltage [V]
            temperature = row[2]  # Temperature [degC]

            # Предсказание напряжения на основе текущего SOC
            predicted_voltage = self.predict_voltage(current, temperature, current_soc)

            # Обновление оценки SOC с помощью фильтра Калмана
            current_soc = self.kf.estimate_soc(
                current=current,
                capacity=self.capacity,
                voltage_measurement=voltage,
                predicted_voltage=predicted_voltage
            )

            # Ограничение SOC в диапазоне [0, 1]
            current_soc = np.clip(current_soc, 0.0, 1.0)
            soc_estimates.append(current_soc)

        return np.array(soc_estimates)

# Dataset класс для последовательностей данных
class BatteryDatasetKalman(Dataset):
    def __init__(self, features, targets, source_files, time_steps):
        # Преобразование признаков в numpy array если это тензор PyTorch
        self.features = features.cpu().numpy() if torch.is_tensor(features) else features
        # Преобразование целевой переменной в numpy array если это тензор PyTorch
        self.targets = targets.cpu().numpy() if torch.is_tensor(targets) else targets
        self.source_files = source_files # список файлов-источников для каждого элемента данных
        self.time_steps = time_steps # сохраняем все временные метки

        # создание последовательностей данных при инициализации
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        sequences = [] # список для хранения всех последовательностей
        file_indices = {}

        # Создание индексов для каждого файла
        current_file = self.source_files[0]
        start_idx = 0

        for i in range(1, len(self.source_files)):
            if self.source_files[i] != current_file:
                file_indices[current_file] = (start_idx, i - 1)
                current_file = self.source_files[i]
                start_idx = i
        file_indices[current_file] = (start_idx, len(self.source_files) - 1)

        # Создание последовательностей для каждого файла
        for file_name, (start, end) in file_indices.items():
            file_length = end - start + 1
            if file_length >= self.sequence_length:
                for i in range(start, end - self.sequence_length + 2):
                    seq_end = min(i + self.sequence_length, end + 1)
                    sequences.append((i, seq_end))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start_idx, end_idx = self.sequences[idx]
        features_seq = self.features[start_idx:end_idx]
        targets_seq = self.targets[start_idx:end_idx]

        # Для фильтра Калмана нам нужна вся последовательность
        return (torch.FloatTensor(features_seq),
                torch.FloatTensor(targets_seq),
                self.source_files[start_idx:end_idx],
                self.time_steps[start_idx:end_idx])


# функции работающие напрямую с данными
def data_loader_and_standarder(temperatures_directory, directory):
    frames = []
    # 1) загрузка данных в датасет
    for temp_folder in os.listdir(directory):  # проходимся по всем температурным директориям
        if temp_folder in temperatures_directory:  # функция фильтрующая какие именно директории будут взяты
            temp_path = os.path.join(directory, temp_folder)
            for file in os.listdir(temp_path):  # проходимся по всем файлам в директории
                if 'Charge' in file or 'Dis' in file:
                    continue  # Пропускаем файлы постоянного заряда/разряда
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(temp_path, file))  # Загрузка и обработка каждого CSV файла
                    df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']  # Расчет новых признаков Мощность (Power)
                    df['SourceFile'] = file
                    frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # 2) стандартизация данных
    scaler = StandardScaler()
    data[features_cols] = scaler.fit_transform(data[features_cols])
    return data, scaler

def data_spliter(data, percents):
    # Итоговое распределение:
    # Train: 80% файлов
    # Validation: 10% файлов (50% от 20%)
    # Test: 10% файлов (50% от 20%)
    test_size_for_test, test_size_for_val = percents
    unique_files = np.array(list(set(data['SourceFile'])))
    train_files, temp_files = train_test_split(unique_files, test_size=test_size_for_test, random_state=24)
    val_files, test_files = train_test_split(temp_files, test_size=test_size_for_val, random_state=24)

    train_data = data[data['SourceFile'].isin(train_files)]
    val_data = data[data['SourceFile'].isin(val_files)]
    test_data = data[data['SourceFile'].isin(test_files)]

    return train_data, val_data, test_data

# функции работающие с сохраненными гиперпараметрами
def save_hyperparams(hyperparams, file_path):
    with open(file_path, 'w') as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Гиперпараметры сохранены в {file_path}")

# чтение гиперпараметров из файла
def load_hyperparams(file_path):
    # чтение файла
    with open(file_path, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

# проверка на наличие файла и гиперпараметров в нем
def hyperparams_exist(file_path):
    # Проверяем существование файла
    if not os.path.exists(file_path):
        print(f"📁 Файл {file_path} не найден")
        return False

    hyperparams = load_hyperparams(file_path)
    # Проверяем только наличие ключевых полей
    required = ['process_noise', 'measurement_noise', 'capacity']
    return all(key in hyperparams for key in required)

'''
существенно не меняется от модели к модели ↑
'''

# подготавливает данные для фильтра Калмана
def data_for_kalman_transmuter(train_data, val_data, test_data):
    train_dataset = BatteryDatasetKalman(
        torch.tensor(train_data[features_cols].values, dtype=torch.float32).to(device),
        torch.tensor(train_data[target_variable].values, dtype=torch.float32).to(device),
        train_data['SourceFile'].values,
        train_data['Time [s]'].values
    )

    val_dataset = BatteryDatasetKalman(
        torch.tensor(val_data[features_cols].values, dtype=torch.float32).to(device),
        torch.tensor(val_data[target_variable].values, dtype=torch.float32).to(device),
        val_data['SourceFile'].values,
        val_data['Time [s]'].values
    )

    test_dataset = BatteryDatasetKalman(
        torch.tensor(test_data[features_cols].values, dtype=torch.float32).to(device),
        torch.tensor(test_data[target_variable].values, dtype=torch.float32).to(device),
        test_data['SourceFile'].values,
        test_data['Time [s]'].values
    )

    # разбиение на батчи для ускорения прогнозирования
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Batch size 1 для последовательностей
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader

# обучение модели (для фильтра Калмана - настройка параметров)
def train_kalman_model(model, train_loader, val_loader, epochs, patience=20, min_delta=0.001):
    """
    Обучение/настройка фильтра Калмана
    """
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_count = 0

        for sequences, labels, _, _ in train_loader:
            sequences = sequences.squeeze(0).numpy()  # [1, seq_len, features] -> [seq_len, features]
            labels = labels.squeeze(0).numpy()  # [1, seq_len] -> [seq_len]

            # Инициализируем фильтр с начальным SOC
            initial_soc = labels[0] if len(labels) > 0 else 1.0
            model.initialize_filter(initial_soc=initial_soc)

            # Получаем предсказания SOC
            soc_predictions = model.estimate(sequences)

            # Вычисляем потери (только для последовательностей одинаковой длины)
            min_len = min(len(soc_predictions), len(labels))
            if min_len > 0:
                loss = np.mean((soc_predictions[:min_len] - labels[:min_len]) ** 2)
                train_loss += loss
                train_count += 1

        train_loss = train_loss / train_count if train_count > 0 else float('inf')
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0

        for sequences, labels, _, _ in val_loader:
            sequences = sequences.squeeze(0).numpy()
            labels = labels.squeeze(0).numpy()

            initial_soc = labels[0] if len(labels) > 0 else 1.0
            model.initialize_filter(initial_soc=initial_soc)

            soc_predictions = model.estimate(sequences)

            min_len = min(len(soc_predictions), len(labels))
            if min_len > 0:
                loss = np.mean((soc_predictions[:min_len] - labels[:min_len]) ** 2)
                val_loss += loss
                val_count += 1

        val_loss = val_loss / val_count if val_count > 0 else float('inf')
        history['val_loss'].append(val_loss)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Ранняя остановка
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f'Эпоха {epoch + 1}/{epochs}, Потеря обучения: {train_loss:.6f}, Потеря проверки: {val_loss:.6f}')
        print(f'Время, затраченное на эпоху: {epoch_time:.4f} секунд')

        if epochs_no_improve >= patience:
            print('Сработала досрочная остановка')
            break

    return history

# подбор гиперпараметров для фильтра Калмана
def hyperparameters_selectioner(train_loader, val_loader):
    # 0) функция, которую мы оптимизируем, дающая гиперпараметры, запускающая прогнозирование и расчитывание ошибки
    def objective(trial):
        # 0.1) предлагаемые гиперпараметры для фильтра Калмана
        process_noise = trial.suggest_float('process_noise', 1e-5, 1e-1, log=True) # шум процесса
        measurement_noise = trial.suggest_float('measurement_noise', 1e-5, 1e-1, log=True) # шум измерений
        capacity = trial.suggest_float('capacity', 2.5, 3.5)  # Емкость батареи в Ah

        # 0.2) создание модели с предлагаемыми гиперпараметрами
        model = KalmanSOCModel(process_noise=process_noise, measurement_noise=measurement_noise, capacity=capacity)

        # 0.3) обучаем модель [модель, обучающие данные, оценивающие даты, эпохи, device]
        history = train_kalman_model(model, train_loader, val_loader, EPOCHS)

        # 0.4) Извлечь последнюю потерю проверки
        last_val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
        return last_val_loss

    # 1) запускаем оптуну для подбора гиперпараметра, минимизируем ошибку
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # 2) извлекаем лучшую версию
    best_trial = study.best_trial
    print(f"Best trial: {best_trial}")
    best_hyperparams = study.best_trial.params
    print('Best hyperparameters:', best_hyperparams)

    # 3) Визуализация процесса оптимизации
    # Построение истории оптимизации
    optimization_history = plot_optimization_history(study)
    optimization_history.show()
    # График важности гиперпараметров
    param_importances = plot_param_importances(study)
    param_importances.show()

    return best_trial, best_hyperparams

# финальное-итоговое обучение модели
def finale_model_trainer(best_hyperparams, train_loader, val_loader, model_path):
    # Создание модели с лучшими гиперпараметрами
    model = KalmanSOCModel(
        process_noise=best_hyperparams['process_noise'],
        measurement_noise=best_hyperparams['measurement_noise'],
        capacity=best_hyperparams['capacity']
    )

    # Обучение модели
    history = train_kalman_model(model, train_loader, val_loader, EPOCHS)

    # Сохраняем модель и гиперпараметры
    model_data = {
        'hyperparams': best_hyperparams,
        'model_type': 'kalman'
    }
    torch.save(model_data, model_path)
    print(f"Модель сохранена в {model_path}")

# тестировка модели
def finale_model_tester(best_hyperparams, test_loader, model_path):
    # Создание модели с лучшими гиперпараметрами
    model = KalmanSOCModel(
        process_noise=best_hyperparams['process_noise'],
        measurement_noise=best_hyperparams['measurement_noise'],
        capacity=best_hyperparams['capacity']
    )

    test_predictions = []
    test_labels = []

    # Тестирование модели
    model.eval()
    for sequences, labels, _, _ in test_loader:
        sequences = sequences.squeeze(0).numpy()
        labels = labels.squeeze(0).numpy()

        initial_soc = labels[0] if len(labels) > 0 else 1.0
        model.initialize_filter(initial_soc=initial_soc)

        soc_predictions = model.estimate(sequences)

        # Сохраняем предсказания и метки
        test_predictions.extend(soc_predictions)
        test_labels.extend(labels[:len(soc_predictions)])

    # Расчет ошибок
    test_predictions_np = np.array(test_predictions)
    test_labels_np = np.array(test_labels)

    mse = mean_squared_error(test_labels_np, test_predictions_np)
    mae = mean_absolute_error(test_labels_np, test_predictions_np)

    print(f"Mean Squared Error on Test Set: {mse:.6f}")
    print(f"Mean Absolute Error on Test Set: {mae:.6f}")

    return test_predictions_np, test_labels_np

'''
существенно не меняется от модели к модели ↓
'''
def main(data_directory_dict, model_path, hyperparams_path):
    # 1) загружаем наши данные, а также совершаем предобработку
    directory = data_directory_dict["LG_HG2_processed"]
    temperatures_directory = [folder for folder in os.listdir(directory) if 'degC' in folder]
    data, scaler = data_loader_and_standarder(temperatures_directory, directory)

    # 1.1) разделяем на тестовую и обучающие выборки
    percents = [0.2, 0.5]
    train_data, val_data, test_data = data_spliter(data, percents)

    # 1.2) преобразуем данные для чтения их моделью
    train_loader, val_loader, test_loader = data_for_kalman_transmuter(train_data, val_data, test_data)

    if not hyperparams_exist(hyperparams_path):
        # 2) подбираем гиперпараметры для улучшения модели
        print("Подберем гиперпараметры для фильтра Калмана")
        best_trial, best_hyperparams = hyperparameters_selectioner(train_loader, val_loader)
        # 2.1) сохраняем гиперпараметры
        save_hyperparams(best_hyperparams, hyperparams_path)
    else:
        # 3) создаем модель на основе подобранных параметров
        print("Реализуем прогноз с помощью фильтра Калмана")
        best_hyperparams = load_hyperparams(hyperparams_path)
        finale_model_trainer(best_hyperparams, train_loader, val_loader, model_path)
        # 4) тестируем модель на наших данных
        finale_model_tester(best_hyperparams, test_loader, model_path)

    return

if __name__ == "__main__":
    model_path = "/Users/nierra/Desktop/диплом-2/датасет_2/kalman_soc_model.pth"
    hyperparams_path = "/Users/nierra/Desktop/диплом-2/датасет_2/kalman_hyperparams.json"
    main_directory = "/Users/nierra/Desktop/диплом-2/датасет_2/Data"
    data_directory_dict = {"LG_HG2_processed": f"{main_directory}/LG_HG2_processed"}
    main(data_directory_dict, model_path, hyperparams_path)