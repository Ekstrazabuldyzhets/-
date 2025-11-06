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

# Импорт библитеки с моделью
import NN_class as nero

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

# функции работающие напрямую с данными
def data_loader_and_standarder(temperatures_directory, directory):
    frames = []
    # 1) загрузка данных в датасет
    for temp_folder in os.listdir(directory): # проходимся по всем температурным директориям
        if temp_folder in temperatures_directory: # функция фильтрующая какие именно директории будут взяты
            temp_path = os.path.join(directory, temp_folder)
            for file in os.listdir(temp_path): # проходимся по всем файлам в директории
                if 'Charge' in file or 'Dis' in file:
                    continue  # Пропускаем файлы постоянного заряда/разряда
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(temp_path, file)) # Загрузка и обработка каждого CSV файла
                    df['Power [W]'] = df['Voltage [V]'] * df['Current [A]'] # Расчет новых признаков Мощность (Power)
                    df['SourceFile'] = file
                    frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # 2) стандартизация данных
    scaler = StandardScaler()
    data[features_cols] = scaler.fit_transform(data[features_cols])
    return data

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

# подготавливает данные для сверточных нейросетей (CNN)
def data_for_cnn_transmuter(train_data, val_data, test_data):
    train_dataset = nero.BatteryDatasetCNN(
        torch.tensor(train_data[features_cols].values, dtype=torch.float32).to(device),
        torch.tensor(train_data[target_variable].values, dtype=torch.float32).to(device),
        SEQUENCE_LENGTH,
        train_data['SourceFile'].values,
        train_data['Time [s]'].values
    )

    val_dataset = nero.BatteryDatasetCNN(
        torch.tensor(val_data[features_cols].values, dtype=torch.float32).to(device),
        torch.tensor(val_data[target_variable].values, dtype=torch.float32).to(device),
        SEQUENCE_LENGTH,
        val_data['SourceFile'].values,
        val_data['Time [s]'].values
    )

    test_dataset = nero.BatteryDatasetCNN(
        torch.tensor(test_data[features_cols].values, dtype=torch.float32).to(device),
        torch.tensor(test_data[target_variable].values, dtype=torch.float32).to(device),
        SEQUENCE_LENGTH,
        test_data['SourceFile'].values,
        test_data['Time [s]'].values
    )

    # разбиение на батчи для ускорения прогнозирования
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

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
    required = ['hidden_size', 'num_layers', 'learning_rate']
    return all(key in hyperparams for key in required)

# обучение модели
def train_education(model, criterion, optimizer, train_loader, val_loader, epochs, device, patience=20,
                       min_delta=0.001):
    # 1) инициализация
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')  # отслеживает лучшую validation loss
    epochs_no_improve = 0

    # 2) эпоха - один полный проход через все тренировочные данные.
    for epoch in range(epochs):
        # 2.1) обучаем на тренировочных данных
        model.train()
        train_loss = 0.0

        epoch_start_time = time.time() # (начало отчета) замеряем время потраченное на эпоху
        for sequences, labels, _, _ in train_loader:
            # Подготовка данных
            sequences, labels = sequences.to(device), labels.to(device)
            labels = labels.unsqueeze(1)  # [128] → [128, 1], чтоб labels и outputs были совместимы
            optimizer.zero_grad()  # Обнуление градиентов Что происходит внутри  и зачем - ?
            # Прямой проход (Forward Pass)
            outputs = model(sequences)  # Выход: [128, 1] - предсказанные SOC
            loss = criterion(outputs, labels)  # Вычисление потерь
            # Обратный проход (Backward Pass)
            loss.backward()
            optimizer.step() # Обновление весов
            train_loss += loss.item() # Накопление потерь

        epoch_end_time = time.time() # (конец отчета) замеряем время потраченное на эпоху
        epoch_time = epoch_end_time - epoch_start_time # длительность эпохи

        train_loss = train_loss/len(train_loader) # подсчет общей ошибки(по всем батчам)
        history['train_loss'].append(train_loss)

        # 2.2) проводим валидация; мы хотим одинаковые результаты на одних и тех же данных
        model.eval()  # Переключение в режим оценки
        val_loss = 0.0
        with torch.no_grad(): # это "режим экзамена" для модели, где она только демонстрирует свои знания, но не получает новых!
            for sequences, labels, _, _ in val_loader:
                # Подготовка данных
                sequences, labels = sequences.to(device), labels.to(device)
                labels = labels.unsqueeze(1) # [128] → [128, 1], чтоб labels и outputs были совместимы
                outputs = model(sequences) #  прямой проход - предсказанные SOC
                loss = criterion(outputs, labels) # вычисление потерь
                val_loss += loss.item() # накопление потерь

        val_loss /= len(val_loader) # подсчет общей ошибки(по всем батчам)
        history['val_loss'].append(val_loss)

        # 2.3) проверка на переобучение; рання остановка
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f'Эпоха {epoch + 1}/{epochs}, Потеря обучения: {train_loss}, Потеря проверки: {val_loss}')
        print(f'Время, затраченное на эпоху: {epoch_time:.8f} секунд')

        if epochs_no_improve >= patience: # 20 эпох без улучшений
            print('Сработала досрочная остановка')
            break

    return history

# подбор гиперпараметров
def hyperparameters_selectioner(train_loader, val_loader):
    # 0) функция, которую мы оптимизируем, дающая гиперпараметры, запускающая прогнозирование и расчитывание ошибки
    def objective(trial):
        # 0.1) предлагаемые гиперпараметры (задается нижний и верхнмий диапазон)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 264]) # количество фильтров (feature maps) в сверточных слоях
        num_layers = trial.suggest_int('num_layers', 1, 3) # количество сверточных слоев в сети
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True) # cкорость обучения - шаг, с которым обновляются веса сети
        kernel_size = trial.suggest_categorical('kernel_size', [3]) # размер ядра свертки
        dropout = trial.suggest_float('dropout', 0.1, 0.5) # вероятность отключения нейрона во время обучения

        # 0.2) создание модели с предлагаемыми гиперпараметрами
        model = nero.SoCCNN(input_size=len(features_cols), hidden_size=hidden_size, num_layers=num_layers,
                            kernel_size=kernel_size, dropout=dropout).type(torch.float32).to(device)

        # 0.3) определяет свою функцию потерь и оптимизатор с помощью предлагаемых гиперпараметров
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # 0.4) обучаем модель [модель, метрика, оптимизатор, обучающие данные, оценивающие даты, эпохи, device]
        history = train_education(model, criterion, optimizer, train_loader, val_loader, EPOCHS, device)

        # 0.5) Извлечь последнюю потерю проверки
        last_val_loss = history['val_loss'][-1]
        return last_val_loss

    # 1) запускаем оптуну для подбора гиперпараметра, минимизируем ошибку
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)

    # 2) извлекаем лучшую версию
    best_trial = study.best_trial
    print(f"Best trial: {best_trial}")
    best_hyperparams = study.best_trial.params
    print('Best CNN hyperparameters:', best_hyperparams)

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
    # 1) предлагаемые гиперпараметры (задается нижний и верхнмий диапазон)
    hidden_size = best_hyperparams['hidden_size']
    num_layers = best_hyperparams['num_layers']
    learning_rate = best_hyperparams['learning_rate']
    kernel_size = best_hyperparams.get('kernel_size', 3)
    dropout = best_hyperparams.get('dropout', 0.2)
    epochs = 20

    # 2) создание модели с предлагаемыми гиперпараметрами
    model = nero.SoCCNN(input_size=len(features_cols), hidden_size=hidden_size, num_layers=num_layers,
                        kernel_size=kernel_size, dropout=dropout).type(torch.float32).to(device)

    # 3) определите свою функцию потерь и оптимизатор с помощью предлагаемых гиперпараметров
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 4) обучаем модель [модель, метрика, оптимизатор, обучающие данные, оценивающие даты, эпохи, device]
    history = train_education(model, criterion, optimizer, train_loader, val_loader, epochs, device)

    # 5) Сохраняем модель
    torch.save({'model_state_dict': model.state_dict(), 'input_size': len(features_cols)}, model_path)

# тестировка модели
def finale_model_tester(best_hyperparams, test_loader, model_path):
    # 1) предлагаемые гиперпараметры (задается нижний и верхнмий диапазон)
    hidden_size = best_hyperparams['hidden_size']
    num_layers = best_hyperparams['num_layers']
    kernel_size = best_hyperparams.get('kernel_size', 3)
    dropout = best_hyperparams.get('dropout', 0.2)

    # 1.1) загружаем сохраненную модель
    loaded_model = nero.SoCCNN(input_size=len(features_cols), hidden_size=hidden_size, num_layers=num_layers,
                               kernel_size=kernel_size, dropout=dropout).type(torch.float32).to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    loaded_model.to(device)
    loaded_model.eval()

    # 1.2) обучаем модель
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        # Прямой проход - получение предсказаний
        for sequences, labels, _, _ in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = loaded_model(sequences)
            # Сохранение результатов
            test_predictions.extend(outputs.cpu().view(-1).tolist())
            test_labels.extend(labels.cpu().view(-1).tolist())

    # 1.3) расчитываем ошибки
    # Преобразовать прогнозы и метки в массивы numpy для расчета ошибок
    test_predictions_np = np.array(test_predictions)
    test_labels_np = np.array(test_labels)

    # MSE и MAE
    mse = mean_squared_error(test_labels_np, test_predictions_np)
    mae = mean_absolute_error(test_labels_np, test_predictions_np)

    print(f"CNN Mean Squared Error on Test Set: {mse:.6f}")
    print(f"CNN Mean Absolute Error on Test Set: {mae:.6f}")

def main(data_directory_dict, model_path, hyperparams_path):
    # 1) загружаем наши данные, а также совершаем предобработку
    directory = data_directory_dict["LG_HG2_processed"]
    # temperatures_directory = [folder for folder in os.listdir(directory) if 'degC' in folder]
    temperatures_directory = [folder for folder in os.listdir(directory) if 'degC' in folder]
    data = data_loader_and_standarder(temperatures_directory, directory)
    # 1.1) разделяем на тестовую и обучающие выборки
    percents = [0.2, 0.5]
    train_data, val_data, test_data = data_spliter(data, percents)
    # 1.2) преобразуем данные для чтения их моделью, так как требует последовательностей определенной длины.
    train_loader, val_loader, test_loader = data_for_cnn_transmuter(train_data, val_data, test_data)

    if not hyperparams_exist(hyperparams_path):
        # 2) подбираем гиперпараметры для улучшения модели
        print("подберем гиперпараметры")
        best_trial, best_hyperparams = hyperparameters_selectioner(train_loader, val_loader)
        # 2.1) сохраняем гиперпараметры
        save_hyperparams(best_hyperparams, hyperparams_path)
    else:
        # 3) создаем модель на основе подобранных параметров
        print("реализуем прогноз")
        best_hyperparams = load_hyperparams(hyperparams_path)
        finale_model_trainer(best_hyperparams, train_loader, val_loader, model_path)
        # 4) тестируем модель на наших данных
        finale_model_tester(best_hyperparams, test_loader, model_path)

    return

if __name__ == "__main__":
    model_path = "/Users/nierra/Desktop/диплом-2/датасет_2/soc_lstm_model.pth"
    cnn_hyperparams_path = "/Users/nierra/Desktop/диплом-2/датасет_2/cnn_best_hyperparams.json"
    main_directory = "/Users/nierra/Desktop/диплом-2/датасет_2/Data"
    data_directory_dict = {"LG_HG2_processed": f"{main_directory}/LG_HG2_processed"}
    main(data_directory_dict, model_path, cnn_hyperparams_path)