# Импорты и настройка библиотек
import os
import pandas as pd
import numpy as np
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Импорт библитеки с моделью
import NN_class as nero

# глобальные переменные
features_cols = ["Voltage [V]", "Current [A]", "Temperature [degC]", "Power [W]", "Capacity [Ah]"]
target_variable = "SOC [-]"

# Функции работы с данными
def data_loader_and_standarder(temperatures_directory, directory):
    frames = []
    # 1) загрузка данных в датасет
    for temp_folder in os.listdir(directory):
        if temp_folder in temperatures_directory:
            temp_path = os.path.join(directory, temp_folder)
            for file in os.listdir(temp_path):
                if 'Charge' in file or 'Dis' in file:
                    continue
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(temp_path, file))
                    df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']
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
    with open(file_path, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams

# проверка на наличие файла и гиперпараметров в нем
def hyperparams_exist(file_path):
    print(file_path)
    # Проверяем существование файла
    if not os.path.exists(file_path):
        print(f" Файл {file_path} не найден")
        return False

    hyperparams = load_hyperparams(file_path)
    # Проверяем только наличие ключевых полей
    required = ['process_noise', 'measurement_noise', 'capacity']
    return all(key in hyperparams for key in required)

'''
существенно не меняется от модели к модели ↑
'''

# подготавливает данные для филтра Калмана
def data_for_kalman_transmuter(train_data, val_data, test_data):
    def calculate_delta_time(time_steps_seconds):
        delta_times = np.ones_like(time_steps_seconds)
        for i in range(1, len(time_steps_seconds)):
            delta_t = time_steps_seconds[i] - time_steps_seconds[i - 1]
            if delta_t > 0:
                delta_times[i] = delta_t
            else:
                delta_times[i] = delta_times[i - 1] if i > 1 else 1.0
        return delta_times

    train_data_dict = {
        'features': train_data[features_cols].values,
        'targets': train_data[target_variable].values,
        'source_files': train_data['SourceFile'].values,
        'time_steps': calculate_delta_time(train_data['Time [s]'].values)
    }
    val_data_dict = {
        'features': val_data[features_cols].values,
        'targets': val_data[target_variable].values,
        'source_files': val_data['SourceFile'].values,
        'time_steps': calculate_delta_time(val_data['Time [s]'].values)
    }
    test_data_dict = {
        'features': test_data[features_cols].values,
        'targets': test_data[target_variable].values,
        'source_files': test_data['SourceFile'].values,
        'time_steps': calculate_delta_time(test_data['Time [s]'].values)
    }
    return train_data_dict, val_data_dict, test_data_dict

# группирует данные по файлам
def group_data_by_files(data_dict):
    """
    Группировка данных по файлам
    """
    grouped_data = []
    unique_files = np.unique(data_dict['source_files'])
    for file_name in unique_files:
        file_mask = data_dict['source_files'] == file_name
        file_data = {
            'file_name': file_name,
            'features': data_dict['features'][file_mask],
            'targets': data_dict['targets'][file_mask],
            'time_steps': data_dict['time_steps'][file_mask]
        }
        grouped_data.append(file_data)
    return grouped_data

# функции автоматического подбора параметров
def analyze_battery_characteristics(data_dict):
    # 1) группируем данные по файлам
    grouped_data = group_data_by_files(data_dict)
    voltage_stats = [] # стандартные отклонения напряжения
    current_stats = [] # стандартные отклонения тока
    soc_ranges = [] # диапазоны SOC
    time_characteristics = [] # временные шаги

    # 2) едем по каждому файлу в сгруппированном наборе
    for file_data in grouped_data:
        # 2.0) достаем данные
        features = file_data['features']
        targets = file_data['targets']
        time_steps = file_data['time_steps']

        # 2.1) статистику получаем
        # статистика напряжения np.std() - рассчитывает стандартное отклонение
        voltage_stats.append(np.std(features[:, 0]))
        # статистика тока
        current_stats.append(np.std(features[:, 1]))
        # диапазон SOC
        soc_range = np.max(targets) - np.min(targets)

        # 2.2) только значимые диапазоны SOC
        if soc_range > 0.1:
            soc_ranges.append(soc_range)

        # 2.3) временные характеристики, как часто проводились измерения
        if len(time_steps) > 1:
            avg_time_step = np.mean(time_steps[1:])
            time_characteristics.append(avg_time_step)

    # 3) анализ собранной статистики
    avg_voltage_std = np.mean(voltage_stats) if voltage_stats else 0.1
    avg_current_std = np.mean(current_stats) if current_stats else 0.1
    avg_soc_range = np.mean(soc_ranges) if soc_ranges else 0.5
    avg_time_step = np.mean(time_characteristics) if time_characteristics else 1.0

    print("Анализ характеристик батареи:")
    print(f"  Среднее отклонение напряжения: {avg_voltage_std:.3f} V")
    print(f"  Среднее отклонение тока: {avg_current_std:.3f} A")
    print(f"  Средний диапазон SOC: {avg_soc_range:.3f}")
    print(f"  Средний временной шаг: {avg_time_step:.3f} s")

    return {'voltage_std': avg_voltage_std, 'current_std': avg_current_std, 'soc_range': avg_soc_range, 'time_step': avg_time_step}

# расчет оптимальных параметров фильтра Калмана на основе статистики обучающих данных
def calculate_optimal_parameters(train_data_dict):
    # 1) анализируем характеристики батареи из обучающих данных
    # [стандартное отклонение напряжения, стандартное отклонение тока, диапазон SOC, средний временной шаг]
    battery_chars = analyze_battery_characteristics(train_data_dict)

    # 2) считаем шум измерений (R) - на основе волатильности напряжения
    measurement_noise = max(0.01, min(0.5, battery_chars['voltage_std'] * 2))

    # 3) считаем шум процесса (Q) - на основе волатильности тока
    process_noise = max(0.001, min(0.1, battery_chars['current_std'] * 0.1))

    # 4) емкость батареи - оценка на основе данных
    grouped_data = group_data_by_files(train_data_dict) # для этого группируем данные по файлам
    capacities = []

    for file_data in grouped_data: # идем по всем данным в файлах
        # 4.1) берем данные
        currents = file_data['features'][:, 1]
        time_steps = file_data['time_steps']
        soc_values = file_data['targets']

        # 4.2) интеграл тока по времени для оценки емкости
        total_charge = np.sum(np.abs(currents) * time_steps) / 3600
        # 4.3) подсчитываем диапазон SOC в файле
        soc_range = np.max(soc_values) - np.min(soc_values)

        # 4.3.1) проверяем, что диапазон SOC достаточно большой для надежной оценки
        if soc_range > 0.1: # минимум 10% изменения SOC
            estimated_capacity = total_charge / soc_range
            # проверяем, что оценка емкости реалистична для литиевых батарей
            if 2.0 < estimated_capacity < 5.0:
                capacities.append(estimated_capacity)

    # 5) используем медиану оценок или значение по умолчанию
    if capacities:
        capacity = np.median(capacities)
    else:
        capacity = 3.0

    # 6) визуализация для большего понимания
    print("\nАвтоматически рассчитанные параметры:")
    print(f"  Шум процесса (Q): {process_noise:.6f}")
    print(f"  Шум измерений (R): {measurement_noise:.6f}")
    print(f"  Емкость батареи: {capacity:.3f} Ah")

    return {'process_noise': process_noise, 'measurement_noise': measurement_noise, 'capacity': capacity}

# оценка модели (без вывода в консоль)
def evaluate_kalman_model_simple(model, data_dict):
    # 1) группируем данные по файлам
    grouped_data = group_data_by_files(data_dict)
    all_predictions = []
    all_labels = []

    # 2) едем по каждому файлу в сгруппированном наборе
    for file_data in grouped_data:
        # 2.0) достаем данные
        sequences = file_data['features']
        labels = file_data['targets']
        time_steps = file_data['time_steps']

        # 2.1) # берем первое истинное значение SOC
        initial_soc = labels[0]
        # 2.2) фильтр Калмана начальным значением SOC
        model.initialize_filter(initial_soc=initial_soc)
        # 2.3) получаем предсказания
        results = model.estimate(sequences, time_steps)
        soc_predictions = results['soc_estimates']
        # 2.4) защита если несоответствие длин предсказаний и истин
        min_len = min(len(soc_predictions), len(labels))
        # 2.5) сохраняем данные предсказания
        all_predictions.extend(soc_predictions[:min_len])
        all_labels.extend(labels[:min_len])

    return all_predictions, all_labels

# подбор гиперпараметров путем оценки на валидационных данных
def hyperparameters_selectioner(train_data_dict, val_data_dict):
    # 1.0) cначала получаем автоматически рассчитанные параметры как базовые и фиксируем как лучшие(пока)
    auto_params = calculate_optimal_parameters(train_data_dict)
    best_params = auto_params
    # 1.1) вначале ошибка худашая - бесконечность
    best_val_loss = float('inf')

    # 2) проводим оценку на подобранных параметрах(создаем модель->рассчитываем ошибку на валидационных данных)
    model = nero.KalmanSOCModel(process_noise=auto_params['process_noise'],
                                measurement_noise=auto_params['measurement_noise'],
                                capacity=auto_params['capacity'])

    # 3) рассчитываем среднеквадратичную ошибку (MSE) между предсказаниями и истинными значениями
    val_predictions, val_labels = evaluate_kalman_model_simple(model, val_data_dict)
    best_val_loss = mean_squared_error(val_labels, val_predictions)
    print(f"Автоматические параметры - MSE: {best_val_loss:.6f}")

    # 4) диапазоны для поиска вокруг автоматических значений
    # 4.1) создаем список значений шума процесса для тестирования
    process_noises = [
        auto_params['process_noise'] * 0.1,  # 10% от автоматического значения
        auto_params['process_noise'] * 0.5,  # 50% от автоматического значения
        auto_params['process_noise'],  # 100% - исходное автоматическое значение
        auto_params['process_noise'] * 2,  # 200% от автоматического значения
        auto_params['process_noise'] * 5  # 500% от автоматического значения
    ]
    # 4.2) создаем список значений шума измерений для тестирования
    measurement_noises = [
        auto_params['measurement_noise'] * 0.1,  # 10% от автоматического значения
        auto_params['measurement_noise'] * 0.5,  # 50% от автоматического значения
        auto_params['measurement_noise'],  # 100% - исходное автоматическое значение
        auto_params['measurement_noise'] * 2,  # 200% от автоматического значения
        auto_params['measurement_noise'] * 5  # 500% от автоматического значения
    ]
    # 4.3) создаем список значений емкости батареи для тестирования
    capacities = [
        auto_params['capacity'] * 0.9,  # 90% от автоматической емкости
        auto_params['capacity'],  # 100% - исходная автоматическая емкость
        auto_params['capacity'] * 1.1  # 110% от автоматической емкости
    ]

    # 5) начинаем перебор всех возможных комбинаций параметров
    sch = 0
    for process_noise in process_noises:  # по всем значениям шума процесса
        for measurement_noise in measurement_noises: # по всем значениям шума измерений
            for capacity in capacities: # по всем значениям емкости
                sch = sch + 1
                # 5.1) проводим оценку на подобранных параметрах(создаем модель->рассчитываем ошибку на валидационных данных)
                model = nero.KalmanSOCModel(process_noise=process_noise,
                                            measurement_noise=measurement_noise,
                                            capacity=capacity)
                # 5.2) рассчитываем среднеквадратичную ошибку (MSE) между предсказаниями и истинными значениями
                val_predictions, val_labels = evaluate_kalman_model_simple(model, val_data_dict)
                val_loss = mean_squared_error(val_labels, val_predictions)
                # 5.3) сравниваем ошибки и ищем лучший вариант
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {'process_noise': process_noise, 'measurement_noise': measurement_noise, 'capacity': capacity}

    # 6) выводим лучшие параметры
    print(f"Лучшие параметры: {best_params}")
    print(f"Лучшая MSE на валидации: {best_val_loss:.6f}")

    return best_params

# создание и обучение модели, а также тестировка
def evaluate_kalman_model(model, data_dict, dataset_name="данных"):
    all_predictions, all_labels = evaluate_kalman_model_simple(model, data_dict)
    mse = mean_squared_error(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)

    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  Количество образцов: {len(all_predictions)}")
    return all_predictions, all_labels

def create_and_evaluate_final_model(best_hyperparams, train_data_dict, val_data_dict, test_data_dict):
    # Создание финальной модели
    model = nero.KalmanSOCModel(process_noise=best_hyperparams['process_noise'],
                                measurement_noise=best_hyperparams['measurement_noise'],
                                capacity=best_hyperparams['capacity'])
    # Оценка на всех наборах данных
    print("\n1. Обучающие данные:")
    print(f"Оценка на обучающих данных")
    evaluate_kalman_model(model, train_data_dict)
    print("\n2. Валидационные данные:")
    print(f"Оценка на валидационных данных")
    evaluate_kalman_model(model, val_data_dict)
    print("\n3. Тестовые данные:")
    print(f"Оценка на тестовых данных")
    evaluate_kalman_model(model, test_data_dict)

    return model

'''
существенно не меняется от модели к модели ↓
'''
# Основная функция
def main(data_directory_dict, hyperparams_path):
    # 1.0) Загрузка и подготовка данных
    directory = data_directory_dict["LG_HG2_processed"]
    temperatures_directory = [folder for folder in os.listdir(directory) if 'degC' in folder]
    data, scaler = data_loader_and_standarder(temperatures_directory, directory)

    # 1.1) Разделение данных
    percents = [0.2, 0.5]
    train_data, val_data, test_data = data_spliter(data, percents)

    # 1.2) Преобразование данных для фильтра Калмана
    train_data_dict, val_data_dict, test_data_dict = data_for_kalman_transmuter(train_data, val_data, test_data)

    # Проверяем, есть ли уже сохраненные параметры
    if not hyperparams_exist(hyperparams_path):
        # 2) подбираем гиперпараметры
        print("подберем гиперпараметры")
        best_hyperparams = hyperparameters_selectioner(train_data_dict, val_data_dict)
        # 2.1) сохраняем гиперпараметры
        save_hyperparams(best_hyperparams, hyperparams_path)
        main(data_directory_dict, hyperparams_path)
    else:
        # 3) создаем модель на основе подобранных параметров
        print("реализуем прогноз, оценку модели")
        best_hyperparams = load_hyperparams(hyperparams_path)
        create_and_evaluate_final_model(best_hyperparams, train_data_dict, val_data_dict, test_data_dict)

if __name__ == "__main__":
    # Настройки путей
    hyperparams_path = "/Users/nierra/Desktop/диплом-2/датасет_2/kalman_classic_hyperparams.json"
    main_directory = "/Users/nierra/Desktop/диплом-2/датасет_2/Data"
    data_directory_dict = {"LG_HG2_processed": f"{main_directory}/LG_HG2_processed"}

    # Запуск основной функции
    start_time = time.time()
    main(data_directory_dict, hyperparams_path)
    end_time = time.time()
    print(f"\nОбщее время выполнения: {end_time - start_time:.2f} секунд")