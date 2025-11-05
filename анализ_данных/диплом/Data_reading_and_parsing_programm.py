# Импорты и настройка библиотек
import os
from multiprocessing import Pool # паралельные пулы для ускорения работы процессов
from tqdm import tqdm # на отслеживает прогресс длительных задач, рассчитывая прошедшее и оставшееся время
import pandas as pd
from scipy.interpolate import interp1d

# глобальные переменные
col_list = ["Timestamp", "Time [min]", "Time [s]", "Voltage [V]", "Current [A]",
            "Temperature [degC]", "Capacity [Ah]", "Cumulative_Capacity_Ah", "SOC [-]"]

# функция созадющая функции интерполяции для связи напряжения
def get_pOCV_SOC_interp_fn(file_path: str):
    df = pd.read_csv(file_path)
    # обработка данных РАЗРЯДА; Отбираются все строки где ток отрицательный (разряд)
    df_discharge = df[df[col_list[4]] < 0].copy()
    # Нормализация емкости
    df_discharge[col_list[6]] = df_discharge[col_list[6]] - df_discharge[col_list[6]].iloc[0]

    # Расчет SOC для разряда Формула: SOC = 1 - (|текущая_емкость| / |максимальная_емкость_разряда|)
    df_discharge[col_list[8]] = 1 - abs(df_discharge[col_list[6]] / df_discharge[col_list[6]].iloc[-1])
    # Удаляем возможные выбросы напряжения выше максимального - ?
    max_voltage_discharge = df_discharge[col_list[3]].max()
    df_discharge = df_discharge[df_discharge[col_list[3]] <= max_voltage_discharge]
    # Создание функции интерполяции для разряда
    discharge_interp = interp1d(df_discharge[col_list[3]], df_discharge[col_list[8]], bounds_error=False,
                                fill_value="extrapolate")

    # Фильтрация данных заряда  - отбираются все строки где ток положительный (заряд)
    df_charge = df[df[col_list[4]] > 0].copy()
    # Нормализация емкости заряда
    df_charge[col_list[6]] = df_charge[col_list[6]] - df_charge[col_list[6]].iloc[0]
    # Расчет SOC для заряда SOC = |текущая_емкость| / максимальная_емкость_заряда
    df_charge[col_list[8]] = abs(df_charge[col_list[6]]) / df_charge[col_list[6]].iloc[-1]

    # Удаляем возможные выбросы напряжения выше максимального - ?
    max_voltage_charge = df_charge[col_list[3]].max()
    df_charge = df_charge[df_charge[col_list[3]] <= max_voltage_charge]

    # Создание функции интерполяции для заряда
    charge_interp = interp1d(df_charge[col_list[3]], df_charge[col_list[8]], bounds_error=False, fill_value="extrapolate")

    return charge_interp, discharge_interp

def get_max_capacities(c20_file_path):
    # Загрузка данных C20
    df_c20 = pd.read_csv(c20_file_path)

    # Нахождение точки перехода разряд→заряд
    charge_start_index = df_c20[df_c20[col_list[4]] > 0].index[0]

    # Разделение данных на две части
    df_discharge = df_c20.iloc[:charge_start_index]
    df_charge = df_c20.iloc[charge_start_index:]

    # Расчет максимальных емкостей как для ЗАРЯДА так и для РАЗРЯДА
    max_discharge_capacity = df_discharge[col_list[6]].max() - df_discharge[col_list[6]].min()
    max_charge_capacity = df_charge[col_list[6]].max() - df_charge[col_list[6]].min()

    return max_charge_capacity, max_discharge_capacity

# определяет начальное состояние заряда (SOC) батареи в начале теста.
def get_initial_soc(df, charge_soc_fn, discharge_soc_fn, current_col, voltage_col):
    # Получение начального напряжения
    initial_voltage = df[voltage_col].iloc[0]

    # Поиск первого ненулевого тока с детальной информацией
    non_zero_current = df[df[current_col] != 0]
    first_non_zero_index = non_zero_current.index[0]

    # Получение значения тока
    first_non_zero_current = df[current_col].iloc[first_non_zero_index]

    # Определение SOC по виду тока;
    # если батарея разряжается используем функцию разряда
    # иначе функцию заряда
    if first_non_zero_current < 0:
        return discharge_soc_fn(initial_voltage)
    else:
        return charge_soc_fn(initial_voltage)

# функции работающие напрямую с файлами
def check_directories_exist(data_directory_dict):
    existing_non_empty_dirs = []
    missing_or_empty_dirs = []

    for name, path in data_directory_dict.items():
        if os.path.exists(path) and os.path.isdir(path):
            # Проверяем что директория не пустая
            if any(os.listdir(path)):
                existing_non_empty_dirs.append(name)
            else:
                missing_or_empty_dirs.append((name, "directory_empty"))
        else:
            missing_or_empty_dirs.append((name, "not_exists"))

    return existing_non_empty_dirs, missing_or_empty_dirs

def parse_crude_data(file_path):
    # Чтение и фильтрация строк
    with open(file_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    # Поиск заголовков
    header_idx = next(i for i, line in enumerate(lines) if 'Time Stamp' in line)
    columns = lines[header_idx].split(',')

    # Создаем DataFrame из данных
    data = [line.split(',') for line in lines[header_idx + 2:]]
    df = pd.DataFrame(data, columns=columns)

    # Обработка времени
    time_parts = df["Prog Time"].str.split(':', expand=True).astype(float)
    total_seconds = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
    start_seconds = total_seconds.iloc[0]

    parse_sd = pd.DataFrame({
        col_list[0]: pd.to_datetime(df["Time Stamp"], format="mixed", errors="coerce"),
        col_list[1]: (total_seconds - start_seconds) / 60,
        col_list[2]: total_seconds - start_seconds,
        col_list[3]: pd.to_numeric(df["Voltage"], errors="coerce").fillna(0),
        col_list[4]: pd.to_numeric(df["Current"], errors="coerce").fillna(0),
        col_list[5]: pd.to_numeric(df["Temperature"], errors="coerce").fillna(0),
        col_list[6]: pd.to_numeric(df["Capacity"], errors="coerce").fillna(0),
    })
    return parse_sd

# функция работы с файлами для приведения их к виду parsed, где не будет лишней информации
def files_processor_first(args, data_and_parsed_directory_dict):
    csv_file_name, temperature = args
    # создаем пути для каждой из дериктории
    for dir_name, dir_path in list(data_and_parsed_directory_dict.items())[1:]:
        dir_dir = os.path.join(dir_path, temperature)
        if not os.path.exists(dir_dir):
            os.makedirs(dir_dir, exist_ok=True)

    # чтение данных и сохранение обновленных
    # cоздаем путь к файлу
    crude_file_path = os.path.join(data_and_parsed_directory_dict["LG_HG2_data"], temperature , f"{csv_file_name}.csv")
    df = parse_crude_data(crude_file_path)

    # Есть ли ненулевые значения тока? Если нет - пропускаем файл
    if (df[col_list[4]] == 0).all():
        return

    # # сохраняем измененные файлы
    parsed_file_path = os.path.join(data_and_parsed_directory_dict["LG_HG2_parsed"], temperature, f"{csv_file_name}_parsed.csv")
    df.to_csv(parsed_file_path, index=False)
    return

# функция работы с файлами для приведения их к виду processed, где будет высчитана дополнительная информация на основе имеющихся данных
def files_processor_second(args, data_and_parsed_directory_dict):
    csv_file_name, temperature = args
    # Создаем директории для обработанных данных
    processed_dir = os.path.join(data_and_parsed_directory_dict["LG_HG2_processed"], temperature)
    parsed_dir = os.path.join(data_and_parsed_directory_dict["LG_HG2_parsed"], temperature)
    parsed_file_path = os.path.join(parsed_dir, f"{csv_file_name}_parsed.csv")

    if not os.path.exists(parsed_file_path):
        print(f"Файл не найден: {parsed_file_path}")
        return

    # Загружаем данные
    df = pd.read_csv(parsed_file_path)

    # Найти файл C20 для функций интерполяции pOCV-SOC
    c20_file = next((f for f in os.listdir(parsed_dir) if 'C20' in f), None)
    if c20_file:
        c20_file_path = os.path.join(parsed_dir, c20_file)
        charge_soc_fn, discharge_soc_fn = get_pOCV_SOC_interp_fn(c20_file_path)
        max_charge_capacity, max_discharge_capacity = get_max_capacities(c20_file_path)

        # Кулоновский подсчет Формула: Ёмкость (Ah) = Ток (A) × Время (ч)
        df['Time_diff'] = df[col_list[2]].diff().fillna(0) / 3600
        df['Cumulative_Capacity_Ah'] = (df[col_list[4]] * df['Time_diff']).cumsum()

        # Определение начального SOC
        initial_soc = get_initial_soc(df, charge_soc_fn, discharge_soc_fn, col_list[4], col_list[3])

        # Итерационный расчет SOC
        soc_values = []
        for index, row in df.iterrows():
            cum_capacity = row['Cumulative_Capacity_Ah']
            if row[col_list[4]] < 0:  # Разряд
                soc = initial_soc - (abs(cum_capacity) / abs(max_discharge_capacity))
            else:  # Заряд
                soc = initial_soc + (cum_capacity / max_charge_capacity)

            soc = max(0, min(soc, 1))
            soc_values.append(soc)

        df[col_list[8]] = soc_values

        # постобработка Экспоненциальное скользящее среднее убирает шум в данных SOC.
        alpha = 0.1
        df[col_list[8]] = df[col_list[8]].ewm(alpha=alpha).mean()

        # Округление времени и удаление дубликатов
        df['Rounded_Time'] = df[col_list[2]].round().astype(int)
        df_processed = df.drop_duplicates(subset='Rounded_Time')

        # Сохраняем обработанные данные
        processed_file_path = os.path.join(processed_dir, f'{csv_file_name}_processed.csv')
        df_processed.to_csv(processed_file_path, index=False)
    else:
        print(f"C20 файл не найден в директории {parsed_dir}")

    return

# функция через которую запускаются все остальные
def main(data_directory_dict):
    # 0) запускаем функцию если таких файлов не обнаружено
    existing, missing = check_directories_exist(data_directory_dict)
    print(existing, missing)
    if missing:
        print(f"Обнаружены пустые/отсутствующие директории: {missing}")
        # 1) проходимся по нашим данным, беря деректории содержащие только degC
        temperatures_directory = [folder for folder in os.listdir(data_directory_dict["LG_HG2_data"]) if
                                  'degC' in folder]
        # создаем дериктории если их нет
        print("Создаю директории:")
        for dir_name, dir_path in list(data_directory_dict.items())[1:]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

        num_processes = 4
        # 2) подготавливаем задачи для обработки всех CSV файлов во всех температурных директориях:
        # проходимся по всем найденным папкам temperatures и фиксируем пути к каждой конкретной температурной директории;
        # ищем все CSV файлы в дерикториях; обрабатываем каждый файл

        tasks_args = []
        for temp in temperatures_directory:
            raw_data_T_directory = os.path.join(data_directory_dict["LG_HG2_data"], temp)
            csv_files = [f for f in os.listdir(raw_data_T_directory) if f.endswith('.csv')]

            for csv_file in csv_files:
                csv_file_name = csv_file.split(".csv")[0]
                tasks_args.append((csv_file_name, temp))

        # 3) исследуем файлы перебирая tasks_args, реализуя параллельную обработку всех тестовых файлов в parsed
        with Pool(num_processes) as pool:
            tasks_args = [(task, data_directory_dict) for task in tasks_args]  # Создаем список аргументов для каждой задачи
            list(tqdm(pool.starmap(files_processor_first, tasks_args), total=len(temperatures_directory)))  # Запускаем все задачи с прогресс-баром

        tasks_args = []
        for temp in temperatures_directory:
            raw_data_T_directory = os.path.join(data_directory_dict["LG_HG2_data"], temp)
            csv_files = [f for f in os.listdir(raw_data_T_directory) if f.endswith('.csv')]

            for csv_file in csv_files:
                csv_file_name = csv_file.split(".csv")[0]
                tasks_args.append((csv_file_name, temp))

        # 4) исследуем файлы перебирая tasks_args, реализуя параллельную обработку всех тестовых файлов в processed
        with Pool(num_processes) as pool:
            tasks_args = [(task, data_directory_dict) for task in tasks_args]  # Создаем список аргументов для каждой задачи
            list(tqdm(pool.starmap(files_processor_second, tasks_args), total=len(temperatures_directory)))  # Запускаем все задачи с прогресс-баром

    else:
        print("Все директории существуют. Программа не запускается.")

    return

if __name__ == '__main__':
    main_directory = "/Users/nierra/Desktop/диплом-2/датасет_2/Data"
    data_directory_dict = {"LG_HG2_data": f"{main_directory}/LG_HG2_data",
                           "LG_HG2_parsed": f"{main_directory}/LG_HG2_parsed",
                           "LG_HG2_processed": f"{main_directory}/LG_HG2_processed",
                           "LG_HG2_parsed_plots": f"{main_directory}/LG_HG2_parsed_plots",
                           "LG_HG2_processed_plots": f"{main_directory}/LG_HG2_processed_plots"}
    main(data_directory_dict)