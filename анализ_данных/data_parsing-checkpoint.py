import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import Manager

import logging

# Set up logging
logging.basicConfig(filename='processing_log.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Define columns
timestamp_col = 'Timestamp'
time_col = 'Time [min]'
time_col_s = 'Time [s]'
voltage_col = 'Voltage [V]'
current_col = 'Current [A]'
temperature_col = 'Temperature [degC]'
capacity_col = 'Capacity [Ah]'
capacity_cc_col = 'Cumulative_Capacity_Ah'
soc_col = 'SOC [-]'

# Directory paths
raw_data_directory = r'..\dataset\LG_HG2_data'
parsed_data_directory = r'..\dataset\LG_HG2_parsed'
processed_data_directory = r'..\dataset\LG_HG2_processed'
parsed_plots_directory = r'..\dataset\LG_HG2_parsed_plots'
processed_plots_directory = r'..\dataset\LG_HG2_processed_plots'

os.makedirs(parsed_data_directory, exist_ok=True)
os.makedirs(processed_data_directory, exist_ok=True)
os.makedirs(parsed_plots_directory, exist_ok=True)
os.makedirs(processed_plots_directory, exist_ok=True)


# Function to parse raw data
def parse_raw_data(file_path: str) -> pd.DataFrame:
    with open(file_path) as f:
        lines = f.readlines()

    column_index = lines.index(next(filter(lambda l: 'Time Stamp' in l, lines)))
    column_line = lines[column_index].split(',')
    data_lines = [l.split(',') for l in lines[column_index + 2:]]

    abs_timestamp_data = []
    timestamp_data_seconds = []
    for l in data_lines:
        abs_timestamp_data.append(pd.Timestamp(l[column_line.index('Time Stamp')]))
        timestamp_str = [float(s) for s in l[column_line.index('Prog Time')].split(':')]
        timestamp_seconds = timestamp_str[0] * 3600 + timestamp_str[1] * 60 + timestamp_str[2]
        timestamp_data_seconds.append(timestamp_seconds)

    df = pd.DataFrame({
        timestamp_col: abs_timestamp_data,
        time_col: [(t - timestamp_data_seconds[0]) / 60 for t in timestamp_data_seconds],  # Time in minutes
        time_col_s: [(t - timestamp_data_seconds[0]) for t in timestamp_data_seconds],  # Time in seconds
        voltage_col: [float(l[column_line.index('Voltage')]) for l in data_lines],
        current_col: [float(l[column_line.index('Current')]) for l in data_lines],
        temperature_col: [float(l[column_line.index('Temperature')]) for l in data_lines],
        capacity_col: [float(l[column_line.index('Capacity')]) for l in data_lines],
    })

    return df

# рисует графики
def generate_and_save_plot(data_df: pd.DataFrame, save_file_path: str, fig_title: str = '',
                           plot_soc: bool = False) -> None:
    num_plots = 5 if plot_soc else 4
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, num_plots * 2.5))

    axs[0].plot(data_df[time_col], data_df[voltage_col], label='Voltage')
    axs[1].plot(data_df[time_col], data_df[current_col], label='Current')
    axs[2].plot(data_df[time_col], data_df[temperature_col], label='Temperature')
    axs[3].plot(data_df[time_col], data_df[capacity_col], label='Capacity')
    if plot_soc:
        axs[4].plot(data_df[time_col], data_df[soc_col], label='SoC')

    for ax in axs:
        ax.legend()
        ax.set_xlabel(time_col)
        ax.set_ylabel('Value')

    fig.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_file_path, format='pdf')
    plt.close(fig)

# Function to create pseudo OCV-SOC interpolation function
def get_pOCV_SOC_interp_fn(file_path: str) -> (interp1d, interp1d):
    df = pd.read_csv(file_path)

    # Process discharge data
    df_discharge = df[df[current_col] < 0].copy()
    df_discharge[capacity_col] = df_discharge[capacity_col] - df_discharge[capacity_col].iloc[0]
    df_discharge[soc_col] = 1 - abs(df_discharge[capacity_col] / df_discharge[capacity_col].iloc[-1])
    max_voltage_discharge = df_discharge[voltage_col].max()
    df_discharge = df_discharge[df_discharge[voltage_col] <= max_voltage_discharge]
    discharge_interp = interp1d(df_discharge[voltage_col], df_discharge[soc_col], bounds_error=False,
                                fill_value="extrapolate")

    # Process charge data
    df_charge = df[df[current_col] > 0].copy()
    df_charge[capacity_col] = df_charge[capacity_col] - df_charge[capacity_col].iloc[0]
    df_charge[soc_col] = abs(df_charge[capacity_col]) / df_charge[capacity_col].iloc[-1]
    max_voltage_charge = df_charge[voltage_col].max()
    df_charge = df_charge[df_charge[voltage_col] <= max_voltage_charge]
    charge_interp = interp1d(df_charge[voltage_col], df_charge[soc_col], bounds_error=False, fill_value="extrapolate")

    return charge_interp, discharge_interp


def get_max_capacities(c20_file_path):
    df_c20 = pd.read_csv(c20_file_path)

    # Find the index where the discharge phase ends and the charge phase begins
    charge_start_index = df_c20[df_c20[current_col] > 0].index[0]

    # Split the DataFrame into discharge and charge phases
    df_discharge = df_c20.iloc[:charge_start_index]
    df_charge = df_c20.iloc[charge_start_index:]

    # Calculate max capacities for discharge and charge phases
    max_discharge_capacity = df_discharge[capacity_col].max() - df_discharge[capacity_col].min()
    max_charge_capacity = df_charge[capacity_col].max() - df_charge[capacity_col].min()

    return max_charge_capacity, max_discharge_capacity


# почему C20 - ?
def process_c20_files(T):
    # logging в Python — это стандартный модуль для ведения журналов (логов),
    # который позволяет записывать информацию о работе программы в различные выходные потоки,
    # такие как файлы или консоль.
    try:
        logging.info(f'Начинается обработка файлов C20 для: {T}')

        # Убедитесь, что каталоги существуют parsed_data_directory - LG_HG2_parsed
        # os.path.join - Интеллектуальное объединение одного или нескольких сегментов пути.
        # Возвращаемое значение представляет собой конкатенацию пути и всех элементов *paths,
        # с одним разделителем каталогов после каждой непустой части, кроме последней.
        parsed_dir = os.path.join(parsed_data_directory, T)
        # Создание директории
        # exist_ok=True - Не вызывает ошибку если директория существует
        os.makedirs(parsed_dir, exist_ok=True)

        # Найдите файл C20 в каталоге необработанных данных для указанной температуры.
        raw_dir = os.path.join(raw_data_directory, T)
        # os.listdir(raw_dir) - возвращает список всех файлов в директории;
        # (f for f in os.listdir(raw_dir) if 'C20' in f) - Итерируется по всем файлам в raw_dir,
        # оставляет только файлы, содержащие 'C20' в названии
        # Берет первый элемент из генератора
        c20_file = next(
            (f for f in os.listdir(raw_dir) if 'C20' in f),  # Генератор
            None  # Значение по умолчанию
        )

        # Проверка наличия файла
        if c20_file:

            raw_c20_file_path = os.path.join(raw_dir, c20_file)
            # Функция parse_raw_data() преобразует сырой CSV в структурированный DataFrame
            # Извлекает колонки: Timestamp, Time, Voltage, Current, Temperature, Capacity
            df = parse_raw_data(raw_c20_file_path)

            # Формирование пути для сохранения
            parsed_c20_file_path = os.path.join(parsed_dir, c20_file.split(".csv")[0] + "_parsed.csv")
            # Сохранение обработанных данных
            df.to_csv(parsed_c20_file_path, index=False)

        logging.info(f'Completed processing C20 files for: {T}')
        return 1  # Return 1 on successful completion

    except Exception as e:
        logging.error(f'Error processing C20 files for: {T} - {e}')
        return 0  # Return 0 on error


def get_initial_soc(df, charge_soc_fn, discharge_soc_fn, current_col, voltage_col):
    initial_voltage = df[voltage_col].iloc[0]

    # Find the index of the first non-zero current
    first_non_zero_index = df[df[current_col] != 0].index[0]
    first_non_zero_current = df[current_col].iloc[first_non_zero_index]

    # Determine SOC based on the sign of the first non-zero current
    if first_non_zero_current < 0:
        return discharge_soc_fn(initial_voltage)
    else:
        return charge_soc_fn(initial_voltage)

# Эта функция обрабатывает отдельный CSV файл: берет данные, рассчитывает SOC и сохраняет результаты.
# !!!!!!!!!!!! откуда инфа по nominal_capacity_ah = 3
'''
Вход: ../dataset/LG_HG2_data/25degC/test1.csv
Выход:
../dataset/LG_HG2_parsed/25degC/test1_parsed.csv (данные без SOC)
../dataset/LG_HG2_processed/25degC/test1_processed.csv (данные с SOC)
../dataset/LG_HG2_parsed_plots/25degC/test1_parsed_plot.pdf (график без SOC)
../dataset/LG_HG2_processed_plots/25degC/test1_processed_plot.pdf (график с SOC)
'''
def process_file(args):
    # csv_file_name - имя файла без расширения (например, "test1")
    # T - температура (например, "25degC")
    # nominal_capacity_ah = 3 - номинальная емкость батареи (3 Ah)
    csv_file_name, T = args
    nominal_capacity_ah = 3
    try:
        logging.info(f'Starting processing for: {csv_file_name} @ {T}')

        # Создание директорий
        parsed_dir = os.path.join(parsed_data_directory, T)
        processed_dir = os.path.join(processed_data_directory, T)
        parsed_plots_dir = os.path.join(parsed_plots_directory, T)
        processed_plots_dir = os.path.join(processed_plots_directory, T)
        os.makedirs(parsed_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(parsed_plots_dir, exist_ok=True)
        os.makedirs(processed_plots_dir, exist_ok=True)

        #чтение данных и сохранение
        raw_file_path = os.path.join(raw_data_directory, T, f'{csv_file_name}.csv')
        df = parse_raw_data(raw_file_path)
        parsed_file_path = os.path.join(parsed_dir, f'{csv_file_name}_parsed.csv')
        df.to_csv(parsed_file_path, index=False)

        # графическое отображение данных
        parsed_plot_file_path = os.path.join(parsed_plots_dir, f'{csv_file_name}_parsed_plot.pdf')
        generate_and_save_plot(df, parsed_plot_file_path, fig_title=f'{csv_file_name} Parsed @ {T}', plot_soc=False)

        # Найти файл C20 для функций интерполяции pOCV-SOC
        c20_file = next((f for f in os.listdir(parsed_dir) if 'C20' in f), None)
        if c20_file:
            # Находит обработанный C20 файл для калибровки
            # Создает интерполяционные функции SOC-напряжение
            # Определяет максимальные емкости заряда/разряда
            c20_file_path = os.path.join(parsed_dir, c20_file)
            charge_soc_fn, discharge_soc_fn = get_pOCV_SOC_interp_fn(c20_file_path)
            max_charge_capacity, max_discharge_capacity = get_max_capacities(c20_file_path)

            # Кулоновский подсчет Формула: Ёмкость (Ah) = Ток (A) × Время (ч)
            df['Time_diff'] = df[time_col].diff().fillna(0) / 60  # Time difference in hours
            df['Cumulative_Capacity_Ah'] = (df[current_col] * df['Time_diff']).cumsum()

            # Определение начального SOC
            initial_soc = get_initial_soc(df, charge_soc_fn, discharge_soc_fn, current_col, voltage_col)

            # Итерационный расчет SOC
            '''
            Разряд: SOC = начальный_SOC - (накопленная_емкость / максимальная_емкость_разряда)
            Заряд: SOC = начальный_SOC + (накопленная_емкость / максимальная_емкость_заряда)
            '''
            for index, row in df.iterrows():
                cum_capacity = row['Cumulative_Capacity_Ah']
                if row[current_col] < 0: # Разряд
                    soc = initial_soc - (abs(cum_capacity) / abs(max_discharge_capacity))
                else:  # Заряд
                    soc = initial_soc + (cum_capacity / max_charge_capacity)

                soc = max(0, min(soc, 1))  # Clamping SoC between 0 and 1
                df.loc[index, soc_col] = soc
        else:
            logging.warning(f'C20 SOC data file missing in directory: {parsed_dir}')

        # постобработка Экспоненциальное скользящее среднее убирает шум в данных SOC.
        alpha = 0.1
        df[soc_col] = df[soc_col].ewm(alpha=alpha).mean()
        #  Округление времени и удаление дубликатов
        df['Rounded_Time'] = df[time_col_s].round().astype(int)
        df_processed = df.drop_duplicates(subset='Rounded_Time')

        # Generating and saving SOC plots
        soc_plot_file_path = os.path.join(processed_plots_dir, f'{csv_file_name}_processed_plot.pdf')
        generate_and_save_plot(df_processed, soc_plot_file_path, fig_title=f'{csv_file_name} SOC @ {T}', plot_soc=True)

        # Save processed data with SOC
        processed_file_path = os.path.join(processed_dir, f'{csv_file_name}_processed.csv')
        df_processed.to_csv(processed_file_path, index=False)

        logging.info(f'Completed processing for: {csv_file_name} @ {T}')
        return 1  # Return 1 on successful completion

    except Exception as e:
        logging.error(f'Error processing: {csv_file_name} @ {T} - {e}')
        return 0  # Return 0 on error


def check_missing_c20_files(directory: str):
    temperatures = [folder for folder in os.listdir(directory) if 'degC' in folder]
    missing_files = []

    for T in temperatures:
        parsed_dir = os.path.join(parsed_data_directory, T)
        c20_file = next((f for f in os.listdir(parsed_dir) if 'C20' in f), None)

        if not c20_file:
            missing_files.append(f'C20 SOC data file missing in directory: {parsed_dir}')

    return missing_files


def update_progress(result, pbar):
    """Update the progress bar by one step."""
    pbar.update(1)

if __name__ == '__main__':
    num_processes = 4
    tasks = []
    # os.listdir - Возвращает список всех файлов и папок в директории (...)
    # for folder in os.listdir(raw_data_directory) - проходимся по каждому элементу
    # if 'degC' in folder Фильтрует только те элементы, которые содержат подстроку 'degC'
    temperatures = [folder for folder in os.listdir(raw_data_directory) if 'degC' in folder]
    # temperatures = ['25degC', '35degC', '45degC']

    # Обработка C20 файлов (калибровочных)
    # Создание пула; Все температурные папки обрабатываются параллельно, а не последовательно,
    # что значительно ускоряет обработку данных.
    with Pool(num_processes) as pool:
        # визуальзация обратной связи
        with tqdm(total=len(temperatures)) as pbar:
            for T in temperatures:
                # T - файл внутри temperatures
                # process_c20_files - функция для выполнения
                # args=(T,) - аргументы для функции (кортеж)
                # callback=... - функция, вызываемая после завершения
                pool.apply_async(process_c20_files, args=(T,), callback=lambda x: update_progress(x, pbar))
            pool.close() # запрещает добавление новых задач
            pool.join() # блокирует выполнение until все задачи завершатся


    # Этот код подготавливает задачи для обработки всех CSV файлов во всех температурных директориях.
    tasks = []  # список очищается после предыдущего использования (для C20 файлов)
    for T in temperatures:
        # Формирование пути
        raw_data_T_directory = os.path.join(raw_data_directory, T)
        # Поиск всех CSV файлов
        '''
        ['C20_calibration.csv', 'test1.csv', 'test2.csv', 'readme.txt', 'config.ini']
        csv_files = ['C20_calibration.csv', 'test1.csv', 'test2.csv']  # результат
        '''
        csv_files = [f for f in os.listdir(raw_data_T_directory) if f.endswith('.csv')]
        for csv_file in csv_files:
            # csv_file = "test1.csv"
            # csv_file_name = "test1"  # после split(".csv")[0]
            csv_file_name = csv_file.split(".csv")[0]
            # (имя_файла, температура(дериктория))
            tasks.append((csv_file_name, T))

    total_tasks = len(tasks)

    with Pool(num_processes) as pool:
        with tqdm(total=total_tasks) as pbar:
            for task in tasks:
                # process_file - Эта функция обрабатывает отдельный CSV файл: парсит данные,
                # рассчитывает SOC и сохраняет результаты.
                pool.apply_async(process_file, args=(task,), callback=lambda x: update_progress(x, pbar))
            pool.close()
            pool.join()