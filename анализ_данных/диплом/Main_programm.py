# подключаемые библиотеки:
import NN_class as method_find
import Drow_programm as drow
import matplotlib.pyplot as plt

# подключакем файлы подпрограмм main-a
import Data_reading_and_parsing_programm as read_data
import LSTM_programm as lstm_start

# здесь программа запускается и задаются все главные параметры:
def main():
    # 1) начальная обработка данных
    # задаем начальные пути к дерикториям
    # а) словарь дериторий:
    main_directory = "/Users/nierra/Desktop/диплом-2/датасет_2"
    data_directory_dict = {"LG_HG2_data": f"{main_directory}/LG_HG2_data",
                           "LG_HG2_parsed": f"{main_directory}/LG_HG2_parsed",
                           "LG_HG2_processed": f"{main_directory}/LG_HG2_processed",
                           "LG_HG2_parsed_plots": f"{main_directory}/LG_HG2_parsed_plots",
                           "LG_HG2_processed_plots": f"{main_directory}/LG_HG2_processed_plots"}
    # запускаем функцию считывающую сырые данные и преобразующая их в новые, читаемые для нейросетей(если конечно данные уже не были преобразованы)

    # 2) запускаем выбор модели обучения
    # 2.1) первая модель - LSTM
    lstm_start.main()

    return

if __name__ == "__main__":
    main()
