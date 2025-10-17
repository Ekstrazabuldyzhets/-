import torch
print(torch.__version__)
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# библиотеки
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from torch.profiler import profile, ProfilerActivity
import time
from torch.cuda.amp import GradScaler, autocast

# слабо понимаю что происходит
# Check if the environment variable is set to 'True'
# Синтаксис: os.getenv(имя_переменной, значение_по_умолчанию)
skip_training = os.getenv('SKIP_TRAINING', 'False') == 'True'
skip_optuna = os.getenv('SKIP_OPTUNA', 'False') == 'False'
# skip_optuna = os.getenv('SKIP_OPTUNA', 'False') == 'True'

# путь в дерректорию
PROCESSED_DATA_DIR = '/Users/nierra/Desktop/диплом-2/датасет_2/LG_HG2_processed'
# Список признаков (features) для обучения модели
FEATURE_COLS = ['Voltage [V]', 'Current [A]', 'Temperature [degC]', 'Power [W]', 'CC_Capacity [Ah]']
# Целевая переменная (что мы предсказываем)
LABEL_COL = 'SOC [-]'
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.0001
SEQUENCE_LENGTH = 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Создание Dataset объектов - ?
'''


# Custom dataset class for LSTM
class BatteryDatasetLSTM(Dataset):
    def __init__(self, data_tensor, labels_tensor, sequence_length=50, filenames=None, times=None):
        self.sequence_length = sequence_length
        self.features = data_tensor
        self.labels = labels_tensor
        self.filenames = filenames
        self.times = times

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        idx_end = idx + self.sequence_length
        sequence = self.features[idx:idx_end]
        label = self.labels[idx_end - 1]
        filename = self.filenames[idx_end - 1]
        time = self.times[idx_end - 1]

        sequence = sequence.clone().detach()
        label = label.clone().detach()

        return sequence, label, filename, time

    def get_unique_filenames(self):
        return set(self.filenames)

    def get_times(self):
        return self.times

'''
загрузка и первая обработка данных(добавление новых характеристик на основе имеющихся)
'''

# функция для загрузки данных
def load_data(directory, temperatures):
    frames = []
    # os.listdir(directory) - получает список всех папок в директории
    for temp_folder in os.listdir(directory):
        # if temp_folder in temperatures - фильтрует только нужные температуры
        if temp_folder in temperatures:
            temp_path = os.path.join(directory, temp_folder)
            for file in os.listdir(temp_path):
                # Файлы с постоянным зарядом/разрядом не представляют ценности для обучения модели временных рядов.
                if 'Charge' in file or 'Dis' in file:
                    continue  # Пропускаем файлы постоянного заряда/разряда
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(temp_path, file))
                    df['SourceFile'] = file

                    # Calculate power
                    df['Power [W]'] = df['Voltage [V]'] * df['Current [A]']

                    # Initialize CC_Capacity [Ah] column
                    df['CC_Capacity [Ah]'] = 0.0

                    # Integrating current over time to calculate cumulative capacity
                    df['CC_Capacity [Ah]'] = (df['Current [A]'] * df['Time [s]'].diff().fillna(0) / 3600).cumsum()

                    frames.append(df)
    return pd.concat(frames, ignore_index=True)

temperatures_to_process = ['25degC', '0degC', 'n10degC', 'n20degC', '10degC', '40degC']
data = load_data(PROCESSED_DATA_DIR, temperatures_to_process)
print(data)

'''
подготавливаем данные для обучения модели (нормализация разделение на тестовую и обучающие выборки)
'''
# StandardScaler() стандартизирует данные: вычитает среднее и делит на стандартное отклонение
# Приводит все признаки к одинаковому масштабу, что ускоряет обучение нейросети
scaler = StandardScaler()
data[FEATURE_COLS] = scaler.fit_transform(data[FEATURE_COLS])
print(data)

# Разделение на train/val/test
# Все файлы (100%)
#     ↓
# train_files (80%) + temp_files (20%)
#                     ↓
# val_files (10%) + test_files (10%)
unique_files = np.array(list(set(data['SourceFile'])))
train_files, temp_files = train_test_split(unique_files, test_size=0.2, random_state=24)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=24)

# фильтрация - создание новых датасетов
def filter_data_by_filenames(df, filenames):
    return df[df['SourceFile'].isin(filenames)]

train_data = filter_data_by_filenames(data, train_files)
val_data = filter_data_by_filenames(data, val_files)
test_data = filter_data_by_filenames(data, test_files)

# Конвертация в тензоры и загрузка в GPU - ускоряет обучение но я пока не понял ка кименно
train_tensor = torch.tensor(train_data[FEATURE_COLS].values, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_data[LABEL_COL].values, dtype=torch.float32).to(device)

val_tensor = torch.tensor(val_data[FEATURE_COLS].values, dtype=torch.float32).to(device)
val_labels = torch.tensor(val_data[LABEL_COL].values, dtype=torch.float32).to(device)

test_tensor = torch.tensor(test_data[FEATURE_COLS].values, dtype=torch.float32).to(device)
test_labels = torch.tensor(test_data[LABEL_COL].values, dtype=torch.float32).to(device)

'''
Создание Dataset объектов - ?
'''
train_dataset = BatteryDatasetLSTM(
    torch.tensor(train_data[FEATURE_COLS].values, dtype=torch.float32).to(device),
    torch.tensor(train_data[LABEL_COL].values, dtype=torch.float32).to(device),
    SEQUENCE_LENGTH,
    train_data['SourceFile'].values,
    train_data['Time [s]'].values
)

val_dataset = BatteryDatasetLSTM(
    torch.tensor(val_data[FEATURE_COLS].values, dtype=torch.float32).to(device),
    torch.tensor(val_data[LABEL_COL].values, dtype=torch.float32).to(device),
    SEQUENCE_LENGTH,
    val_data['SourceFile'].values,
    val_data['Time [s]'].values
)

test_dataset = BatteryDatasetLSTM(
    torch.tensor(test_data[FEATURE_COLS].values, dtype=torch.float32).to(device),
    torch.tensor(test_data[LABEL_COL].values, dtype=torch.float32).to(device),
    SEQUENCE_LENGTH,
    test_data['SourceFile'].values,
    test_data['Time [s]'].values
)

# Создание DataLoader
# Автоматически создает батчи для обучения
# shuffle=True для тренировочных данных - перемешивает данные каждый эпох
# shuffle=False для validation/test - сохраняет порядок для воспроизводимости
# Все данные: [запись1, запись2, ..., запись441453]
#     ↓
# Батчи: [[1-128], [129-256], [257-384], ...]
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

train_files = train_dataset.get_unique_filenames()
val_files = val_dataset.get_unique_filenames()
test_files = test_dataset.get_unique_filenames()

print("Training files:", train_files)
print("\nValidation files:", val_files)
print("\nTesting files:", test_files)
print("Train features shape:", train_tensor.shape)
print("Test features shape:", test_tensor .shape)
print("Train labels shape:", train_labels.shape)
print("Test labels shape:", test_labels.shape)

'''
работа с данными - подбор гиперпараметров при помощи OPTUNA
'''
# SoCLSTM Model
class SoCLSTM(nn.Module):
    # input_size=5 - 5 входных признаков (напряжение, ток, температура, мощность, емкость)
    # hidden_size - размер скрытого состояния (настраивается Optuna)
    # num_layers - количество LSTM слоев (настраивается Optuna)
    # batch_first=True - входные данные в формате [batch, sequence, features]
    # fc - полносвязный слой для преобразования выхода LSTM в одно число (SOC)
    def __init__(self, input_size, hidden_size, num_layers):
        super(SoCLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # LSTM выдает 50 значений (по одному для каждого нейрона), а нам нужно одно число - предсказанный SOC.
        # SOC = w₁·h₁ + w₂·h₂ + ... + w₅₀·h₅₀ + b
        self.fc = nn.Linear(hidden_size, 1)
    # Вход: [batch_size, 20, 5] - 20 последовательных измерений, 5 признаков
    #     ↓ LSTM
    # Выход LSTM: [batch_size, 20, hidden_size] - скрытые состояния для всех 20 шагов
    #     ↓ Берем только последний шаг
    # Выход: [batch_size, hidden_size] - только последнее скрытое состояние
    #     ↓ Полносвязный слой
    # Предсказание: [batch_size, 1] - одно число (SOC)

    def forward(self, x):
        # x - входной тензор с размерностью: [batch_size, sequence_length, input_size]
        # Инициализация скрытых состояний h0 (hidden state) - краткосрочная память:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        # Инициализация скрытых состояний c0 (cell state) - долгосрочная память:
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)

        # моделим РАЗОБРАТЬСЯ В LSTM
        out, _ = self.lstm(x, (h0, c0)) # Берем out, игнорируем hn, cn
        # out = out[:, -1, :]  # Берем только последний временной шаг
        out = self.fc(out[:, -1, :]) # fc = nn.Linear(94, 1)
        return out


'''
ОБУЧЕНИЕ
model -	nn.Module -	LSTM модель для обучения
criterion -	loss function -	Функция потерь (MSE)
optimizer -	Optimizer -	Алгоритм оптимизации (Adam)
train_loader -	DataLoader - Загрузчик тренировочных данных
val_loader - DataLoader - Загрузчик валидационных данных
epochs - Максимальное количество эпох
device - torch.device - CPU или GPU для вычислений
patience - Терпение для early stopping
min_delta - Минимальное значимое улучшение
'''
def train_and_validate(model, criterion, optimizer, train_loader, val_loader, epochs, device, patience=20,
                       min_delta=0.001):

    history = {'train_loss': [], 'val_loss': []} # хранит историю ошибок для графиков
    best_val_loss = float('inf') # отслеживает лучшую validation loss
    epochs_no_improve = 0 # счетчик эпох без улучшений

    # Эпоха - один полный проход через все тренировочные данные.
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_iterations = len(train_loader)

        epoch_start_time = time.time()
        # Данные: [128, 20, 5] → LSTM → [128, 1] (предсказания)
        #     ↓
        # Сравнение: [128, 1] (предсказания) vs [128, 1] (реальные SOC)
        #     ↓
        # Вычисление MSE Loss → Градиенты → Обновление весов
        for sequences, labels, _, _ in train_loader:
            # Подготовка данных
            sequences, labels = sequences.to(device), labels.to(device)
            labels = labels.unsqueeze(1) # [128] → [128, 1], чтоб labels и outputs были совместимы


            optimizer.zero_grad() # Обнуление градиентов Что происходит внутри  и зачем - ?

            # Прямой проход (Forward Pass)
            outputs = model(sequences) # Выход: [128, 1] - предсказанные SOC
            loss = criterion(outputs, labels) #  Вычисление потерь

            # Обратный проход (Backward Pass)
            loss.backward()
            # Обновление весов
            optimizer.step()

            # Накопление потерь
            train_loss += loss.item()

        # Мониторинг производительности: 154 секунды на эпоху
        # Прогноз общего времени: 20 эпох × 154 сек = ~51 минута
        # Выявление проблем: Внезапное увеличение времени может указывать на проблемы
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        # Статистика по эпохе
        # # Предположим:
        # # - 3,448 батчей в train_loader
        # # - total train_loss = 1.289
        # # Тогда:
        # train_loss = 1.289 / 3,448 ≈ 0.000374

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Валидация; При валидации мы хотим одинаковые результаты на одних и тех же данных
        model.eval() # Переключение в режим оценки
        '''
        ===========================================================================
        В режиме model.train() (обучение):
        ===========================================================================
        # Dropout слои: Активированы
        # Пример: 50% нейронов случайно отключаются
        # input → [нейрон1, нейрон2, нейрон3, нейрон4] 
        # dropout → [нейрон1, 0, нейрон3, 0]  # 50% отключено
        
        # BatchNorm слои: Используют текущий батч для статистики
        # mean = текущий_батч.mean(), std = текущий_батч.std()
        ===========================================================================
        В режиме model.eval() (валидация):
        ===========================================================================
        # Dropout слои: ОТКЛЮЧЕНЫ
        # input → [нейрон1, нейрон2, нейрон3, нейрон4] 
        # eval → [нейрон1, нейрон2, нейрон3, нейрон4]  # ВСЕ нейроны активны
        
        # BatchNorm слои: Используют НАКОПЛЕННУЮ статистику
        # mean = running_mean, std = running_std (из обучения)
        ===========================================================================
        '''
        val_loss = 0.0
        with torch.no_grad():

            for sequences, labels, _, _ in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                labels = labels.unsqueeze(1)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # Логика ранней остановки
        # patience=20 - ждать 20 эпох без улучшений
        # min_delta=0.001 - минимальное улучшение для сброса счетчика
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f'Эпоха {epoch + 1}/{epochs}, Потеря обучения: {train_loss}, Потеря проверки: {val_loss}')
        print(f'Время, затраченное на эпоху: {epoch_time:.8f} секунд')

        if epochs_no_improve >= patience:
            print('Сработала досрочная остановка')
            # break

    return history

if not skip_optuna:
    print("подберем гиперпараметры")
    # Определить целевую функцию Optuna
    def objective(trial):
        # Предложить гиперпараметры
        hidden_size = trial.suggest_int('hidden_size', 10, 100)
        num_layers = trial.suggest_int('num_layers', 1, 5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

        # Создание модели с предлагаемыми гиперпараметрами
        model = SoCLSTM(input_size=len(FEATURE_COLS), hidden_size=hidden_size, num_layers=num_layers).type(torch.float32).to(device)

        # Определите свою функцию потерь и оптимизатор с помощью предлагаемых гиперпараметров
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Вызовите train и проверьте функцию
        history = train_and_validate(model, criterion, optimizer, train_loader, val_loader, EPOCHS, device)

        # Извлечь последнюю потерю проверки
        last_val_loss = history['val_loss'][-1]
        return last_val_loss

    # Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # Извлечь лучшую пробную версию
    best_trial = study.best_trial
    print(f"Best trial: {best_trial}")

    best_hyperparams = study.best_trial.params
    print('Best hyperparameters:', best_hyperparams)

    # Визуализируйте процесс оптимизации
    from optuna.visualization import plot_optimization_history, plot_param_importances

    # Построение истории оптимизации
    optimization_history = plot_optimization_history(study)
    optimization_history.show()

    # График важности гиперпараметров
    param_importances = plot_param_importances(study)
    param_importances.show()

if not skip_training:
    print("обучим")
    # Использование лучших гиперпараметров
    '''
    hidden_size=73: 73 нейрона в каждом LSTM слое
    num_layers=4: 4 LSTM слоя (глубокая архитектура)
    learning_rate=0.001318: Скорость обучения (~1.3e-3)
    '''
    best_hyperparams = {'hidden_size': 73, 'num_layers': 4, 'learning_rate': 0.00131803348155665}
    hidden_size = best_hyperparams['hidden_size']
    num_layers = best_hyperparams['num_layers']
    EPOCHS = 20
    LEARNING_RATE = best_hyperparams['learning_rate']

    model = SoCLSTM(input_size=len(FEATURE_COLS), hidden_size=hidden_size, num_layers=num_layers)
    model.to(device).type(torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Train and validate the model
    history = train_and_validate(model, criterion, optimizer, train_loader, val_loader, EPOCHS, device)

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the model
    model_path = "soc_lstm_model.pth"
    torch.save({'model_state_dict': model.state_dict(), 'input_size': len(FEATURE_COLS)}, model_path)

from sklearn.metrics import mean_squared_error, mean_absolute_error

model_path = "soc_lstm_model.pth"
'''
ТЕСТИРОВКА
'''
def load_lstm_model(model_path, input_size, hidden_size, num_layers):
    model = SoCLSTM(input_size=len(FEATURE_COLS), hidden_size=hidden_size, num_layers=num_layers).to(device).type(
        torch.float32)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# Load and test the model
best_hyperparams = {'hidden_size': 73, 'num_layers': 4, 'learning_rate': 0.00131803348155665}
hidden_size = best_hyperparams['hidden_size']
num_layers = best_hyperparams['num_layers']
loaded_model = load_lstm_model(model_path, input_size=len(FEATURE_COLS), hidden_size=hidden_size, num_layers=num_layers)


def test_model(model, test_loader, device):
    model.eval()
    test_predictions = []
    test_labels = []

    with torch.no_grad():
        for inputs, labels, _, _ in test_loader:
            outputs = model(inputs)
            test_predictions.extend(outputs.cpu().view(-1).tolist())
            test_labels.extend(labels.cpu().view(-1).tolist())

    return test_predictions, test_labels


# Evaluate the model
test_predictions, test_labels = test_model(loaded_model, test_loader, device)

# Convert predictions and labels to numpy arrays for error calculation
test_predictions_np = np.array(test_predictions)
test_labels_np = np.array(test_labels)

# Calculate MSE and MAE
mse = mean_squared_error(test_labels_np, test_predictions_np)
mae = mean_absolute_error(test_labels_np, test_predictions_np)

print(f"Mean Squared Error on Test Set:: {mse}")
print(f"Mean Absolute Error on Test Set: {mae}")

plt.figure(figsize=(8, 8))
plt.scatter(test_labels, test_predictions, alpha=0.5)
plt.xlabel('True Values [SOC]')
plt.ylabel('Predictions [SOC]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot([0, 1], [0, 1], color='red')
plt.title('Predicted SOC vs True SOC')
plt.show()