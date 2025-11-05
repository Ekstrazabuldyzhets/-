import torch.nn as nn
import torch

# Custom dataset class for LSTM
class BatteryDatasetLSTM():
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

    def get_unique_filenames(self):
        return set(self.filenames)

    def get_times(self):
        return self.times

# SoCLSTM Model Функция возвращает тензор предсказанных значений SOC для каждой входной последовательности в батче.
class SoCLSTM(nn.Module):
    # input_size=5 - 5 входных признаков (напряжение, ток, температура, мощность, емкость)
    def __init__(self, input_size, hidden_size, num_layers):
        super(SoCLSTM, self).__init__()
        self.hidden_size = hidden_size # hidden_size - размер скрытого состояния (настраивается Optuna)
        self.num_layers = num_layers # num_layers - количество LSTM слоев (настраивается Optuna)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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