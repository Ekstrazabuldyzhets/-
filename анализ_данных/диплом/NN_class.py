import torch.nn as nn
import torch
import torch.nn.functional as F

'''
BatteryDataset пользовательский класс набора данных
'''
# Общий базовый класс
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

    def get_unique_filenames(self):
        return set(self.filenames)

    def get_times(self):
        return self.times

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