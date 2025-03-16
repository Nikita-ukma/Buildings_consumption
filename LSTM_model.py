import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Завантаження тренувальних даних
energy_data_train = pd.read_csv('data/monthly_consumption.csv', sep=',')
building_data = pd.read_csv('data/power-laws-forecasting-energy-consumption-metadata.csv', sep=';')
temperature_data = pd.read_csv('data/monthly_weather.csv', sep=',')
holidays_data = pd.read_csv('data/monthly_holidays.csv', sep=',')

# Об'єднання тренувальних даних
data_train = pd.merge(energy_data_train, building_data, on='SiteId')
data_train = pd.merge(data_train, temperature_data, on=['SiteId', 'Month'])
data_train = pd.merge(data_train, holidays_data, on=['SiteId', 'Month'])

# Перетворення Timestamp у datetime
data_train['Timestamp'] = pd.to_datetime(data_train['Month'])
data_train['Month'] = data_train['Timestamp'].dt.month
data_train['Year'] = data_train['Timestamp'].dt.year

# Додавання нових ознак
data_train['value_lag1'] = data_train.groupby('SiteId')['value'].shift(1)
data_train['value_lag2'] = data_train.groupby('SiteId')['value'].shift(2)
data_train['value_lag3'] = data_train.groupby('SiteId')['value'].shift(3)
data_train['value_rolling_mean'] = data_train.groupby('SiteId')['value'].rolling(window=3).mean().reset_index(level=0, drop=True)

# Видалення пропущених значень
data_train.dropna(inplace=True)

# Масштабування даних
scaler = MinMaxScaler()
numerical_features = data_train.select_dtypes(include=['float64', 'int64']).columns
data_train[numerical_features] = scaler.fit_transform(data_train[numerical_features])

# Підготовка тренувальних даних
X_train = data_train.drop(columns=["value"])
y_train = data_train["value"]

X_train = X_train.select_dtypes(include=['float64', 'int64'])

# Функція для створення часових послідовностей
def create_dataset(X, y, time_step=1):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        v = X.iloc[i:(i + time_step)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_step])
    return np.array(Xs), np.array(ys)

time_step = 6  # Використання довшої історії для прогнозу
X_train, y_train = create_dataset(X_train, y_train, time_step)

# Побудова LSTM моделі з покращеннями
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01)),  # Додатковий шар LSTM
    Dropout(0.3),
    LSTM(50, return_sequences=False, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(50, activation='relu'),
    Dense(1)
])

# Компіляція моделі з меншим learning rate
model.compile(optimizer='adam', loss='mean_squared_error')

# Додавання EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Навчання моделі
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Завантаження тестових даних
energy_data_test = pd.read_csv('data/monthly_consumption_test.csv', sep=',')
data_test = pd.merge(energy_data_test, building_data, on='SiteId')
data_test = pd.merge(data_test, temperature_data, on=['SiteId', 'Month'])
data_test = pd.merge(data_test, holidays_data, on=['SiteId', 'Month'])

# Обробка тестових даних
data_test['Timestamp'] = pd.to_datetime(data_test['Month'])
data_test['Month'] = data_test['Timestamp'].dt.month
data_test['Year'] = data_test['Timestamp'].dt.year

data_test['value_lag1'] = data_test.groupby('SiteId')['value'].shift(1)
data_test['value_lag2'] = data_test.groupby('SiteId')['value'].shift(2)
data_test['value_lag3'] = data_test.groupby('SiteId')['value'].shift(3)
data_test['value_rolling_mean'] = data_test.groupby('SiteId')['value'].rolling(window=3).mean().reset_index(level=0, drop=True)

# Видалення пропущених значень
data_test.dropna(inplace=True)

# Масштабування тестових даних за допомогою того самого scaler
data_test[numerical_features] = scaler.transform(data_test[numerical_features])

# Підготовка тестових даних
X_test = data_test.drop(columns=["value"])
y_test = data_test["value"]

X_test = X_test.select_dtypes(include=['float64', 'int64'])

# Перетворення тестових даних у формат, прийнятний для LSTM
X_test, y_test = create_dataset(X_test, y_test, time_step)

# Прогнозування
y_pred = model.predict(X_test)

# Оцінка моделі
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📊 Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# Візуалізація результатів
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()