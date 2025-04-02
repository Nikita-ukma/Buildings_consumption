import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Завантаження даних
data = pd.read_csv("residential_energy_normalized.csv")

# Обробка часу
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['Hour'] = data['Timestamp'].dt.hour

# Вибір ознак
features = [
    'Temperature (°C)', 
    'Humidity (%)', 
    'Occupancy Rate (%)',
    'Energy Price ($/kWh)', 
    'Month', 
    'DayOfWeek', 
    'Hour',
    'Energy Consumption (kWh) Normalized'  # Цільова змінна
]

data = data[features]

# Нормалізація даних
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Підготовка даних для LSTM
def create_dataset(data, look_back=24):
    X, y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), :-1])  # Всі ознаки крім останньої (цільової)
        y.append(data[i+look_back, -1])       # Цільова змінна
    return np.array(X), np.array(y)

look_back = 24  # Використовуємо 24 години для прогнозу
X, y = create_dataset(data_scaled, look_back)

# Розділення на тренувальний та тестовий набори
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Побудова LSTM моделі
model = Sequential([
    LSTM(64, input_shape=(look_back, X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Навчання моделі з ранньою зупинкою
early_stop = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Прогнозування
y_pred = model.predict(X_test)

# Зворотнє масштабування прогнозів
y_test_actual = y_test * (data['Energy Consumption (kWh) Normalized'].max() - data['Energy Consumption (kWh) Normalized'].min()) + data['Energy Consumption (kWh) Normalized'].min()
y_pred_actual = y_pred * (data['Energy Consumption (kWh) Normalized'].max() - data['Energy Consumption (kWh) Normalized'].min()) + data['Energy Consumption (kWh) Normalized'].min()

# Оцінка моделі
rmse = mean_squared_error(y_test_actual, y_pred_actual, squared=False)
r2 = r2_score(y_test_actual, y_pred_actual)
print(f"RMSE (LSTM): {rmse:.2f}")
print(f"R² (LSTM): {r2:.2f}")

# Візуалізація результатів
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Реальні значення')
plt.plot(y_pred_actual, label='Прогнозовані значення')
plt.xlabel('Час (години)')
plt.ylabel('Енерговитрати (кВт·г)')
plt.legend()
plt.title('LSTM: Прогноз vs Реальність')
plt.show()

# Створення DataFrame для агрегації
results = pd.DataFrame({
    'Timestamp': data['Timestamp'].iloc[-len(y_test):],  # Відповідає останнім time_steps точкам
    'Actual': y_test_actual.flatten(),
    'Predicted': y_pred.flatten()
})

# Прогнозування годинних витрат
hourly_pred = model.predict(X_test)

# Агрегація прогнозів за місяць
test_data = X_test.copy()
test_data['Actual_Hourly'] = y_test
test_data['Predicted_Hourly'] = hourly_pred

monthly_actual = test_data.groupby('Month')['Actual_Hourly'].sum()
monthly_pred = test_data.groupby('Month')['Predicted_Hourly'].sum()

# Оцінка на місячному рівні
rmse = mean_squared_error(monthly_actual, monthly_pred, squared=False)
r2 = r2_score(monthly_actual, monthly_pred)
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

import matplotlib.pyplot as plt
# Прогнози vs реальні значення (після агрегації за місяць)
plt.figure(figsize=(10, 6))
plt.plot(monthly_actual.index, monthly_actual, label='Реальні')
plt.plot(monthly_actual.index, monthly_pred, label='Прогноз')
plt.xlabel('Місяць')
plt.ylabel('Енерговитрати (кВт·г)')
plt.legend()
plt.title('Прогноз vs Реальність')
plt.show()

# Візуалізація втрат
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Тренувальна втрата')
plt.plot(history.history['val_loss'], label='Валідаційна втрата')
plt.xlabel('Епоха')
plt.ylabel('Втрата (MSE)')
plt.legend()
plt.title('Графік втрат під час навчання')
plt.show()