import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Завантаження тренувального датасету
energy_data_train = pd.read_csv('data/monthly_consumption.csv', sep=',')
building_data = pd.read_csv('data/power-laws-forecasting-energy-consumption-metadata.csv', sep=';')
temperature_data = pd.read_csv('data/monthly_weather.csv', sep=',')
holidays_data = pd.read_csv('data/monthly_holidays.csv', sep=',')

# Об'єднання тренувального датасету
data_train = pd.merge(energy_data_train, building_data, on='SiteId')
data_train = pd.merge(data_train, temperature_data, on=['SiteId', 'Month'])
data_train = pd.merge(data_train, holidays_data, on=['SiteId', 'Month'])

# Перетворення Timestamp у datetime
data_train['Timestamp'] = pd.to_datetime(data_train['Month'])

# Додаткові ознаки
data_train['Month'] = data_train['Timestamp'].dt.month
data_train['Year'] = data_train['Timestamp'].dt.year

# Додавання нових ознак (середнє значення за минулі 3 місяці)
data_train['value_lag1'] = data_train.groupby('SiteId')['value'].shift(1)
data_train['value_lag2'] = data_train.groupby('SiteId')['value'].shift(2)
data_train['value_lag3'] = data_train.groupby('SiteId')['value'].shift(3)
data_train['value_rolling_mean'] = data_train.groupby('SiteId')['value'].rolling(window=3).mean().reset_index(level=0, drop=True)

# Видалення рядків з пропущеними значеннями після додавання нових ознак
data_train.dropna(inplace=True)

# Обробка викидів (заміна на медіану)
Q1 = data_train['value'].quantile(0.25)
Q3 = data_train['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_train['value'] = np.where(
    (data_train['value'] < lower_bound) | (data_train['value'] > upper_bound),
    data_train['value'].median(),
    data_train['value']
)

# Видалення непотрібних стовпців
data_train = data_train.drop(columns=['obs_id', 'Timestamp'])

# Перевірка на наявність пропущених значень
data_train.isnull().sum()

# Нормалізація лише числових ознак
numerical_features = data_train.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
data_train[numerical_features] = scaler.fit_transform(data_train[numerical_features])

# Заповнення пропущених значень
data_train.fillna(data_train.median(), inplace=True)

# Розділення на навчальну та тестову вибірки (для тренування моделі)
X_train = data_train.drop(columns=["value"])  # Ознаки
y_train = data_train["value"]  # Цільова змінна

# Завантаження тестового датасету
energy_data_test = pd.read_csv('data/monthly_consumption_test.csv', sep=',')

# Об'єднання тестового датасету
data_test = pd.merge(energy_data_test, building_data, on='SiteId')
data_test = pd.merge(data_test, temperature_data, on=['SiteId', 'Month'])
data_test = pd.merge(data_test, holidays_data, on=['SiteId', 'Month'])

# Перетворення Timestamp у datetime
data_test['Timestamp'] = pd.to_datetime(data_test['Month'])

# Додаткові ознаки
data_test['Month'] = data_test['Timestamp'].dt.month
data_test['Year'] = data_test['Timestamp'].dt.year

# Додавання нових ознак (середнє значення за минулі 3 місяці)
data_test['value_lag1'] = data_test.groupby('SiteId')['value'].shift(1)
data_test['value_lag2'] = data_test.groupby('SiteId')['value'].shift(2)
data_test['value_lag3'] = data_test.groupby('SiteId')['value'].shift(3)
data_test['value_rolling_mean'] = data_test.groupby('SiteId')['value'].rolling(window=3).mean().reset_index(level=0, drop=True)

# Видалення рядків з пропущеними значеннями після додавання нових ознак
data_test.dropna(inplace=True)

# Обробка викидів (заміна на медіану)
Q1 = data_test['value'].quantile(0.25)
Q3 = data_test['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_test['value'] = np.where(
    (data_test['value'] < lower_bound) | (data_test['value'] > upper_bound),
    data_test['value'].median(),
    data_test['value']
)

# Видалення непотрібних стовпців
data_test = data_test.drop(columns=['obs_id', 'Timestamp'])

# Перевірка на наявність пропущених значень
data_test.isnull().sum()

# Нормалізація тестового датасету за допомогою того самого scaler
data_test[numerical_features] = scaler.transform(data_test[numerical_features])

# Заповнення пропущених значень
data_test.fillna(data_test.median(), inplace=True)

# Підготовка тестового датасету
X_test = data_test.drop(columns=["value"])  # Ознаки
y_test = data_test["value"]  # Цільова змінна

# Перетворення даних у формат, придатний для GRU
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12  # Використовуємо 12 місяців для прогнозування
X_train_seq, y_train_seq = create_sequences(X_train, seq_length)
X_test_seq, y_test_seq = create_sequences(X_test, seq_length)

# Перевірка типів даних
print("Типи даних у X_train_seq:", X_train_seq.dtype)
print("Типи даних у y_train_seq:", y_train_seq.dtype)

# Перетворення даних у float32
X_train_seq = X_train_seq.astype(np.float32)
y_train_seq = y_train_seq.astype(np.float32)
X_test_seq = X_test_seq.astype(np.float32)
y_test_seq = y_test_seq.astype(np.float32)

# Побудова моделі GRU
model = Sequential()
model.add(GRU(100, activation='relu', input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))  # Збільшено кількість нейронів
model.add(Dropout(0.2))
model.add(Dense(1))

# Компіляція моделі
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Навчання моделі
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,  # Збільшено кількість епох
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq),
    verbose=1
)

# Оцінка моделі на тестовому датасеті
y_pred = model.predict(X_test_seq)

# Перевірка форм
print("Форма y_test_seq:", y_test_seq.shape)
print("Форма y_pred:", y_pred.shape)

# Метрики на тестовому датасеті
mae = mean_absolute_error(y_test_seq, y_pred)
mape = mean_absolute_percentage_error(y_test_seq, y_pred)

print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📊 Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.plot(y_test_seq, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted (Test Dataset)")
plt.show()