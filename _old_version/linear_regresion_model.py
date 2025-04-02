import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import numpy as np
#📉 Mean Absolute Error (MAE): 0.75
#📊 Mean Absolute Percentage Error (MAPE): 123.78%
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

# Побудова моделі лінійної регресії
model = LinearRegression()

# Навчання моделі
model.fit(X_train, y_train)

# Оцінка моделі на тестовому датасеті
y_pred = model.predict(X_test)

# Метрики на тестовому датасеті
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📊 Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted (Test Dataset)")
plt.show()