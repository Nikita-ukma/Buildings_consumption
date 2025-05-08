import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Завантаження тренувальних даних
energy_data_train = pd.read_csv('data/monthly_consumption.csv', sep=',')
# Візуалізація результатів
print(energy_data_train.shape)  # Розмірність датасету

building_data = pd.read_csv('data/power-laws-forecasting-energy-consumption-metadata.csv', sep=';')
print(building_data.shape)  # Розмірність датасету
temperature_data = pd.read_csv('data/monthly_weather.csv', sep=',')
print(temperature_data.shape)  # Розмірність датасету

holidays_data = pd.read_csv('data/monthly_holidays.csv', sep=',')
print(holidays_data.shape)  # Розмірність датасету



# Об'єднання тренувальних даних
data_train = pd.merge(energy_data_train, building_data, on='SiteId')
data_train = pd.merge(data_train, temperature_data, on=['SiteId', 'Month'])
data_train = pd.merge(data_train, holidays_data, on=['SiteId', 'Month'])

print(f"energy_data_train: {energy_data_train.shape}")
print(f"building_data: {building_data.shape}")
print(f"temperature_data: {temperature_data.shape}")
print(f"holidays_data: {holidays_data.shape}")

print(f"After merging: {data_train.shape}")  # Після об'єднання таблиць


# Перетворення Timestamp у datetime та встановлення його як індексу
data_train['Timestamp'] = pd.to_datetime(data_train['Month'])
data_train.set_index('Timestamp', inplace=True)

# Видалення дублікатів у часовому індексі
data_train = data_train[~data_train.index.duplicated(keep='first')]

# Встановлення частоти для індексу (місячна частота)
data_train = data_train.asfreq('MS')  # 'MS' для місячного інтервалу


print(f"After merging: {data_train.shape}")  # Після об'єднання таблиць
# Додавання нових ознак
data_train['value_lag1'] = data_train.groupby('SiteId')['value'].shift(1)
data_train['value_lag2'] = data_train.groupby('SiteId')['value'].shift(2)
data_train['value_lag3'] = data_train.groupby('SiteId')['value'].shift(3)
data_train['value_rolling_mean'] = data_train.groupby('SiteId')['value'].rolling(window=3).mean().reset_index(level=0, drop=True)

# Видалення пропущених значень
data_train.dropna(inplace=True)

# Підготовка тренувальних даних
y_train = data_train["value"]

# # Побудова та навчання моделі ARIMA (ручний підбір параметрів)
# model = ARIMA(y_train, order=(5, 1, 0))  # Приклад параметрів (p, d, q)
# model_fit = model.fit()

# Завантаження тестових даних
energy_data_test = pd.read_csv('data/monthly_consumption_test.csv', sep=',')

# Об'єднання тестового датасету
data_test = pd.merge(energy_data_test, building_data, on='SiteId')
data_test = pd.merge(data_test, temperature_data, on=['SiteId', 'Month'])
data_test = pd.merge(data_test, holidays_data, on=['SiteId', 'Month'])

# Перетворення Timestamp у datetime та встановлення його як індексу
data_test['Timestamp'] = pd.to_datetime(data_test['Month'])
data_test.set_index('Timestamp', inplace=True)

# Видалення дублікатів у часовому індексі
data_test = data_test[~data_test.index.duplicated(keep='first')]

# Встановлення частоти для індексу (місячна частота)
data_test = data_test.asfreq('MS')  # 'MS' для місячного інтервалу

# Перевірка, чи індекс монотонний
if not data_test.index.is_monotonic_increasing:
    data_test = data_test.sort_index()  # Сортування за індексом, якщо він не монотонний

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

# # Підготовка тестового датасету
# y_test = data_test["value"]  # Цільова змінна
# print(data_train.shape)  # Розмірність датасету
# # Прогнозування на тестовому датасеті
# y_pred = model_fit.forecast(steps=len(y_test))

# # Оцінка моделі
# mae = mean_absolute_error(y_test, y_pred)
# mape = mean_absolute_percentage_error(y_test, y_pred)
# print(data_train.shape)  # Розмірність датасету
# print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
# print(f"📊 Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# Візуалізація результатів
print(data_train.shape)  # Розмірність датасету

# plt.figure(figsize=(12, 6))
# plt.plot(y_test.index, y_test.values, label='Actual')
# plt.plot(y_test.index, y_pred, label='Predicted')
# plt.legend()
# plt.title("Actual vs Predicted (ARIMA)")
# plt.show()
