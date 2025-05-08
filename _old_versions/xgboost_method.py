import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

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

time_step = 6
X_train_seq, y_train_seq = create_dataset(X_train, y_train, time_step)

# Reshape X_train_seq to 2D
X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], -1)

print("🚀 Навчання моделі XGBoost...")
model = xgb.XGBRegressor(random_state=42)

# Параметри для GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
}

# Пошук найкращих параметрів
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_seq, y_train_seq)
print("Найкращі параметри:", grid_search.best_params_)

# Використання найкращих параметрів
best_model = grid_search.best_estimator_

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

# Масштабування тестових даних
data_test[numerical_features] = scaler.transform(data_test[numerical_features])

# Підготовка тестових даних
X_test = data_test.drop(columns=["value"])
y_test = data_test["value"]
X_test = X_test.select_dtypes(include=['float64', 'int64'])

# Створення часових послідовностей для тестових даних
X_test_seq, y_test_seq = create_dataset(X_test, y_test, time_step)

# Reshape X_test_seq to 2D
X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], -1)

# Прогнозування
y_pred = best_model.predict(X_test_seq)

# Оцінка моделі
mae = mean_absolute_error(y_test_seq, y_pred)
mape = mean_absolute_percentage_error(y_test_seq, y_pred)
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")
print(f"📊 Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")

# Візуалізація прогнозу
plt.figure(figsize=(12, 6))
plt.plot(y_test_seq, label="Actual", linestyle='dashed', color='blue')
plt.plot(y_pred, label="Predicted", linestyle='dashed', color='red')
plt.xlabel("Time Steps")
plt.ylabel("Energy Consumption (Normalized)")
plt.title("Actual vs. Predicted Energy Consumption")
plt.legend()
plt.grid(True)
plt.show()