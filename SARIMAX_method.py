import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Завантаження обробленого датасету
df_monthly = pd.read_csv("processed_dataset.csv", parse_dates=["Year-Month"], index_col="Year-Month")
#MAPE: 0.0692, RMSE: 2145.0523

# Визначаємо предиктори (всі змінні, крім цільової)
exog_vars = [col for col in df_monthly.columns if col != "Energy Consumption (kWh)"]

# Перетворення екзогенних змінних у числовий формат (якщо є текстові значення)
df_monthly[exog_vars] = df_monthly[exog_vars].apply(pd.to_numeric, errors='coerce')

# Перевірка на наявність некоректних типів даних
print("Перевірка типів даних у предикторах:")
print(df_monthly[exog_vars].dtypes)

# Розділення на тренувальний та тестовий набір
train_size = int(len(df_monthly) * 0.8)
train, test = df_monthly.iloc[:train_size], df_monthly.iloc[train_size:]

# Встановлення фіксованих параметрів SARIMAX (налаштуй за необхідності)
order = (1, 1, 1)  # (p, d, q)
seasonal_order = (1, 1, 1, 12)  # (P, D, Q, s)
print(f"Використовуємо фіксовані параметри: order={order}, seasonal_order={seasonal_order}")

# Налаштування SARIMAX
model = SARIMAX(train["Energy Consumption (kWh)"].astype(float), 
                exog=train[exog_vars].astype(float),
                order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False)

# Навчання моделі
sarimax_result = model.fit()

# Прогнозування
forecast = sarimax_result.predict(start=len(train), end=len(df_monthly)-1, exog=test[exog_vars].astype(float), dynamic=False)

# Обчислення метрик
mape = mean_absolute_percentage_error(test["Energy Consumption (kWh)"], forecast)
rmse = np.sqrt(mean_squared_error(test["Energy Consumption (kWh)"], forecast))
print(f"MAPE: {mape:.4f}, RMSE: {rmse:.4f}")

# Візуалізація
plt.figure(figsize=(10,5))
plt.plot(train.index, train["Energy Consumption (kWh)"], label="Train")
plt.plot(test.index, test["Energy Consumption (kWh)"], label="Test")
plt.plot(test.index, forecast, label="SARIMAX Forecast", linestyle="dashed")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Energy Consumption (kWh)")
plt.title("SARIMAX Energy Consumption Forecast")
plt.show()

print("Прогнозування завершено!")
