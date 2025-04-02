import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 🔹 Завантаження даних
df = pd.read_csv("residential_energy_normalized.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df.set_index('Timestamp', inplace=True)

# 🔹 Конвертація всіх колонок у числовий формат
df = df.apply(pd.to_numeric, errors='coerce')

# 🔹 Видалення об'єктних колонок (якщо залишилися)
df_numeric = df.select_dtypes(include=[np.number])

# 🔹 Ресемплінг **залишаємо на 12-годинному рівні**
df_12h = df_numeric.resample('12h').mean()

# 🔹 Визначення ознак та цільової змінної
features = [
    "Temperature (°C)", "Humidity (%)", "Occupancy Rate (%)",
    "Lighting Consumption (kWh)", "HVAC Consumption (kWh)",
    "Energy Price ($/kWh)", "Carbon Emission Rate (g CO2/kWh)",
    "Voltage Levels (V)", "Indoor Temperature (°C)", "Building Age (years)",
    "Equipment Age (years)", "Energy Efficiency Rating", "Building Size (m²)",
    "Window-to-Wall Ratio (%)", "Insulation Quality Score", 
    "Historical Energy Consumption (kWh)", "Solar Irradiance (W/m²)",
    "Smart Plug Usage (kWh)", "Water Usage (liters)"
]
target = "Energy Consumption (kWh) Normalized"

# 🔹 Видалення пропущених значень
df_12h = df_12h[[target] + features].dropna()

# 🔹 Масштабування ознак
scaler = StandardScaler()
df_12h[features] = scaler.fit_transform(df_12h[features])

# 🔹 Перевірка стаціонарності
result = adfuller(df_12h[target])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# 🔹 Диференціювання, якщо ряд нестаціонарний
if result[1] > 0.05:
    df_12h[target] = df_12h[target].diff().dropna()
    df_12h = df_12h.dropna()

# 🔹 Встановлення частоти індексу (12 годин)
df_12h = df_12h.asfreq('12h')

# 🔹 Поділ на тренувальні та тестові дані (80/20)
train_size = int(len(df_12h) * 0.8)
train, test = df_12h.iloc[:train_size], df_12h.iloc[train_size:]

# 🔹 Параметри моделі SARIMAX
order = (1, 1, 1)  # Авторегресія, диференціювання, MA
seasonal_order = (1, 1, 1, 14)  # Тижнева сезонність (14 точок = 7 днів по 12H)

# 🔹 Побудова моделі
model = SARIMAX(
    train[target], 
    exog=train[features],  
    order=order, 
    seasonal_order=seasonal_order,
    enforce_stationarity=False, 
    enforce_invertibility=False
)
model_fit = model.fit()

# 🔹 Функція для **рекурсивного прогнозування** на 30 днів (~60 точок)
def recursive_forecast(model_fit, test_exog, steps=60):
    forecast = []
    last_data = test_exog.iloc[:1]  # Початковий екзогенний вектор

    for i in range(steps):
        fc = model_fit.get_forecast(steps=1, exog=last_data).predicted_mean
        forecast.append(fc.iloc[0])

        # Оновлюємо екзогенні змінні для наступного прогнозу
        last_data = test_exog.iloc[i+1:i+2] if i+1 < len(test_exog) else last_data

    return pd.Series(forecast, index=test_exog.index[:steps])

# 🔹 Виконання прогнозування на 60 точок (~30 днів)
forecast_values = recursive_forecast(model_fit, test[features], steps=60)

# 🔹 Візуалізація прогнозу
plt.figure(figsize=(12, 6))
plt.plot(train.index[-100:], train[target].iloc[-100:], label="Тренувальні дані")
plt.plot(test.index[:60], test[target][:60], label="Реальні значення (30 днів)")
plt.plot(test.index[:60], forecast_values, label="Прогноз", linestyle="--")
plt.legend()
plt.title("Прогноз енергоспоживання на 30 днів (SARIMAX)")
plt.show()

# 🔹 Оцінка якості прогнозу
print("MAE:", mean_absolute_error(test[target][:60], forecast_values))
print("RMSE:", np.sqrt(mean_squared_error(test[target][:60], forecast_values)))