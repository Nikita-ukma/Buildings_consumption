import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Завантаження та підготовка даних
data = pd.read_csv("data/electricity_dataset.csv", parse_dates=['Timestamp'], index_col='Timestamp')

# 2. Створення необхідних змінних
data['Hour'] = data.index.hour
data['DayOfWeek'] = data.index.dayofweek
data['IsWeekend'] = data['DayOfWeek'].isin([5,6]).astype(int)

# Обов'язкова перевірка наявності колонок
required_columns = ['Occupancy Rate (%)', 'Building Size (m²)', 'Temperature (°C)', 'Humidity (%)']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Відсутня обов'язкова колонка: {col}")

data['Occupancy_Ratio'] = data['Occupancy Rate (%)'] / (data['Building Size (m²)'] + 1e-6)

# 3. Визначення exog змінних
exog_vars = ['Temperature (°C)', 'Humidity (%)', 'Occupancy_Ratio', 'Hour', 'DayOfWeek', 'IsWeekend']

# 4. Розділення на навчальну та тестову вибірки
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# 5. Побудова SARIMAX моделі з фіксованими параметрами
print("Побудова SARIMAX моделі з параметрами (0,0,0)x(1,0,1,24)...")
model = sm.tsa.SARIMAX(
    endog=train_data['Energy Consumption (kWh)'],
    exog=train_data[exog_vars],
    order=(0, 0, 0),
    seasonal_order=(1, 0, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False,
    freq='H'
)

# 6. Оптимізоване навчання моделі
result = model.fit(
    method='powell',  # Ефективний метод оптимізації
    maxiter=100,      # Збільшена кількість ітерацій
    disp=True,
    full_output=True
)

print(result.summary())

# 7. Прогнозування на тестовій вибірці
print("\nГенерація прогнозів...")
forecast = result.get_forecast(
    steps=len(test_data),
    exog=test_data[exog_vars]
)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# 8. Оцінка якості моделі
metrics = {
    'RMSE': mean_squared_error(test_data['Energy Consumption (kWh)'], forecast_values, squared=False),
    'MAE': mean_absolute_error(test_data['Energy Consumption (kWh)'], forecast_values),
    'MAPE': np.mean(np.abs((test_data['Energy Consumption (kWh)'] - forecast_values) / test_data['Energy Consumption (kWh)'])) * 100
}

print("\nМетрики якості моделі:")
for name, value in metrics.items():
    print(f"{name}: {value:.2f}")

# 9. Візуалізація результатів
plt.figure(figsize=(14, 7))
plt.plot(train_data.index[-100:], train_data['Energy Consumption (kWh)'][-100:], label='Training Data')
plt.plot(test_data.index, test_data['Energy Consumption (kWh)'], label='Actual', color='blue')
plt.plot(test_data.index, forecast_values, label='Forecast', color='red')
plt.fill_between(test_data.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='pink', alpha=0.3, label='95% CI')
plt.title('Energy Consumption Forecast with Optimized SARIMAX')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.show()

# 10. Діагностика залишків
result.plot_diagnostics(figsize=(12, 8))
plt.tight_layout()
plt.show()

# 11. Збереження результатів
output = pd.DataFrame({
    'Timestamp': test_data.index,
    'Actual': test_data['Energy Consumption (kWh)'],
    'Predicted': forecast_values,
    'Lower CI': conf_int.iloc[:, 0],
    'Upper CI': conf_int.iloc[:, 1]
})
output.to_csv('sarimax_forecast_results.csv', index=False)