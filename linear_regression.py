import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv("residential_energy_normalized.csv")

# Обробка часу
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['Hour'] = data['Timestamp'].dt.hour

# Додаємо сумарні енерговитрати за місяць
monthly_energy = data.groupby(['Building Type', 'Month'])['Energy Consumption (kWh) Normalized'].transform('sum')
data['Monthly_Energy'] = monthly_energy

# Вибір ознак
features = [
    'Temperature (°C)',
    'Humidity (%)',
    'Occupancy Rate (%)',
    'Energy Price ($/kWh)',
    'Month',
    'DayOfWeek',
    'Hour',
    'Energy Savings Potential (%)',
    'Voltage Levels (V)',
    'Demand Response Participation',
    'Lighting Consumption (kWh)',
    'Water Usage (liters)',
    'Thermal Comfort Index',
    'Power Factor'
]

X = data[features]
y = data['Energy Consumption (kWh) Normalized']

# Нормалізація даних
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Розділення даних
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Створення та навчання моделі
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозування
hourly_pred = model.predict(X_test)

# Агрегація за місяць
test_data = pd.DataFrame(X_test, columns=features)
test_data['Actual_Hourly'] = y_test.values
test_data['Predicted_Hourly'] = hourly_pred

monthly_actual = test_data.groupby('Month')['Actual_Hourly'].sum()
monthly_pred = test_data.groupby('Month')['Predicted_Hourly'].sum()

# Оцінка моделі
rmse = mean_squared_error(monthly_actual, monthly_pred, squared=False)
r2 = r2_score(monthly_actual, monthly_pred)
print(f"RMSE (Linear Regression): {rmse:.6f}")
print(f"R² (Linear Regression): {r2:.6f}")

# Важливість ознак
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nВажливість ознак (коефіцієнти):")
print(coefficients)

# Візуалізація
plt.figure(figsize=(12, 6))
plt.barh(coefficients['Feature'], coefficients['Coefficient'])
plt.xlabel('Коефіцієнт')
plt.title('Важливість ознак у Linear Regression')
plt.show()

# Прогнози vs реальні значення
plt.figure(figsize=(10, 6))
plt.plot(monthly_actual.index, monthly_actual, marker='o', label='Реальні')
plt.plot(monthly_actual.index, monthly_pred, marker='x', label='Прогноз')
plt.xlabel('Місяць')
plt.ylabel('Енерговитрати (кВт·г)')
plt.legend()
plt.title('Linear Regression: Прогноз vs Реальність')
plt.grid(True)
plt.show()