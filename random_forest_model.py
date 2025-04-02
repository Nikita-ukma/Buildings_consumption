import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv("residential_energy_normalized.csv")
# Завантажуємо дані (припускаємо, що вони вже завантажені)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek  
data['Hour'] = data['Timestamp'].dt.hour  

# Додаємо сумарні енерговитрати за місяць як цільову змінну
monthly_energy = data.groupby(['Building Type', 'Month'])['Energy Consumption (kWh) Normalized'].transform('sum')
data['Monthly_Energy'] = monthly_energy
features = [
    'Temperature (°C)',
    'Humidity (%)',
    'Occupancy Rate (%)',
    'Energy Price ($/kWh)',
    'Month',
    'DayOfWeek',
    'Hour',
    # Додані нові важливі фічі
    'Energy Savings Potential (%)',
    'Voltage Levels (V)',
    'Demand Response Participation',
    'Lighting Consumption (kWh)',
    'Water Usage (liters)',
    'Thermal Comfort Index',
    'Power Factor'
]

X = data[features]
y = data['Energy Consumption (kWh) Normalized']  # Годинні витрати як ціль

# Розділення даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчання моделі
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

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