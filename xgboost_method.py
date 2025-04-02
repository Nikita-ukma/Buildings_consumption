import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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
    'Hour'
]

X = data[features]
y = data['Energy Consumption (kWh) Normalized']

# Розділення даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Налаштування та навчання XGBoost
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Прогнозування
hourly_pred = model.predict(X_test)

# Агрегація за місяць
test_data = X_test.copy()
test_data['Actual_Hourly'] = y_test
test_data['Predicted_Hourly'] = hourly_pred

monthly_actual = test_data.groupby('Month')['Actual_Hourly'].sum()
monthly_pred = test_data.groupby('Month')['Predicted_Hourly'].sum()

# Оцінка
rmse = mean_squared_error(monthly_actual, monthly_pred, squared=False)
r2 = r2_score(monthly_actual, monthly_pred)
print(f"RMSE (XGBoost): {rmse:.2f}")
print(f"R² (XGBoost): {r2:.2f}")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(monthly_actual.index, monthly_actual, marker='o', label='Реальні')
plt.plot(monthly_actual.index, monthly_pred, marker='x', label='Прогноз XGBoost')
plt.xlabel('Місяць')
plt.ylabel('Енерговитрати (кВт·г)')
plt.legend()
plt.grid(True)
plt.title('XGBoost: Прогноз vs Реальність')
plt.show()

# Важливість ознак
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, height=0.8)
plt.title('Важливість ознак (XGBoost)')
plt.show()