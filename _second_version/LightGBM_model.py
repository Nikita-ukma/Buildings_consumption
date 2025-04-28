'''
RESULT:
RMSE: 2.86 kWh
MAE: 1.07 kWh
MAPE: 5.72%
R²: 0.9807
'''


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv("data/electricity_dataset.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# 1. Обробка рядкових значень
string_features = ['Building Type', 'Occupancy Schedule', 'Building Orientation', 
                  'Carbon Emission Reduction Category', 'Peak Demand Reduction Indicator']

# Кодування категоріальних змінних
label_encoders = {}
for col in string_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 2. Нормалізація цільової змінної
energy_scaler = MinMaxScaler()
data['Energy Consumption (kWh) Normalized'] = energy_scaler.fit_transform(
    data[['Energy Consumption (kWh)']])

# 3. Додавання часових фіч
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek  
data['Hour'] = data['Timestamp'].dt.hour
data['IsWeekend'] = data['DayOfWeek'].isin([5,6]).astype(int)
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour']/24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour']/24)

# 4. Додавання взаємодій
data['Temp_Humidity'] = data['Temperature (°C)'] * data['Humidity (%)']
data['Occupancy_Ratio'] = data['Occupancy Rate (%)'] / (data['Building Size (m²)'] + 1e-6)

# 5. Вибір фіч
features = [
    # Основні фічі
    'Temperature (°C)', 'Humidity (%)', 'Occupancy Rate (%)',
    'Energy Price ($/kWh)', 'Power Factor', 'Voltage Levels (V)',
    
    # Категоріальні
    'Building Type', 'Occupancy Schedule', 'Building Orientation',
    
    # Часові
    'Month', 'DayOfWeek', 'Hour', 'IsWeekend', 'Hour_sin', 'Hour_cos',
    
    # Технічні
    'Building Size (m²)', 'Window-to-Wall Ratio (%)', 'Insulation Quality Score',
    'Building Age (years)', 'Equipment Age (years)',
    
    # Взаємодії
    'Temp_Humidity', 'Occupancy_Ratio',
    
    # Інше
    'Demand Response Participation', 'Thermal Comfort Index',
    'Carbon Emission Reduction Category'
]

X = data[features]
y = data['Energy Consumption (kWh) Normalized']

# 6. Категоріальні фічі для LightGBM
cat_features = ['Building Type', 'Occupancy Schedule', 'Building Orientation', 
               'Month', 'DayOfWeek', 'Hour', 'IsWeekend', 
               'Carbon Emission Reduction Category']
for col in cat_features:
    X[col] = X[col].astype('category')

# 7. Розділення даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Навчання моделі LightGBM
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[test_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100)
    ]
)

# 9. Прогнозування та де-нормалізація
hourly_pred_normalized = model.predict(X_test)
hourly_pred = energy_scaler.inverse_transform(hourly_pred_normalized.reshape(-1, 1)).flatten()
y_test_original = energy_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

# 10. Оцінка моделі
rmse = mean_squared_error(y_test_original, hourly_pred, squared=False)
mae = mean_absolute_error(y_test_original, hourly_pred)
mape = mean_absolute_percentage_error(y_test_original, hourly_pred)
r2 = r2_score(y_test_original, hourly_pred)

print("\n=== Model Evaluation ===")
print(f"RMSE: {rmse:.2f} kWh")
print(f"MAE: {mae:.2f} kWh")
print(f"MAPE: {mape:.2%}")
print(f"R²: {r2:.4f}")

# 11. Агрегація за місяць для аналізу
test_data = X_test.copy()
test_data['Actual'] = y_test_original
test_data['Predicted'] = hourly_pred
test_data['Month'] = data.loc[X_test.index, 'Month']  # Додаємо місяць з оригінальних даних

monthly_actual = test_data.groupby('Month')['Actual'].sum()
monthly_pred = test_data.groupby('Month')['Predicted'].sum()

# 12. Візуалізація
plt.figure(figsize=(12, 6))
plt.plot(monthly_actual.index, monthly_actual, 'o-', label='Actual')
plt.plot(monthly_actual.index, monthly_pred, 'x--', label='Predicted')
plt.title('Monthly Energy Consumption: Actual vs Predicted')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.show()

# Важливість фіч
lgb.plot_importance(model, figsize=(10, 8), max_num_features=20)
plt.title('Feature Importance')
plt.show()