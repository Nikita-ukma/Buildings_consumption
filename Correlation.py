# A program, which shows correlation variables to main variable (Energy Consumption (kWh) Normalized)

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv(".csv")
# Завантажуємо дані (припускаємо, що вони вже завантажені)
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek  
data['Hour'] = data['Timestamp'].dt.hour  

# Додаємо сумарні енерговитрати за місяць як цільову змінну
monthly_energy = data.groupby(['Building Type', 'Month'])['Energy Consumption (kWh) Normalized'].transform('sum')
data['Monthly_Energy'] = monthly_energy

columns_to_drop = [
    'Building Type'
]


data.drop(columns=columns_to_drop, inplace=True)
data["Occupancy Schedule"] = data["Occupancy Schedule"].map({"Vacant": 0, "Occupied": 1}).astype(float)
# Обчислюємо кореляцію всіх ознак з цільовою змінною
corr_matrix = data.corr()
# Визначаємо цільову змінну
target_variable = "Energy Consumption (kWh) Normalized"

# Обчислюємо кореляцію всіх змінних із цільовою змінною
correlations = data.corr()[target_variable].drop(target_variable)

# Сортуємо за абсолютним значенням кореляції
top_correlations = correlations.abs().sort_values(ascending=False).head(10)

# Виводимо результати
print("Топ-10 змінних за кореляцією з цільовою змінною:\n")
for feature, corr_value in top_correlations.items():
    print(f"{feature}: {corr_value:.4f}")