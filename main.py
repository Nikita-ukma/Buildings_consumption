import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# Завантаження даних
file_path = "residential_energy_normalized.csv"  # Замініть на шлях до вашого файлу
df = pd.read_csv(file_path)

# Приклад коду для очищення:
columns_to_drop = [
    'Building Orientation'
]

df.drop(columns=columns_to_drop, inplace=True)
# Обробка неможливих значень
df.loc[df['Humidity (%)'] > 100, 'Humidity (%)'] = 100
df.loc[df['Occupancy Rate (%)'] > 100, 'Occupancy Rate (%)'] = 100

# Перетворення часу
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month

output_file = "residential_energy_normalized.csv"
df.to_csv(output_file, index=False)