import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv("data/electricity_dataset.csv", parse_dates=['Timestamp'], index_col='Timestamp')

# Вибір лише числових колонок для кореляції
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Розрахунок кореляційної матриці
corr_matrix = data[numeric_cols].corr()

# Вибір кореляцій тільки з цільовою змінною
target_corr = corr_matrix[['Energy Consumption (kWh)']].sort_values(
    by='Energy Consumption (kWh)', 
    ascending=False
)

# Візуалізація теплової карти
plt.figure(figsize=(10, 12))
sns.heatmap(target_corr, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            fmt=".2f",
            linewidths=.5)
plt.title('Кореляція змінних з Energy Consumption (kWh)')
plt.tight_layout()
plt.show()

# Вивід топ-15 найбільш корельованих змінних
print("Топ-15 найбільш корельованих змінних:")
print(target_corr.iloc[:15])

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Вибір ключових змінних для аналізу
features_to_analyze = ['HVAC Consumption (kWh)', 'Lighting Consumption (kWh)', 
                      'Occupancy Rate (%)', 'Temperature (°C)', 'Building Size (m²)']

# Непараметрична кореляція Спірмена
print("Рангова кореляція Спірмена:")
for feature in features_to_analyze:
    corr, p = spearmanr(data['Energy Consumption (kWh)'], data[feature])
    print(f"{feature}: {corr:.3f} (p-value: {p:.4f})")

# Добова сезонність
data['Hour'] = data.index.hour
hourly_pattern = data.groupby('Hour')['Energy Consumption (kWh)'].mean()

plt.figure(figsize=(12,6))
hourly_pattern.plot()
plt.title('Добова сезонність споживання енергії')
plt.ylabel('Середнє споживання (kWh)')
plt.grid(True)
plt.show()

# Тижнева сезонність
data['DayOfWeek'] = data.index.dayofweek
weekday_pattern = data.groupby('DayOfWeek')['Energy Consumption (kWh)'].mean()

plt.figure(figsize=(12,6))
weekday_pattern.plot(kind='bar')
plt.title('Тижнева сезонність споживання енергії')
plt.xticks(ticks=range(7), labels=['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Нд'])
plt.ylabel('Середнє споживання (kWh)')
plt.show()

# Автокореляційний аналіз
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(12,6))
plot_acf(data['Energy Consumption (kWh)'], lags=48, alpha=0.05)
plt.title('Автокореляція (добовий цикл)')
plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Підготовка даних для кластеризації
cluster_features = ['Energy Consumption (kWh)', 'HVAC Consumption (kWh)',
                   'Lighting Consumption (kWh)', 'Occupancy Rate (%)',
                   'Temperature (°C)', 'Building Size (m²)']

# Нормалізація даних
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[cluster_features])

# Визначення оптимальної кількості кластерів (метод ліктя)
inertia = []
for k in range(1, 6):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 6), inertia, marker='o')
plt.title('Метод ліктя для визначення кількості кластерів')
plt.xlabel('Кількість кластерів')
plt.ylabel('Inertia')
plt.show()

# Кластеризація (припустимо, що оптимально k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Аналіз кластерів
cluster_analysis = data.groupby(['Building Type', 'Cluster'])[cluster_features].mean()
print(cluster_analysis)

# Візуалізація кластерів
sns.pairplot(data, vars=cluster_features[:4], hue='Cluster', palette='viridis')
plt.suptitle('Розподіл кластерів у просторі ознак', y=1.02)
plt.show()

# Аналіз за типами будівель
for btype in data['Building Type'].unique():
    subset = data[data['Building Type'] == btype]
    print(f"\nАналіз для типу будівлі: {btype}")
    print(subset.groupby('Cluster')['Energy Consumption (kWh)'].describe())
    
    # Графік розподілу споживання по кластерах
    sns.boxplot(x='Cluster', y='Energy Consumption (kWh)', data=subset)
    plt.title(f'Розподіл споживання для {btype}')
    plt.show()