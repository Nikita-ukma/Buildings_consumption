import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose

# === 1. Завантаження даних ===
print("🔹 Завантаження даних...")
df = pd.read_csv("data/monthly_consumption.csv", sep=',', parse_dates=["Month"])

# === 2. Перетворення Timestamp у формат YYYY-MM ===
df["Month"] = df["Month"].dt.to_period("M")  # Формат YYYY-MM

# === 3. Групування за місяцем та обчислення середнього значення ===
monthly_avg = df.groupby("Month")["value"].mean().reset_index()
monthly_avg.rename(columns={"value": "avg_value"}, inplace=True)

# Перетворюємо Month назад у рядок (для зручності)
monthly_avg["Month"] = monthly_avg["Month"].astype(str)

# === 4. Побудова тренду ===
# Лінійна регресія для середніх значень
X = np.arange(len(monthly_avg)).reshape(-1, 1)
y = monthly_avg["avg_value"].values
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

# Візуалізація тренду
plt.figure(figsize=(10, 6))
plt.plot(monthly_avg["Month"], y, label="Середнє енергоспоживання")
plt.plot(monthly_avg["Month"], trend, label="Тренд", color="red")
plt.title("Загальний тренд енергоспоживання (середнє значення за місяць)")
plt.xlabel("Місяць")
plt.ylabel("Енергоспоживання")
plt.xticks(rotation=45)  # Поворот підписів на осі X для кращої читабельності
plt.legend()
plt.show()

# === 5. Декомпозиція часового ряду ===
# Перетворюємо дані у часовий ряд (для statsmodels)
monthly_avg.set_index("Month", inplace=True)
monthly_avg.index = pd.to_datetime(monthly_avg.index)

# Виконуємо декомпозицію (адитивна модель)
decomposition = seasonal_decompose(monthly_avg["avg_value"], model='additive', period=12)  # period=12 для місячних даних

# Візуалізація декомпозиції
plt.figure(figsize=(12, 8))

# Оригінальний ряд
plt.subplot(411)
plt.plot(decomposition.observed, label="Оригінальний ряд")
plt.legend()

# Тренд
plt.subplot(412)
plt.plot(decomposition.trend, label="Тренд")
plt.legend()

# Сезонність
plt.subplot(413)
plt.plot(decomposition.seasonal, label="Сезонність")
plt.legend()

# Залишки (шум)
plt.subplot(414)
plt.plot(decomposition.resid, label="Залишки (шум)")
plt.legend()

plt.tight_layout()
plt.show()

# === 6. Аналіз шуму ===
# Статистичний аналіз залишків
residuals = decomposition.resid.dropna()  # Видаляємо NaN
print("\nСтатистичний аналіз залишків (шуму):")
print(f"Середнє значення: {residuals.mean()}")
print(f"Дисперсія: {residuals.var()}")
print(f"Стандартне відхилення: {residuals.std()}")

# Візуалізація розподілу залишків
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.title("Розподіл залишків (шуму)")
plt.xlabel("Залишки")
plt.ylabel("Частота")
plt.show()

from scipy.signal import periodogram

# Спектральний аналіз
frequencies, spectrum = periodogram(monthly_avg["avg_value"])
plt.figure(figsize=(10, 6))
plt.plot(frequencies, spectrum)
plt.title("Спектральна щільність")
plt.xlabel("Частота")
plt.ylabel("Інтенсивність")
plt.show()