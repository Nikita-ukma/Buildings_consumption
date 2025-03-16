import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

# === 1. Завантаження даних ===
print("🔹 Завантаження даних...")
df = pd.read_csv("data/power-laws-forecasting-energy-consumption-training-data.csv", sep=';', parse_dates=["Timestamp"])
metadata = pd.read_csv("data/power-laws-forecasting-energy-consumption-metadata.csv", sep=';')
weather = pd.read_csv("data/power-laws-forecasting-energy-consumption-weather.csv", sep=';', parse_dates=["Timestamp"])
holidays = pd.read_csv("data/power-laws-forecasting-energy-consumption-holidays.csv", sep=';', parse_dates=["Date"])

# === 2. Часові фічі ===
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek  # (0 - понеділок, 6 - неділя)
df["Month"] = df["Timestamp"].dt.month
df["Year"] = df["Timestamp"].dt.year
df["Date"] = df["Timestamp"].dt.date  # Для об'єднання зі святами

# Додаткові фічі
df["IsWeekend"] = df["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)  # Вихідний день

# === 3. Додавання мета-даних про будівлі ===
df = df.merge(metadata, on="SiteId", how="left")

# === 4. Об'єднання з погодними даними ===
weather = weather.sort_values(["SiteId", "Timestamp", "Distance"]).drop_duplicates(["SiteId", "Timestamp"])
df = df.merge(weather, on=["SiteId", "Timestamp"], how="left")

# === 5. Додавання інформації про свята ===
holidays["HolidayFlag"] = 1

# Перетворення типів даних для об'єднання
df["Date"] = pd.to_datetime(df["Date"])
holidays["Date"] = pd.to_datetime(holidays["Date"])

# Додавання свята
df = df.merge(holidays[["Date", "SiteId", "HolidayFlag"]], on=["Date", "SiteId"], how="left")

# Заповнення NaN значень (де немає свята → ставимо 0)
df["HolidayFlag"] = df["HolidayFlag"].fillna(0)

# === 6. Фільтрація за 2017 рік (або взяти частину вибірки) ===
df = df[df["Year"] == 2017]  # Можна змінити на df.sample(frac=0.5, random_state=42)

# === 7. Видалення непотрібних стовпців ===
df.drop(columns=["Timestamp", "Date", "Year"], inplace=True)

# === 8. Заповнення пропущених значень ===
df.fillna(df.median(), inplace=True)

# === 9. Розділення на навчальну та тестову вибірки ===
X = df.drop(columns=["Value"])  # Фічі
y = df["Value"]  # Цільова змінна (споживання)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 10. Тюнінг гіперпараметрів XGBoost ===
print("🚀 Навчання моделі XGBoost...")
model = xgb.XGBRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
print("Найкращі параметри:", grid_search.best_params_)

# Використання найкращих параметрів
best_model = grid_search.best_estimator_

# === 11. Оцінка моделі ===
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📉 Mean Absolute Error (MAE): {mae:.2f}")

mean_value = df["Value"].mean()
print(f"Середнє значення енерговитрат: {mean_value:.2f}")

relative_error = (mae / mean_value) * 100
print(f"Відносна похибка: {relative_error:.2f}%")

# === 12. Візуалізація важливості фіч ===
plt.figure(figsize=(10, 5))
sns.barplot(x=best_model.feature_importances_, y=X.columns)
plt.xlabel("Важливість фіч")
plt.ylabel("Фічі")
plt.title("Важливість фіч для XGBoost")
plt.show()