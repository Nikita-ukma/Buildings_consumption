import pandas as pd

# === 1. Завантаження даних ===
print("🔹 Завантаження даних...")
weather = pd.read_csv("data/power-laws-forecasting-energy-consumption-weather.csv", sep=';', parse_dates=["Timestamp"])

# === 2. Перетворення Timestamp у формат YYYY-MM ===
weather["Month"] = weather["Timestamp"].dt.to_period("M")  # Формат YYYY-MM

# === 3. Групування за SiteId та місяцем ===
weather_grouped = weather.groupby(["SiteId", "Month"]).agg(
    average_temperature=("Temperature", "mean")  # Середня температура за місяць
).reset_index()

# Перетворюємо Month назад у рядок (для зручності зберігання)
weather_grouped["Month"] = weather_grouped["Month"].astype(str)

# === 4. Збереження у новий CSV-файл ===
output_file = "data/monthly_weather.csv"
weather_grouped.to_csv(output_file, index=False)
print(f"✅ Нові дані збережено у файл: {output_file}")

# Виведемо перші 5 рядків для перевірки
print(weather_grouped.head())