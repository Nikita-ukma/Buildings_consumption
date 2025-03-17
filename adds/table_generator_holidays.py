import pandas as pd

# === 1. Завантаження даних ===
print("🔹 Завантаження даних...")
holidays = pd.read_csv("data/power-laws-forecasting-energy-consumption-holidays.csv", sep=';', parse_dates=["Date"])

# === 2. Перетворення Date у формат YYYY-MM ===
holidays["Month"] = holidays["Date"].dt.to_period("M")  # Формат YYYY-MM

# === 3. Групування за SiteId та місяцем ===
holidays_grouped = holidays.groupby(["SiteId", "Month"]).agg(
    number_of_holidays=("Holiday", "count")  # Кількість свят за місяць
).reset_index()

# Перетворюємо Month назад у рядок (для зручності зберігання)
holidays_grouped["Month"] = holidays_grouped["Month"].astype(str)

# === 4. Збереження у новий CSV-файл ===
output_file = "data/monthly_holidays.csv"
holidays_grouped.to_csv(output_file, index=False)
print(f"✅ Нові дані збережено у файл: {output_file}")

# Виведемо перші 5 рядків для перевірки
print(holidays_grouped.head())