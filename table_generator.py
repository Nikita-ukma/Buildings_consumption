import pandas as pd

# === 1. Завантаження даних ===
print("🔹 Завантаження даних...")
df = pd.read_csv("data/power-laws-forecasting-energy-consumption-test-data.csv", sep=';', parse_dates=["Timestamp"])

# === 2. Перетворення Timestamp у формат YYYY-MM ===
df["Month"] = df["Timestamp"].dt.to_period("M")  # Формат YYYY-MM

# === 3. Групування за SiteId та місяцем ===
monthly_data = df.groupby(["SiteId", "Month"]).agg(
    number_of_fixations=("Value", "count"),  # Кількість фіксацій за місяць
    total_value=("Value", "sum")            # Сумарні витрати за місяць
).reset_index()

# Обчислюємо середнє значення витрат за місяць
monthly_data["value"] = monthly_data["total_value"] / monthly_data["number_of_fixations"]

# Видаляємо тимчасовий стовпець total_value
monthly_data.drop(columns=["total_value"], inplace=True)

# Перейменуємо стовпці для зручності
monthly_data.rename(columns={"SiteId": "obs_id", "Month": "Timestamp"}, inplace=True)

# Додаємо стовпець SiteId
monthly_data["SiteId"] = monthly_data["obs_id"]

# Перетворюємо Timestamp назад у рядок (для зручності зберігання)
monthly_data["Timestamp"] = monthly_data["Timestamp"].astype(str)

# === 4. Збереження у новий CSV-файл ===
output_file = "data/monthly_consumption_test.csv"
monthly_data.to_csv(output_file, index=False)
print(f"✅ Нові дані збережено у файл: {output_file}")

# Виведемо перші 5 рядків для перевірки
print(monthly_data.head())