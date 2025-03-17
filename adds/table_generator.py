import pandas as pd

# === 1. Завантаження даних ===
print("🔹 Завантаження даних...")
df = pd.read_csv("data\monthly_consumption.csv")

# === 4. Видалення SiteId з менше ніж 5 місяцями ===
# Підраховуємо кількість місяців для кожного SiteId
month_count = df.groupby("SiteId")["Month"].nunique()
# Фільтруємо ті SiteId, які мають менше 5 місяців
valid_site_ids = month_count[month_count >= 5].index
# Залишаємо тільки записи з valid_site_ids
monthly_data = df[df["SiteId"].isin(valid_site_ids)]

# === 5. Збереження у новий CSV-файл ===
output_file = "data/monthly_consumption.csv"
monthly_data.to_csv(output_file, index=False)
print(f"✅ Нові дані збережено у файл: {output_file}")

# Виведемо перші 5 рядків для перевірки
print(monthly_data.head())