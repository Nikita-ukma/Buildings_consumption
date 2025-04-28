import pandas as pd
import numpy as np
import pandas as pd

df = pd.read_csv('data/FINAL_dataset.csv')  # Замініть на шлях до вашого файлу

# Переконуємося, що дати у правильному форматі
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Створюємо колонку для сезону на основі місяця
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Додаємо сезон до датафрейму
df['Season'] = df['Datetime'].apply(get_season)

# Обчислюємо середнє значення COMED_MW за сезонами
seasonal_means = df.groupby('Season')['COMED_MW'].mean()

print("Середні значення COMED_MW за сезонами:")
print(seasonal_means)

# Також можна обчислити медіану та інші статистики, якщо потрібно
seasonal_stats = df.groupby('Season')['COMED_MW'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
print("\nДодаткові статистики за сезонами:")
print(seasonal_stats)

# Зберігаємо розширені статистики (опціонально)
seasonal_stats.to_csv('seasonal_stats_COMED_MW.csv')

# Якщо хочете візуалізувати результати
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
seasonal_means.plot(kind='bar', color='skyblue')
plt.title('Середнє значення COMED_MW за сезонами')
plt.ylabel('COMED_MW (середнє)')
plt.xlabel('Сезон')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('seasonal_means_plot.png')  # Зберегти графік (опціонально)
plt.show()  # Показати графік (опціонально)