import pandas as pd
import numpy as np

# Фіксуємо сид для відтворюваності
np.random.seed(42)

# Створюємо логічні закономірності
students = 60
study_time = np.linspace(1, 10, students)  # Час підготовки від 1 до 10 годин
sleep_hours = np.linspace(8, 3.5, students)  # Більше навчання -> менше сну
previous_scores = np.linspace(40, 90, students).astype(int)  # Попередні оцінки зростають

# Функція для логічного розрахунку оцінки:
# Вплив навчання (+2 за годину)
# Вплив сну (оптимально 6-7 год, менше 4 - мінус бали)
# Вплив попереднього балу (високий бал - вища оцінка)
test_scores = (previous_scores + (study_time * 2) - (np.abs(sleep_hours - 6) * 1.5) 
               + np.random.randint(-3, 4, students)).astype(int)

# Створюємо датафрейм
df = pd.DataFrame({
    "Study Time (hours)": study_time.round(1),
    "Sleep Hours (hours)": sleep_hours.round(1),
    "Previous Test Score": previous_scores,
    "Test Score (points)": test_scores
})

# Зберігаємо як CSV
df.to_csv("student_scores_60.csv", index=False)

print("CSV file 'student_scores_60.csv' has been created successfully!")