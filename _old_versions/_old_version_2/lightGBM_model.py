import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Завантаження даних
final_df = pd.read_csv('data/final_dataset.csv', sep=';')

# Обробка дати
final_df['Month'] = pd.to_datetime(final_df['Month'])
final_df['year'] = final_df['Month'].dt.year
final_df['month'] = final_df['Month'].dt.month
final_df['days_in_month'] = final_df['Month'].dt.days_in_month

# Додаткові ознаки
final_df['temp_diff'] = final_df['AvgTemperature'] - final_df['BaseTemperature']
final_df['temp_ratio'] = final_df['AvgTemperature'] / (final_df['BaseTemperature'] + 1e-6)
final_df['month_sin'] = np.sin(2 * np.pi * final_df['month']/12)
final_df['month_cos'] = np.cos(2 * np.pi * final_df['month']/12)
final_df['value_per_surface'] = final_df['Value'] / (final_df['Surface'] + 1e-6)

# Вибір ознак
features = [
    'SiteId', 'month', 'days_in_month', 'AvgTemperature', 
    'Surface', 'BaseTemperature', 'temp_diff', 'temp_ratio',
    'month_sin', 'month_cos', 'value_per_surface'
]

# Видалення викидів (збереження оригінальних індексів)
Q1 = final_df['Value'].quantile(0.05)
Q3 = final_df['Value'].quantile(0.95)
filtered_df = final_df[(final_df['Value'] >= Q1) & (final_df['Value'] <= Q3)].copy()

# Логарифмічне перетворення цільової змінної
y = np.log1p(filtered_df['Value'])
X = filtered_df[features]

# Скидання індексу для уникнення проблем
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Розділення даних
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, groups=X['SiteId']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Навчання моделі
model = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=31,
    learning_rate=0.01,
    n_estimators=1000,
    random_state=42
)

model.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='mape',
    categorical_feature=['SiteId', 'month'],
    early_stopping_rounds=50,
    verbose=10
)

# Прогнозування
y_pred = np.expm1(model.predict(X_test))
y_true = np.expm1(y_test)

# Розрахунок помилок
errors = pd.DataFrame({
    'SequenceID': X_test['SequenceID'],
    'LastMonth': X_test['Month'].dt.strftime('%Y-%m'),
    'TrueValue': y_true,
    'PredValue': y_pred,
    'Error%': np.abs(y_true - y_pred) / y_true * 100
})

# Аналіз помилок
print("\nСередня помилка: {:.1f}%".format(errors['Error%'].mean()))
print("Мінімальна помилка: {:.1f}%".format(errors['Error%'].min()))
print("Максимальна помилка: {:.1f}%".format(errors['Error%'].max()))
print("Медіанна помилка: {:.1f}%".format(errors['Error%'].median()))

# Важливість ознак
plt.figure(figsize=(12, 8))
lgb.plot_importance(model, max_num_features=15, importance_type='gain')
plt.title('Feature Importance (Gain)')
plt.show()