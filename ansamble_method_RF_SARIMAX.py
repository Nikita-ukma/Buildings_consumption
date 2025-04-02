import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. Завантаження та підготовка даних
def load_data():
    df = pd.read_csv("residential_energy_normalized.csv", parse_dates=['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='all')
    return df

# 2. Генерація часових ознак
def create_time_features(df):
    df = df.copy()
    df['hour_of_week'] = (df.index.dayofweek * 24) + df.index.hour//12
    df['time_idx'] = (df.index - df.index[0]).days * 2 + (df.index.hour//12)
    return df

# 3. Ресемплінг та підготовка наборів даних
df = load_data()
df_12h = df.resample('12h').mean().dropna()
df_12h = create_time_features(df_12h)

target = "Energy Consumption (kWh) Normalized"
features = [col for col in df_12h.columns if col not in [target, 'hour_of_week', 'time_idx']]

# Розділення на тренувальний/тестовий набори
test_size = 60  # 30 днів (60 періодів)
train = df_12h.iloc[:-test_size]
test = df_12h.iloc[-test_size:]

# 4. Навчання SARIMAX
sarimax_model = SARIMAX(
    endog=train[target],
    exog=train[features],
    order=(1,1,1),
    seasonal_order=(1,1,1,14)
)
sarimax_fit = sarimax_model.fit(disp=False)

# 5. Навчання Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train[['time_idx', 'hour_of_week'] + features], train[target])

# 6. Генерація майбутніх ознак
def generate_future_features(last_index, steps):
    future_dates = pd.date_range(
        start=last_index + pd.Timedelta(hours=12),
        periods=steps,
        freq='12h'
    )
    future_df = pd.DataFrame(index=future_dates)
    future_df['hour_of_week'] = (future_df.index.dayofweek * 24) + future_df.index.hour//12
    future_df['time_idx'] = (future_df.index - df_12h.index[0]).days * 2 + (future_df.index.hour//12)
    return future_df

future_exog = generate_future_features(train.index[-1], test_size)
future_exog[features] = test[features]  # Використовуємо реальні значення ознак

# 7. Прогнозування
sarimax_fc = sarimax_fit.get_forecast(steps=test_size, exog=future_exog[features]).predicted_mean
rf_fc = rf.predict(future_exog[['time_idx', 'hour_of_week'] + features])
final_fc = 0.7 * sarimax_fc + 0.3 * rf_fc

# 8. Агрегація до місячного рівня
def aggregate_to_monthly(series):
    return series.resample('ME').sum()

# Прогнозовані місячні значення
forecast_monthly = aggregate_to_monthly(pd.Series(final_fc, index=future_exog.index))

# Реальні місячні значення
actual_monthly = aggregate_to_monthly(df[target].resample('12H').mean().dropna())

# Порівняння з останнім місяцем
last_month = forecast_monthly.index[-1]
monthly_comparison = pd.DataFrame({
    'Actual': actual_monthly.loc[last_month],
    'Forecast': forecast_monthly.loc[last_month]
})

# 9. Візуалізація
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
actual_monthly.plot(label='Фактичні місячні дані', marker='o')
forecast_monthly.plot(label='Прогнозовані', marker='x', linestyle='--')
plt.title('Порівняння місячного прогнозу з реальними даними')
plt.ylabel('Енергоспоживання (кВт*год)')
plt.grid(True)
plt.legend()
plt.show()

# 10. Оцінка якості
common_index = actual_monthly.index.intersection(forecast_monthly.index)
mae_monthly = mean_absolute_error(actual_monthly.loc[common_index], forecast_monthly.loc[common_index])
rmse_monthly = np.sqrt(mean_squared_error(actual_monthly.loc[common_index], forecast_monthly.loc[common_index]))

print(f"MAE на місячному рівні: {mae_monthly:.2f}")
print(f"RMSE на місячному рівні: {rmse_monthly:.2f}")
print("\nПорівняння для останнього місяця:")
print(monthly_comparison)