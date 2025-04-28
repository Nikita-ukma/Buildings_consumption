import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Покращена підготовка даних з нормалізацією
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Обробка пропущених значень
    df = df.resample('h').mean().interpolate(method='time')
    
    # Додавання основних часових ознак
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    
    # Створення циклічних ознак для годин (щоб урахувати циклічність часу)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Створення ознаки вихідного дня
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # Температурні змінні (якщо є в датасеті)
    if 'Chicago_temp' in df.columns:
        # Нормалізація температури (переведення в Цельсії)
        df['temp_c'] = (df['Chicago_temp'] - 273.15)
        
        # Квадрат температури для врахування нелінійності
        df['temp_c_squared'] = df['temp_c'] ** 2
        
        # Різні температурні режими (холодно/помірно/жарко)
        df['temp_cold'] = np.where(df['temp_c'] < 5, 1, 0)
        df['temp_hot'] = np.where(df['temp_c'] > 25, 1, 0)
        
        # Взаємодія температури з часом доби
        df['temp_morning'] = df['temp_c'] * ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['temp_afternoon'] = df['temp_c'] * ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['temp_evening'] = df['temp_c'] * ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
        df['temp_night'] = df['temp_c'] * ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        
        # Взаємодія температури з вихідними
        df['temp_weekend'] = df['temp_c'] * df['is_weekend']
    
    # Сезонні ознаки
    seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 
               6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 
               11: 'Fall', 12: 'Winter'}
    df['season'] = df['month'].map(seasons)
    
    # Створення dummy-змінних для сезонів
    df['is_summer'] = (df['season'] == 'Summer').astype(int)
    df['is_winter'] = (df['season'] == 'Winter').astype(int)
    
    # Лагові змінні (попередні значення споживання)
    df['lag_1day'] = df['COMED_MW'].shift(24)
    df['lag_1week'] = df['COMED_MW'].shift(24*7)
    
    # Середні значення за попередні періоди
    df['mean_last_24h'] = df['COMED_MW'].rolling(window=24).mean().shift(1)
    
    # Додавання взаємодії між лагами та часовими змінними
    df['lag_1day_weekend'] = df['lag_1day'] * df['is_weekend']
    
    # Заповнення пропущених значень (лаги створюють NaN значення на початку)
    df = df.fillna(method='bfill')
    
    return df.loc['2015-01-01':]

# 2. Оцінка моделі з детальним аналізом сезонності
def evaluate_model(y_true, y_pred, target_scaler=None):
    # Якщо дані були нормалізовані, повертаємо їх у вихідний масштаб для оцінки
    if target_scaler is not None:
        y_true_original = pd.Series(target_scaler.inverse_transform(y_true.values.reshape(-1, 1)).flatten(), index=y_true.index)
        y_pred_original = target_scaler.inverse_transform(y_pred.values.reshape(-1, 1)).flatten()
    else:
        y_true_original = y_true
        y_pred_original = y_pred
    
    # Базові метрики
    rmse = mean_squared_error(y_true_original, y_pred_original, squared=False)
    mae = mean_absolute_error(y_true_original, y_pred_original)
    r2 = r2_score(y_true_original, y_pred_original)
    mape = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
    
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW") 
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    # Створення DataFrame для аналізу помилок
    errors_df = pd.DataFrame({
        'y_true': y_true_original,
        'y_pred': y_pred_original,
        'error': np.abs(y_true_original - y_pred_original),
        'rel_error': np.abs((y_true_original - y_pred_original) / y_true_original) * 100,
        'month': pd.DatetimeIndex(y_true.index).month,
        'hour': pd.DatetimeIndex(y_true.index).hour,
        'day_of_week': pd.DatetimeIndex(y_true.index).dayofweek
    }, index=y_true.index)
    
    # Визначення сезонів
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }
    
    errors_df['season'] = errors_df['month'].apply(
        lambda x: next(season for season, months in seasons.items() if x in months)
    )
    
    # Оцінка за сезонами
    print("\nСезонна оцінка помилок:")
    seasonal_metrics = {}
    for season in seasons.keys():
        season_data = errors_df[errors_df['season'] == season]
        season_rmse = mean_squared_error(season_data['y_true'], season_data['y_pred'], squared=False)
        season_mae = mean_absolute_error(season_data['y_true'], season_data['y_pred'])
        season_mape = np.mean(season_data['rel_error'])
        seasonal_metrics[season] = {
            'RMSE': season_rmse,
            'MAE': season_mae,
            'MAPE': season_mape
        }
        print(f"{season}:")
        print(f"  RMSE: {season_rmse:.2f} MW")
        print(f"  MAE: {season_mae:.2f} MW")
        print(f"  MAPE: {season_mape:.2f}%")
    
    # Аналіз за годинами доби
    hour_errors = errors_df.groupby('hour')['rel_error'].mean().sort_values(ascending=False)
    print("\nТоп-5 годин з найбільшими помилками:")
    for hour, error in hour_errors.head(5).items():
        print(f"  Година {hour}: {error:.2f}%")
    
    return seasonal_metrics, errors_df

# 3. Основна функція покращеної моделі SARIMAX з нормалізацією
def improved_sarimax_model(df):
    """
    Покращена модель SARIMAX з фіксованими параметрами та нормалізацією
    """
    # Фіксовані параметри моделі
    order = (2, 0, 2)
    seasonal_order = (1, 1, 1, 24)  # Добова сезонність
    
    # Розділення на тренувальний та тестовий набори
    train = df.loc[:'2016-12-31']
    test = df.loc['2017-01-01':]
    
    # Вибір найважливіших екзогенних змінних
    important_vars = [
        'hour_sin', 'hour_cos',      # Циклічне кодування годин
        'is_weekend', 'is_holiday',  # Спеціальні дні (якщо є в даних)
        'is_summer', 'is_winter',    # Сезонні ознаки
        'lag_1day', 'lag_1week',     # Лагові змінні
        'mean_last_24h'              # Ковзне середнє
    ]
    
    # Перевірка наявності змінної is_holiday
    if 'is_holiday' not in df.columns:
        important_vars.remove('is_holiday')
    
    # Додавання температурних змінних, якщо вони є
    if 'temp_c' in df.columns:
        temp_vars = [
            'temp_c', 'temp_c_squared',  # Температура і її квадрат
            'temp_morning', 'temp_afternoon', 'temp_evening', 'temp_night',  # Взаємодія з часом доби
            'temp_weekend',  # Взаємодія з вихідними
            'temp_cold', 'temp_hot'  # Температурні режими
        ]
        important_vars.extend(temp_vars)
    
    # Перевірка наявності всіх змінних у датасеті
    exog_vars = [var for var in important_vars if var in df.columns]
    
    print(f"Використовуємо {len(exog_vars)} екзогенних змінних:")
    for var in exog_vars:
        print(f"  - {var}")
    
    # Підготовка даних для моделі
    exog_train = train[exog_vars]
    exog_test = test[exog_vars]
    
    # Нормалізація екзогенних змінних
    exog_scaler = StandardScaler()
    exog_train_scaled = pd.DataFrame(exog_scaler.fit_transform(exog_train), 
                                   index=exog_train.index, 
                                   columns=exog_train.columns)
    exog_test_scaled = pd.DataFrame(exog_scaler.transform(exog_test), 
                                  index=exog_test.index, 
                                  columns=exog_test.columns)
    
    # НОРМАЛІЗАЦІЯ ЦІЛЬОВОЇ ЗМІННОЇ (новий код)
    print("\nЗастосування нормалізації до цільової змінної...")
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Підготовка даних для тренування з нормалізованою цільовою змінною
    train_target = pd.Series(
        target_scaler.fit_transform(train['COMED_MW'].values.reshape(-1, 1)).flatten(),
        index=train.index
    )
    
    test_target = pd.Series(
        target_scaler.transform(test['COMED_MW'].values.reshape(-1, 1)).flatten(),
        index=test.index
    )
    
    print(f"Діапазон оригінальних даних: [{train['COMED_MW'].min():.2f}, {train['COMED_MW'].max():.2f}]")
    print(f"Діапазон нормалізованих даних: [{train_target.min():.2f}, {train_target.max():.2f}]")
    
    # Навчання моделі на нормалізованих даних
    print(f"\nНавчання SARIMAX моделі з параметрами {order}{seasonal_order}...")
    model = SARIMAX(train_target, 
                  exog=exog_train_scaled,
                  order=order,
                  seasonal_order=seasonal_order,
                  enforce_stationarity=False,
                  enforce_invertibility=False)
    
    model_results = model.fit(disp=False)
    print("Модель навчена.")
    
    # Прогнозування на нормалізованих даних
    print("Виконання прогнозу...")
    forecast = model_results.get_forecast(steps=len(test), exog=exog_test_scaled)
    forecast_mean_scaled = pd.Series(forecast.predicted_mean.values, index=test.index)
    
    # Оцінка моделі з використанням нормалізованих даних
    print("\nДетальне оцінювання моделі (на нормалізованих даних):")
    seasonal_metrics_scaled, errors_df_scaled = evaluate_model(test_target, forecast_mean_scaled)
    
    # Повернення прогнозів до оригінального масштабу для візуалізації
    forecast_mean_original = pd.Series(
        target_scaler.inverse_transform(forecast_mean_scaled.values.reshape(-1, 1)).flatten(),
        index=test.index
    )
    
    # Оцінка моделі з використанням даних в оригінальному масштабі
    print("\nДетальне оцінювання моделі (на оригінальних даних):")
    seasonal_metrics, errors_df = evaluate_model(test['COMED_MW'], forecast_mean_original)
    
    # Візуалізація результатів (в оригінальному масштабі)
    # 1. Загальне порівняння прогнозу з реальними даними
    plt.figure(figsize=(14, 7))
    plt.plot(train.index[-30*24:], train['COMED_MW'][-30*24:], label='Тренувальні дані')
    plt.plot(test.index[:30*24], test['COMED_MW'][:30*24], label='Реальні значення')
    plt.plot(test.index[:30*24], forecast_mean_original[:30*24], label='Прогноз', alpha=0.7)
    plt.title(f'SARIMAX{order}{seasonal_order} з нормалізацією та оптимізованими змінними (січень 2017)')
    plt.xlabel('Дата')
    plt.ylabel('Споживання (МВт)')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 2. Детальний погляд на один тиждень
    one_week = test.loc['2017-01-01':'2017-01-07', 'COMED_MW']
    forecast_week = forecast_mean_original.loc['2017-01-01':'2017-01-07']
    
    plt.figure(figsize=(16, 6))
    plt.plot(one_week.index, one_week, label='Реальні значення')
    plt.plot(one_week.index, forecast_week, label='Прогноз', alpha=0.7)
    plt.title('Детальний погляд на перший тиждень 2017 року')
    plt.xlabel('Дата')
    plt.ylabel('Споживання (МВт)')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 3. Порівняння оригінальних та нормалізованих даних
    plt.figure(figsize=(14, 6))
    
    # Підготовка даних для візуалізації (вибірка останніх 7 днів тренувальних даних)
    sample_dates = df.index[-7*24:]
    original_values = df.loc[sample_dates, 'COMED_MW']
    
    # Нормалізувати ці дані для порівняння
    normalized_values = pd.Series(
        target_scaler.transform(original_values.values.reshape(-1, 1)).flatten(),
        index=original_values.index
    )
    
    plt.plot(sample_dates, original_values, label='Оригінальні дані')
    plt.plot(sample_dates, normalized_values, label='Нормалізовані дані', alpha=0.7)
    plt.title('Порівняння оригінальних та нормалізованих даних (останній тиждень)')
    plt.xlabel('Дата')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 4. Порівняння помилок за сезонами
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='season', y='rel_error', data=errors_df, order=['Winter', 'Spring', 'Summer', 'Fall'])
    plt.title('Відносні помилки за сезонами')
    plt.xlabel('Сезон')
    plt.ylabel('Відносна помилка (%)')
    plt.grid(axis='y')
    plt.show()
    
    # 5. Середні помилки за годинами доби
    hourly_errors = errors_df.groupby('hour')['rel_error'].mean()
    
    plt.figure(figsize=(14, 6))
    hourly_errors.plot(kind='bar')
    plt.title('Середня відносна помилка за годинами доби')
    plt.xlabel('Година')
    plt.ylabel('Середня відносна помилка (%)')
    plt.grid(axis='y')
    plt.show()
    
    # 6. Графік залишків
    plt.figure(figsize=(14, 6))
    plt.scatter(test.index, test['COMED_MW'] - forecast_mean_original, alpha=0.5, s=2)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Залишки моделі')
    plt.xlabel('Дата')
    plt.ylabel('Залишок (МВт)')
    plt.grid()
    plt.show()
    
    # 7. Порівняння фактичних і прогнозованих значень
    plt.figure(figsize=(12, 8))
    plt.scatter(test['COMED_MW'], forecast_mean_original, alpha=0.5)
    plt.plot([test['COMED_MW'].min(), test['COMED_MW'].max()], 
             [test['COMED_MW'].min(), test['COMED_MW'].max()], 'r--')
    plt.title('Порівняння фактичних і прогнозованих значень')
    plt.xlabel('Фактичне споживання (МВт)')
    plt.ylabel('Прогнозоване споживання (МВт)')
    plt.grid()
    plt.show()
    
    # Повернення результатів
    return model_results, forecast_mean_original, exog_vars, target_scaler

# 4. Основна функція
def main():
    print("Завантаження та підготовка даних...")
    df = load_and_prepare_data("SARIMAX_dataset.csv")
    
    print("Запуск покращеної моделі SARIMAX з нормалізацією даних...")
    model_results, forecast, exog_vars, target_scaler = improved_sarimax_model(df)
    
    print("\nГотово!")

if __name__ == "__main__":
    main()