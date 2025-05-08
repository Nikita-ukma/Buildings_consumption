import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings

# Налаштування відображення помилок
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None  # Вимкнення SettingWithCopyWarning

def clean_data(df):
    """Функція для очищення даних від inf і NaN"""
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Заповнення пропущених значень для числових колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

def load_and_preprocess_data(train_test='train'):
    """Завантаження та підготовка даних"""
    # Завантаження даних
    energy_data = pd.read_csv(f'data/monthly_consumption{"_test" if train_test=="test" else ""}.csv', sep=',')
    building_data = pd.read_csv('data/power-laws-forecasting-energy-consumption-metadata.csv', sep=';')
    temperature_data = pd.read_csv('data/monthly_weather.csv', sep=',')
    holidays_data = pd.read_csv('data/monthly_holidays.csv', sep=',')
    
    # Об'єднання даних
    data = energy_data.merge(building_data, on='SiteId', how='left')
    data = data.merge(temperature_data, on=['SiteId', 'Month'], how='left')
    data = data.merge(holidays_data, on=['SiteId', 'Month'], how='left')
    
    # Конвертація дати та встановлення індексу
    data['Timestamp'] = pd.to_datetime(data['Month'])
    data.set_index(['SiteId', 'Timestamp'], inplace=True)
    data = data.sort_index()
    
    # Очищення даних
    data = clean_data(data)
    
    # Додавання лагів та ковзаючого середнього
    for site_id in data.index.get_level_values('SiteId').unique():
        site_mask = data.index.get_level_values('SiteId') == site_id
        data.loc[site_mask, 'value_lag1'] = data.loc[site_mask, 'value'].shift(1)
        data.loc[site_mask, 'value_lag2'] = data.loc[site_mask, 'value'].shift(2)
        data.loc[site_mask, 'value_lag3'] = data.loc[site_mask, 'value'].shift(3)
        data.loc[site_mask, 'value_rolling_mean'] = data.loc[site_mask, 'value'].rolling(3, min_periods=1).mean()
    
    # Повторне очищення після створення нових змінних
    data = clean_data(data)
    
    return data

def train_and_predict(train_site, test_site):
    """Навчання моделі та прогнозування для одного об'єкта"""
    try:
        # Визначення екзогенних змінних
        exog_features = ['average_temperature', 'number_of_holidays', 
                        'Surface', 'Sampling', 'BaseTemperature',
                        'value_lag1', 'value_lag2', 'value_lag3',
                        'value_rolling_mean']
        
        # Вибір лише доступних та коректних змінних
        available_features = [f for f in exog_features if f in train_site.columns]
        
        # Перевірка на наявність достатньої кількості даних
        if len(train_site) < 12 or len(test_site) == 0:
            return None
        
        # Навчання моделі
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(train_site['value'],
                          exog=train_site[available_features] if available_features else None,
                          order=(1, 1, 1),
                          seasonal_order=(1, 1, 1, 12),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            
            model_fit = model.fit(disp=False)
        
        # Прогнозування
        if available_features:
            # Додаткова перевірка на наявність inf/nan у тестових даних
            if not np.isfinite(test_site[available_features].values).all():
                return None
                
            forecast = model_fit.get_forecast(steps=len(test_site), 
                                            exog=test_site[available_features])
        else:
            forecast = model_fit.get_forecast(steps=len(test_site))
        
        # Зберігання результатів
        test_site['prediction'] = forecast.predicted_mean.values
        return test_site[['value', 'prediction']]
    
    except Exception as e:
        return None

# Основна частина програми
data_train = load_and_preprocess_data('train')
data_test = load_and_preprocess_data('test')

results = []
for site_id in data_train.index.get_level_values('SiteId').unique():
    try:
        train_site = data_train.xs(site_id, level='SiteId').copy()
        test_site = data_test.xs(site_id, level='SiteId').copy()
        
        result = train_and_predict(train_site, test_site)
        if result is not None:
            results.append(result)
    
    except Exception as e:
        continue

# Обробка результатів
if results:
    full_results = pd.concat(results)
    
    # Фільтрація нескінченних значень у результатах (на випадок, якщо щось пропустили)
    full_results = full_results.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(full_results) > 0:
        mae = mean_absolute_error(full_results['value'], full_results['prediction'])
        mape = mean_absolute_percentage_error(full_results['value'], full_results['prediction'])
        
        print(f"\n📊 Результати для {len(results)} об'єктів:")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape*100:.2f}%")
        
        # Візуалізація
        plt.figure(figsize=(12, 6))
        for site_id in full_results.index.get_level_values('SiteId').unique()[:10]:  # Обмежуємо кількість для читабельності
            site_data = full_results.xs(site_id, level='SiteId')
            plt.plot(site_data.index, site_data['value'], label=f'Site {site_id} - Actual')
            plt.plot(site_data.index, site_data['prediction'], '--', label=f'Site {site_id} - Predicted')
        
        plt.title("Порівняння прогнозу з фактичними данами (перші 10 об'єктів)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    else:
        print("Немає коректних даних для оцінки")
else:
    print("Не вдалося отримати результати для жодного об'єкта")