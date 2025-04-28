'''
RMSE: 710.49 MW
MAE: 500.06 MW
R²: 0.8859
Winter RMSE: 736.39 MW
Spring RMSE: 504.24 MW
Summer RMSE: 914.86 MW
Fall RMSE: 620.66 MW
'''
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# 1. Завантаження та попередня обробка даних
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Додавання ознаки кварталу року
    df['quarter'] = df.index.quarter
    
    # Додавання місяця як категоріальної ознаки
    df['month'] = df.index.month
    
    # Додавання ознаки літнього періоду (червень-серпень)
    df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
    
    # Додавання ознаки дня року для вловлення сезонності
    df['day_of_year'] = df.index.dayofyear
    
    # Додавання ознаки години дня як синусоїдальної функції для врахування циклічності
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Розрахунок cooling degree days (CDD)
    # Базова температура, при якій зазвичай вмикають кондиціонери
    base_temp = 21  # Це значення може потребувати калібрування
    df['cooling_degree'] = df['Chicago_temp'].apply(lambda x: max(0, x - base_temp))
    
    # Квадрат cooling degree для врахування нелінійного зростання споживання при високих температурах
    df['cooling_degree_squared'] = df['cooling_degree'] ** 2
    
    # Додавання взаємодії між humidity і temperature (важливо для відчуття "душності")
    df['temp_humidity_interaction'] = df['Chicago_temp'] * df['Chicago_humidity'] / 100
    
    # Додавання бінарної ознаки для екстремальних температур (>30°C)
    df['extreme_heat'] = (df['Chicago_temp'] > 30).astype(int)
    
    # Додавання ознаки для пікових годин використання кондиціонерів (12-18 годин)
    df['ac_peak_hours'] = ((df.index.hour >= 12) & (df.index.hour <= 18)).astype(int)
    
    # Взаємодія між літнім періодом і температурою
    df['summer_temp_interaction'] = df['is_summer'] * df['Chicago_temp']
    
    return df

# 2. Розрахунок лагів БЕЗ витоку даних
def calculate_features(df):
    # Створюємо копію для безпечного додавання ознак
    features_df = df.copy()

    # Лаг 720 годин (30 днів)
    features_df['lag_720h'] = features_df['COMED_MW'].shift(720)
    
    # Лаг за попередній рік (8760 годин = 365 днів)
    features_df['lag_1y'] = features_df['COMED_MW'].shift(8760)
    
    # Лаг за позаминулий рік (17520 годин = 730 днів)
    features_df['lag_2y'] = features_df['COMED_MW'].shift(17520)
    
    # Додаємо лаги для cooling_degree
    features_df['cooling_degree_lag_24h'] = features_df['cooling_degree'].shift(24)
    features_df['cooling_degree_lag_168h'] = features_df['cooling_degree'].shift(168)
    
    # Додаємо ковзні середні для температури за останні дні
    features_df['temp_rolling_24h'] = features_df['Chicago_temp'].rolling(window=24).mean().shift(1)
    features_df['temp_rolling_72h'] = features_df['Chicago_temp'].rolling(window=72).mean().shift(1)
    
    # Додаємо ковзне максимальне значення температури
    features_df['temp_rolling_max_24h'] = features_df['Chicago_temp'].rolling(window=24).max().shift(1)
    
    # Додаємо ковзну різницю температур (мінлива погода)
    features_df['temp_volatility_24h'] = features_df['Chicago_temp'].rolling(window=24).std().shift(1)
    
    return features_df

# 3. Розділення даних на тренувальний (до 2016р.) та тестовий (2017р.) набори
def split_data(df, test_start='2017-01-01', test_end='2017-12-31'):
    train = df.loc[:test_start].iloc[:-1]  # до початку 2017 року, не враховуючи сам тестовий день
    test = df.loc[test_start:test_end]     # весь 2017 рік
    return train, test

# 4. Навчання моделі - замінюємо LightGBM на XGBoost
def train_model(X_train, y_train, X_val=None, y_val=None):
    params = {
        'objective': 'reg:squarederror',  # Для регресії у XGBoost
        'eval_metric': 'rmse',
        'max_depth': 7,                  # Глибина дерева
        'eta': 0.03,                     # Швидкість навчання
        'subsample': 0.8,                # Підвибірка рядків
        'colsample_bytree': 0.9,         # Підвибірка колонок для кожного дерева
        'alpha': 0.1,                    # L1 регуляризація
        'lambda': 0.1,                   # L2 регуляризація
        'min_child_weight': 20           # Аналог min_data_in_leaf
    }
    
    if X_val is not None and y_val is not None:
        # Створення валідаційного набору
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Налаштування для раннього зупинення
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        # Навчання моделі
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        print(f"Модель навчена на {len(X_train)} + {len(X_val)} зразках з раннім зупиненням")
        
    else:
        # Якщо немає валідаційного набору, використовуємо весь тренувальний
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            verbose_eval=100
        )
        print(f"Модель навчена на {len(X_train)} зразках")
    
    return model

# 5. Оцінка моделі
def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW") 
    print(f"R²: {r2:.4f}")
    
    # Розрахунок помилки по сезонах
    # Створюємо DataFrame з наростаючими індексами щоб уникнути дублювання
    errors_df = pd.DataFrame({
        'month': pd.DatetimeIndex(y_true.index).month,
        'error': np.abs(y_pred - y_true.values)
    }, index=range(len(y_true)))
    
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }
    
    for season, months in seasons.items():
        season_errors = errors_df[errors_df['month'].isin(months)]['error']
        if len(season_errors) > 0:
            season_rmse = np.sqrt(np.mean(season_errors ** 2))
            print(f"{season} RMSE: {season_rmse:.2f} MW")
    
    return rmse, mae, r2

# 6. Візуалізація
def plot_predictions(test, y_true, y_pred):
    plt.figure(figsize=(16, 8))
    plt.plot(test.index, y_true, label='Actual', color='blue')
    plt.plot(test.index, y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs Predicted Energy Consumption (2017)')
    plt.xlabel('Date')
    plt.ylabel('MW')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Додаткова візуалізація - помісячна похибка
    plt.figure(figsize=(16, 6))
    
    # Створюємо DataFrame без індексу для безпечного ресемплінгу
    y_true_series = pd.Series(y_true, index=test.index)
    y_pred_series = pd.Series(y_pred, index=test.index)
    
    # Ресемплінг на місячні дані
    monthly_actual = y_true_series.resample('ME').mean()
    monthly_predicted = y_pred_series.resample('ME').mean()
    monthly_error = monthly_predicted - monthly_actual
    
    plt.bar(monthly_error.index, monthly_error.values, color='skyblue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Monthly Average Prediction Error (2017)')
    plt.xlabel('Month')
    plt.ylabel('Error (MW)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Додавання візуалізації помилок за температурою
    temp_errors = pd.DataFrame({
        'Temperature': test['Chicago_temp'].values,
        'Error': y_pred - y_true.values
    })
    
    plt.figure(figsize=(14, 6))
    
    # Створюємо температурні біни
    temp_bins = pd.cut(temp_errors['Temperature'], bins=10)
    
    # Використовуємо boxplot з seaborn
    ax = sns.boxplot(x=temp_bins, y=temp_errors['Error'])
    plt.title('Prediction Error by Temperature Range')
    plt.xlabel('Temperature Range (°C)')
    plt.ylabel('Error (MW)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# Аналіз перших двох тижнів січня
def plot_january_first_two_weeks(test, y_test, y_pred):
    # Конвертація y_pred в pandas Series з тим самим індексом, що і test
    y_pred_series = pd.Series(y_pred, index=test.index)
    
    # Фільтрація за перші два тижні січня 2017
    jan_start = '2017-01-01'
    jan_end = '2017-01-14'
    
    # Отримання даних за цей період
    jan_data = test.loc[jan_start:jan_end].copy()
    
    # Додавання фактичних та прогнозованих значень до датафрейму
    jan_data['Actual'] = y_test.loc[jan_start:jan_end]
    jan_data['Predicted'] = y_pred_series.loc[jan_start:jan_end]
    
    # Розрахунок погодинної помилки
    jan_data['Error'] = jan_data['Predicted'] - jan_data['Actual']
    jan_data['AbsError'] = abs(jan_data['Error'])
    
    # Побудова графіку порівняння
    plt.figure(figsize=(16, 10))
    
    # Перший підграфік: Фактичне vs Прогнозоване
    plt.subplot(2, 1, 1)
    plt.plot(jan_data.index, jan_data['Actual'], label='Actual', linewidth=2)
    plt.plot(jan_data.index, jan_data['Predicted'], label='Predicted', linewidth=2, alpha=0.7)
    plt.title('Actual vs Predicted Energy Consumption - First Two Weeks of January 2017')
    plt.ylabel('Energy Consumption (MW)')
    plt.legend()
    plt.grid(True)
    
    # Другий підграфік: Помилка
    plt.subplot(2, 1, 2)
    plt.bar(jan_data.index, jan_data['Error'], color='darkred', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Prediction Error (Predicted - Actual)')
    plt.ylabel('Error (MW)')
    plt.xlabel('Date')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Виведення статистики за цей період
    mean_error = jan_data['Error'].mean()
    mean_abs_error = jan_data['AbsError'].mean()
    rmse_value = np.sqrt(np.mean(jan_data['Error'] ** 2))
    
    print("\nFirst Two Weeks of January 2017 Statistics:")
    print(f"Mean Error: {mean_error:.2f} MW")
    print(f"Mean Absolute Error: {mean_abs_error:.2f} MW")
    print(f"RMSE: {rmse_value:.2f} MW")
    
    # Аналіз добових патернів
    plt.figure(figsize=(12, 6))
    
    # Групування даних за годиною доби
    hourly_data = jan_data.groupby(jan_data.index.hour).agg({
        'Actual': 'mean',
        'Predicted': 'mean',
        'Error': 'mean'
    })
    
    plt.plot(hourly_data.index, hourly_data['Actual'], 'o-', label='Actual')
    plt.plot(hourly_data.index, hourly_data['Predicted'], 'x-', label='Predicted')
    plt.title('Average Hourly Pattern - First Two Weeks of January 2017')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Energy Consumption (MW)')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Функція для відображення важливості ознак у XGBoost моделі
def plot_feature_importance(model, features):
    # Отримання важливості ознак (дві методики)
    importance_gain = model.get_score(importance_type='gain')
    importance_weight = model.get_score(importance_type='weight')
    
    # Перетворення на DataFrame для зручності візуалізації
    importance_df_gain = pd.DataFrame({
        'Feature': list(importance_gain.keys()),
        'Importance': list(importance_gain.values())
    }).sort_values('Importance', ascending=False)
    
    importance_df_weight = pd.DataFrame({
        'Feature': list(importance_weight.keys()),
        'Importance': list(importance_weight.values())
    }).sort_values('Importance', ascending=False)
    
    # Візуалізація важливості за "gain" (внесок у зменшення помилки)
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.barh(importance_df_gain['Feature'][:20], importance_df_gain['Importance'][:20])
    plt.xlabel('Gain (вклад в покращення моделі)')
    plt.ylabel('Ознака')
    plt.title('Важливість ознак за Gain (топ-20)')
    plt.gca().invert_yaxis()  # Показати найважливішу ознаку зверху
    
    # Візуалізація важливості за "weight" (кількість використань в деревах)
    plt.subplot(2, 1, 2)
    plt.barh(importance_df_weight['Feature'][:20], importance_df_weight['Importance'][:20])
    plt.xlabel('Weight (частота використання в деревах)')
    plt.ylabel('Ознака')
    plt.title('Важливість ознак за Weight (топ-20)')
    plt.gca().invert_yaxis()  # Показати найважливішу ознаку зверху
    
    plt.tight_layout()
    plt.show()
    
    return importance_df_gain, importance_df_weight

# Основна виконавча частина
if __name__ == "__main__":
    try:
        # Завантаження даних
        print("Завантаження даних...")
        df = load_and_preprocess('data/FINAL_dataset.csv')
        
        # Розрахунок всіх ознак для всього датасету
        print("Розрахунок ознак...")
        df_with_features = calculate_features(df)
        
        # Розділення на тренувальний (до 2016) та тестовий (2017) набори
        print("Розділення даних...")
        train, test = split_data(df_with_features, test_start='2017-01-01', test_end='2017-12-31')
        
        # Видалення рядків з пропущеними значеннями
        train = train.dropna()
        
        # Визначення ознак та цільової змінної
        features = [
            'hour', 'weekday', 'quarter', 'month', 'is_holiday', 'is_summer',
            'hour_sin', 'hour_cos', 'day_of_year',
            'Chicago_temp', 'Chicago_humidity', 'cooling_degree', 'cooling_degree_squared',
            'temp_humidity_interaction', 'extreme_heat', 'ac_peak_hours', 'summer_temp_interaction',
            'lag_720h', 'lag_1y', 'lag_2y',
            'cooling_degree_lag_24h', 'cooling_degree_lag_168h',
            'temp_rolling_24h', 'temp_rolling_72h', 'temp_rolling_max_24h', 'temp_volatility_24h'
        ]
        
        target = 'COMED_MW'
        
        # Перевіряємо, що тестовий набір має всі потрібні ознаки без NaN
        print(f"Тестовий набір містить NaN: {test[features].isna().sum().sum() > 0}")
        if test[features].isna().sum().sum() > 0:
            # Виведемо колонки з NaN значеннями
            nan_columns = test[features].columns[test[features].isna().any()].tolist()
            print(f"Колонки з NaN: {nan_columns}")
            
        test = test.dropna(subset=features)
        print(f"Розмір тестового набору після видалення NaN: {test.shape}")
        
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]
        
        # Тренування моделі
        print("Тренування моделі XGBoost...")
        
        # Створення валідаційного набору за останні 3 місяці тренувальних даних
        val_start = pd.to_datetime('2016-10-01')
        X_val = X_train[X_train.index >= val_start]
        y_val = y_train[y_train.index >= val_start]
        X_train_final = X_train[X_train.index < val_start]
        y_train_final = y_train[y_train.index < val_start]
        
        model = train_model(X_train_final, y_train_final, X_val, y_val)
        
        # Прогнозування
        print("Виконання прогнозування...")
        # Перетворення даних для XGBoost
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        
        # Оцінка
        print("\nПродуктивність моделі:")
        evaluate(y_test, y_pred)
        
        # Візуалізація
        print("Побудова графіків...")
        plot_predictions(test, y_test, y_pred)
        
        # Важливість ознак
        importance_gain, importance_weight = plot_feature_importance(model, features)
        print("\nТоп-10 найважливіших ознак за Gain:")
        print(importance_gain.head(10))
        
        # Аналіз залежності прогнозу від температури
        plt.figure(figsize=(14, 6))
        
        # Створюємо новий DataFrame для аналізу
        temp_data = pd.DataFrame({
            'Temperature': test['Chicago_temp'].values,
            'Actual': y_test.values,
            'Predicted': y_pred
        })
        
        # Створюємо температурні групи
        temp_data['temp_group'] = pd.cut(temp_data['Temperature'], bins=20)
        
        # Групуємо дані за температурними групами
        grouped = temp_data.groupby('temp_group')[['Actual', 'Predicted']].mean().reset_index()
        
        # Конвертуємо категоріальні біни до строк для нормального відображення на графіку
        temp_cats = [str(x) for x in grouped['temp_group']]
        
        plt.plot(range(len(temp_cats)), grouped['Actual'], marker='o', label='Actual')
        plt.plot(range(len(temp_cats)), grouped['Predicted'], marker='x', label='Predicted')
        plt.title('Середнє споживання за температурними діапазонами')
        plt.xlabel('Температурний діапазон (°C)')
        plt.ylabel('Середнє споживання (MW)')
        plt.xticks(range(len(temp_cats)), temp_cats, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Аналіз перших двох тижнів січня
        print("Analyzing first two weeks of January...")
        plot_january_first_two_weeks(test, y_test, y_pred)
        
        # Додатково: візуалізація дерева (тільки одне для прикладу)
        plt.figure(figsize=(20, 10))
        xgb.plot_tree(model, num_trees=0)
        plt.title('Перше дерево в XGBoost моделі')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Помилка виконання: {str(e)}")
        import traceback
        traceback.print_exc()