import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import xgboost as xgb
from pandas.tseries.holiday import USFederalHolidayCalendar
import logging
import warnings
from pyswarm import pso  


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_preprocess(filepath):
    
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise ValueError("Файл порожній")
        
        logger.info("Завантажено дані з файлу")
        
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.set_index('Datetime').sort_index()
        
        
        logger.info("Обробка пропущених даних...")
        missing_before = df.isna().sum().sum()
        df = df.interpolate(method='linear').fillna(method='ffill')
        missing_after = df.isna().sum().sum()
        logger.info(f"Пропущені значення: {missing_before} до обробки, {missing_after} після")
        
        
        df = detect_outliers(df, 'COMED_MW')
        df = detect_outliers(df, 'Chicago_temp')
        
        
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['weekday'] = df.index.dayofweek
        df['hour'] = df.index.hour
        df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
        df['day_of_year'] = df.index.dayofyear
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        
        base_temp = 21
        df['cooling_degree'] = df['Chicago_temp'].apply(lambda x: max(0, x - base_temp))
        df['cooling_degree_squared'] = df['cooling_degree'] ** 2
        df['temp_humidity_interaction'] = df['Chicago_temp'] * df['Chicago_humidity'] / 100
        df['extreme_heat'] = (df['Chicago_temp'] > 30).astype(int)
        df['ac_peak_hours'] = ((df.index.hour >= 12) & (df.index.hour <= 18)).astype(int)
        df['summer_temp_interaction'] = df['is_summer'] * df['Chicago_temp']
        
        
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df.index.min(), end=df.index.max())
        df['is_holiday'] = df.index.isin(holidays).astype(int)
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        
        
        df['heat_index'] = calculate_heat_index(df['Chicago_temp'], df['Chicago_humidity'])
        
        return df
    
    except FileNotFoundError:
        logger.error(f"Файл {filepath} не знайдено")
        raise
    except Exception as e:
        logger.error(f"Помилка при обробці даних: {e}")
        raise


def detect_outliers(df, column):
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[f'{column}_outlier'] = ((df[column] < lower_bound) | (df[column] > upper_bound)).astype(int)
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    logger.info(f"Оброблено аномалії в колонці {column}")
    return df


def calculate_heat_index(temp, humidity):
    
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6
    
    temp_f = temp * 9/5 + 32
    hi = (c1 + c2 * temp_f + c3 * humidity + c4 * temp_f * humidity +
          c5 * temp_f**2 + c6 * humidity**2 + c7 * temp_f**2 * humidity +
          c8 * temp_f * humidity**2 + c9 * temp_f**2 * humidity**2)
    return hi * 5/9 - 32/9


def calculate_features(df):
    features_df = df.copy()
    
    features_df['lag_720h'] = features_df['COMED_MW'].shift(720)
    features_df['lag_1y'] = features_df['COMED_MW'].shift(8760)
    features_df['lag_2y'] = features_df['COMED_MW'].shift(17520)
    features_df['cooling_degree_lag_24h'] = features_df['cooling_degree'].shift(24)
    features_df['cooling_degree_lag_168h'] = features_df['cooling_degree'].shift(168)
    features_df['temp_rolling_24h'] = features_df['Chicago_temp'].rolling(window=24).mean().shift(1)
    features_df['temp_rolling_72h'] = features_df['Chicago_temp'].rolling(window=72).mean().shift(1)
    features_df['temp_rolling_max_24h'] = features_df['Chicago_temp'].rolling(window=24).max().shift(1)
    features_df['temp_volatility_24h'] = features_df['Chicago_temp'].rolling(window=24).std().shift(1)
    
    return features_df


def split_data(df, test_start='2017-01-01', test_end='2017-12-31'):
    train = df.loc[:test_start].iloc[:-1]
    test = df.loc[test_start:test_end]
    return train, test


def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X.iloc[i:i+time_steps].values)
        y_seq.append(y.iloc[i+time_steps])
    return np.array(X_seq), np.array(y_seq)


def train_lstm_model(X_train, y_train, X_val=None, y_val=None, time_steps=24):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    X_train_seq, y_train_seq = create_sequences(
        pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index),
        pd.Series(y_train_scaled, index=y_train.index),
        time_steps
    )
    
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
        X_val_seq, y_val_seq = create_sequences(
            pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index),
            pd.Series(y_val_scaled, index=y_val.index),
            time_steps
        )
        validation_data = (X_val_seq, y_val_seq)
    else:
        validation_data = None
    
    
    inputs = tf.keras.Input(shape=(time_steps, X_train.shape[1]))
    lstm_out = Bidirectional(LSTM(128, activation='tanh', return_sequences=True, 
                                 kernel_regularizer=l2(0.01)))(inputs)
    
    
    attention_out = Attention()([lstm_out, lstm_out])
    
    
    combined = Concatenate()([lstm_out, attention_out])
    
    
    lstm_final = Bidirectional(LSTM(64, activation='tanh', kernel_regularizer=l2(0.01)))(combined)
    
    
    dropout = Dropout(0.3)(lstm_final)
    dense = Dense(32, activation='relu')(dropout)
    output = Dense(1)(dense)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')
    ]
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_data=validation_data if validation_data else None,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info(f"LSTM модель навчена на {len(X_train_seq)} послідовностях")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Тренувальна похибка')
    if validation_data:
        plt.plot(history.history['val_loss'], label='Валідаційна похибка')
    plt.title('Похибка LSTM моделі по епохам')
    plt.xlabel('Епоха')
    plt.ylabel('Похибка (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_training_history.png')
    
    return model, scaler_X, scaler_y, time_steps


def train_xgboost_model(X_train, y_train, X_val=None, y_val=None):
    
    def objective_function(params):
        max_depth, learning_rate, subsample, colsample_bytree = params
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            n_estimators=1000,
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=50,
            random_state=42
        )
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
        
        y_pred = model.predict(X_val if X_val is not None else X_train)
        mse = mean_squared_error(y_val if y_val is not None else y_train, y_pred)
        return mse
    
    
    lb = [3, 0.01, 0.5, 0.5]  
    ub = [10, 0.3, 1.0, 1.0]  
    
    
    logger.info("Запуск PSO для оптимізації гіперпараметрів XGBoost...")
    best_params, best_score = pso(
        objective_function,
        lb,
        ub,
        swarmsize=20,
        maxiter=20,
        debug=True,
        minfunc=1e-6
    )
    
    
    best_params = [int(best_params[0]), best_params[1], best_params[2], best_params[3]]
    
    logger.info(f"Найкращі параметри XGBoost (PSO): max_depth={best_params[0]}, "
                f"learning_rate={best_params[1]:.4f}, subsample={best_params[2]:.4f}, "
                f"colsample_bytree={best_params[3]:.4f}")
    logger.info(f"Найкраще значення MSE: {best_score:.4f}")
    
    
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=1000,
        max_depth=best_params[0],
        learning_rate=best_params[1],
        subsample=best_params[2],
        colsample_bytree=best_params[3],
        early_stopping_rounds=50,
        random_state=42
    )
    
    if X_val is not None and y_val is not None:
        best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
    else:
        best_model.fit(X_train, y_train, verbose=100)
    
    logger.info(f"XGBoost модель навчена на {len(X_train)} зразках")
    
    return best_model


def predict_lstm(model, X_test, scaler_X, scaler_y, time_steps):
    X_test_scaled = scaler_X.transform(X_test)
    X_test_seq = np.array([X_test_scaled[i:i+time_steps] for i in range(len(X_test_scaled)-time_steps)])
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    return y_pred, time_steps


def predict_xgboost(model, X_test):
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest) if isinstance(model, xgb.core.Booster) else model.predict(X_test)
    return y_pred, 0


def train_meta_model(lstm_preds, xgb_preds, y_true):
    meta_X = pd.DataFrame({
        'lstm_pred': lstm_preds,
        'xgb_pred': xgb_preds
    })
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X, y_true)
    
    logger.info(f"LSTM коефіцієнт: {meta_model.coef_[0]:.4f}")
    logger.info(f"XGBoost коефіцієнт: {meta_model.coef_[1]:.4f}")
    logger.info(f"Зсув: {meta_model.intercept_:.4f}")
    
    return meta_model


def evaluate_model(y_true, y_pred, model_name="Модель"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"--- {model_name} ---")
    logger.info(f"RMSE: {rmse:.2f} MW")
    logger.info(f"MAE: {mae:.2f} MW")
    logger.info(f"R²: {r2:.4f}")
    
    errors_df = pd.DataFrame({
        'month': pd.DatetimeIndex(y_true.index).month,
        'error': np.abs(y_pred - y_true.values)
    })
    
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
            logger.info(f"{season} RMSE: {season_rmse:.2f} MW")
    
    return rmse, mae, r2


def plot_all_predictions(test, y_true, lstm_pred, xgb_pred, hybrid_pred):
    plt.figure(figsize=(16, 10))
    plt.plot(y_true.index, y_true, label='Фактичні значення', color='black', linewidth=2)
    plt.plot(y_true.index, lstm_pred, label='LSTM', color='blue', alpha=0.7)
    plt.plot(y_true.index, xgb_pred, label='XGBoost', color='green', alpha=0.7)
    plt.plot(y_true.index, hybrid_pred, label='Гібридна модель', color='red', alpha=0.7)
    plt.title('Порівняння моделей прогнозування енергоспоживання (2017)')
    plt.xlabel('Дата')
    plt.ylabel('Споживання (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('all_predictions.png')
    
    plt.figure(figsize=(16, 12))
    
    y_true_series = pd.Series(y_true, index=y_true.index)
    lstm_series = pd.Series(lstm_pred, index=y_true.index)
    xgb_series = pd.Series(xgb_pred, index=y_true.index)
    hybrid_series = pd.Series(hybrid_pred, index=y_true.index)
    
    monthly_actual = y_true_series.resample('ME').mean()
    monthly_lstm = lstm_series.resample('ME').mean()
    monthly_xgb = xgb_series.resample('ME').mean()
    monthly_hybrid = hybrid_series.resample('ME').mean()
    
    lstm_error = monthly_lstm - monthly_actual
    xgb_error = monthly_xgb - monthly_actual
    hybrid_error = monthly_hybrid - monthly_actual
    
    plt.subplot(3, 1, 1)
    plt.bar(lstm_error.index, lstm_error.values, color='blue', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('LSTM - Середня місячна похибка')
    plt.ylabel('Похибка (MW)')
    plt.grid(True, axis='y')
    
    plt.subplot(3, 1, 2)
    plt.bar(xgb_error.index, xgb_error.values, color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('XGBoost - Середня місячна похибка')
    plt.ylabel('Похибка (MW)')
    plt.grid(True, axis='y')
    
    plt.subplot(3, 1, 3)
    plt.bar(hybrid_error.index, hybrid_error.values, color='red', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Гібридна модель - Середня місячна похибка')
    plt.xlabel('Місяць')
    plt.ylabel('Похибка (MW)')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('monthly_errors.png')
    
    plt.figure(figsize=(10, 6))
    abs_errors = pd.DataFrame({
        'LSTM': np.abs(lstm_pred - y_true.values),
        'XGBoost': np.abs(xgb_pred - y_true.values),
        'Гібридна': np.abs(hybrid_pred - y_true.values)
    })
    
    sns.boxplot(data=abs_errors)
    plt.title('Порівняння розподілу абсолютних похибок')
    plt.ylabel('Абсолютна похибка (MW)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('error_distribution.png')


def analyze_by_temperature(test, y_true, lstm_pred, xgb_pred, hybrid_pred):
    temp_data = pd.DataFrame({
        'Temperature': test['Chicago_temp'],
        'Actual': y_true,
        'LSTM': lstm_pred,
        'XGBoost': xgb_pred,
        'Hybrid': hybrid_pred
    })
    
    temp_data['temp_group'] = pd.cut(temp_data['Temperature'], bins=10)
    grouped = temp_data.groupby('temp_group')[['Actual', 'LSTM', 'XGBoost', 'Hybrid']].mean()
    
    rmse_by_temp = pd.DataFrame(index=pd.unique(temp_data['temp_group']))
    
    for temp_group in rmse_by_temp.index:
        group_data = temp_data[temp_data['temp_group'] == temp_group]
        rmse_by_temp.loc[temp_group, 'LSTM'] = np.sqrt(mean_squared_error(
            group_data['Actual'], group_data['LSTM']))
        rmse_by_temp.loc[temp_group, 'XGBoost'] = np.sqrt(mean_squared_error(
            group_data['Actual'], group_data['XGBoost']))
        rmse_by_temp.loc[temp_group, 'Hybrid'] = np.sqrt(mean_squared_error(
            group_data['Actual'], group_data['Hybrid']))
    
    plt.figure(figsize=(14, 6))
    plt.plot(range(len(grouped)), grouped['Actual'], marker='o', label='Actual', color='black', linewidth=2)
    plt.plot(range(len(grouped)), grouped['LSTM'], marker='x', label='LSTM', color='blue')
    plt.plot(range(len(grouped)), grouped['XGBoost'], marker='+', label='XGBoost', color='green')
    plt.plot(range(len(grouped)), grouped['Hybrid'], marker='*', label='Hybrid', color='red')
    plt.title('Середнє споживання за температурними діапазонами')
    plt.xlabel('Температурний діапазон (°C)')
    plt.ylabel('Середнє споживання (MW)')
    plt.xticks(range(len(grouped)), [str(x) for x in grouped.index], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('temp_analysis_mean.png')
    
    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(len(rmse_by_temp))-0.2, rmse_by_temp['LSTM'], width=0.2, color='blue', alpha=0.7, label='LSTM')
    plt.bar(np.arange(len(rmse_by_temp)), rmse_by_temp['XGBoost'], width=0.2, color='green', alpha=0.7, label='XGBoost')
    plt.bar(np.arange(len(rmse_by_temp))+0.2, rmse_by_temp['Hybrid'], width=0.2, color='red', alpha=0.7, label='Hybrid')
    plt.title('RMSE за температурними діапазонами')
    plt.xlabel('Температурний діапазон (°C)')
    plt.ylabel('RMSE (MW)')
    plt.xticks(range(len(rmse_by_temp)), [str(x) for x in rmse_by_temp.index], rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('temp_analysis_rmse.png')
    
    return rmse_by_temp


def analyze_hourly_patterns(test, y_true, lstm_pred, xgb_pred, hybrid_pred):
    hourly_data = pd.DataFrame({
        'hour': test.index.hour,
        'Actual': y_true,
        'LSTM': lstm_pred,
        'XGBoost': xgb_pred,
        'Hybrid': hybrid_pred
    })
    
    hourly_means = hourly_data.groupby('hour').mean()
    rmse_by_hour = pd.DataFrame(index=range(24))
    
    for hour in range(24):
        hour_data = hourly_data[hourly_data['hour'] == hour]
        if not hour_data.empty:
            rmse_by_hour.loc[hour, 'LSTM'] = np.sqrt(mean_squared_error(
                hour_data['Actual'], hour_data['LSTM']))
            rmse_by_hour.loc[hour, 'XGBoost'] = np.sqrt(mean_squared_error(
                hour_data['Actual'], hour_data['XGBoost']))
            rmse_by_hour.loc[hour, 'Hybrid'] = np.sqrt(mean_squared_error(
                hour_data['Actual'], hour_data['Hybrid']))
    
    plt.figure(figsize=(14, 6))
    plt.plot(hourly_means.index, hourly_means['Actual'], marker='o', label='Actual', color='black', linewidth=2)
    plt.plot(hourly_means.index, hourly_means['LSTM'], marker='x', label='LSTM', color='blue')
    plt.plot(hourly_means.index, hourly_means['XGBoost'], marker='+', label='XGBoost', color='green')
    plt.plot(hourly_means.index, hourly_means['Hybrid'], marker='*', label='Hybrid', color='red')
    plt.title('Середнє споживання за годинами доби')
    plt.xlabel('Година доби')
    plt.ylabel('Середнє споживання (MW)')
    plt.xticks(range(24))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hourly_analysis_mean.png')
    
    plt.figure(figsize=(14, 6))
    plt.bar(np.arange(24)-0.2, rmse_by_hour['LSTM'], width=0.2, color='blue', alpha=0.7, label='LSTM')
    plt.bar(np.arange(24), rmse_by_hour['XGBoost'], width=0.2, color='green', alpha=0.7, label='XGBoost')
    plt.bar(np.arange(24)+0.2, rmse_by_hour['Hybrid'], width=0.2, color='red', alpha=0.7, label='Hybrid')
    plt.title('RMSE за годинами доби')
    plt.xlabel('Година доби')
    plt.ylabel('RMSE (MW)')
    plt.xticks(range(24))
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('hourly_analysis_rmse.png')
    
    return rmse_by_hour


def main():
    logger.info("Початок виконання програми...")
    
    
    logger.info("Завантаження та попередня обробка даних...")
    df = load_and_preprocess('data/FINAL_dataset.csv')
    
    logger.info("Розрахунок ознак часових рядів...")
    features_df = calculate_features(df)
    
    features_df = features_df.dropna()
    
    logger.info("Розділення даних...")
    train_df, test_df = split_data(features_df)
    
    target_col = 'COMED_MW'
    feature_cols = [col for col in features_df.columns if col != target_col]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    logger.info(f"Розмір тренувальних даних: {X_train.shape}")
    logger.info(f"Розмір тестових даних: {X_test.shape}")
    
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=False
    )
    
    logger.info("\nТренування LSTM моделі...")
    lstm_model, scaler_X, scaler_y, time_steps = train_lstm_model(
        X_train_main, y_train_main, X_val, y_val, time_steps=24
    )
    
    logger.info("\nТренування XGBoost моделі з PSO...")
    xgb_model = train_xgboost_model(X_train_main, y_train_main, X_val, y_val)
    
    logger.info("\nОтримання прогнозів від LSTM моделі...")
    lstm_predictions, lstm_offset = predict_lstm(lstm_model, X_test, scaler_X, scaler_y, time_steps)
    lstm_predictions_df = pd.Series(lstm_predictions, index=y_test.index[lstm_offset:])
    
    logger.info("\nОтримання прогнозів від XGBoost моделі...")
    xgb_predictions, xgb_offset = predict_xgboost(xgb_model, X_test)
    xgb_predictions_df = pd.Series(xgb_predictions, index=y_test.index[xgb_offset:])
    
    common_index = lstm_predictions_df.index.intersection(xgb_predictions_df.index)
    y_test_aligned = y_test.loc[common_index]
    lstm_predictions_aligned = lstm_predictions_df.loc[common_index]
    xgb_predictions_aligned = xgb_predictions_df.loc[common_index]
    
    logger.info("\nНавчання мета-моделі...")
    meta_model = train_meta_model(
        lstm_predictions_aligned.values, 
        xgb_predictions_aligned.values, 
        y_test_aligned.values
    )
    
    hybrid_predictions = meta_model.predict(pd.DataFrame({
        'lstm_pred': lstm_predictions_aligned.values,
        'xgb_pred': xgb_predictions_aligned.values
    }))
    
    logger.info("\n--- Оцінка моделей ---")
    lstm_metrics = evaluate_model(y_test_aligned, lstm_predictions_aligned, "LSTM модель")
    xgb_metrics = evaluate_model(y_test_aligned, xgb_predictions_aligned, "XGBoost модель")
    hybrid_metrics = evaluate_model(y_test_aligned, hybrid_predictions, "Гібридна модель")
    
    logger.info("\nВізуалізація результатів...")
    plot_all_predictions(
        test_df.loc[common_index], 
        y_test_aligned, 
        lstm_predictions_aligned.values, 
        xgb_predictions_aligned.values, 
        hybrid_predictions
    )
    
    logger.info("\nАналіз результатів за температурними діапазонами...")
    temp_analysis = analyze_by_temperature(
        test_df.loc[common_index], 
        y_test_aligned, 
        lstm_predictions_aligned.values, 
        xgb_predictions_aligned.values, 
        hybrid_predictions
    )
    
    logger.info("\nАналіз результатів за годинами доби...")
    hourly_analysis = analyze_hourly_patterns(
        test_df.loc[common_index], 
        y_test_aligned, 
        lstm_predictions_aligned.values, 
        xgb_predictions_aligned.values, 
        hybrid_predictions
    )
    
    logger.info("\nПрогнозування та аналіз завершені!")
    
    return {
        "lstm": lstm_metrics,
        "xgboost": xgb_metrics,
        "hybrid": hybrid_metrics
    }

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8')
    
    try:
        results = main()
        logger.info("\nРезультати метрик:")
        for model, (rmse, mae, r2) in results.items():
            logger.info(f"{model.upper()}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
    except Exception as e:
        logger.error(f"Помилка під час виконання: {e}")
        import traceback
        traceback.print_exc()