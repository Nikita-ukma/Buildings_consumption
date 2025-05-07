import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['timestamp'] = df.index
    return df


def add_lag_features(df, lags=[3, 6, 12, 24, 48, 168]):
    for lag in lags:
        df[f'lag_{lag}'] = df['COMED_MW'].shift(lag)
    return df

def add_rolling_features(df):
    df['rolling_mean_24h'] = df['COMED_MW'].shift(1).rolling(window=24).mean()
    df['rolling_std_24h'] = df['COMED_MW'].shift(1).rolling(window=24).std()
    df['rolling_mean_7d'] = df['COMED_MW'].shift(1).rolling(window=168).mean()
    df['rolling_std_7d'] = df['COMED_MW'].shift(1).rolling(window=168).std()
    return df


def split_data(df, test_start='2017-07-01'):
    train = df[df.index < test_start]
    test = df[df.index >= test_start]
    return train, test


def train_model(X_train, y_train):
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model


def recursive_forecast(model, last_history_df, forecast_horizon, features, lags):
    history = last_history_df.copy()
    forecasts = []

    for _ in tqdm(range(forecast_horizon)):
        last_row = history.iloc[-1].copy()
        new_time = last_row['timestamp'] + pd.Timedelta(hours=1)

        new_row = {
            'timestamp': new_time,
            'hour': new_time.hour,
            'weekday': new_time.weekday(),
            'month': new_time.month,
            'dayofyear': new_time.dayofyear,
            'weekofyear': new_time.isocalendar().week,
            'is_weekend': int(new_time.weekday() >= 5),
        }

        
        for lag in lags:
            new_row[f'lag_{lag}'] = history['COMED_MW'].iloc[-lag]

        
        new_row['rolling_mean_24h'] = history['COMED_MW'].iloc[-24:].mean()
        new_row['rolling_std_24h'] = history['COMED_MW'].iloc[-24:].std()
        new_row['rolling_mean_7d'] = history['COMED_MW'].iloc[-168:].mean()
        new_row['rolling_std_7d'] = history['COMED_MW'].iloc[-168:].std()

        
        for col in ['Chicago_temp', 'Chicago_humidity', 'is_holiday']:
            new_row[col] = history[col].iloc[-1]

        
        new_row_df = pd.DataFrame([new_row])
        y_pred = model.predict(new_row_df[features])[0]
        new_row_df['COMED_MW'] = y_pred

        
        history = pd.concat([history, new_row_df], ignore_index=True)

        
        forecasts.append({'timestamp': new_time, 'predicted_MW': y_pred})

    forecast_df = pd.DataFrame(forecasts)
    return forecast_df


def plot_forecast(original_df, forecast_df, forecast_start_time):
    plt.figure(figsize=(16, 6))
    plt.plot(original_df.index[-500:], original_df['COMED_MW'].iloc[-500:], label='Actual')
    plt.plot(forecast_df['timestamp'], forecast_df['predicted_MW'], label='Recursive Forecast', color='red')
    plt.axvline(x=forecast_start_time, color='gray', linestyle='--', label='Forecast Start')
    plt.title("Recursive Forecast for 1 Month")
    plt.xlabel("Date")
    plt.ylabel("MW")
    plt.legend()
    plt.grid()
    plt.show()
def evaluate_forecast(forecast_df, actual_df):
    
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp']).dt.floor('H')
    actual_df = actual_df.copy()
    actual_df.index = pd.to_datetime(actual_df.index).floor('H')

    
    merged = forecast_df.merge(actual_df[['COMED_MW']], left_on='timestamp', right_index=True, how='inner')
    merged = merged.dropna()

    if len(merged) == 0:
        print("⚠️ УВАГА: немає спільних часових точок між прогнозом і реальністю!")
        print("🔎 Перевір діапазони часу.")
        return pd.DataFrame(), np.nan, np.nan, np.nan

    y_true = merged['COMED_MW']
    y_pred = merged['predicted_MW']

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\n📊 Forecast Evaluation:")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE:  {mae:.2f} MW")
    print(f"MAPE: {mape:.2f}%")

    return merged, rmse, mae, mape


def plot_forecast_comparison(merged_df, forecast_start_time):
    plt.figure(figsize=(16, 6))
    plt.plot(merged_df['timestamp'], merged_df['COMED_MW'], label='Actual', color='blue', alpha=1)
    plt.plot(merged_df['timestamp'], merged_df['predicted_MW'], label='Forecast', color='red', alpha=0.7)
    plt.axvline(x=forecast_start_time, color='gray', linestyle='--', label='Forecast Start')
    plt.fill_between(merged_df['timestamp'], merged_df['COMED_MW'], merged_df['predicted_MW'], color='gray', alpha=0.2, label='Error')
    plt.title("Recursive Forecast vs Actual (1 Month Ahead)")
    plt.xlabel("Time")
    plt.ylabel("MW")
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    
    df = load_and_preprocess("FINAL_dataset.csv")

    
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna()

    
    train_df, test_df = split_data(df, test_start="2017-07-01")

    
    features = [
        'hour', 'weekday', 'month', 'dayofyear', 'weekofyear', 'is_weekend',
        'is_holiday', 'Chicago_temp', 'Chicago_humidity',
        'lag_3', 'lag_6', 'lag_12', 'lag_24', 'lag_48', 'lag_168',
        'rolling_mean_24h', 'rolling_std_24h', 'rolling_mean_7d', 'rolling_std_7d'
    ]
    lags = [3, 6, 12, 24, 48, 168]

    X_train = train_df[features]
    y_train = train_df['COMED_MW']

    
    print("🔧 Training model...")
    model = train_model(X_train, y_train)

    
    forecast_hours = 24 * 30  
    last_window = test_df.iloc[-168:].copy()  

    print("📈 Generating recursive forecast...")
    forecast_df = recursive_forecast(
        model=model,
        last_history_df=last_window,
        forecast_horizon=forecast_hours,
        features=features,
        lags=lags
    )

    
    forecast_start_time = last_window['timestamp'].iloc[-1]
    plot_forecast(df, forecast_df, forecast_start_time)
     
    merged_df, rmse, mae, mape = evaluate_forecast(forecast_df, df)

    
    plot_forecast_comparison(merged_df, forecast_start_time)

