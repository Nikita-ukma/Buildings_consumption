import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Завантаження даних
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    return df

# 2. Розділення на train / test
def split_data(df, test_start='2017-01-01'):
    train = df.loc[:test_start].copy()
    test = df.loc[test_start:].copy()
    return train, test

# 3. Додавання lag-ознак
def add_lag_features(df, lags=[3, 6]):
    for lag in lags:
        df[f'lag_{lag}'] = df['COMED_MW'].shift(lag)
    return df

# 4. Навчання моделі
def train_model(X_train, y_train):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=300)
    return model

# 5. Ітеративний autoregressive прогноз
def autoregressive_forecast(model, train_df, test_df, features, target, lags):
    history = train_df.copy()
    predictions = []

    for i in range(len(test_df)):
        current = test_df.iloc[i:i+1].copy()
        
        # Додаємо лаги з history
        for lag in lags:
            current[f'lag_{lag}'] = history[target].iloc[-lag]

        # Переконаємося, що всі потрібні фічі є
        X_current = current[features]

        # Прогноз
        y_pred = model.predict(X_current)[0]
        predictions.append(y_pred)

        # Додаємо передбачення в історію для наступних лагів
        new_row = current.copy()
        new_row[target] = y_pred
        history = pd.concat([history, new_row], axis=0)

    return predictions

# 6. Оцінка
def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nRMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW")
    print(f"R²: {r2:.4f}")
    return rmse, mae, r2

# 7. Візуалізація
def plot_predictions(test_index, y_true, y_pred):
    plt.figure(figsize=(14, 6))
    plt.plot(test_index, y_true, label='Actual', color='blue')
    plt.plot(test_index, y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs Predicted Energy Consumption')
    plt.xlabel('Date')
    plt.ylabel('MW')
    plt.legend()
    plt.grid()
    plt.show()

# 8. Побудова важливості ознак
def plot_feature_importance(model):
    lgb.plot_importance(model, importance_type='gain', figsize=(12, 6))
    plt.title('Feature Importance')
    plt.show()

# ==== MAIN ====
if __name__ == "__main__":
    # Шлях до датасету
    filepath = 'FINAL_dataset.csv'

    # Параметри
    lag_features = [3, 6]
    feature_cols = ['hour', 'weekday', 'is_holiday', 'Chicago_temp', 'Chicago_humidity'] + [f'lag_{l}' for l in lag_features]
    target_col = 'COMED_MW'

    # 1. Завантаження
    df = load_and_preprocess(filepath)

    # 2. Розділення
    train_df, test_df = split_data(df)

    # 3. Генерація lag-ознак у train
    train_df = add_lag_features(train_df, lag_features)
    train_df = train_df.dropna()

    # 4. Навчання
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    model = train_model(X_train, y_train)

    # 5. Прогнозування autoregressive style
    y_test_true = test_df[target_col].values
    y_test_pred = autoregressive_forecast(model, train_df, test_df, feature_cols, target_col, lag_features)

    # 6. Оцінка
    print("\nModel Performance:")
    evaluate(y_test_true, y_test_pred)

    # 7. Візуалізація
    plot_predictions(test_df.index, y_test_true, y_test_pred)

    # 8. Важливість ознак
    plot_feature_importance(model)