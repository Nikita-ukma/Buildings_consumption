import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    return df


def calculate_features(train, test):
    
    for lag in [3, 6, 12, 24]:
        train[f'lag_{lag}'] = train['COMED_MW'].shift(lag)
    
    
    
    test_features = test.copy()
    for lag in [3, 6, 12, 24]:
        test_features[f'lag_{lag}'] = np.nan
    
    test_features['rolling_mean_3h'] = np.nan
    test_features['rolling_mean_6h'] = np.nan
    
    
    for i in range(len(test_features)):
        
        for lag in [3, 6, 12, 24]:
            if i >= lag:
                test_features.iloc[i, test_features.columns.get_loc(f'lag_{lag}')] = test_features['COMED_MW'].iloc[i-lag]
        
        
        if i >= 3:
            window_3h = test_features['COMED_MW'].iloc[max(0, i-3):i+1]
            test_features.iloc[i, test_features.columns.get_loc('rolling_mean_3h')] = window_3h.mean()
        
        if i >= 6:
            window_6h = test_features['COMED_MW'].iloc[max(0, i-6):i+1]
            test_features.iloc[i, test_features.columns.get_loc('rolling_mean_6h')] = window_6h.mean()
    
    return train, test_features


def split_data(df, test_start='2017-01-01'):
    train = df.loc[:test_start]
    test = df.loc[test_start:]
    return train, test


def train_model(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    return model


def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW") 
    print(f"R²: {r2:.4f}")
    
    return rmse, mae, r2


def plot_predictions(test, y_true, y_pred):
    plt.figure(figsize=(14, 6))
    plt.plot(test.index, y_true, label='Actual', color='blue')
    plt.plot(test.index, y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs Predicted Energy Consumption')
    plt.xlabel('Date')
    plt.ylabel('MW')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    
    df = load_and_preprocess('FINAL_dataset.csv')
    
    
    train_raw, test_raw = split_data(df)
    
    
    train, test = calculate_features(train_raw, test_raw)
    
    
    train = train.dropna()
    test = test.dropna()
    
    
    features = ['hour', 'weekday', 'is_holiday', 'Chicago_temp', 
               'Chicago_humidity', 'lag_3', 'lag_6']
    target = 'COMED_MW'
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    
    print("Training model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    
    y_pred = model.predict(X_test)
    
    
    print("\nModel Performance:")
    evaluate(y_test, y_pred)
    
    
    plot_predictions(test, y_test, y_pred)
    
    
    lgb.plot_importance(model, importance_type='gain', figsize=(12, 6))
    plt.title('Feature Importance')
    plt.show()