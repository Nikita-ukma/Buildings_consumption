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
    
    
    df['quarter'] = df.index.quarter
    
    return df


def calculate_features(df):
    
    features_df = df.copy()
    return features_df


def split_data(df, test_start='2018-01-01', test_end='2018-12-31'):
    train = df.loc[:test_start].iloc[:-1]  
    test = df.loc[test_start:test_end]     
    return train, test


def train_model(X_train, y_train, X_val=None, y_val=None):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
    else:
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500
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
    plt.figure(figsize=(16, 8))
    plt.plot(test.index, y_true, label='Actual', color='blue')
    plt.plot(test.index, y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs Predicted Energy Consumption (2018)')
    plt.xlabel('Date')
    plt.ylabel('MW')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(16, 6))
    monthly_errors = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Error': y_pred - y_true
    }, index=test.index)
    
    monthly_errors = monthly_errors.resample('M').mean()
    plt.bar(monthly_errors.index, monthly_errors['Error'], color='skyblue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Monthly Average Prediction Error (2018)')
    plt.xlabel('Month')
    plt.ylabel('Error (MW)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    df = load_and_preprocess('FINAL_dataset.csv')
    
    
    df_with_features = calculate_features(df)
    
    
    train, test = split_data(df_with_features)
    
    
    train = train.dropna()
    
    
    features = [
        'hour', 'weekday', 'quarter', 'is_holiday', 
        'Chicago_temp', 'Chicago_humidity',
    ]
    
    target = 'COMED_MW'
    
    
    print(f"Тестовий набір містить NaN: {test[features].isna().sum().sum() > 0}")
    test = test.dropna(subset=features)
    print(f"Розмір тестового набору після видалення NaN: {test.shape}")
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    
    print("Тренування моделі...")
    
    
    val_start = pd.to_datetime('2017-10-01')
    X_val = X_train[X_train.index >= val_start]
    y_val = y_train[y_train.index >= val_start]
    X_train_final = X_train[X_train.index < val_start]
    y_train_final = y_train[y_train.index < val_start]
    
    model = train_model(X_train_final, y_train_final, X_val, y_val)
    
    
    y_pred = model.predict(X_test)
    
    
    print("\nПродуктивність моделі:")
    evaluate(y_test, y_pred)
    
    
    plot_predictions(test, y_test, y_pred)
    
    
    plt.figure(figsize=(12, 8))
    lgb.plot_importance(model, importance_type='gain', figsize=(12, 8))
    plt.title('Важливість ознак')
    plt.tight_layout()
    plt.show()