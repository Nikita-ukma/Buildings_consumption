

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    
    df['quarter'] = df.index.quarter
    
    
    df['month'] = df.index.month
    
    
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
    
    
    df['weekday'] = df.index.dayofweek
    
    
    df['hour'] = df.index.hour
    
    return df


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


def train_model(X_train, y_train, X_val=None, y_val=None, time_steps=24):
    
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
    
    
    model = Sequential([
        GRU(128, activation='tanh', return_sequences=True, 
            input_shape=(time_steps, X_train.shape[1])),
        Dropout(0.2),
        GRU(64, activation='tanh'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )
    
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_gru_model.h5', save_best_only=True, monitor='val_loss')
    ]
    
    
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=100,
        batch_size=32,
        validation_data=validation_data if validation_data else None,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"Model trained on {len(X_train_seq)} sequences")
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if validation_data:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('GRU Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, scaler_X, scaler_y, time_steps


def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW") 
    print(f"R²: {r2:.4f}")
    
    
    
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


def plot_predictions(test, y_true, y_pred):
    plt.figure(figsize=(16, 8))
    plt.plot(test.index[len(test)-len(y_pred):], y_true, label='Actual', color='blue')
    plt.plot(test.index[len(test)-len(y_pred):], y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title('Actual vs Predicted Energy Consumption (2017)')
    plt.xlabel('Date')
    plt.ylabel('MW')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    plt.figure(figsize=(16, 6))
    
    
    y_true_series = pd.Series(y_true, index=test.index[len(test)-len(y_pred):])
    y_pred_series = pd.Series(y_pred, index=test.index[len(test)-len(y_pred):])
    
    
    monthly_actual = y_true_series.resample('M').mean()
    monthly_predicted = y_pred_series.resample('M').mean()
    monthly_error = monthly_predicted - monthly_actual
    
    plt.bar(monthly_error.index, monthly_error.values, color='skyblue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Monthly Average Prediction Error (2017)')
    plt.xlabel('Month')
    plt.ylabel('Error (MW)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    
    temp_values = test['Chicago_temp'].values[len(test)-len(y_pred):]
    temp_errors = pd.DataFrame({
        'Chicago_temp': temp_values,
        'Error': y_pred - y_true
    })
    
    plt.figure(figsize=(14, 6))
    
    
    temp_bins = pd.cut(temp_errors['Chicago_temp'], bins=10)
    
    
    ax = sns.boxplot(x=temp_bins, y=temp_errors['Error'])
    plt.title('Prediction Error by Temperature Range')
    plt.xlabel('Temperature Range (°C)')
    plt.ylabel('Error (MW)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, features, X_test_scaled, scaler_y, time_steps):
    
    feature_importance = []
    
    
    X_test_seq = np.array([X_test_scaled[i:i+time_steps] for i in range(len(X_test_scaled)-time_steps)])
    baseline_prediction = model.predict(X_test_seq)
    
    
    for i, feature in enumerate(features):
        
        X_test_modified = X_test_scaled.copy()
        X_test_modified[:, i] += 0.1  
        
        
        X_test_mod_seq = np.array([X_test_modified[j:j+time_steps] for j in range(len(X_test_modified)-time_steps)])
        
        
        modified_prediction = model.predict(X_test_mod_seq)
        
        
        importance = np.mean(np.abs(modified_prediction - baseline_prediction))
        feature_importance.append(importance)
    
    
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    
    plt.figure(figsize=(14, 10))
    plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
    plt.xlabel('Sensitivity (Mean Absolute Change in Prediction)')
    plt.ylabel('Feature')
    plt.title('GRU Feature Importance by Sensitivity Analysis (top-20)')
    plt.gca().invert_yaxis()  
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df


if __name__ == "__main__":
    try:
        
        np.random.seed(42)
        tf.random.set_seed(42)
        
        
        print("Loading data...")
        df = load_and_preprocess('FINAL_dataset.csv')
        
        
        print("Calculating features...")
        df_with_features = calculate_features(df)
        
        
        print("Splitting data...")
        train, test = split_data(df_with_features, test_start='2017-01-01', test_end='2017-12-31')
        
        
        train = train.dropna()
        
        
        features = [
            'hour', 'weekday', 'quarter', 'month', 'is_summer',
            'hour_sin', 'hour_cos', 'day_of_year',
            'Chicago_temp', 'Chicago_humidity', 'cooling_degree', 'cooling_degree_squared',
            'temp_humidity_interaction', 'extreme_heat', 'ac_peak_hours', 'summer_temp_interaction',
            'lag_720h', 'lag_1y', 'lag_2y',
            'cooling_degree_lag_24h', 'cooling_degree_lag_168h',
            'temp_rolling_24h', 'temp_rolling_72h', 'temp_rolling_max_24h', 'temp_volatility_24h'
        ]
        
        target = 'COMED_MW'
        
        
        print(f"Test set contains NaN: {test[features].isna().sum().sum() > 0}")
        if test[features].isna().sum().sum() > 0:
            
            nan_columns = test[features].columns[test[features].isna().any()].tolist()
            print(f"Columns with NaN: {nan_columns}")
            
        test = test.dropna(subset=features)
        print(f"Test set size after removing NaN: {test.shape}")
        
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]
        
        
        time_steps = 24  
        
        
        val_start = pd.to_datetime('2016-10-01')
        X_val = X_train[X_train.index >= val_start]
        y_val = y_train[y_train.index >= val_start]
        X_train_final = X_train[X_train.index < val_start]
        y_train_final = y_train[y_train.index < val_start]
        
        
        print("Training the GRU model...")
        model, scaler_X, scaler_y, time_steps = train_model(
            X_train_final, y_train_final, X_val, y_val, time_steps=time_steps
        )
        
        
        X_test_scaled = scaler_X.transform(X_test)
        
        
        X_test_seq = np.array([X_test_scaled[i:i+time_steps] for i in range(len(X_test_scaled)-time_steps)])
        
        
        print("Making predictions...")
        y_pred_scaled = model.predict(X_test_seq)
        
        
        y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        
        y_true = y_test.iloc[time_steps:].values
        
        
        print("\nModel performance:")
        rmse, mae, r2 = evaluate(y_test.iloc[time_steps:], y_pred)
        
        
        print("Building charts...")
        plot_predictions(test, y_true, y_pred)
        
        
        feature_importance = plot_feature_importance(model, features, X_test_scaled, scaler_y, time_steps)
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        
        plt.figure(figsize=(14, 6))
        
        
        test_temps = test['Chicago_temp'].iloc[time_steps:].values
        
        
        temp_data = pd.DataFrame({
            'Chicago_temp': test_temps,
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        
        temp_data['temp_group'] = pd.cut(temp_data['Chicago_temp'], bins=20)
        
        
        grouped = temp_data.groupby('temp_group')[['Actual', 'Predicted']].mean().reset_index()
        
        
        temp_cats = [str(x) for x in grouped['temp_group']]
        
        plt.plot(range(len(temp_cats)), grouped['Actual'], marker='o', label='Actual')
        plt.plot(range(len(temp_cats)), grouped['Predicted'], marker='x', label='Predicted')
        plt.title('Average Consumption by Temperature Range')
        plt.xlabel('Temperature Range (°C)')
        plt.ylabel('Average Consumption (MW)')
        plt.xticks(range(len(temp_cats)), temp_cats, rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Execution error: {str(e)}")
        import traceback
        traceback.print_exc()