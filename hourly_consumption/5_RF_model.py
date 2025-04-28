'''
RMSE: 783.74 MW
MAE: 566.03 MW
R²: 0.8611
Winter RMSE: 753.17 MW
Spring RMSE: 628.11 MW
Summer RMSE: 983.60 MW
Fall RMSE: 725.11 MW

Top 10 most important features:
                    Feature  Importance
0                      hour    0.111895
8              Chicago_temp    0.109714
10           cooling_degree    0.108439
11   cooling_degree_squared    0.105940
17                   lag_1y    0.080273
15  summer_temp_interaction    0.063341
5                  hour_sin    0.062685
1                   weekday    0.059821
21         temp_rolling_24h    0.058638
23     temp_rolling_max_24h    0.046184
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# 1. Loading and preprocessing data
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Add quarter of year feature
    df['quarter'] = df.index.quarter
    
    # Add month as categorical feature
    df['month'] = df.index.month
    
    # Add summer period indicator (June-August)
    df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
    
    # Add day of year for seasonality
    df['day_of_year'] = df.index.dayofyear
    
    # Add hour of day as sinusoidal function to capture cyclical patterns
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # Calculate cooling degree days (CDD)
    # Base temperature at which AC is typically turned on
    base_temp = 21  # This value may need calibration
    df['cooling_degree'] = df['Chicago_temp'].apply(lambda x: max(0, x - base_temp))
    
    # Square of cooling degree to account for non-linear energy consumption at high temperatures
    df['cooling_degree_squared'] = df['cooling_degree'] ** 2
    
    # Add interaction between humidity and temperature (important for "mugginess" feeling)
    df['temp_humidity_interaction'] = df['Chicago_temp'] * df['Chicago_humidity'] / 100
    
    # Add binary feature for extreme temperatures (>30°C)
    df['extreme_heat'] = (df['Chicago_temp'] > 30).astype(int)
    
    # Add feature for AC peak usage hours (12-18 hours)
    df['ac_peak_hours'] = ((df.index.hour >= 12) & (df.index.hour <= 18)).astype(int)
    
    # Interaction between summer period and temperature
    df['summer_temp_interaction'] = df['is_summer'] * df['Chicago_temp']
    
    # Add weekday feature
    df['weekday'] = df.index.dayofweek
    
    # Add hour feature
    df['hour'] = df.index.hour
    
    return df

# 2. Calculate lag features WITHOUT data leakage
def calculate_features(df):
    # Create a copy for safely adding features
    features_df = df.copy()

    # Lag of 720 hours (30 days)
    features_df['lag_720h'] = features_df['COMED_MW'].shift(720)
    
    # Lag of previous year (8760 hours = 365 days)
    features_df['lag_1y'] = features_df['COMED_MW'].shift(8760)
    
    # Lag of two years ago (17520 hours = 730 days)
    features_df['lag_2y'] = features_df['COMED_MW'].shift(17520)
    
    # Add lags for cooling_degree
    features_df['cooling_degree_lag_24h'] = features_df['cooling_degree'].shift(24)
    features_df['cooling_degree_lag_168h'] = features_df['cooling_degree'].shift(168)
    
    # Add rolling averages for temperature
    features_df['temp_rolling_24h'] = features_df['Chicago_temp'].rolling(window=24).mean().shift(1)
    features_df['temp_rolling_72h'] = features_df['Chicago_temp'].rolling(window=72).mean().shift(1)
    
    # Add rolling maximum temperature
    features_df['temp_rolling_max_24h'] = features_df['Chicago_temp'].rolling(window=24).max().shift(1)
    
    # Add rolling temperature difference (volatile weather)
    features_df['temp_volatility_24h'] = features_df['Chicago_temp'].rolling(window=24).std().shift(1)
    
    return features_df

# 3. Split data into training (before 2017) and test (2017) sets
def split_data(df, test_start='2017-01-01', test_end='2017-12-31'):
    train = df.loc[:test_start].iloc[:-1]  # up to the start of 2017, not including the test day itself
    test = df.loc[test_start:test_end]     # all of 2017
    return train, test

# 4. Train the model
def train_model(X_train, y_train, X_val=None, y_val=None):
    # Random Forest parameters optimized for time series forecasting
    params = {
        'n_estimators': 200,           # Number of trees in the forest
        'max_depth': 20,               # Maximum depth of the trees
        'min_samples_split': 10,       # Minimum samples required to split a node
        'min_samples_leaf': 5,         # Minimum samples required at each leaf node
        'max_features': 'sqrt',        # Number of features to consider at each split
        'bootstrap': True,             # Whether bootstrap samples are used
        'n_jobs': -1,                  # Use all available processors
        'random_state': 42,            # For reproducibility
        'verbose': 1,                  # Output training details
        'warm_start': False            # Whether to reuse previous solution
    }
    
    # Create and train the Random Forest model
    model = RandomForestRegressor(**params)
    
    if X_val is not None and y_val is not None:
        # Combine training and validation data for final model
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])
        model.fit(X_combined, y_combined)
        print(f"Model trained on {len(X_combined)} samples")
    else:
        # Train on just the training data
        model.fit(X_train, y_train)
        print(f"Model trained on {len(X_train)} samples")
    
    return model

# 5. Evaluate the model
def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW") 
    print(f"R²: {r2:.4f}")
    
    # Calculate error by season
    # Create DataFrame with sequential indices to avoid duplication
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

# 6. Visualization
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
    
    # Additional visualization - monthly error
    plt.figure(figsize=(16, 6))
    
    # Create DataFrame without index for safe resampling
    y_true_series = pd.Series(y_true, index=test.index)
    y_pred_series = pd.Series(y_pred, index=test.index)
    
    # Resample to monthly data
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
    
    # Add visualization of errors by temperature
    temp_errors = pd.DataFrame({
        'Chicago_temp': test['Chicago_temp'].values,
        'Error': y_pred - y_true.values
    })
    
    plt.figure(figsize=(14, 6))
    
    # Create temperature bins
    temp_bins = pd.cut(temp_errors['Chicago_temp'], bins=10)
    
    # Use boxplot with seaborn
    ax = sns.boxplot(x=temp_bins, y=temp_errors['Error'])
    plt.title('Prediction Error by Temperature Range')
    plt.xlabel('Temperature Range (°C)')
    plt.ylabel('Error (MW)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# Function to display feature importance for Random Forest
def plot_feature_importance(model, features):
    # Extract feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(14, 10))
    plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Random Forest Feature Importance (top-20)')
    plt.gca().invert_yaxis()  # Display most important feature on top
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df

# Main execution part
if __name__ == "__main__":
    try:
        # Load data
        print("Loading data...")
        df = load_and_preprocess('data/FINAL_dataset.csv')
        
        # Calculate all features for the entire dataset
        print("Calculating features...")
        df_with_features = calculate_features(df)
        
        # Split into training (before 2017) and test (2017) sets
        print("Splitting data...")
        train, test = split_data(df_with_features, test_start='2017-01-01', test_end='2017-12-31')
        
        # Remove rows with missing values
        train = train.dropna()
        
        # Define features and target variable
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
        
        # Check that the test set has all required features without NaN
        print(f"Test set contains NaN: {test[features].isna().sum().sum() > 0}")
        if test[features].isna().sum().sum() > 0:
            # Print columns with NaN values
            nan_columns = test[features].columns[test[features].isna().any()].tolist()
            print(f"Columns with NaN: {nan_columns}")
            
        test = test.dropna(subset=features)
        print(f"Test set size after removing NaN: {test.shape}")
        
        X_train = train[features]
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]
        
        # Train the model
        print("Training the model...")
        
        # Create validation set from the last 3 months of training data
        val_start = pd.to_datetime('2016-10-01')
        X_val = X_train[X_train.index >= val_start]
        y_val = y_train[y_train.index >= val_start]
        X_train_final = X_train[X_train.index < val_start]
        y_train_final = y_train[y_train.index < val_start]
        
        model = train_model(X_train_final, y_train_final, X_val, y_val)
        
        # Prediction
        print("Making predictions...")
        y_pred = model.predict(X_test)
        
        # Evaluation
        print("\nModel performance:")
        evaluate(y_test, y_pred)
        
        # Visualization
        print("Building charts...")
        plot_predictions(test, y_test, y_pred)
        
        # Feature importance for Random Forest
        feature_importance = plot_feature_importance(model, features)
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # Analysis of prediction dependence on temperature
        plt.figure(figsize=(14, 6))
        
        # Create new DataFrame for analysis
        temp_data = pd.DataFrame({
            'Chicago_temp': test['Chicago_temp'].values,
            'Actual': y_test.values,
            'Predicted': y_pred
        })
        
        # Create temperature groups
        temp_data['temp_group'] = pd.cut(temp_data['Chicago_temp'], bins=20)
        
        # Group data by temperature groups
        grouped = temp_data.groupby('temp_group', observed=False)[['Actual', 'Predicted']].mean().reset_index()
        
        # Convert categorical bins to strings for proper display on the chart
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
        # Аналіз перших двох тижнів січня
        print("\nАналіз перших двох тижнів січня 2017:")
        jan_start = '2017-01-01'
        jan_end = '2017-01-14'
        jan_data = test[(test.index >= jan_start) & (test.index <= jan_end)]
        jan_X = jan_data[features]
        jan_y_true = jan_data[target]
        jan_y_pred = model.predict(jan_X)

        # Обчислення метрик для перших двох тижнів
        jan_rmse = mean_squared_error(jan_y_true, jan_y_pred, squared=False)
        jan_mae = mean_absolute_error(jan_y_true, jan_y_pred)
        jan_r2 = r2_score(jan_y_true, jan_y_pred)

        print(f"Перші два тижні січня:")
        print(f"RMSE: {jan_rmse:.2f} MW")
        print(f"MAE: {jan_mae:.2f} MW") 
        print(f"R²: {jan_r2:.4f}")

        # Візуалізація прогнозу для перших двох тижнів
        plt.figure(figsize=(16, 8))
        plt.plot(jan_data.index, jan_y_true, label='Actual')
        plt.plot(jan_data.index, jan_y_pred, label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted Energy Consumption (First Two Weeks of January 2017)')
        plt.xlabel('Date')
        plt.ylabel('MW')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Аналіз похибок по годинах доби для перших двох тижнів
        jan_analysis = pd.DataFrame({
            'Datetime': jan_data.index,
            'Hour': jan_data.index.hour,
            'Actual': jan_y_true,
            'Predicted': jan_y_pred,
            'Error': jan_y_pred - jan_y_true,
            'AbsError': np.abs(jan_y_pred - jan_y_true),
            'Temperature': jan_data['Chicago_temp']
        })

        # Середня похибка по годинах
        hourly_error = jan_analysis.groupby('Hour')['AbsError'].mean()

        plt.figure(figsize=(14, 6))
        plt.bar(hourly_error.index, hourly_error.values, color='skyblue')
        plt.title('Average Absolute Error by Hour (First Two Weeks of January 2017)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Absolute Error (MW)')
        plt.xticks(range(0, 24))
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

        # Аналіз залежності навантаження від температури для перших двох тижнів
        plt.figure(figsize=(14, 6))
        plt.scatter(jan_analysis['Temperature'], jan_analysis['Actual'], label='Actual', alpha=0.6)
        plt.scatter(jan_analysis['Temperature'], jan_analysis['Predicted'], label='Predicted', alpha=0.6)
        plt.title('Consumption vs Temperature (First Two Weeks of January 2017)')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Consumption (MW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Аналіз середнього споживання по днях тижня
        jan_analysis['Weekday'] = jan_analysis['Datetime'].dt.dayofweek
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        jan_analysis['WeekdayName'] = jan_analysis['Weekday'].apply(lambda x: weekday_names[x])

        weekday_actual = jan_analysis.groupby('WeekdayName')['Actual'].mean()
        weekday_predicted = jan_analysis.groupby('WeekdayName')['Predicted'].mean()

        plt.figure(figsize=(14, 6))
        x = np.arange(len(weekday_names))
        width = 0.35
        plt.bar(x - width/2, weekday_actual.reindex(weekday_names), width, label='Actual')
        plt.bar(x + width/2, weekday_predicted.reindex(weekday_names), width, label='Predicted')
        plt.title('Average Daily Consumption by Day of Week (First Two Weeks of January 2017)')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Consumption (MW)')
        plt.xticks(x, weekday_names)
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Execution error: {str(e)}")
        import traceback
        traceback.print_exc()