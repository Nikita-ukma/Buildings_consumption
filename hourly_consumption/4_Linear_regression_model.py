'''
RMSE: 1149.67 MW
MAE: 894.72 MW
R²: 0.7012
Winter RMSE: 1011.26 MW
Spring RMSE: 897.15 MW
Summer RMSE: 1422.34 MW
Fall RMSE: 1196.10 MW
                      Feature    Importance
11     cooling_degree_squared  35770.635077
15    summer_temp_interaction  26305.978015
4                   is_summer  25892.454048
8                Chicago_temp  17951.072660
10             cooling_degree  17951.072660
3                       month    962.073482
12  temp_humidity_interaction    957.503190
9            Chicago_humidity    828.031216
5                    hour_sin    767.658797
2                     quarter    565.207435
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
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

# 4. Train the linear regression model
def train_model(X_train, y_train, X_val=None, y_val=None):
    # Standardize features for better linear regression performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and train the Linear Regression model
    model = LinearRegression(n_jobs=-1)  # Use all available processors
    
    if X_val is not None and y_val is not None:
        # Combine training and validation data for final model
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])
        X_combined_scaled = scaler.transform(X_combined)
        model.fit(X_combined_scaled, y_combined)
        print(f"Model trained on {len(X_combined)} samples")
    else:
        # Train on just the training data
        model.fit(X_train_scaled, y_train)
        print(f"Model trained on {len(X_train)} samples")
    
    return model, scaler

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

# Function to display coefficients for Linear Regression model
def plot_feature_importance(model, features):
    # Extract feature coefficients
    coefficients = model.coef_
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(coefficients)  # Using absolute value for importance
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance (coefficients)
    plt.figure(figsize=(14, 10))
    plt.barh(feature_importance_df['Feature'][:20], feature_importance_df['Importance'][:20])
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Linear Regression Feature Importance (top-20)')
    plt.gca().invert_yaxis()  # Display most important feature on top
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df
# Add visualization of actual vs predicted values for the first two weeks of January
def plot_january_first_two_weeks(test, y_test, y_pred):
    # Convert y_pred to a pandas Series with the same index as test
    y_pred_series = pd.Series(y_pred, index=test.index)
    
    # Filter for first two weeks of January 2017
    jan_start = '2017-01-01'
    jan_end = '2017-01-14'
    
    # Get the data for the period
    jan_data = test.loc[jan_start:jan_end].copy()
    
    # Add actual and predicted values to the dataframe
    jan_data['Actual'] = y_test.loc[jan_start:jan_end]
    jan_data['Predicted'] = y_pred_series.loc[jan_start:jan_end]
    
    # Calculate hourly error
    jan_data['Error'] = jan_data['Predicted'] - jan_data['Actual']
    jan_data['AbsError'] = abs(jan_data['Error'])
    
    # Plot the comparison
    plt.figure(figsize=(16, 10))
    
    # First subplot: Actual vs Predicted
    plt.subplot(2, 1, 1)
    plt.plot(jan_data.index, jan_data['Actual'], label='Actual', linewidth=2)
    plt.plot(jan_data.index, jan_data['Predicted'], label='Predicted', linewidth=2, alpha=0.7)
    plt.title('Actual vs Predicted Energy Consumption - First Two Weeks of January 2017')
    plt.ylabel('Energy Consumption (MW)')
    plt.legend()
    plt.grid(True)
    
    # Second subplot: Error
    plt.subplot(2, 1, 2)
    plt.bar(jan_data.index, jan_data['Error'], color='darkred', alpha=0.6)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Prediction Error (Predicted - Actual)')
    plt.ylabel('Error (MW)')
    plt.xlabel('Date')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics for this period
    mean_error = jan_data['Error'].mean()
    mean_abs_error = jan_data['AbsError'].mean()
    rmse_value = np.sqrt(np.mean(jan_data['Error'] ** 2))
    
    print("\nFirst Two Weeks of January 2017 Statistics:")
    print(f"Mean Error: {mean_error:.2f} MW")
    print(f"Mean Absolute Error: {mean_abs_error:.2f} MW")
    print(f"RMSE: {rmse_value:.2f} MW")
    
    # Daily pattern analysis
    plt.figure(figsize=(12, 6))
    
    # Group data by hour of day
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
        
        model, scaler = train_model(X_train_final, y_train_final, X_val, y_val)
        
        # Prediction - first scale test data with the same scaler
        print("Making predictions...")
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        
        # Evaluation
        print("\nModel performance:")
        evaluate(y_test, y_pred)
        
        # Visualization
        print("Building charts...")
        plot_predictions(test, y_test, y_pred)
        
        # Feature importance for Linear Regression (coefficients)
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
        grouped = temp_data.groupby('temp_group')[['Actual', 'Predicted']].mean().reset_index()
        
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
                # Add this after other visualization calls
        print("Analyzing first two weeks of January...")
        plot_january_first_two_weeks(test, y_test, y_pred)
    except Exception as e:
        print(f"Execution error: {str(e)}")
        import traceback
        traceback.print_exc()
