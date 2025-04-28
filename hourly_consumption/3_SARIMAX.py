import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Data Loading & Preparation
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Resample and handle missing values
    df = df.resample('h').mean().interpolate(method='time')
    
    # Prepare exogenous variables
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday  # Monday=0, Sunday=6
    df['month'] = df.index.month
    
    # Створення циклічних ознак для годин (щоб урахувати циклічність часу)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    # Add season feature
    df['season'] = df.index.month % 12 // 3
    # 0: Winter (Dec, Jan, Feb)
    # 1: Spring (Mar, Apr, May)
    # 2: Summer (Jun, Jul, Aug)
    # 3: Fall (Sep, Oct, Nov)
    
    # Assign season names for easier interpretation
    season_map = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Fall'}
    df['season_name'] = df['season'].map(season_map)
    
    # is_holiday is already in the data
    
    return df.loc['2015-01-01':]

# 2. Enhanced Visualization
def plot_consumption(df, title):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df.reset_index(), x='Datetime', y='COMED_MW')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('MW')
    plt.grid(True)
    plt.show()

# 3. Stationarity Check & Differencing
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    return result[1] <= 0.05

def make_stationary(series):
    d = 0
    while not check_stationarity(series) and d < 2:
        d += 1
        series = series.diff().dropna()
        print(f'\nAfter {d} order differencing:')
        check_stationarity(series)
    return series, d

# 4. Enhanced evaluation metrics
def calculate_metrics(actual, predicted):
    """Calculate comprehensive evaluation metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    # MAPE calculation with handling for zeros
    mape = np.mean(np.abs((actual - predicted) / np.maximum(np.ones(len(actual)), np.abs(actual)))) * 100
    r2 = r2_score(actual, predicted)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2
    }
    
    return metrics

def calculate_seasonal_metrics(test_df, forecast_mean):
    """Calculate metrics by season"""
    test_df = test_df.copy()
    test_df['Prediction'] = forecast_mean
    
    seasonal_metrics = {}
    for season_id, season_name in [(0, 'Winter'), (1, 'Spring'), (2, 'Summer'), (3, 'Fall')]:
        season_data = test_df[test_df['season'] == season_id]
        if len(season_data) > 0:
            season_rmse = np.sqrt(mean_squared_error(season_data['COMED_MW'], season_data['Prediction']))
            seasonal_metrics[f'{season_name}_RMSE'] = season_rmse
    
    return seasonal_metrics

# 5. Model Training & Evaluation with Exogenous Variables
def train_evaluate_sarimax(train, test, order, seasonal_order=None):
    # Clear previous figures
    plt.close('all')
    
    # Prepare exogenous variables - FIXED: corrected hour_cos, hour_sin
    exog_train = train[['hour', 'weekday', 'is_holiday', 'month', 'hour_sin', 'hour_cos']]
    exog_test = test[['hour', 'weekday', 'is_holiday', 'month', 'hour_sin', 'hour_cos']]
    
    # Model training
    model = SARIMAX(train['COMED_MW'], 
                   exog=exog_train,
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    results = model.fit(disp=False)
    print(results.summary())
    
    # Forecasting
    forecast = results.get_forecast(steps=len(test), exog=exog_test)
    forecast_mean = forecast.predicted_mean
    
    # Comprehensive evaluation
    metrics = calculate_metrics(test['COMED_MW'], forecast_mean)
    seasonal_metrics = calculate_seasonal_metrics(test, forecast_mean)
    
    # Combine all metrics
    all_metrics = {**metrics, **seasonal_metrics}
    
    # Print detailed metrics
    print("\n===== MODEL EVALUATION METRICS =====")
    for metric_name, value in all_metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['COMED_MW'], label='Training Data')
    plt.plot(test.index, test['COMED_MW'], label='Actual Values')
    plt.plot(test.index, forecast_mean, label='Forecast', alpha=0.7)
    plt.fill_between(test.index,
                    forecast.conf_int().iloc[:, 0],
                    forecast.conf_int().iloc[:, 1],
                    color='gray', alpha=0.2)
    
    # Include key metrics in title
    metrics_text = f"RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE']:.2f}%, R²: {metrics['R²']:.3f}"
    plt.title(f'SARIMAX{order}{seasonal_order} Forecast\n{metrics_text}')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Create seasonal performance visualization
    plot_seasonal_performance(test, forecast_mean, seasonal_metrics)
    
    return results, forecast_mean, all_metrics

# 6. New function for seasonal performance visualization
def plot_seasonal_performance(test, forecast_mean, seasonal_metrics):
    """Create visualization of model performance by season"""
    test = test.copy()
    test['Prediction'] = forecast_mean
    
    # Plot seasonal RMSE as bar chart
    plt.figure(figsize=(10, 6))
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    values = [seasonal_metrics.get(f'{season}_RMSE', 0) for season in seasons]
    
    bars = plt.bar(seasons, values, color=['lightblue', 'lightgreen', 'salmon', 'wheat'])
    
    # Add values on top of bars
    for bar, value in zip(bars, values):
        if value > 0:
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 5, 
                    f'{value:.1f}', 
                    ha='center')
    
    plt.title('RMSE by Season')
    plt.ylabel('RMSE (MW)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(values) * 1.2)  # Add some space for the text
    plt.show()
    
    # Create boxplot of errors by season
    test['Error'] = test['COMED_MW'] - test['Prediction']
    test['AbsError'] = np.abs(test['Error'])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='season_name', y='AbsError', data=test, order=seasons)
    plt.title('Distribution of Absolute Errors by Season')
    plt.xlabel('Season')
    plt.ylabel('Absolute Error (MW)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 7. Main Workflow
if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data("FINAL_dataset.csv")
    plot_consumption(df, 'COMED Hourly Energy Consumption (2015-2018)')
    
    # Train-test split
    train = df.loc[:'2016-12-31']
    test = df.loc['2017-01-01':]
    
    # Stationarity check
    print("\nStationarity Analysis:")
    stationary_series, d = make_stationary(train['COMED_MW'])
    
    # Parameter Selection
    print("\nACF/PACF Analysis:")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(stationary_series, lags=48, ax=ax1)
    plot_pacf(stationary_series, lags=48, ax=ax2)
    plt.show()
    
    # Best parameters from previous analysis
    order = (2, 0, 2)  # Replace with optimal parameters from ACF/PACF
    
    # Model training and evaluation with exogenous variables
    # FIXED: Capturing forecast_mean and metrics from function
    model_results, forecast_mean, metrics = train_evaluate_sarimax(
        train,
        test,
        order=order,
        seasonal_order=(1, 1, 1, 24)  # Daily seasonality
    )
    
    # Hourly RMSE analysis
    test['Prediction'] = forecast_mean
    
    # Prepare data for boxplot - need to melt the dataframe correctly
    plot_data = test.reset_index()[['hour', 'COMED_MW', 'Prediction']].melt(
        id_vars=['hour'], 
        value_vars=['COMED_MW', 'Prediction'],
        var_name='Type',
        value_name='MW'
    )
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(
        x='hour', 
        y='MW', 
        data=plot_data, 
        hue='Type'
    )
    plt.title('Hourly Distribution: Actual vs Predicted')
    plt.ylabel('MW')
    plt.grid()
    plt.show()
    
    # Calculate errors by hour
    test['Error'] = test['COMED_MW'] - test['Prediction']
    hourly_rmse = test.groupby('hour')['Error'].apply(lambda x: np.sqrt(np.mean(x**2)))
    
    plt.figure(figsize=(14, 6))
    hourly_rmse.plot(kind='bar')
    plt.title('RMSE by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('RMSE (MW)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()