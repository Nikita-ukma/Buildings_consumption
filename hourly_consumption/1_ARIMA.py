import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    df = df.resample('h').mean().interpolate(method='time')
    return df.loc['2015-01-01':]

def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    return result[1] <= 0.05

def plot_results(train, test, predictions, title):
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Values')
    plt.plot(test.index, predictions, label='Forecast', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Consumption (MW)')
    plt.legend()
    plt.grid()
    plt.show()

def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAE: {mae:.2f} MW") 
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    errors_df = pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred.values,
        'error': np.abs(y_pred - y_true.values),
        'month': pd.DatetimeIndex(y_true.index).month,
        'hour': pd.DatetimeIndex(y_true.index).hour,
        'day_of_week': pd.DatetimeIndex(y_true.index).dayofweek
    }, index=y_true.index)
    
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Fall': [9, 10, 11]
    }
    
    errors_df['season'] = errors_df['month'].apply(
        lambda x: next(season for season, months in seasons.items() if x in months)
    )
    
    seasonal_metrics = {}
    print("\nSeasonal Error Assessment:")
    for season in seasons.keys():
        season_data = errors_df[errors_df['season'] == season]
        season_rmse = mean_squared_error(season_data['y_true'], season_data['y_pred'], squared=False)
        season_mae = mean_absolute_error(season_data['y_true'], season_data['y_pred'])
        season_mape = np.mean(np.abs((season_data['y_true'] - season_data['y_pred']) / season_data['y_true'])) * 100
        seasonal_metrics[season] = {
            'RMSE': season_rmse,
            'MAE': season_mae,
            'MAPE': season_mape
        }
        print(f"{season}:")
        print(f"  RMSE: {season_rmse:.2f} MW")
        print(f"  MAE: {season_mae:.2f} MW")
        print(f"  MAPE: {season_mape:.2f}%")
    
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='season', y='error', data=errors_df, order=['Winter', 'Spring', 'Summer', 'Fall'])
    plt.title('Error Distribution by Season')
    plt.ylabel('Absolute Error (MW)')
    plt.grid(axis='y')
    plt.show()
    
    plt.figure(figsize=(18, 8))
    sns.boxplot(x='hour', y='error', data=errors_df)
    plt.title('Error Distribution by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Absolute Error (MW)')
    plt.grid(axis='y')
    plt.show()
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(16, 8))
    sns.boxplot(x='day_of_week', y='error', data=errors_df)
    plt.xticks(range(7), days)
    plt.title('Error Distribution by Day of Week')
    plt.ylabel('Absolute Error (MW)')
    plt.grid(axis='y')
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual Values (MW)')
    plt.ylabel('Predicted Values (MW)')
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(14, 8))
    errors = y_pred - y_true
    sns.histplot(errors, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Forecast Error Distribution')
    plt.xlabel('Error (MW)')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
    monthly_errors = errors_df.groupby(errors_df.index.month)['error'].mean()
    months_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    
    plt.figure(figsize=(14, 8))
    monthly_errors.plot(kind='bar')
    plt.title('Mean Absolute Error by Month')
    plt.xticks(range(12), months_names, rotation=45)
    plt.ylabel('Mean Absolute Error (MW)')
    plt.grid(axis='y')
    plt.show()
    
    return seasonal_metrics

if __name__ == "__main__":
    df = load_data("COMED_hourly.csv")
    
    train = df.loc[:'2016-12-31']
    test = df.loc['2017-01-01':]
    
    print("Checking stationarity:")
    if not check_stationarity(train['COMED_MW']):
        print("\nSeries is not stationary. Applying differencing...")
        train_diff = train['COMED_MW'].diff().dropna()
        check_stationarity(train_diff)
    else:
        print("\nSeries is stationary. No differencing needed.")
    
    print("\nACF/PACF Analysis:")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(train['COMED_MW'], lags=48, ax=ax1)
    plot_pacf(train['COMED_MW'], lags=48, ax=ax2)
    plt.show()
    
    p, d, q = 2, 0, 2
    
    print(f"\nTraining ARIMA({p},{d},{q}) model...")
    model = ARIMA(train['COMED_MW'], order=(p, d, q))
    model_fit = model.fit()
    print(model_fit.summary())
    
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    print("\nDetailed Model Evaluation:")
    seasonal_metrics = evaluate(test['COMED_MW'], forecast_mean)
    
    rmse = mean_squared_error(test['COMED_MW'], forecast_mean, squared=False)
    mae = mean_absolute_error(test['COMED_MW'], forecast_mean)
    r2 = r2_score(test['COMED_MW'], forecast_mean)
    plot_results(train['COMED_MW'], test['COMED_MW'], forecast_mean,
               f'ARIMA({p},{d},{q}) Forecast\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}')
    
    short_test = test.loc['2017-01-01':'2017-01-14']
    short_forecast = forecast_mean.loc['2017-01-01':'2017-01-14']
    
    plt.figure(figsize=(14, 7))
    plt.plot(short_test.index, short_test, label='Actual Values')
    plt.plot(short_test.index, short_forecast, label='Forecast')
    plt.fill_between(short_test.index,
                    conf_int.loc['2017-01-01':'2017-01-14'].iloc[:, 0],
                    conf_int.loc['2017-01-01':'2017-01-14'].iloc[:, 1],
                    color='gray', alpha=0.2)
    plt.title('ARIMA Forecast (First 2 Weeks of January 2017)')
    plt.legend()
    plt.grid()
    plt.show()