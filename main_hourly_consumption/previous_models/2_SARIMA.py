import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    df = df.resample('h').mean().interpolate(method='time')
    return df.loc['2015-01-01':]

def plot_consumption(df, title):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df.reset_index(), x='Datetime', y='COMED_MW')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('MW')
    plt.grid(True)
    plt.show()

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
        'y_pred': y_pred,
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

def train_evaluate_sarima(train, test, order, seasonal_order=None):
    plt.close('all')
    
    model = SARIMAX(train, 
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    results = model.fit(disp=False)
    print(results.summary())
    
    forecast = results.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    
    print("\nDetailed Model Evaluation:")
    seasonal_metrics = evaluate(test, forecast_mean)
    
    rmse = mean_squared_error(test, forecast_mean, squared=False)
    mae = mean_absolute_error(test, forecast_mean)
    r2 = r2_score(test, forecast_mean)
    
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Actual Values')
    plt.plot(test.index, forecast_mean, label='Forecast', alpha=0.7)
    plt.fill_between(test.index,
                    forecast.conf_int().iloc[:, 0],
                    forecast.conf_int().iloc[:, 1],
                    color='gray', alpha=0.2)
    
    if seasonal_order:
        plt.title(f'SARIMA{order}{seasonal_order} Forecast\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}')
    else:
        plt.title(f'ARIMA{order} Forecast\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}')
    plt.legend()
    plt.grid()
    plt.show()
    
    short_test = test.loc['2017-01-01':'2017-01-14']
    short_forecast = forecast_mean.loc['2017-01-01':'2017-01-14']
    
    plt.figure(figsize=(14, 7))
    plt.plot(short_test.index, short_test, label='Actual Values')
    plt.plot(short_test.index, short_forecast, label='Forecast')
    plt.fill_between(short_test.index,
                    forecast.conf_int().loc['2017-01-01':'2017-01-14'].iloc[:, 0],
                    forecast.conf_int().loc['2017-01-01':'2017-01-14'].iloc[:, 1],
                    color='gray', alpha=0.2)
    
    if seasonal_order:
        plt.title(f'SARIMA{order}{seasonal_order} (First 2 Weeks of January 2017)')
    else:
        plt.title(f'ARIMA{order} (First 2 Weeks of January 2017)')
    plt.legend()
    plt.grid()
    plt.show()
    
    return results, forecast_mean, seasonal_metrics

if __name__ == "__main__":
    df = load_and_prepare_data("data/COMED_hourly.csv")
    plot_consumption(df, 'COMED Hourly Energy Consumption (2015-2018)')
    
    train = df.loc[:'2016-12-31']
    test = df.loc['2017-01-01':]
    
    print("\nStationarity Analysis:")
    stationary_series, d = make_stationary(train['COMED_MW'])
    
    print("\nACF/PACF Analysis:")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(stationary_series, lags=48, ax=ax1)
    plot_pacf(stationary_series, lags=48, ax=ax2)
    plt.show()
    
    order = (3, 0, 6)
    seasonal_order = (1, 1, 1, 24)
    
    model_results, forecast, seasonal_metrics = train_evaluate_sarima(
        train['COMED_MW'],
        test['COMED_MW'],
        order=order,
        seasonal_order=seasonal_order
    )
    
    print("\nHourly Pattern Analysis:")
    test_with_predictions = test.copy()
    test_with_predictions['Hour'] = test_with_predictions.index.hour
    test_with_predictions['Prediction'] = forecast
    test_with_predictions['Error'] = np.abs(test_with_predictions['COMED_MW'] - test_with_predictions['Prediction'])
    
    plt.figure(figsize=(16, 8))
    hourly_actual = test_with_predictions.groupby('Hour')['COMED_MW'].mean()
    hourly_predicted = test_with_predictions.groupby('Hour')['Prediction'].mean()
    
    plt.plot(hourly_actual.index, hourly_actual, 'b-', label='Actual')
    plt.plot(hourly_predicted.index, hourly_predicted, 'r--', label='Predicted')
    plt.title('Average Hourly Consumption: Actual vs Forecast')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Consumption (MW)')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(16, 8))
    hourly_error = test_with_predictions.groupby('Hour')['Error'].mean()
    plt.bar(hourly_error.index, hourly_error)
    plt.title('Mean Forecast Error by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Mean Absolute Error (MW)')
    plt.grid(axis='y')
    plt.show()
    
    print("\nSeasonal Deviation Assessment:")
    print(f"Summer RMSE: {seasonal_metrics['Summer']['RMSE']:.2f} MW")
    print(f"Spring RMSE: {seasonal_metrics['Spring']['RMSE']:.2f} MW") 
    print(f"Winter RMSE: {seasonal_metrics['Winter']['RMSE']:.2f} MW")
    print(f"Fall RMSE: {seasonal_metrics['Fall']['RMSE']:.2f} MW")

    print("\nSeasonal MAPE Assessment:")
    print(f"Summer MAPE: {seasonal_metrics['Summer']['MAPE']:.2f}%")
    print(f"Spring MAPE: {seasonal_metrics['Spring']['MAPE']:.2f}%")
    print(f"Winter MAPE: {seasonal_metrics['Winter']['MAPE']:.2f}%")
    print(f"Fall MAPE: {seasonal_metrics['Fall']['MAPE']:.2f}%")