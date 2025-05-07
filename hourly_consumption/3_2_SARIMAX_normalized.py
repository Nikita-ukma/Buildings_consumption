import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    
    df = df.resample('h').mean().interpolate(method='time')
    
    
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday  
    
    
    
    scaler = MinMaxScaler()
    df['COMED_MW_normalized'] = scaler.fit_transform(df[['COMED_MW']])
    
    return df.loc['2015-01-01':], scaler


def plot_consumption(df, title):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df.reset_index(), x='Datetime', y='COMED_MW_normalized')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('Normalized MW')
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


def train_evaluate_sarimax(train, test, order, seasonal_order=None):
    
    plt.close('all')
    
    
    exog_train = train[['hour', 'weekday', 'is_holiday']]
    exog_test = test[['hour', 'weekday', 'is_holiday']]
    
    
    model = SARIMAX(train['COMED_MW_normalized'], 
                   exog=exog_train,
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    results = model.fit(disp=False)
    print(results.summary())
    
    
    forecast = results.get_forecast(steps=len(test), exog=exog_test)
    forecast_mean = forecast.predicted_mean
    
    
    rmse = np.sqrt(mean_squared_error(test['COMED_MW_normalized'], forecast_mean))
    mae = mean_absolute_error(test['COMED_MW_normalized'], forecast_mean)
    
    
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['COMED_MW_normalized'], label='Training Data')
    plt.plot(test.index, test['COMED_MW_normalized'], label='Actual Values')
    plt.plot(test.index, forecast_mean, label='Forecast', alpha=0.7)
    plt.fill_between(test.index,
                    forecast.conf_int().iloc[:, 0],
                    forecast.conf_int().iloc[:, 1],
                    color='gray', alpha=0.2)
    plt.title(f'SARIMAX{order}{seasonal_order} Forecast\nRMSE: {rmse:.4f}, MAE: {mae:.4f} (normalized)')
    plt.legend()
    plt.grid()
    plt.show()
    
    return results


if __name__ == "__main__":
    
    df, scaler = load_and_prepare_data("energy_data_prepared.csv")
    plot_consumption(df, 'Normalized COMED Hourly Energy Consumption (2015-2018)')
    
    
    train = df.loc[:'2016-12-31']
    test = df.loc['2017-01-01':]
    
    
    print("\nStationarity Analysis:")
    stationary_series, d = make_stationary(train['COMED_MW_normalized'])
    
    
    print("\nACF/PACF Analysis:")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(stationary_series, lags=48, ax=ax1)
    plot_pacf(stationary_series, lags=48, ax=ax2)
    plt.show()
    
    
    order = (3, 0, 6)  
    
    
    model_results = train_evaluate_sarimax(
        train,
        test,
        order=order,
        seasonal_order=(1, 1, 1, 24)  
    )
    
    
    test['Prediction_normalized'] = model_results.predict(
        start=test.index[0], 
        end=test.index[-1],
        exog=test[['hour', 'weekday', 'is_holiday']]
    )
    
    
    test['COMED_MW'] = scaler.inverse_transform(test[['COMED_MW_normalized']])
    test['Prediction'] = scaler.inverse_transform(test[['Prediction_normalized']])
    
    
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
    plt.title('Hourly Distribution: Actual vs Predicted (Original Scale)')
    plt.ylabel('MW')
    plt.grid()
    plt.show()