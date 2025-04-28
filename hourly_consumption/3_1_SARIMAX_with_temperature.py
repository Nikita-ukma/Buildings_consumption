import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler

# 1. Data Loading & Preparation with Normalization
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index()
    
    # Normalize COMED_MW to create 'value' column
    scaler = MinMaxScaler()
    df['value'] = scaler.fit_transform(df[['COMED_MW']])
    
    # Handle missing values
    df = df.interpolate(method='time')
    
    return df, scaler

# 2. Enhanced Visualization
def plot_consumption(df, title):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df.reset_index(), x='Datetime', y='COMED_MW')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('MW')
    plt.grid(True)
    plt.show()

# 4. Model Training & Evaluation with Extended Exogenous Variables
def train_evaluate_sarimax(train, test, order, seasonal_order=None):
    plt.close('all')
    
    # Prepare all exogenous variables
    exog_cols = ['Chicago_temp', 'Chicago_humidity', 'Chicago_pressure', 
                 'Chicago_wind_dir', 'Chicago_wind_speed', 'hour', 'weekday', 'is_holiday']
    
    exog_train = train[exog_cols]
    exog_test = test[exog_cols]
    
    # Model training
    model = SARIMAX(train['COMED_MW'], 
                   exog=exog_train,
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    results = model.fit(disp=False, maxiter=100)
    print(results.summary())
    
    # Forecasting
    forecast = results.get_forecast(steps=len(test), exog=exog_test)
    forecast_mean = forecast.predicted_mean
    
    # Evaluation
    rmse = np.sqrt(mean_squared_error(test['COMED_MW'], forecast_mean))
    mae = mean_absolute_error(test['COMED_MW'], forecast_mean)
    
    # Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['COMED_MW'], label='Training Data')
    plt.plot(test.index, test['COMED_MW'], label='Actual Values')
    plt.plot(test.index, forecast_mean, label='Forecast', alpha=0.7)
    plt.fill_between(test.index,
                    forecast.conf_int().iloc[:, 0],
                    forecast.conf_int().iloc[:, 1],
                    color='gray', alpha=0.2)
    plt.title(f'SARIMAX{order}{seasonal_order} Forecast\nRMSE: {rmse:.2f}, MAE: {mae:.2f}')
    plt.legend()
    plt.grid()
    plt.show()
    
    return results

# 5. Main Workflow
if __name__ == "__main__":
    # Load data
    df, scaler = load_and_prepare_data("data/FINAL_dataset.csv")
    plot_consumption(df, 'COMED Hourly Energy Consumption with Weather Data')
    
    # Train-test split
    train = df.loc[:'2016-12-31']
    test = df.loc['2017-01-01':]

    
    # Model parameters
    order = (2, 0, 2)  # From your previous analysis
    seasonal_order = (1, 1, 1, 24)  # Daily seasonality
    
    # Model training and evaluation
    model_results = train_evaluate_sarimax(
        train,
        test,
        order=order,
        seasonal_order=seasonal_order
    )
    
    # Hourly RMSE analysis
    test['Prediction'] = model_results.predict(
        start=test.index[0], 
        end=test.index[-1],
        exog=test[['Chicago_temp', 'Chicago_humidity', 'Chicago_pressure', 
                  'Chicago_wind_dir', 'Chicago_wind_speed', 'hour', 'weekday', 'is_holiday']]
    )
    
    plot_data = test.reset_index()[['hour', 'COMED_MW', 'Prediction']].melt(
        id_vars=['hour'], 
        value_vars=['COMED_MW', 'Prediction'],
        var_name='Type',
        value_name='MW'
    )
    
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='hour', y='MW', data=plot_data, hue='Type')
    plt.title('Hourly Distribution: Actual vs Predicted')
    plt.ylabel('MW')
    plt.grid()
    plt.show()

    # 4. Model Training & Evaluation with Extended Exogenous Variables
def train_evaluate_sarimax(train, test, order, seasonal_order=None):
    plt.close('all')
    
    # Prepare all exogenous variables
    exog_cols = ['Chicago_temp', 'Chicago_humidity', 'Chicago_pressure', 'Chicago_wind_speed', 'hour', 'weekday', 'is_holiday']
    
    exog_train = train[exog_cols]
    exog_test = test[exog_cols]
    
    # Model training
    model = SARIMAX(train['COMED_MW'], 
                   exog=exog_train,
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)
    
    results = model.fit(disp=False, maxiter=100)
    print(results.summary())
    
    # Forecasting
    forecast = results.get_forecast(steps=len(test), exog=exog_test)
    forecast_mean = forecast.predicted_mean
    
    # 11. Model evaluation
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.utils.validation import check_array
    
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    y_test_original = test['COMED_MW']
    hourly_pred = forecast_mean
    
    rmse = mean_squared_error(y_test_original, hourly_pred, squared=False)
    mae = mean_absolute_error(y_test_original, hourly_pred)
    mape = mean_absolute_percentage_error(y_test_original, hourly_pred)
    r2 = r2_score(y_test_original, hourly_pred)

    print("\n=== Model Evaluation ===")
    print(f"RMSE: {rmse:.2f} kWh")
    print(f"MAE: {mae:.2f} kWh")
    print(f"MAPE: {mape:.2%}")
    print(f"R²: {r2:.4f}")
    
    # Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(train.index, train['COMED_MW'], label='Training Data')
    plt.plot(test.index, test['COMED_MW'], label='Actual Values')
    plt.plot(test.index, forecast_mean, label='Forecast', alpha=0.7)
    plt.fill_between(test.index,
                    forecast.conf_int().iloc[:, 0],
                    forecast.conf_int().iloc[:, 1],
                    color='gray', alpha=0.2)
    plt.title(f'SARIMAX{order}{seasonal_order} Forecast\nRMSE: {rmse:.2f}, MAE: {mae:.2f}')
    plt.legend()
    plt.grid()
    plt.show()
    
    return results, forecast_mean