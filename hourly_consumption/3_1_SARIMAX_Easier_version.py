import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def prepare_data(df):
    if df.index.duplicated().any():
        print(f"Found {df.index.duplicated().sum()} duplicate indices. Removing...")
        df = df[~df.index.duplicated(keep='first')]
    
    df = df.sort_index()
    
    df = df.interpolate(method='time')
    
    return df

def train_evaluate_sarimax(train, test):
    train = prepare_data(train)
    test = prepare_data(test)
    
    exog_cols = ['Chicago_temp', 'Chicago_humidity', 'hour', 'weekday', 'is_holiday']
    
    try:
        model = SARIMAX(train['COMED_MW'],
                      exog=train[exog_cols],
                      order=(2, 0, 2),
                      seasonal_order=(0, 1, 1, 24),
                      enforce_stationarity=False,
                      enforce_invertibility=True)
        
        results = model.fit(disp=True, maxiter=50, method='lbfgs')
        
        forecast = results.get_forecast(steps=len(test), exog=test[exog_cols])
        forecast_mean = forecast.predicted_mean
        forecast_mean.index = test.index
        
        y_test = test['COMED_MW']
        rmse = mean_squared_error(y_test, forecast_mean, squared=False)
        mae = mean_absolute_error(y_test, forecast_mean)
        mape = mean_absolute_percentage_error(y_test, forecast_mean)
        r2 = r2_score(y_test, forecast_mean)
        
        print("\n=== Model Evaluation ===")
        print(f"RMSE: {rmse:.2f} kWh")
        print(f"MAE: {mae:.2f} kWh")
        print(f"MAPE: {mape:.2%}")
        print(f"R²: {r2:.4f}")
        
        return results, forecast_mean
    
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/FINAL_dataset.csv", parse_dates=['Datetime'], index_col='Datetime')
        df = prepare_data(df)
        
        train = df.loc[:'2016-12-31'].copy()
        test = df.loc['2017-01-01':].copy()
        
        model, preds = train_evaluate_sarimax(train, test)
        
        if model is not None and preds is not None:
            results = test[['COMED_MW']].copy()
            results['Prediction'] = preds
            results.to_csv("sarimax_optimized_results.csv")
            print("Results saved to file sarimax_optimized_results.csv")
    
    except Exception as e:
        print(f"Main error: {e}")