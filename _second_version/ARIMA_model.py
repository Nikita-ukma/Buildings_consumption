'''
RESULT: 
Model Evaluation Metrics:
RMSE: 20.31
MAE: 16.16
MAPE: 86.79
'''

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 1. Load and prepare data
data = pd.read_csv("data/electricity_dataset.csv", parse_dates=['Timestamp'], index_col='Timestamp')

# 2. Create necessary features
data['Hour'] = data.index.hour
data['DayOfWeek'] = data.index.dayofweek
data['IsWeekend'] = data['DayOfWeek'].isin([5,6]).astype(int)

# Check required columns
required_columns = ['Occupancy Rate (%)', 'Building Size (m²)', 'Temperature (°C)', 'Humidity (%)']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

data['Occupancy_Ratio'] = data['Occupancy Rate (%)'] / (data['Building Size (m²)'] + 1e-6)

# 3. Define exogenous variables
exog_vars = ['Temperature (°C)', 'Humidity (%)', 'Occupancy_Ratio', 'Hour', 'DayOfWeek', 'IsWeekend']

# 4. Split data
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# 5. Manual parameter selection function
def evaluate_arima(endog, exog, order):
    try:
        model = ARIMA(
            endog=endog,
            exog=exog,
            order=order
        )
        result = model.fit()
        return result.aic, result.bic, result
    except:
        return np.inf, np.inf, None

# 6. Grid search for best parameters
def grid_search_arima(endog, exog):
    p_values = range(0, 3)  # AR order
    d_values = range(0, 2)  # Differencing
    q_values = range(0, 3)  # MA order
    
    best_aic = np.inf
    best_order = None
    best_model = None
    
    print("Performing grid search for ARIMA parameters...")
    for p, d, q in product(p_values, d_values, q_values):
        aic, bic, model = evaluate_arima(endog, exog, (p, d, q))
        if aic < best_aic:
            best_aic = aic
            best_order = (p, d, q)
            best_model = model
            print(f"ARIMA{best_order} - AIC:{best_aic:.2f}, BIC:{bic:.2f}")
    
    return best_order, best_model

# 7. Find best parameters
best_order, best_model = grid_search_arima(
    train_data['Energy Consumption (kWh)'],
    train_data[exog_vars]
)

print(f"\nBest ARIMA order: {best_order}")
print(best_model.summary())

# 8. Forecasting
forecast = best_model.get_forecast(
    steps=len(test_data),
    exog=test_data[exog_vars]
)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# 9. Model evaluation
metrics = {
    'RMSE': mean_squared_error(test_data['Energy Consumption (kWh)'], forecast_values, squared=False),
    'MAE': mean_absolute_error(test_data['Energy Consumption (kWh)'], forecast_values),
    'MAPE': np.mean(np.abs((test_data['Energy Consumption (kWh)'] - forecast_values) / 
                   test_data['Energy Consumption (kWh)'])) * 100
}

print("\nModel Evaluation Metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.2f}")

# 10. Visualization
plt.figure(figsize=(14, 7))
plt.plot(train_data.index[-100:], train_data['Energy Consumption (kWh)'][-100:], label='Training Data')
plt.plot(test_data.index, test_data['Energy Consumption (kWh)'], label='Actual', color='blue')
plt.plot(test_data.index, forecast_values, label='Forecast', color='red')
plt.fill_between(test_data.index,
                conf_int.iloc[:, 0],
                conf_int.iloc[:, 1],
                color='pink', alpha=0.3, label='95% CI')
plt.title(f'Energy Consumption Forecast with ARIMA{best_order}')
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.show()

# 11. Residual diagnostics
residuals = pd.Series(best_model.resid)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals over Time')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Residuals')

axes[0, 1].hist(residuals, bins=30)
axes[0, 1].set_title('Residuals Distribution')
axes[0, 1].set_xlabel('Residuals')

sm.qqplot(residuals, line='s', ax=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

sm.graphics.tsa.plot_acf(residuals, lags=40, ax=axes[1, 1])
axes[1, 1].set_title('ACF of Residuals')

plt.tight_layout()
plt.show()

# 12. Save results
output = pd.DataFrame({
    'Timestamp': test_data.index,
    'Actual': test_data['Energy Consumption (kWh)'],
    'Predicted': forecast_values,
    'Lower CI': conf_int.iloc[:, 0],
    'Upper CI': conf_int.iloc[:, 1]
})
output.to_csv('arima_manual_forecast_results.csv', index=False)