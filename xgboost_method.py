'''
RESULT:
=== Model Evaluation ===
RMSE: 2.82 kWh
MAE: 0.95 kWh
MAPE: 9.95%
R²: 0.9813
'''


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data/electricity_dataset.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# 1. Process string values
string_features = ['Building Type', 'Occupancy Schedule', 'Building Orientation', 
                  'Carbon Emission Reduction Category', 'Peak Demand Reduction Indicator']

# Encode categorical variables
label_encoders = {}
for col in string_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 2. Normalize target variable
energy_scaler = MinMaxScaler()
data['Energy Consumption (kWh) Normalized'] = energy_scaler.fit_transform(
    data[['Energy Consumption (kWh)']])

# 3. Add time features
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek  
data['Hour'] = data['Timestamp'].dt.hour
data['IsWeekend'] = data['DayOfWeek'].isin([5,6]).astype(int)
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour']/24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour']/24)

# 4. Add feature interactions
data['Temp_Humidity'] = data['Temperature (°C)'] * data['Humidity (%)']
data['Occupancy_Ratio'] = data['Occupancy Rate (%)'] / (data['Building Size (m²)'] + 1e-6)
data['Energy_Intensity'] = data['Energy Consumption (kWh)'] / (data['Building Size (m²)'] + 1e-6)

# 5. Feature selection
features = [
    # Basic features
    'Temperature (°C)', 'Humidity (%)', 'Occupancy Rate (%)',
    'Energy Price ($/kWh)', 'Power Factor', 'Voltage Levels (V)',
    
    # Categorical
    'Building Type', 'Occupancy Schedule', 'Building Orientation',
    
    # Time-based
    'Month', 'DayOfWeek', 'Hour', 'IsWeekend', 'Hour_sin', 'Hour_cos',
    
    # Technical
    'Building Size (m²)', 'Window-to-Wall Ratio (%)', 'Insulation Quality Score',
    'Building Age (years)', 'Equipment Age (years)',
    
    # Interactions
    'Temp_Humidity', 'Occupancy_Ratio', 'Energy_Intensity',
    
    # Other
    'Demand Response Participation', 'Thermal Comfort Index',
    'Carbon Emission Reduction Category'
]

X = data[features]
y = data['Energy Consumption (kWh) Normalized']

# 6. Convert categorical features (XGBoost handles these differently than LightGBM)
# For XGBoost, we'll one-hot encode or just leave as numeric (since we already label encoded)
cat_features = ['Building Type', 'Occupancy Schedule', 'Building Orientation', 
               'Month', 'DayOfWeek', 'Hour', 'IsWeekend', 
               'Carbon Emission Reduction Category']
for col in cat_features:
    X[col] = X[col].astype('int')

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=False)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=False)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'eta': 0.1,
    'seed': 42
}

eval_list = [(dtrain, 'train'), (dtest, 'eval')]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=eval_list,
    early_stopping_rounds=50,
    verbose_eval=100
)

# 9. Predictions and denormalization
hourly_pred_normalized = model.predict(dtest)
hourly_pred = energy_scaler.inverse_transform(hourly_pred_normalized.reshape(-1, 1)).flatten()
y_test_original = energy_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()

# 10. Model evaluation
rmse = mean_squared_error(y_test_original, hourly_pred, squared=False)
mae = mean_absolute_error(y_test_original, hourly_pred)
mape = mean_absolute_percentage_error(y_test_original, hourly_pred)
r2 = r2_score(y_test_original, hourly_pred)

print("\n=== Model Evaluation ===")
print(f"RMSE: {rmse:.2f} kWh")
print(f"MAE: {mae:.2f} kWh")
print(f"MAPE: {mape:.2%}")
print(f"R²: {r2:.4f}")

# 11. Aggregate by month for analysis
test_data = X_test.copy()
test_data['Actual'] = y_test_original
test_data['Predicted'] = hourly_pred
test_data['Month'] = data.loc[X_test.index, 'Month']  # Add month from original data

monthly_actual = test_data.groupby('Month')['Actual'].sum()
monthly_pred = test_data.groupby('Month')['Predicted'].sum()

# 12. Visualization
plt.figure(figsize=(12, 6))
plt.plot(monthly_actual.index, monthly_actual, 'o-', label='Actual')
plt.plot(monthly_actual.index, monthly_pred, 'x--', label='Predicted')
plt.title('Monthly Energy Consumption: Actual vs Predicted')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.show()

# Feature importance
xgb.plot_importance(model, max_num_features=20, height=0.8)
plt.title('Feature Importance')
plt.show()