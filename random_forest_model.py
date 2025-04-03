'''
RESULT:
=== Model Evaluation ===
RMSE: 9.18 kWh
MAE: 6.36 kWh
MAPE: 37.23%
R²: 0.8018
'''



import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
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

# 6. Convert categorical features (Random Forest can handle label encoded categories)
cat_features = ['Building Type', 'Occupancy Schedule', 'Building Orientation', 
               'Month', 'DayOfWeek', 'Hour', 'IsWeekend', 
               'Carbon Emission Reduction Category']
for col in cat_features:
    X[col] = X[col].astype('category')

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Create and train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

# 9. Predictions and denormalization
hourly_pred_normalized = rf_model.predict(X_test)
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
plt.title('Monthly Energy Consumption: Actual vs Predicted (Random Forest)')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.show()

# 13. Feature importance
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importances['Feature'].head(20), feature_importances['Importance'].head(20))
plt.title('Top 20 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()