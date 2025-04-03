'''
RESULT:
RMSE: 20.31 kWh
MAE: 16.15 kWh
MAPE: 87.46%
R²: -0.0009
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv("data/electricity_dataset.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# 1. Process categorical features
cat_features = ['Building Type', 'Occupancy Schedule', 'Building Orientation', 
               'Carbon Emission Reduction Category', 'Peak Demand Reduction Indicator']
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 2. Create time features
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek  
data['Hour'] = data['Timestamp'].dt.hour
data['IsWeekend'] = data['DayOfWeek'].isin([5,6]).astype(int)
data['Hour_sin'] = np.sin(2 * np.pi * data['Hour']/24)
data['Hour_cos'] = np.cos(2 * np.pi * data['Hour']/24)

# 3. Create feature interactions
data['Temp_Humidity'] = data['Temperature (°C)'] * data['Humidity (%)']
data['Occupancy_Ratio'] = data['Occupancy Rate (%)'] / (data['Building Size (m²)'] + 1e-6)
data['Energy_Intensity'] = data['Energy Consumption (kWh)'] / (data['Building Size (m²)'] + 1e-6)

# 4. Define features
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

# Verify all features exist
missing = [f for f in features if f not in data.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

# 5. Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Energy Consumption (kWh)'] + features])

# 6. Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data)-seq_length-1):
        X.append(data[i:(i+seq_length)])
        y.append(data[i+seq_length, 0])  # Target is energy consumption
    return np.array(X), np.array(y)

SEQ_LENGTH = 24  # 24-hour sequences
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# 7. Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 8. Build LSTM model
model = Sequential([
    LSTM(128, input_shape=(SEQ_LENGTH, len(features)+1), 
         return_sequences=True, recurrent_dropout=0.1),
    BatchNormalization(),
    LSTM(64, recurrent_dropout=0.1),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='mse',
             metrics=['mae'])

# 9. Train with callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 10. Predict and evaluate
y_pred = model.predict(X_test)

# Inverse scaling function
def inverse_scale(scaler, data, n_features):
    dummy = np.zeros((len(data), n_features))
    dummy[:, 0] = data.ravel()
    return scaler.inverse_transform(dummy)[:, 0]

y_test_orig = inverse_scale(scaler, y_test, len(features)+1)
y_pred_orig = inverse_scale(scaler, y_pred, len(features)+1)

# Calculate metrics
rmse = mean_squared_error(y_test_orig, y_pred_orig, squared=False)
mae = mean_absolute_error(y_test_orig, y_pred_orig)
mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)

print("\n=== LSTM Model Evaluation ===")
print(f"RMSE: {rmse:.2f} kWh")
print(f"MAE: {mae:.2f} kWh")
print(f"MAPE: {mape:.2%}")
print(f"R²: {r2:.4f}")

# 11. Visualization
plt.figure(figsize=(14, 6))
plt.plot(y_test_orig[:500], label='Actual', linewidth=1)
plt.plot(y_pred_orig[:500], label='Predicted', linewidth=1, alpha=0.7)
plt.title('Energy Consumption: Actual vs LSTM Prediction (First 500 Samples)')
plt.xlabel('Time Steps')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()