import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Завантаження та обробка даних
data = pd.read_csv("residential_energy_normalized.csv")
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['Hour'] = data['Timestamp'].dt.hour

# Вибір ознак з урахуванням нових важливих фіч
features = [
    'Temperature (°C)',
    'Humidity (%)',
    'Occupancy Rate (%)',
    'Energy Price ($/kWh)',
    'Month',
    'DayOfWeek',
    'Hour',
    # Додані нові важливі фічі
    'Energy Savings Potential (%)',
    'Voltage Levels (V)',
    'Demand Response Participation',
    'Lighting Consumption (kWh)',
    'Water Usage (liters)',
    'Thermal Comfort Index',
    'Power Factor'
]

target = 'Energy Consumption (kWh) Normalized'

X = data[features].values
y = data[target].values

# Нормалізація даних
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Створення послідовностей для GRU
def create_sequences(X, y, time_steps=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 24  # Кількість годин у послідовності
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Розділення на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, shuffle=False
)

# Побудова моделі GRU з більш складною архітектурою
n_features = X_train.shape[2]
gru_units = 64

model = Sequential([
    GRU(gru_units, activation='tanh', recurrent_activation='sigmoid',
        return_sequences=True, input_shape=(time_steps, n_features)),
    Dropout(0.2),
    GRU(gru_units//2, activation='tanh', recurrent_activation='sigmoid',
        return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# Компіляція моделі
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Вивід архітектури
model.summary()

# Навчання моделі з ранньою зупинкою
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Візуалізація історії навчання
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progress')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Прогнозування та оцінка
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# Оцінка моделі
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)
print(f"RMSE (GRU): {rmse:.2f}")
print(f"R² (GRU): {r2:.2f}")

# Візуалізація результатів
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.title('GRU: Actual vs Predicted Energy Consumption')
plt.xlabel('Time Steps')
plt.ylabel('Energy Consumption (kWh) Normalized')
plt.legend()
plt.show()

# Агрегація за місяць
results = pd.DataFrame({
    'Timestamp': data['Timestamp'].iloc[-len(y_test):],
    'Actual': y_test_actual.flatten(),
    'Predicted': y_pred.flatten()
})

results['Month'] = results['Timestamp'].dt.month

monthly_actual = results.groupby('Month')['Actual'].sum()
monthly_pred = results.groupby('Month')['Predicted'].sum()

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(monthly_actual.index, monthly_actual, marker='o', label='Actual')
plt.plot(monthly_actual.index, monthly_pred, marker='x', label='Predicted (GRU)')
plt.xlabel('Month')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Monthly Energy Consumption: Actual vs GRU Prediction')
plt.legend()
plt.grid(True)
plt.show()