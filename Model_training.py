import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

# ------------------------
# Load Processed Data
# ------------------------

# Traditional ML data
df_classical = pd.read_csv("processed_data_classical.csv")
X_ml = df_classical.drop(columns=["CO(GT)"])
y_ml = df_classical["CO(GT)"]

# LSTM data
X_train_lstm = np.load("X_train.npy")
X_test_lstm = np.load("X_test.npy")
y_train_lstm = np.load("y_train.npy")
y_test_lstm = np.load("y_test.npy")

# Use only the target column from y for LSTM
target_idx = df_classical.columns.get_loc("CO(GT)")
y_train_lstm = y_train_lstm[:, target_idx]
y_test_lstm = y_test_lstm[:, target_idx]

# ------------------------
# Train Random Forest Model
# ------------------------
print("\nTraining Random Forest...")
split_idx = int(0.8 * len(X_ml))
X_train_rf, X_test_rf = X_ml[:split_idx], X_ml[split_idx:]
y_train_rf, y_test_rf = y_ml[:split_idx], y_ml[split_idx:]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
rf_preds = rf_model.predict(X_test_rf)

# ------------------------
# Train LSTM Model
# ------------------------
print("\nTraining LSTM...")
lstm_model = Sequential()
lstm_model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5)
lstm_model.fit(X_train_lstm, y_train_lstm,
               epochs=50, batch_size=32,
               validation_split=0.1, callbacks=[early_stop],
               verbose=1)

lstm_preds = lstm_model.predict(X_test_lstm).flatten()

# ------------------------
# Align Prediction Lengths
# ------------------------
min_len = min(len(rf_preds), len(lstm_preds))
rf_preds_aligned = rf_preds[-min_len:]
lstm_preds_aligned = lstm_preds[-min_len:]
true_values_aligned = y_test_rf.values[-min_len:]

# ------------------------
# Combine Predictions
# ------------------------
combined_preds = (rf_preds_aligned + lstm_preds_aligned) / 2

# ------------------------
# Evaluation
# ------------------------
rmse = np.sqrt(mean_squared_error(true_values_aligned, combined_preds))
r2 = r2_score(true_values_aligned, combined_preds)

print("\nModel Evaluation (Combined):")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# ------------------------
# Plotting
# ------------------------
plt.figure(figsize=(10, 5))
plt.plot(true_values_aligned[:200], label="True", color='blue')
plt.plot(combined_preds[:200], label="Combined Prediction", color='orange')
plt.title("True vs Combined Prediction (CO(GT))")
plt.xlabel("Time Steps")
plt.ylabel("CO(GT) Concentration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
