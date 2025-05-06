import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import joblib

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
# Train & Save Random Forest
# ------------------------

print("\nTraining Random Forest...")
split_idx = int(0.8 * len(X_ml))
X_train_rf, X_test_rf = X_ml[:split_idx], X_ml[split_idx:]
y_train_rf, y_test_rf = y_ml[:split_idx], y_ml[split_idx:]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
rf_preds = rf_model.predict(X_test_rf)

# Save RF model
joblib.dump(rf_model, "random_forest_model.pkl")
print("Random Forest model saved as 'random_forest_model.pkl'.")

# ------------------------
# Train & Save LSTM Model
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

# Save LSTM model
lstm_model.save("lstm_model.h5")
print("LSTM model saved as 'lstm_model.h5'.")

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
# Health Risk Alert Logic
# ------------------------

def health_risk_alert(co_value):
    if co_value < 1:
        return "Good air quality"
    elif 1 <= co_value < 5:
        return "Moderate – May cause minor health issues for sensitive people"
    elif 5 <= co_value < 10:
        return "Unhealthy – Risk for sensitive groups"
    else:
        return "Very Unhealthy – Risk for everyone"

latest_pred = combined_preds[-1]
alert_message = health_risk_alert(latest_pred)
print(f"\nHealth Alert Based on Latest Prediction ({latest_pred:.2f} ppm): {alert_message}")

# ------------------------
# Real-time Prediction Function
# ------------------------

def predict_new_input(new_input_array):
    """
    Provide a new input (numpy array) for both RF and LSTM.
    Returns combined prediction and health alert.
    """
    rf_model_loaded = joblib.load("random_forest_model.pkl")
    lstm_model_loaded = load_model("lstm_model.h5")

    rf_prediction = rf_model_loaded.predict(new_input_array.reshape(1, -1))[0]
    lstm_prediction = lstm_model_loaded.predict(new_input_array.reshape(1, new_input_array.shape[0], 1)).flatten()[0]

    combined = (rf_prediction + lstm_prediction) / 2
    alert = health_risk_alert(combined)

    return combined, alert

