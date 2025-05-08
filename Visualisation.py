import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import seaborn as sns

# ------------------------
# Load Data and Models
# ------------------------

# Load classical model data
df_classical = pd.read_csv("processed_data_classical.csv")
X_ml = df_classical.drop(columns=["CO(GT)"])
y_ml = df_classical["CO(GT)"]
split_idx = int(0.8 * len(X_ml))
X_test_rf = X_ml[split_idx:]
y_test_rf = y_ml[split_idx:]

# Load LSTM model data
X_test_lstm = np.load("X_test.npy")
y_test_lstm = np.load("y_test.npy")
target_idx = df_classical.columns.get_loc("CO(GT)")
y_test_lstm = y_test_lstm[:, target_idx]

# Load models
rf_model = joblib.load("random_forest_model.pkl")
lstm_model = load_model("lstm_model.h5", custom_objects={'mse': MeanSquaredError()})

# ------------------------
# Generate Predictions
# ------------------------

rf_preds = rf_model.predict(X_test_rf)
lstm_preds = lstm_model.predict(X_test_lstm).flatten()

# Align lengths
min_len = min(len(rf_preds), len(lstm_preds))
rf_preds = rf_preds[-min_len:]
lstm_preds = lstm_preds[-min_len:]
true_values = y_test_rf.values[-min_len:]

# Combine predictions
combined_preds = (rf_preds + lstm_preds) / 2

# ------------------------
# Model Evaluation Metrics
# ------------------------

# Combined model metrics
rmse = np.sqrt(mean_squared_error(true_values, combined_preds))
r2 = r2_score(true_values, combined_preds)

# Individual model metrics
rf_rmse = np.sqrt(mean_squared_error(true_values, rf_preds))
lstm_rmse = np.sqrt(mean_squared_error(true_values, lstm_preds))
rf_r2 = r2_score(true_values, rf_preds)
lstm_r2 = r2_score(true_values, lstm_preds)

print("\nModel Evaluation Metrics:")
print(f"Combined Model - RMSE: {rmse:.4f}, R²: {r2:.4f}")

print("\nModel Comparison:")
print(f"Random Forest  - RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")
print(f"LSTM           - RMSE: {lstm_rmse:.4f}, R²: {lstm_r2:.4f}")
print(f"Combined       - RMSE: {rmse:.4f}, R²: {r2:.4f}")

# ------------------------
# Confusion Matrix
# ------------------------

threshold = 5
y_true_class = np.where(true_values >= threshold, 1, 0)
y_pred_class = np.where(combined_preds >= threshold, 1, 0)

cm = confusion_matrix(y_true_class, y_pred_class)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Good", "Unhealthy"], yticklabels=["Good", "Unhealthy"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ------------------------
# ROC Curve
# ------------------------

fpr, tpr, thresholds = roc_curve(y_true_class, combined_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# ------------------------
# Feature Importance (Random Forest)
# ------------------------

importances = rf_model.feature_importances_
feature_names = X_ml.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='teal')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# ------------------------
# Residual Plot
# ------------------------

residuals = true_values - combined_preds

plt.figure(figsize=(10, 6))
plt.scatter(range(len(residuals)), residuals, color='orange', alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Residuals of Combined Predictions")
plt.xlabel("Index")
plt.ylabel("Residuals (True - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# Plot Visualization with Threshold Lines
# ------------------------

plt.figure(figsize=(12, 6))
plt.plot(true_values[:200], label="True CO(GT)", color='blue')
plt.plot(rf_preds[:200], label="RF Prediction", color='green', linestyle='dashed')
plt.plot(lstm_preds[:200], label="LSTM Prediction", color='purple', linestyle='dotted')
plt.plot(combined_preds[:200], label="Combined Prediction", color='orange')

plt.axhline(y=1, color='green', linestyle='--', label='Good/Moderate Threshold')
plt.axhline(y=5, color='orange', linestyle='--', label='Moderate/Unhealthy Threshold')
plt.axhline(y=10, color='red', linestyle='--', label='Unhealthy/Very Unhealthy Threshold')

plt.title("CO(GT) Prediction vs Actual with Thresholds")
plt.xlabel("Time Steps")
plt.ylabel("CO(GT) Concentration (ppm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# Interactive Plot using Plotly
# ------------------------

fig = go.Figure()

fig.add_trace(go.Scatter(y=true_values[:200], mode='lines', name='True CO(GT)', line=dict(color='blue')))
fig.add_trace(go.Scatter(y=rf_preds[:200], mode='lines', name='RF Prediction', line=dict(dash='dash', color='green')))
fig.add_trace(go.Scatter(y=lstm_preds[:200], mode='lines', name='LSTM Prediction', line=dict(dash='dot', color='purple')))
fig.add_trace(go.Scatter(y=combined_preds[:200], mode='lines', name='Combined Prediction', line=dict(color='orange')))

fig.add_hline(y=1, line_dash="dash", line_color="green", annotation_text="Good/Moderate", annotation_position="top left")
fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Moderate/Unhealthy", annotation_position="top left")
fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Unhealthy/Very Unhealthy", annotation_position="top left")

fig.update_layout(
    title="Interactive CO(GT) Prediction vs Actual",
    xaxis_title="Time Steps",
    yaxis_title="CO(GT) Concentration (ppm)",
    legend_title="Legend",
    template="plotly_white"
)

fig.show()
