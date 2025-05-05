import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# Load cleaned dataset
from cleaning import load_and_clean_data  # Ensure this is implemented
df = load_and_clean_data()

print("\n Starting Feature Engineering")

# -------------------------------
# Common Feature Engineering
# -------------------------------

# Extract time-based features
df['Hour'] = df['DateTime'].dt.hour
df['Day'] = df['DateTime'].dt.day
df['Month'] = df['DateTime'].dt.month
df['Weekday'] = df['DateTime'].dt.weekday
df['is_weekend'] = df['Weekday'].isin([5, 6]).astype(int)

# Create ratio feature (e.g., NO2 / CO)
if 'NO2(GT)' in df.columns and 'CO(GT)' in df.columns:
    df['NO2_to_CO'] = df['NO2(GT)'] / (df['CO(GT)'] + 1e-5)

# Binning temperature
if 'T' in df.columns:
    df['Temp_Level'] = pd.cut(df['T'], bins=[-10, 0, 10, 20, 30, 50],
                              labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

# --------------------------------
# Traditional ML Feature Pipeline
# --------------------------------
print("\n Traditional ML Feature Engineering")

target = 'CO(GT)'
selected_num_cols = ['T', 'RH', 'AH']  # Adjust based on your dataset

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[selected_num_cols])
poly_feature_names = poly.get_feature_names_out(selected_num_cols)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# Combine polynomial features
df_ml = pd.concat([df, df_poly], axis=1)

# Feature selection
X_ml = df_ml.select_dtypes(include='number').drop(columns=[target])
y_ml = df_ml[target]

selector = SelectKBest(score_func=f_regression, k=10)
X_ml_selected = selector.fit_transform(X_ml, y_ml)
selected_features = X_ml.columns[selector.get_support()]
print(f" Selected Features: {list(selected_features)}")

# Keep only selected features + target
df_selected = df_ml[selected_features.tolist() + [target]]

# Standardization
scaler_std = StandardScaler()
df_ml_final = pd.DataFrame(scaler_std.fit_transform(df_selected), columns=df_selected.columns)

# Optional PCA
pca = PCA(n_components=5)
pca_components = pca.fit_transform(df_ml_final.drop(columns=[target]))
df_pca = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(5)])
df_pca[target] = df_ml_final[target].values

# Save
df_pca.to_csv("processed_data_classical.csv", index=False)
print("Traditional ML Data Ready: processed_data_classical.csv")

# --------------------------------
# LSTM Feature Pipeline
# --------------------------------
print("\nLSTM Feature Engineering")

# Set DateTime as index
df_lstm = df.copy()
df_lstm.set_index('DateTime', inplace=True)

# Drop non-numeric or irrelevant columns for LSTM
df_lstm = df_lstm.select_dtypes(include='number')

# MinMax Scaling
scaler_minmax = MinMaxScaler()
df_lstm_scaled = pd.DataFrame(scaler_minmax.fit_transform(df_lstm), 
                              index=df_lstm.index, columns=df_lstm.columns)

# Create LSTM sequences
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        y.append(data.iloc[i + seq_length].values)
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(df_lstm_scaled, seq_length=24)

# Split 80-20
split = int(0.8 * len(X_lstm))
X_train, X_test = X_lstm[:split], X_lstm[split:]
y_train, y_test = y_lstm[:split], y_lstm[split:]

# Save
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print(" LSTM Data Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")
