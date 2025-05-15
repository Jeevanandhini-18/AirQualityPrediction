import streamlit as st  
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import datetime

# Set Matplotlib backend for Streamlit compatibility
import matplotlib
matplotlib.use('Agg')

# Load the trained models
MODEL_PATHS = {
    "LSTM": './lstm_model.h5',
    "Random Forest": './random_forest_model.pkl'
}
PCA_PATH = './pca_transformer.pkl'  # Path to the PCA transformer

@st.cache_resource
def load_models():
    models = {}
    try:
        models["LSTM"] = load_model(MODEL_PATHS["LSTM"], custom_objects={'mse': MeanSquaredError()})
        models["Random Forest"] = joblib.load(MODEL_PATHS["Random Forest"])
    except Exception as e:
        st.error(f"🚫 Failed to load models: {e}")
    
    try:
        if os.path.exists(PCA_PATH):
            models["PCA"] = joblib.load(PCA_PATH)
        else:
            dummy_data = np.random.rand(100, 6)
            pca = PCA(n_components=5)
            pca.fit(dummy_data)
            joblib.dump(pca, PCA_PATH)
            models["PCA"] = pca
    except Exception as e:
        st.error(f"🚫 Failed to load or train PCA: {e}")

    return models

models = load_models()

def prepare_lstm_input(features):
    lstm_input = np.zeros((1, 24, 19))  # shape expected by LSTM
    lstm_input[0, 0, :6] = features.flatten()
    return lstm_input

def make_prediction(features, model_type='LSTM'):
    if model_type == 'LSTM':
        lstm_features = prepare_lstm_input(features)
        prediction = models["LSTM"].predict(lstm_features)
    elif model_type == 'Random Forest':
        transformed_features = models["PCA"].transform(features)
        prediction = models["Random Forest"].predict(transformed_features)
    return float(prediction.flatten()[0])

def health_alert_and_tips(aqi, sensitivities):
    if aqi > 150:
        alert = ("❌ Very Poor Air Quality! Health warning for all groups.")
        tips = [
            "🚫 Avoid all outdoor activities.",
            "🏡 Use air purifiers indoors.",
            "😷 Wear N95 masks if going outside.",
            "🔒 Keep windows closed."
        ]
        alert_type = "error"
    elif aqi > 100:
        alert = ("⚠ Poor Air Quality! Sensitive groups may experience health effects.")
        tips = [
            "🧘 Limit prolonged outdoor exertion.",
            "😷 Wear masks if sensitive to pollution.",
            "🌬 Keep indoor air clean."
        ]
        alert_type = "warning"
    elif aqi > 50:
        alert = ("🌤 Moderate Air Quality. Be cautious if you have respiratory conditions.")
        tips = [
            "🚶 Consider reducing heavy outdoor exercise.",
            "🩺 Monitor symptoms if you have asthma or allergies."
        ]
        alert_type = "info"
    else:
        alert = ("✅ Good Air Quality. No immediate health risk.")
        tips = ["🌞 Enjoy your day outdoors!"]
        alert_type = "success"
    
    # Customize alerts based on user sensitivities
    if aqi > 50 and sensitivities:
        alert += " ⚠ Based on your health profile, take extra precautions."
        if 'Asthma' in sensitivities:
            tips.append("💨 Carry your inhaler and avoid triggers.")
        if 'Allergies' in sensitivities:
            tips.append("💊 Take allergy medications as prescribed.")
        if 'Heart Disease' in sensitivities:
            tips.append("❤ Limit physical exertion and monitor symptoms.")
    
    return alert, tips, alert_type

def display_alert_and_tips(aqi, sensitivities):
    alert, tips, alert_type = health_alert_and_tips(aqi, sensitivities)
    getattr(st, alert_type)(alert)
    st.markdown("#### 📝 Recommended Precautions:")
    for tip in tips:
        st.write(f"- {tip}")

# --- Streamlit UI ---

# Set page config for better layout
st.set_page_config(page_title="Air Quality Predictor", page_icon="🌍", layout="centered")

# Main Title with some padding
st.markdown(
    """
    <h1 style='text-align: center; color: #2E8B57;'>🌍 Air Quality Prediction & Early Health Risk Alert System</h1>
    """, 
    unsafe_allow_html=True
)
st.markdown("---")

# Description with improved formatting
st.markdown("""
<p style="text-align:center; font-size:16px; line-height:1.5; margin-top:0;">
This application <em>predicts air quality, provides <strong>early health risk alerts</strong>,</em>  
and suggests <em>preventive measures</em> personalized to your health conditions.
</p>
""", unsafe_allow_html=True)

# Sidebar - User Health Profile
st.sidebar.header("🩺 Your Health Profile")

health_conditions_options = ['None', 'Asthma', 'Allergies', 'Heart Disease', 'Other respiratory issues']
user_sensitivities = st.sidebar.multiselect(
    "Do you have any of these health conditions?",
    options=health_conditions_options,
    default=['None']
)

if 'None' in user_sensitivities and len(user_sensitivities) > 1:
    user_sensitivities.remove('None')

email_alert_option = st.sidebar.selectbox(
    "📧 Receive email alerts for poor air quality?",
    options=["No", "Yes"]
)

user_email = ""
if email_alert_option == "Yes":
    user_email = st.sidebar.text_input("✉ Enter your email address:")

# Main Inputs Section with a box
with st.container():
    st.subheader("📍 Environmental Data Input")
    cols = st.columns(3)
    cols[0].number_input("🌡 Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0, key="temp")
    cols[1].number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, key="humidity")
    cols[2].number_input("🌫 PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=35.0, key="pm2_5")

    cols2 = st.columns(3)
    cols2[0].number_input("🌫 PM10 (µg/m³)", min_value=0.0, max_value=500.0, value=50.0, key="pm10")
    cols2[1].number_input("🛢 CO (ppm)", min_value=0.0, max_value=20.0, value=1.0, key="co")
    cols2[2].number_input("🛢 NO2 (ppm)", min_value=0.0, max_value=10.0, value=0.05, key="no2")

    model_type = st.radio("🧠 Select prediction model:", ('LSTM', 'Random Forest'), horizontal=True)

location = st.text_input("📌 Enter your location (optional):", placeholder="City or area name")

features = np.array([[st.session_state.get("temp", 25.0), st.session_state.get("humidity", 60.0),
                      st.session_state.get("pm2_5", 35.0), st.session_state.get("pm10", 50.0),
                      st.session_state.get("co", 1.0), st.session_state.get("no2", 0.05)]])

st.markdown("<br>", unsafe_allow_html=True)

# Button centered horizontally
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    if st.button("🔍 Get Air Quality Prediction"):
        try:
            prediction = make_prediction(features, model_type)
            st.success(f"### 🌟 Predicted Air Quality Index (AQI) using {model_type}: {prediction:.2f}")
            display_alert_and_tips(prediction, user_sensitivities)

            if email_alert_option == "Yes" and user_email:
                st.info(f"📧 An email alert will be sent to {user_email} when air quality is poor (functionality to be implemented).")

        except Exception as e:
            st.error(f"🚫 Prediction failed: {e}")
    else:
        prediction = None

st.markdown("---")

# Historical Data Section with better headings and spacing
st.header("📊 Historical Comparison & Trends")

CLEANED_CSV_PATH = './cleaned_air_quality.csv'

if os.path.exists(CLEANED_CSV_PATH):
    try:
        air_quality_data = pd.read_csv(CLEANED_CSV_PATH)

        st.subheader("📅 Historical Air Quality Data")
        st.dataframe(air_quality_data.tail(), use_container_width=True)

        air_quality_data['DateTime'] = pd.to_datetime(air_quality_data['DateTime'], errors='coerce')

        if 'AQI' not in air_quality_data.columns:
            if 'PM2.5' in air_quality_data.columns:
                air_quality_data['AQI'] = air_quality_data['PM2.5'].fillna(0)
            elif 'CO(GT)' in air_quality_data.columns:
                air_quality_data['AQI'] = air_quality_data['CO(GT)'].fillna(0)
            else:
                air_quality_data['AQI'] = 0

        air_quality_data.dropna(subset=['DateTime', 'AQI'], inplace=True)

        air_quality_data['YearMonth'] = air_quality_data['DateTime'].dt.to_period('M')
        air_quality_data['Month'] = air_quality_data['DateTime'].dt.month
        air_quality_data['Day'] = air_quality_data['DateTime'].dt.day

        monthly_avg = air_quality_data.groupby('YearMonth')['AQI'].mean().reset_index()
        monthly_avg['YearMonth'] = monthly_avg['YearMonth'].dt.to_timestamp()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_avg['YearMonth'], monthly_avg['AQI'], marker='o', linestyle='-', color='#2c3e50')
        ax.set_title('Monthly Average AQI Over Time', fontsize=16, weight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average AQI', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

        if prediction is not None:
            today = pd.Timestamp.now()
            current_month = today.month
            current_day = today.day

            seasonal_avg = air_quality_data[air_quality_data['Month'] == current_month]['AQI'].mean()
            daily_avg = air_quality_data[air_quality_data['Day'] == current_day]['AQI'].mean()

            st.markdown(f"### 🌟 Seasonal & Daily Averages vs Current Prediction")
            st.write(f"- Historical average AQI for month {current_month} (seasonal): {seasonal_avg:.2f}")
            st.write(f"- Historical average AQI for day {current_day} (daily): {daily_avg:.2f}")
            st.write(f"- Current predicted AQI: {prediction:.2f}")

            if prediction > seasonal_avg:
                st.warning(f"⚠ The predicted AQI is higher than the historical seasonal average for month {current_month}.")
            else:
                st.success(f"✅ The predicted AQI is better than or equal to the historical seasonal average for month {current_month}.")

            if prediction > daily_avg:
                st.warning(f"⚠ The predicted AQI is higher than the historical average for day {current_day}.")
            else:
                st.success(f"✅ The predicted AQI is better than or equal to the historical average for day {current_day}.")

    except Exception as e:
        st.error(f"Error processing historical data: {e}")

else:
    st.warning("cleaned_air_quality.csv not found. Please run cleaning.py first.")