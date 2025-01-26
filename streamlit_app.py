import streamlit as st
import requests
from joblib import load
from datetime import datetime
import pytz
import time

# Load models
scaler = load('scaler.pkl')  # Scaler filename
model = load('best_model.pkl')  # Model filename

# URL for fetching data from ThingSpeak
THINGSPEAK_CHANNEL_ID = '2802771'
THINGSPEAK_API_KEY = 'BVPSP1KOVBKQWW5K'
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_API_KEY}"

# Function to convert UTC time to local time
def convert_to_local_time(utc_time_str):
    local_tz = pytz.timezone('Asia/Bangkok')
    utc_time = datetime.strptime(utc_time_str, "%Y-%m-%dT%H:%M:%SZ")
    utc_time = pytz.utc.localize(utc_time)
    local_time = utc_time.astimezone(local_tz)
    return local_time.strftime("%d %B %Y, %H:%M:%S")

# Function to make predictions based on environmental conditions
def make_prediction(age, gender, bmi, heart_rate, o2, body_temperature):
    gender_numeric = 1 if gender.lower() == 'male' else 0
    input_data = [[age, gender_numeric, bmi, heart_rate, o2, body_temperature]]
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return "High Risk" if prediction[0] == 1 else "Low Risk"

# Function to fetch and handle sensor data safely
def get_sensor_data(last_valid_data):
    try:
        response = requests.get(THINGSPEAK_URL)
        response.raise_for_status()
        data = response.json()
        feeds = data.get('feeds', [])
        if feeds:
            latest_feed = feeds[-1]
            timestamp = latest_feed.get('created_at', None)
            heart_rate = latest_feed.get('field1', None)
            o2 = latest_feed.get('field2', None)
            body_temperature = latest_feed.get('field3', None)
            if all(value is not None for value in [timestamp, heart_rate, o2, body_temperature]):
                try:
                    heart_rate = float(heart_rate)
                    o2 = float(o2)
                    body_temperature = float(body_temperature)
                    formatted_time = convert_to_local_time(timestamp)
                    return formatted_time, heart_rate, o2, body_temperature
                except ValueError as e:
                    st.write(f"❌ Error converting sensor data: {e}")
            return last_valid_data
        else:
            st.write("❌ No data available in the channel.")
    except requests.exceptions.RequestException as e:
        st.write(f"❌ Unable to connect to ThingSpeak API: {e}")
    return last_valid_data

# Streamlit UI configuration
st.set_page_config(page_title='Health Risk Prediction', layout='wide')
st.title('💡 Health Risk Prediction System')

# Store previous data in session state to track changes
if 'last_data' not in st.session_state:
    st.session_state.last_data = None

if 'last_valid_data' not in st.session_state:
    st.session_state.last_valid_data = (None, None, None, None)

# Personal Information Input Section
st.header('Step 1: Enter Your Personal Information')
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age (years)', min_value=0, max_value=120, step=1, value=25)

with col2:
    gender = st.selectbox('Gender', ['Male', 'Female'])

with col3:
    weight = st.number_input('Weight (kg)', min_value=10.0, max_value=200.0, step=0.1, value=60.0)
    height = st.number_input('Height (cm)', min_value=50.0, max_value=250.0, step=0.1, value=170.0)
    bmi = round(weight / ((height / 100) ** 2), 2)
    st.markdown(
        f"<div style='background-color: #dff9fb; padding: 10px; border-radius: 8px; text-align: center;'>"
        f"<b>BMI:</b> <span style='font-size: 20px;'>{bmi} kg/m²</span>"
        f"</div>",
        unsafe_allow_html=True
    )

# Sensor Data Input Section
st.header('Step 2: Receive Sensor Data')

# Get the latest sensor data
formatted_time, heart_rate, o2, body_temperature = get_sensor_data(st.session_state.last_valid_data)

# Update last valid data if current data is valid
if all(value is not None for value in [formatted_time, heart_rate, o2, body_temperature]):
    st.session_state.last_valid_data = (formatted_time, heart_rate, o2, body_temperature)

# Display the latest sensor data
formatted_time, heart_rate, o2, body_temperature = st.session_state.last_valid_data

if formatted_time and heart_rate and o2 and body_temperature:
    st.markdown(f"""
    <div style='background-color: #f1f2f6; padding: 20px; border-radius: 8px;'>
        <b>📊 Latest Sensor Data:</b><br>
        <b>Date:</b> {formatted_time}<br>
        <b>Heart Rate:</b> <span style='color: #ff7979;'>{heart_rate} bpm</span><br>
        <b>Oxygen Saturation:</b> <span style='color: #22a6b3;'>{o2} %</span><br>
        <b>Body Temperature:</b> <span style='color: #f0932b;'>{body_temperature} °C</span>
    </div>
    """, unsafe_allow_html=True)

    # Prediction Section
    st.header('Step 3: Risk Prediction')
    risk_status = make_prediction(age, gender, bmi, heart_rate, o2, body_temperature)

    st.markdown(
        f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: {'red' if risk_status == 'High Risk' else 'green'};'>"
        f"Risk Status: {risk_status}</div>",
        unsafe_allow_html=True)
else:
    st.write("❌ Unable to retrieve valid sensor data.")

# Auto refresh every 5 seconds
time.sleep(5)
st.rerun()
