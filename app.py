import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# 1. Page Configuration (Sets up the web page title and width)
st.set_page_config(page_title="Server Load Forecaster", layout="wide")
st.title("🌩️ Alibaba Cloud: Server Load Forecasting")
st.markdown("Predicting task arrival rates to optimize cloud infrastructure and prevent downtime.")

# 2. Data Loading (The @st.cache_data decorator makes the app load instantly after the first run)
@st.cache_data
def load_data():
    # Only load 500k rows to keep the app lightning fast
    df = pd.read_csv('alibaba_mini.csv')
    df = df[['machine_id', 'start_time']]
    
    # Convert Unix seconds to real time and set as index
    df['start_time'] = pd.to_datetime(df['start_time'], unit='s')
    df.set_index('start_time', inplace=True)
    
    # Resample into hourly task counts using lowercase 'h'
    hourly_data = df.resample('h').size()
    return hourly_data

data = load_data()

# 3. Sidebar for User Interaction
st.sidebar.header("Forecast Settings")
st.sidebar.write("Adjust the slider to see further into the future.")

# This creates a slider that lets the user choose how many hours to predict
forecast_hours = st.sidebar.slider("Hours to Forecast", min_value=1, max_value=48, value=12)

# 4. Train Model & Forecast
st.subheader("Live SARIMA Forecast")
with st.spinner("Training Seasonal AI Model (this might take 5-10 seconds)..."):
    # seasonal_order=(0, 1, 1, 24) forces the AI to map the exact shape of yesterday's spikes
    model = SARIMAX(data, order=(2, 0, 2), seasonal_order=(0, 1, 1, 24))
    fitted_model = model.fit(disp=False)
    
    # Predict the future based on the slider value
    forecast = fitted_model.forecast(steps=forecast_hours)
    
# 5. Visualizing the Results
fig, ax = plt.subplots(figsize=(14, 5))

# We plot the last 50 hours of history in blue to give context
ax.plot(data.index[-50:], data.values[-50:], label="Recent History (Last 50 Hrs)", color="#1f77b4", linewidth=2)

# We plot our future forecast in dashed orange
ax.plot(forecast.index, forecast.values, label="SARIMA Forecast", color="#ff7f0e", linestyle="--", linewidth=2)

# Chart styling
ax.set_title(f"Predicting Server Load for the Next {forecast_hours} Hours", fontsize=14)
ax.set_xlabel("Time (Hourly)", fontsize=12)
ax.set_ylabel("Total Tasks Started", fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle="--", alpha=0.7)

# This physically draws the chart on the web page
st.pyplot(fig)

# 6. Let the user see the raw numbers if they want to
if st.checkbox("Show Raw Forecast Data"):
    st.write(forecast)