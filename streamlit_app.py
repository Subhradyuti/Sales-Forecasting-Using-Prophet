import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle

# Load the trained model
with open('prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

def main():
    st.title("Time Series Forecasting with Prophet")

    # User inputs
    periods = st.number_input('Number of future periods', min_value=1, max_value=365, value=15)
    interval = st.selectbox('Select interval', options=['D', 'W', 'M', 'Q', 'Y'], index=1)

    if st.button('Predict'):
        try:
            # Generate future dates based on the selected interval
            freq_map = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q', 'Y': 'Y'}
            freq = freq_map.get(interval, 'D')
            future_data = model.make_future_dataframe(periods=periods, freq=freq)

            # Make predictions
            forecast_data = model.predict(future_data)

            # Back transform predictions
            forecast_data['yhat'] = np.exp(forecast_data['yhat'])
            forecast_data['yhat_lower'] = np.exp(forecast_data['yhat_lower'])
            forecast_data['yhat_upper'] = np.exp(forecast_data['yhat_upper'])

            # Display predictions
            predictions = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
            st.write(predictions)
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
