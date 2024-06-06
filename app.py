from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle

app = Flask(__name__)

# Load the trained model
with open('prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the number of future periods and the selected interval
        periods = int(request.form['periods'])
        interval = request.form['interval']
        
        # Generate future dates based on the selected interval
        freq_map = {'D': 'D', 'W': 'W', 'M': 'M', 'Q': 'Q', 'Y': 'Y'}
        freq = freq_map.get(interval, 'W')  
        future_data = model.make_future_dataframe(periods=periods, freq=freq)
        
        # Make predictions
        forecast_data = model.predict(future_data)
        
        # Back transform predictions
        forecast_data['yhat'] = np.exp(forecast_data['yhat'])
        forecast_data['yhat_lower'] = np.exp(forecast_data['yhat_lower'])
        forecast_data['yhat_upper'] = np.exp(forecast_data['yhat_upper'])
        
        # Convert predictions to dictionary
        predictions = forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict(orient='records')
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
