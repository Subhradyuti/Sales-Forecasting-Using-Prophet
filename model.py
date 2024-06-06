import numpy as np
import pandas as pd
from prophet import Prophet
import pickle

# Load and preprocess data
df = pd.read_csv('Global_Superstore2.csv', encoding="ISO-8859-1", parse_dates=['Order Date'])
df['ds'] = df['Order Date']
df['y_orig'] = df['Sales']
df['Sales'] = np.log(df['Sales'])
df['y'] = df['Sales']
df['Sales'].fillna(df['Sales'].mean(), inplace=True)
df.dropna(subset=['Order Date'], inplace=True)

# Initialize and train Prophet model
model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
model.add_seasonality(name='quarterly', period=90.5, fourier_order=8)
model.fit(df)

# Save the model to a file
with open('prophet_model.pkl1', 'wb') as f:
    pickle.dump(model, f)
