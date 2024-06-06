# Sales Forecasting Using Prophet

This repository contains a comprehensive sales forecasting project utilizing Prophet, an open-source forecasting tool developed by Facebook. The project involves data preprocessing, model training, and deployment using Flask and Streamlit, providing an interactive interface for users to generate and visualize future sales predictions.

## Features
- **Data Preprocessing**: Includes scripts for cleaning and transforming sales data, handling missing values, and feature engineering to prepare the dataset for modeling.
- **Model Training**: Implements a Prophet model with custom seasonality components and optimized hyperparameters to accurately forecast sales.
- **Model Deployment**: Provides a Flask API for backend predictions and a Streamlit app for an interactive frontend interface, allowing users to input parameters and view forecast results.
- **User Interface**: Designed with a responsive HTML/CSS frontend for user-friendly interaction, including dynamic display of predictions and visualization of forecasted sales.

## Technologies Used
- **Python**: Core programming language for data processing, model training, and deployment.
- **Prophet**: Forecasting library used to build the time series model.
- **Flask**: Web framework for creating the RESTful API to serve predictions.
- **Streamlit**: Framework for creating the interactive web application.
- **HTML/CSS**: Frontend technologies for building the user interface.
- **Pandas and NumPy**: Libraries for data manipulation and numerical operations.

## Usage

### Data Preprocessing
1. Preprocess the raw sales data by running the `model.py` script.
2. Clean data, handle missing values, and transform features for modeling.

### Model Training
1. Train the Prophet model with custom seasonality and optimized hyperparameters.
2. Save the trained model using pickle for later use.

### Deployment
1. Deploy the trained model using Flask (`app.py`) to provide an API for predictions.
2. Create an interactive frontend using Streamlit (`streamlit_app.py`) to visualize predictions.

### Running the Application
1. Start the Flask server:
   ```bash
   python app.py
2. Access the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
### Frontend Interaction
1. Access the HTML/CSS frontend by navigating to the root URL of the Flask server.
2. Input prediction parameters (number of periods, interval) and submit to get sales forecasts.
## Contributions
Contributions are welcome! Please open an issue or submit a pull request with your proposed changes or enhancements.
