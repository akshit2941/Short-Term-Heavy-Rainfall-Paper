import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('Weather_data.csv')

# Print column names to check for 'Rainfall'
print(data.columns)

# Create a new 'Rainfall' column from 'precip_mm' (if it's not present)
if 'precip_mm' in data.columns:
    data['Rainfall'] = data['precip_mm']
else:
    raise KeyError("'precip_mm' column is missing. Please check the dataset or add precipitation data.")

# Fill missing values in 'Rainfall'
data['Rainfall'].fillna(0, inplace=True)

# Features and labels (use relevant columns from your dataset)
X = data[['temperature_celsius', 'wind_kph', 'humidity', 'cloud', 'pressure_mb']]  # Adjust based on your dataset
y = data['Rainfall']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Function to predict if rainfall is going to happen (yes/no)
def predict_rainfall(weather_data):
    """
    Predict if rainfall will occur based on input weather conditions.
    
    :param weather_data: A list or array of [Temperature, WindSpeed, Humidity, CloudCover, Pressure]
    :return: "Rainfall expected" or "No rainfall expected"
    """
    weather_data_scaled = scaler.transform([weather_data])
    predicted_rainfall = model.predict(weather_data_scaled)[0]
    
    # If predicted rainfall is greater than 0.1 mm, assume rainfall will occur
    if predicted_rainfall > 0.1:
        return f"Rainfall expected ({predicted_rainfall:.2f} mm)"
    else:
        return "No rainfall expected"

# Test Case 1: Predict rainfall for given weather conditions (Rainfall expected)
test_weather_data_1 = [25.0, 15.0, 90, 80, 1008]  # Example: [Temperature, WindSpeed, Humidity, CloudCover, Pressure]
rainfall_prediction_1 = predict_rainfall(test_weather_data_1)
print(f'Test case 1 result: {rainfall_prediction_1}')

# Test Case 2: Predict rainfall for given weather conditions (No rainfall expected)
test_weather_data_2 = [30.0, 10.0, 40, 10, 1015]  # Example: [Temperature, WindSpeed, Humidity, CloudCover, Pressure]
rainfall_prediction_2 = predict_rainfall(test_weather_data_2)w
print(f'Test case 2 result: {rainfall_prediction_2}')

