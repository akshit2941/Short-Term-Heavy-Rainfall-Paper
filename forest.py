# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor  # Random Forest Regressor

# Load your dataset (ensure 'Rainfall' is the target variable)
data = pd.read_csv('Weather_data.csv')

# Create a new 'Rainfall' column from 'precip_mm' if it doesn't exist
if 'Rainfall' not in data.columns and 'precip_mm' in data.columns:
    data['Rainfall'] = data['precip_mm']
elif 'Rainfall' not in data.columns:
    raise KeyError("'Rainfall' column is missing and 'precip_mm' is also not available. Please check the dataset.")

# One-hot encoding for categorical features
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables into dummy/indicator variables

# Assuming 'Rainfall' is the target variable
X = data.drop(columns=['Rainfall'])  # Features
y = data['Rainfall']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Regressor
print("Random Forest Regressor:")
model_rf = RandomForestRegressor()  # Using Regressor for continuous output
model_rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = model_rf.predict(X_test)

# Calculate metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Output results
print(f'MSE: {mse_rf}, R^2: {r2_rf}\n')
