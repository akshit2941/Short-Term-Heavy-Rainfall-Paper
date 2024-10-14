# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC  # LIBSVM

# Load your dataset (ensure 'Rainfall' is the target variable)
data = pd.read_csv('Weather_data.csv')

# Create a new 'Rainfall' column from 'precip_mm' if it doesn't exist
if 'Rainfall' not in data.columns and 'precip_mm' in data.columns:
    data['Rainfall'] = data['precip_mm']
elif 'Rainfall' not in data.columns:
    raise KeyError("'Rainfall' column is missing and 'precip_mm' is also not available. Please check the dataset.")

# Assuming 'Rainfall' is the target variable and binarizing it for classification
data['Rainfall'] = data['Rainfall'].apply(lambda x: 1 if x > 0 else 0)  # Binarizing for classification

# One-hot encoding for categorical features
data = pd.get_dummies(data, drop_first=True)  # Convert categorical variables into dummy/indicator variables

# Features and target
X = data.drop(columns=['Rainfall'])  # Features
y = data['Rainfall']  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the data (important for SVC)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# LIBSVM (SVC)
print("LIBSVM (SVC):")
model_libsvm = SVC(kernel='linear')  # Using linear kernel for LIBSVM
model_libsvm.fit(X_train_scaled, y_train)

# Make predictions
y_pred_libsvm = model_libsvm.predict(X_test_scaled)

# Calculate metrics
accuracy_libsvm = accuracy_score(y_test, y_pred_libsvm)
confusion_libsvm = confusion_matrix(y_test, y_pred_libsvm)

# Output results
print(f'Accuracy: {accuracy_libsvm}')
print('Confusion Matrix:')
print(confusion_libsvm)
