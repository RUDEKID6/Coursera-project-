# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset (replace 'rainfall_data.csv' with your file)
data = pd.read_csv('rainfall_data.csv')

# Preprocess dataset (example: convert target label Yes/No to 1/0)
data['RainTomorrow'] = data['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Select features and target variable
X = data.drop(['RainTomorrow'], axis=1)
y = data['RainTomorrow']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Evaluate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
