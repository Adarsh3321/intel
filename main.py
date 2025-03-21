import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer



df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display the first few rows
# print(df.head())

# Basic info about the dataset
# print(df.info())

# Summary statistics for numerical columns
# print(df.describe())

# Checking missing values
# print(df.isnull().sum())

# Handling missing values (drop or fill)
df = df.dropna()  # Drop rows with missing values

# Convert categorical columns to dummy/indicator variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Check the data after encoding
# print(df_encoded.head())

# Separate features and target variable
X = df_encoded.drop("Churn_Yes", axis=1)  # Assuming "Churn_Yes" is the target variable
y = df_encoded["Churn_Yes"]  # Churn_Yes is typically 1 if churned, 0 otherwise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# from sklearn.preprocessing import StandardScaler

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# from sklearn.ensemble import RandomForestClassifier

# Initialize the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)


# Predict on the test set
y_pred = rf_model.predict(X_test_scaled)

# print(y_pred)


# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Accuracy score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)



# import seaborn as sns
# import matplotlib.pyplot as plt

# Visualize confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


import joblib

# Save the trained model
joblib.dump(rf_model, "churn_prediction_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")
