import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load preprocessed dataset
df = pd.read_csv("dataset/processed_transactions.csv")

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Class"]), df["Class"], test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, "credit_fraud_model.pkl")
print("Model saved successfully.")
