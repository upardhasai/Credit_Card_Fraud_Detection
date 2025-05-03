import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# Load trained model & dataset
model = joblib.load("credit_fraud_model.pkl")
df = pd.read_csv("dataset/processed_transactions.csv")  # ✅ Load preprocessed dataset

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Class"]), df["Class"], test_size=0.2, random_state=42)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC Curve

# Compute Precision, Recall, F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Fraud Detection Confusion Matrix")
plt.show()

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Fraud Detection")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Fraud Detection")
plt.legend()
plt.show()

importances = model.feature_importances_
features = X_train.columns
plt.figure(figsize=(12, 6))
sns.barplot(x=features, y=importances, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Feature Importance in Credit Card Fraud Detection")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.show()

if "Amount" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, color="green", label="Non-Fraud", kde=True)
    sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, color="red", label="Fraud", kde=True)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Count")
    plt.title("Distribution of Fraud vs. Non-Fraud Transaction Amounts")
    plt.legend()
    plt.show()
