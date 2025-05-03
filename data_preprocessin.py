import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import os

# Ensure the dataset directory exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Load dataset
df = pd.read_csv("/Users/pardha/Downloads/creditcard.csv")

# Handle missing values (Drop rows with NaNs)
df.dropna(inplace=True)

# Verify "Class" column exists
if "Class" not in df.columns:
    raise ValueError("Error: 'Class' column is missing from the dataset!")

# Separate features and target variable
X = df.drop(columns=["Class"])
y = df["Class"]

# Check initial class distribution
print("Original Class Distribution:", Counter(y))

# Apply SMOTE (Oversampling) for class imbalance correction
smote = SMOTE(sampling_strategy="minority", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Class Distribution After SMOTE:", Counter(y_resampled))

# Calculate safe undersampling ratio dynamically
fraud_samples = Counter(y_resampled)[1]
genuine_samples = Counter(y_resampled)[0]

# Adjust sampling_strategy to avoid excessive removal
safe_ratio = max(0.9, fraud_samples / genuine_samples)  # Ensures at least 90% majority class remains

# Apply safer undersampling strategy
undersampler = RandomUnderSampler(sampling_strategy=safe_ratio, random_state=42)
X_under, y_under = undersampler.fit_resample(X_resampled, y_resampled)
print("Class Distribution After Undersampling:", Counter(y_under))

# Save preprocessed data for model training
processed_data = pd.concat([pd.DataFrame(X_under), pd.DataFrame(y_under, columns=["Class"])], axis=1)
processed_data.to_csv("dataset/processed_transactions.csv", index=False)

print("✅ Data Preprocessing Complete! Saved processed_transactions.csv")
