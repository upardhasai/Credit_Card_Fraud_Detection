import pandas as pd

# Load the dataset
df = pd.read_csv("/Users/pardha/Downloads/creditcard.csv")

# Print basic info to confirm successful loading
print("Dataset Loaded Successfully!")
print("\nDataset Info:\n")
print(df.info())
