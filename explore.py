import pandas as pd

print("🔄 Loading Telco Churn dataset...")
df = pd.read_csv('churn.csv')

print("\n📋 Dataset Shape (Rows, Columns):")
print(df.shape)

print("\n🔍 First 5 Rows:")
print(df.head())

print("\nℹ️ Data Types and Missing Values Summary:")
print(df.info())