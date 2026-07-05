import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE  # 🚀 New import

# 1. Load Data
print("🔄 Loading dataset...")
df = pd.read_csv('churn.csv')

# 2. Fix the TotalCharges numeric bug
print("⚙️ Cleaning TotalCharges data type...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# 3. Clean categorical columns and target variable
print("📊 Encoding target and categorical features...")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.drop(['customerID'], axis=1)

X = df.drop('Churn', axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Churn']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Apply SMOTE to balance the training set
print("⚖️ Balancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 6. Train Model on Balanced Data
print("🤖 Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train_balanced, y_train_balanced)

# 7. Evaluate
print("\n📈 Model Evaluation Results (Post-SMOTE):")
y_pred = model.predict(X_test)
print(f"Customer Churn Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))